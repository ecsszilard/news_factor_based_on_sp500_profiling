import logging
import numpy as np
import tensorflow as tf
import mlflow
import os
import warnings

# Suppress TensorFlow/Transformers deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*TensorFlow and JAX classes are deprecated.*')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

logger = logging.getLogger("AdvancedNewsFactor.AttentionBasedNewsFactorModel")
layers = tf.keras.layers
models = tf.keras.models
optimizers = tf.keras.optimizers
regularizers = tf.keras.regularizers
callbacks = tf.keras.callbacks
metrics = tf.keras.metrics
losses = tf.keras.losses

class DirectionalAccuracy(metrics.Metric):
    """
    Measures the model's accuracy in predicting the correct direction of change (delta).
    Compares the sign of predicted_delta with the sign of actual_delta.
    """
    def __init__(self, name='directional_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct_directions = self.add_weight(name='correct_directions', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        y_true: [batch, N, N, 3] - [actual_corr, baseline_corr, affected_companies_mask]
        y_pred: [batch, N, N, 2] - [μ, log(σ²)]
        """
        actual_corr = y_true[..., 0]
        baseline_corr = y_true[..., 1]
        actual_delta = actual_corr - baseline_corr

        predicted_mu = y_pred[..., 0]  # Model predicts Z_utan
        predicted_delta = predicted_mu - baseline_corr

        actual_sign = tf.math.sign(actual_delta)
        predicted_sign = tf.math.sign(predicted_delta)

        non_zero_mask = tf.math.not_equal(actual_sign, 0)

        correct = tf.equal(actual_sign, predicted_sign)
        correct_non_zero = tf.logical_and(correct, non_zero_mask)

        self.correct_directions.assign_add(tf.reduce_sum(tf.cast(correct_non_zero, tf.float32)))
        self.total_samples.assign_add(tf.reduce_sum(tf.cast(non_zero_mask, tf.float32)))

    def result(self):
        return self.correct_directions / (self.total_samples + 1e-7)

    def reset_state(self):
        self.correct_directions.assign(0)
        self.total_samples.assign(0)


class WeightedNegativeLogLikelihoodLoss(losses.Loss):
    """
    Weighted Negative Log-Likelihood Loss
    
    Weights pairs by:
    1. Involvement of ACTUALLY AFFECTED companies (from affected_companies mask)
    2. Magnitude of actual correlation change
    
    This forces the model to prioritize learning pairs that:
    - Are directly affected by the news (not just focus company)
    - Show significant real changes
    """
    def __init__(self, min_sigma=1e-4, affected_weight=3.0, magnitude_weight=2.0, **kwargs):
        super().__init__(**kwargs)
        self.min_sigma = min_sigma
        self.affected_weight = affected_weight  # Weight for pairs involving affected companies
        self.magnitude_weight = magnitude_weight  # Weight for high-change pairs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'min_sigma': self.min_sigma,
            'affected_weight': self.affected_weight,
            'magnitude_weight': self.magnitude_weight,
        })
        return config
    
    def call(self, y_true, y_pred):
        """
        y_true: [batch, N, N, 3] - [actual_corr, baseline_corr, affected_companies_mask]
        y_pred: [batch, N, N, 2] - [μ, log(σ²)]
        """
        # Extract components
        actual_corr = y_true[..., 0]  # [batch, N, N]
        baseline_corr = y_true[..., 1]  # [batch, N, N]
        affected_mask = tf.cast(y_true[..., 2], tf.float32)  # [batch, N, N] - 1 if pair involves affected company
        
        mu = y_pred[..., 0]  # [batch, N, N]
        log_var = y_pred[..., 1]  # [batch, N, N]
        log_var = tf.clip_by_value(log_var, -10.0, 10.0) # to avoid underflow and overflow of exp(), which leads to an explosion of loss
        
        # Ensure numerical stability
        var = tf.exp(log_var) + self.min_sigma
        
        # Standard NLL
        squared_error = tf.square(actual_corr - mu)
        nll = 0.5 * (squared_error / var + log_var)
        
        # WEIGHT 1: Affected company involvement
        # affected_mask is pre-computed: 1.0 if either i or j is in affected_companies, else 0.0
        affected_weights = 1.0 + (self.affected_weight - 1.0) * affected_mask
        # Result: affected pairs get self.affected_weight (e.g., 3.0), others get 1.0
        
        # WEIGHT 2: Magnitude of actual change
        actual_change = tf.abs(actual_corr - baseline_corr)
        magnitude_weights = 1.0 + self.magnitude_weight * actual_change
        
        # Combined weights
        total_weights = affected_weights * magnitude_weights
        # Weighted NLL
        weighted_nll = nll * total_weights
        return tf.reduce_mean(weighted_nll)

class CalibrationMetric(metrics.Metric):
    """
    Measures how well the predicted uncertainty (σ) matches actual errors
    Good calibration: High σ → High error, Low σ → Low error
    """
    def __init__(self, num_bins=10, **kwargs):
        super().__init__(**kwargs)
        self.num_bins = num_bins
        self.bin_errors = self.add_weight(
            name='bin_errors', 
            shape=(num_bins,), 
            initializer='zeros'
        )
        self.bin_counts = self.add_weight(
            name='bin_counts', 
            shape=(num_bins,), 
            initializer='zeros'
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({'num_bins': self.num_bins})
        return config
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Extract actual correlations and predictions (mean, log variance)
        actual_corr = y_true[..., 0]
        mu = y_pred[..., 0]
        log_var = y_pred[..., 1]
        sigma = tf.sqrt(tf.exp(log_var) + 1e-6)  # Calculate standard deviation
        
        # Calculate the absolute errors between actual and predicted values
        errors = tf.abs(actual_corr - mu)
        
        # Flatten the tensors for binning
        sigma_flat = tf.reshape(sigma, [-1])
        errors_flat = tf.reshape(errors, [-1])
        
        # --- Vectorized binning with accumulation ---
        
        # Calculate bin width and normalize sigma values
        sigma_max = tf.reduce_max(sigma_flat)
        sigma_min = tf.reduce_min(sigma_flat)
        bin_width = (sigma_max - sigma_min) / tf.cast(self.num_bins, tf.float32)
        normalized_sigmas = (sigma_flat - sigma_min) / (bin_width + 1e-7)
        
        # Compute bin indices and clamp them to the range [0, num_bins-1]
        bin_indices = tf.cast(normalized_sigmas, tf.int32)
        bin_indices = tf.clip_by_value(bin_indices, 0, self.num_bins - 1)
        
        # Use unsorted_segment_sum for proper accumulation
        errors_to_add = tf.math.unsorted_segment_sum(errors_flat, bin_indices, num_segments=self.num_bins)
        counts_to_add = tf.math.unsorted_segment_sum(tf.ones_like(errors_flat), bin_indices, num_segments=self.num_bins)
        
        # Update bin errors and counts
        self.bin_errors.assign_add(errors_to_add)
        self.bin_counts.assign_add(counts_to_add)
    
    def result(self):
        avg_errors = self.bin_errors / (self.bin_counts + 1e-7)
        return tf.reduce_mean(tf.abs(avg_errors))
    
    def reset_state(self):
        self.bin_errors.assign(tf.zeros_like(self.bin_errors))
        self.bin_counts.assign(tf.zeros_like(self.bin_counts))

class OptimizedAttentionLayer(layers.Layer):
    def __init__(self, num_heads, key_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.attention = None
        self.layer_norm = None
        self.dropout = None
        
        # Scaled attention with learnable temperature
        self.temperature = self.add_weight(
            shape=(), initializer='ones', trainable=True, name='temperature'
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout_rate': self.dropout_rate
        })
        return config
    
    def build(self, input_shape):
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate
        )
        self.layer_norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(self.dropout_rate)
        super().build(input_shape)
    
    def call(self, query, key, value, training=None):
        attn_out = self.attention(query, key, value, training=training)
        attn_out = attn_out * self.temperature  # Temperature scaled attention
        return self.layer_norm(query + self.dropout(attn_out, training=training)) # Residual connection

class AttentionBasedNewsFactorModel:
    """
    Enhanced multi-task learning model with:
    1. Residual Learning for correlation changes
    2. Probabilistic predictions (μ, σ²) for aleatoric uncertainty
    3. Weighted loss focusing on important pairs
    """

    def __init__(self, tokenizer, bert_model, num_companies, max_keywords=128, keyword_dim=256, company_dim=128, latent_dim=256):
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.num_companies = num_companies
        self.max_keywords = max_keywords
        self.keyword_dim = keyword_dim
        self.company_dim = company_dim
        self.latent_dim = latent_dim

        # Simple BERT embedding cache (hash-based)
        self._bert_cache = {}

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("news_correlation_trading_probabilistic")
        
        self.model = self.build_model()
        
        # Custom optimizer with gradient clipping
        optimizer = optimizers.AdamW(
            learning_rate=optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=1e-3,
                first_decay_steps=1000,
                t_mul=2.0,
                m_mul=0.9,
                alpha=0.1
            ),
            weight_decay=1e-4,
            clipnorm=1.0
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss={
                'correlation_changes_probabilistic': WeightedNegativeLogLikelihoodLoss(
                    affected_weight=3.0,  # 3x weight for pairs with affected companies
                    magnitude_weight=2.0  # +2x weight per unit of actual change
                ),
                'price_deviations': losses.Huber(delta=1.0),
                'news_reconstruction': 'mae'
            },
            loss_weights={
                'correlation_changes_probabilistic': 4.0,
                'price_deviations': 0.5,
                'news_reconstruction': 0.5
            },
            metrics={
                'correlation_changes_probabilistic': [CalibrationMetric(), DirectionalAccuracy()],
                'price_deviations': ['mae', 'mse'],
                'news_reconstruction': ['mae']
            }
        )

    def build_model(self):
        """Multi-task model with Residual Learning: baseline + news-induced delta.

        Hybrid news encoder:
        - smooth_global: robust global pooling (avg + max) -> resists small embedding jitter
        - local_edge: pointwise Dense + sigmoid gate -> detects sharp local semantic edges,
                    GlobalMaxPooling extracts the strongest local signal
        - combined -> single moderate Dense -> shared_news_rep (latent)
        """

        # -----------------------------
        # 1. Inputs
        # -----------------------------
        keyword_input = layers.Input(shape=(self.max_keywords,), name='keywords')
        
        # Fisher-z transformed baseline correlation matrix (square NxN)
        baseline_correlation_input = layers.Input(
            shape=(self.num_companies, self.num_companies),
            name='baseline_correlation_input',
            dtype='float32'
        )

        # -----------------------------
        # 2. Keyword encoder (global, not company-specific)
        # -----------------------------
        # Token embedding for keywords (sequence of tokens per news item)
        keyword_embedding = layers.Embedding(
            input_dim=len(self.tokenizer),
            output_dim=self.keyword_dim,
            mask_zero=True,
            embeddings_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name='keyword_embeddings'
        )(keyword_input)  # shape: [batch, seq_len, keyword_dim]

        # Self-attention over keywords (returns [batch, seq_len, keyword_dim])
        attn_keywords = OptimizedAttentionLayer(
            num_heads=8,
            key_dim=max(1, self.keyword_dim // 8),
            dropout_rate=0.2,
            name='keyword_attention'
        )(keyword_embedding, keyword_embedding, keyword_embedding)

        # Residual + LayerNorm: stable per-token representations
        # processed shape: [batch, seq_len, keyword_dim]
        keyword_latent = layers.LayerNormalization(name='keywords_layer_norm')(attn_keywords + keyword_embedding)

        # -----------------------------
        # 3. News representation aggregation
        # -----------------------------
        # Average + Max pooling across tokens -> stable global descriptor
        news_features = layers.Concatenate(name='news_smooth_global')([
            layers.GlobalAveragePooling1D(name='keywords_global_avg')(keyword_latent),
            layers.GlobalMaxPooling1D(name='keywords_global_max')(keyword_latent)
        ]) # [batch, 2*keyword_dim]

        # Optionally reduce dimensionality of smooth branch (small projection)
        smooth_global = layers.Dense(
            self.latent_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(1e-6, 1e-5),
            name='smooth_global_proj'
        )(news_features)  # [batch, latent_dim]

        # --- Local edge branch: detect sharp local semantic flips ---
        # 1D convolutional layer is able to examine n-grams (e.g. kernel 3 = 3 words) at once, making it much more robust in recognizing "semantic edges" and less prone to noise.
        local_conv = layers.Conv1D(
            filters=self.latent_dim,
            kernel_size=3,
            padding='same',  # Megtartja a szekvencia hosszát
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(1e-6, 1e-5),
            name='local_conv_bank'
        )(keyword_latent)  # [batch, seq_len, latent_dim]

        # Gating mechanism: sigmoid gate computed from processed tokens, this enables the network to selectively amplify or silence local signals.
        local_gate = layers.Dense(
            self.latent_dim,
            activation='sigmoid',
            name='local_gate'
        )(keyword_latent)  # [batch, seq_len, latent_dim]

        # Elementwise multiply: gated local features
        gated_local = layers.Multiply(name='gated_local')([local_conv, local_gate])  # [batch, seq_len, latent_dim]

        # Global max over timesteps picks the strongest "edge" activation per latent channel
        local_edge = layers.GlobalMaxPooling1D(name='local_edge_pool')(gated_local)  # [batch, latent_dim]

        # --- Combine smooth and local branches ---
        combined = layers.Concatenate(name='news_combined')([smooth_global, local_edge])  # [batch, 2*latent_dim]

        # Single moderate Dense to produce final shared news representation
        # Keep this layer intentionally small/cautious to avoid creating "bumpiness".
        shared_news_rep = layers.Dense(
            self.latent_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name='shared_news_representation'
        )(combined)  # [batch, latent_dim]

        # Use LayerNorm (not BatchNorm) and small Dropout for stability without erasing edges
        shared_news_rep = layers.LayerNormalization(name='shared_news_layernorm')(shared_news_rep)
        shared_news_rep = layers.Dropout(0.2, name='shared_news_dropout')(shared_news_rep)

        # -----------------------------
        # 4. Company embeddings (for pairwise features only)
        # -----------------------------
        # Create learnable company embeddings for ALL companies
        company_embedding_layer = layers.Embedding(
            input_dim=self.num_companies,
            output_dim=self.company_dim,
            embeddings_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name='company_embeddings'
        )
        
        # Get embeddings for all companies repeating across batch dimension
        batch_size = tf.shape(keyword_input)[0]
        all_company_indices = tf.range(self.num_companies, dtype=tf.int32)
        all_company_indices = tf.expand_dims(all_company_indices, 0)
        all_company_indices = tf.tile(all_company_indices, [batch_size, 1])  # [batch, N]
        all_company_embeddings = company_embedding_layer(all_company_indices)  # [batch, N, company_dim]

        # -----------------------------
        # 5. Pairwise correlation features (all company pairs)
        # -----------------------------
        # Create pairwise features: for each pair (i,j), concatenate [company_i, company_j, news]
        # company_i: [batch, N, 1, company_dim] -> tile to [batch, N, N, company_dim]
        company_i = layers.Lambda(lambda x: tf.expand_dims(x, 2), name='company_i_expand')(all_company_embeddings)
        company_i = layers.Lambda(lambda x: tf.tile(x, [1, 1, self.num_companies, 1]), name='company_i_tile')(company_i)

        # company_j: [batch, 1, N, company_dim] -> tile to [batch, N, N, company_dim]
        company_j = layers.Lambda(lambda x: tf.expand_dims(x, 1), name='company_j_expand')(all_company_embeddings)
        company_j = layers.Lambda(lambda x: tf.tile(x, [1, self.num_companies, 1, 1]), name='company_j_tile')(company_j)

        # shared_news_rep: [batch, latent] -> expand to [batch, N, N, latent]
        news_expanded = layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, 1), 1), name='news_expand')(shared_news_rep)  # [batch,1,1,latent]
        news_for_pairs = layers.Lambda(
            lambda x: tf.tile(x, [1, self.num_companies, self.num_companies, 1]),
            name='news_tile_for_pairs'
        )(news_expanded)  # [batch, N, N, latent]

        # Concatenate company_i, company_j, news_for_pairs along the last axis
        pairwise_features = layers.Concatenate(axis=-1, name='pairwise_concat')([company_i, company_j, news_for_pairs])
        # shape: [batch, N, N, 2*company_dim + latent]

        # -----------------------------
        # 6. PROBABILISTIC RESIDUAL LEARNING: Predict correlation CHANGE (delta)
        # -----------------------------
        # Flatten baseline correlation into a channel for concatenation
        baseline_flat = layers.Reshape((self.num_companies, self.num_companies, 1), name='baseline_reshape')(baseline_correlation_input)
        
        # Merge pairwise features with baseline correlation
        merged_features = layers.Concatenate(axis=-1, name='merged_features_with_baseline')([
            pairwise_features,
            baseline_flat
        ])  # shape: [batch, N, N, 2*company_dim + latent + 1]
        
        # Shared hidden layer for both μ and log-variance predictions
        delta_hidden = layers.Dense(
            128,
            activation='tanh',
            kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name='delta_hidden_shared'
        )(merged_features)
        delta_hidden = layers.Dropout(0.45, name='delta_hidden_dropout')(delta_hidden)
        
        # TWO SEPARATE DENSE LAYERS for different predictions for Mean (μ) and Log-Variance (log σ²)
        
        # μ: expected correlation change
        delta_mu_raw = layers.Dense(
            1,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name='correlation_delta_mu'
        )(delta_hidden)  # [batch, N, N, 1]
        
        delta_mu = layers.Reshape(
            (self.num_companies, self.num_companies),
            name='delta_mu_reshaped'
        )(delta_mu_raw)  # [batch, N, N]

        # log(σ²): aleatoric uncertainty
        delta_log_var_raw = layers.Dense(
            1,
            activation='linear',
            kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name='correlation_delta_log_var'
        )(delta_hidden)  # [batch, N, N, 1]

        delta_log_var = layers.Reshape(
            (self.num_companies, self.num_companies),
            name='delta_log_var_reshaped'
        )(delta_log_var_raw)  # [batch, N, N]

        # Ensure symmetry for μ and log_var (average with transpose)
        delta_mu_symmetric = layers.Lambda(
            lambda m: 0.5 * (m + tf.transpose(m, perm=[0, 2, 1])),
            name='correlation_delta_mu_symmetric'
        )(delta_mu)

        delta_log_var_symmetric = layers.Lambda(
            lambda m: 0.5 * (m + tf.transpose(m, perm=[0, 2, 1])),
            name='correlation_delta_log_var_symmetric'
        )(delta_log_var)

        # Residual connection: baseline + μ_delta -> predicted mean correlation
        correlation_mean = layers.Add(name='correlation_mean_prediction')([
            baseline_correlation_input,
            delta_mu_symmetric
        ])  # [batch, N, N]

        # Stack mean and log-variance for downstream probabilistic loss
        correlation_changes_prob = layers.Lambda(
            lambda tensors: tf.stack(tensors, axis=-1),
            name='correlation_changes_probabilistic'
        )([correlation_mean, delta_log_var_symmetric])  # [batch, N, N, 2]

        # -----------------------------
        # 7. Price deviations (auxiliary task)
        # -----------------------------
        # For price deviation, use shared_news_rep combined with flattened company embeddings
        price_dev_input = layers.Concatenate(name='price_dev_input')([shared_news_rep, layers.Flatten()(all_company_embeddings)])
        price_dev_hidden = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(1e-5), name='price_dev_hidden')(price_dev_input)
        price_dev_hidden = layers.Dropout(0.3, name='price_dev_dropout')(price_dev_hidden)
        price_deviations = layers.Dense(
            self.num_companies,
            activation='tanh',
            name='price_deviations'
        )(price_dev_hidden)  # [batch, N]

        # -----------------------------
        # 8. News reconstruction (auxiliary task)
        # -----------------------------
        # Encourage shared_news_rep to preserve reconstructible information
        recon_out = layers.Dense(
            self.latent_dim,
            activation='tanh',
            name='news_reconstruction'
        )(shared_news_rep)  # [batch, latent_dim]

        # -----------------------------
        # Build and return the Model
        # -----------------------------
        return models.Model(
            inputs=[keyword_input, baseline_correlation_input],
            outputs=[correlation_changes_prob, price_deviations, recon_out],
            name='NewsCorrelationProbabilisticResidualModel'
        )


    def train(self, training_data, validation_data=None, epochs=100, batch_size=32):
        """Train the probabilistic residual correlation model"""
        
        # Create output directory for models
        os.makedirs("./saved_models", exist_ok=True)
        
        with mlflow.start_run():
            mlflow.log_params({
                'epochs': epochs,
                'batch_size': batch_size,
                'keyword_dim': self.keyword_dim,
                'company_dim': self.company_dim,
                'latent_dim': self.latent_dim,
                'max_keywords': self.max_keywords,
                'architecture': 'probabilistic_residual_learning'
            })
            
            class MLflowCallback(callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if logs:
                        for metric, value in logs.items():
                            mlflow.log_metric(metric, value, step=epoch)

            # Train data preparation with augmented targets
            
            X_keywords = np.array(training_data['keyword_sequence'])  # Shape: (batch, seq_len)
            X_baseline_correlation = np.array(training_data['baseline_correlation'])
            
            # Augment y_correlation_changes with baseline and affected_companies_mask for weighted loss
            y_correlation_actual = np.array(training_data['correlation_changes'])
            
            # Get affected companies mask from training data
            affected_companies_masks = np.array(training_data['affected_companies_mask'])  # [batch, N, N]
            
            # Create augmented target: [actual, baseline, affected_mask]
            # Shape: [batch, N, N, 3]
            y_correlation_augmented = np.stack([
                y_correlation_actual,
                X_baseline_correlation,
                affected_companies_masks
            ], axis=-1)
            
            y_price_deviations = np.array(training_data['price_deviations'])
            y_news_targets = np.array(training_data['news_targets'])

            # Validation data preparation
            validation_data_prepared = None
            if validation_data:
                val_X_keywords = np.array(validation_data['keyword_sequence'])              
                val_X_baseline_correlation = np.array(validation_data['baseline_correlation'])
                val_y_correlation_actual = np.array(validation_data['correlation_changes'])
                val_affected_companies_masks = np.array(validation_data['affected_companies_mask'])
                
                val_y_correlation_augmented = np.stack([
                    val_y_correlation_actual,
                    val_X_baseline_correlation,
                    val_affected_companies_masks
                ], axis=-1)
                
                val_y_price_deviations = np.array(validation_data['price_deviations'])
                val_y_news_targets = np.array(validation_data['news_targets'])
                
                validation_data_prepared = (
                    [val_X_keywords, val_X_baseline_correlation],
                    [val_y_correlation_augmented, val_y_price_deviations, val_y_news_targets]
                )
            
            # ---- Train ----
            history = self.model.fit(
                [X_keywords, X_baseline_correlation],
                [y_correlation_augmented, y_price_deviations, y_news_targets],
                validation_data=validation_data_prepared,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[
                    callbacks.EarlyStopping(
                        monitor='val_loss' if validation_data else 'loss',
                        patience=15, 
                        restore_best_weights=True
                    ),
                    callbacks.ReduceLROnPlateau(
                        monitor='val_loss' if validation_data else 'loss', 
                        factor=0.5, 
                        patience=8, 
                        min_lr=1e-6
                    ),
                    MLflowCallback()
                ],
                verbose=1
            )
            
            # Save model manually to avoid MLflow serialization issues
            try:
                model_path = "./saved_models/news_correlation_model.keras"
                self.model.save(model_path, save_format='keras')
                
                # Log only the path as a parameter (not the model itself)
                mlflow.log_param('model_saved_to', model_path)
                logger.info("✅ Model saved successfully to %s", model_path)
                
            except Exception as e:
                logger.warning("Could not save full model: %s", str(e))
                # Fallback: save weights only
                weights_path = "./saved_models/model_weights.h5"
                self.model.save_weights(weights_path)
                mlflow.log_param('weights_saved_to', weights_path)
                logger.info("⚠️ Only weights saved to %s", weights_path)
            
            if validation_data:
                mlflow.log_metric('best_val_loss', min(history.history['val_loss']))
            else:
                mlflow.log_metric('best_train_loss', min(history.history['loss']))     
        return history

    def predict_with_uncertainty(self, text, baseline_correlation, n_samples: int = 10):
        """
        Monte Carlo Dropout uncertainty + reconstruction-based confidence.

        Args:
            keywords: Tokenized keywords
            baseline_correlation: Baseline correlation matrix
            news_target_embedding: Ground truth BERT embedding of the news (for reconstruction error)
            n_samples: Number of MC Dropout samples (default: 10)

        Returns:
            Dict with mean, std (total), epistemic_std, aleatoric_std, and confidence scores
        """

        # ----------------------------
        # 1) Ensure correct shapes
        # ----------------------------
        # keywords: [batch, max_keywords]
        keywords = self.prepare_keyword_sequence(text)
        if keywords.ndim == 1:
            keywords = keywords[None, :]

        # news target embedding: [1, latent_dim]
        news_target_embedding = self.get_bert_embedding(text)
        if news_target_embedding.ndim == 1:
            news_target_embedding = news_target_embedding.reshape(1, -1)
        
        # baseline: must be [batch, N, N]
        if baseline_correlation.ndim == 2:
            baseline_correlation = baseline_correlation[None, ...]

        # ----------------------------
        # 2) Monte Carlo Dropout: multiple forward passes with dropout enabled
        # ----------------------------
        mc_mu, mc_logvar = [], []
        keywords_tf = tf.convert_to_tensor(keywords, dtype=tf.int32)
        baseline_tf = tf.convert_to_tensor(baseline_correlation, dtype=tf.float32)
        for _ in range(n_samples):
            # Use __call__() instead of predict() to respect training=True, this ensures Dropout layers are active during inference
            pred = self.model.call([keywords_tf, baseline_tf], training=True)
            
            mu_sample = pred[0][..., 0].numpy()
            logvar_sample = pred[0][..., 1].numpy()

            mc_mu.append(mu_sample)
            mc_logvar.append(logvar_sample)
        
        mc_mu = np.array(mc_mu)
        mc_logvar = np.array(mc_logvar)

        # ----------------------------
        # 3) Uncertainty decomposition
        # ----------------------------
        epistemic_var = np.var(mc_mu, axis=0) # Epistemic uncertainty: variance across MC samples
        aleatoric_var = np.mean(np.exp(mc_logvar) + 1e-6, axis=0) # Aleatoric uncertainty: average of predicted variances

        sigma_epistemic = np.sqrt(epistemic_var)
        sigma_aleatoric = np.sqrt(aleatoric_var)
        sigma_total = np.sqrt(epistemic_var + aleatoric_var)

        # ----------------------------
        # 4) Reconstruction error (using predict for final deterministic output)
        # ----------------------------
        final_pred = self.model.predict([keywords, baseline_correlation], verbose=0)
        recon_prediction = final_pred[2]  # shape [1, latent_dim]

        min_len = min(recon_prediction.shape[-1], news_target_embedding.shape[-1])
        recon_error = np.mean(np.abs(news_target_embedding[..., :min_len] - recon_prediction[..., :min_len]))

        # ----------------------------
        # 5) Confidence scores
        # ----------------------------
        recon_conf = np.exp(-recon_error)
        unc_conf = 1.0 / (1.0 + sigma_total.mean())

        return {
            "mean": np.mean(mc_mu, axis=0),
            "std": sigma_total,
            "epistemic_std": sigma_epistemic,
            "aleatoric_std": sigma_aleatoric,
            "total_confidence": recon_conf * unc_conf,
            "recon_confidence": recon_conf,
            "uncertainty_confidence": unc_conf,
            "price_deviations": final_pred[1],
            "reconstruction_error": recon_error,
        }
    
    def prepare_keyword_sequence(self, text: str) -> np.ndarray:
        """
        Tokenize text using the model's tokenizer

        Args:
            text (str): The text to be tokenized.
    
        Returns:
            A numpy array containing the tokenized input_ids.
        """
        return self.tokenizer(
            text,
            truncation=True, 
            padding='max_length',
            max_length=self.max_keywords, 
            return_tensors="np"
        )["input_ids"][0]
    
    def get_bert_embedding(self, text: str) -> np.ndarray:
        """
        Get BERT embedding with simple hash-based caching
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector [latent_dim]
        """
        # Simple cache check
        text_key = hash(text.strip().lower())
        if text_key in self._bert_cache:
            return self._bert_cache[text_key].copy()

        # Tokenize to numpy → convert to TF tensors
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_keywords,
            return_tensors="tf"
        )

        # Forward pass (classifier token embedding)
        embedding_full = self.bert_model(inputs).last_hidden_state[:, 0, :].numpy()[0]  # shape (768,)
        embedding_sliced = embedding_full[:self.latent_dim] # shape (256,)

        # Cache the SLICED result
        self._bert_cache[text_key] = embedding_sliced.copy()
        return embedding_sliced