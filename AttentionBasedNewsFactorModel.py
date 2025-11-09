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

    def __init__(self, tokenizer, num_companies, max_keywords, keyword_dim=256, company_dim=128, latent_dim=128):
        self.tokenizer = tokenizer
        self.num_companies = num_companies
        self.max_keywords = max_keywords
        self.keyword_dim = keyword_dim
        self.company_dim = company_dim
        self.latent_dim = latent_dim

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
        """Multi-task model with Residual Learning: baseline + news-induced delta"""

        # -----------------------------
        # 1. Inputs
        # -----------------------------
        keyword_input = layers.Input(shape=(self.max_keywords,), name='keywords')
        
        # Fisher-z transformed baseline correlation matrix
        baseline_correlation_input = layers.Input(
            shape=(self.num_companies, self.num_companies),
            name='baseline_correlation_input',
            dtype='float32'
        )

        # -----------------------------
        # 2. Keyword encoder (global, not company-specific)
        # -----------------------------
        keyword_embedding = layers.Embedding(
            input_dim=len(self.tokenizer.vocab),
            output_dim=self.keyword_dim,
            mask_zero=True,
            embeddings_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name='keyword_embeddings'
        )(keyword_input)

        attn_keywords = OptimizedAttentionLayer(
            num_heads=8,
            key_dim=max(1, self.keyword_dim // 8),
            dropout_rate=0.1,
            name='keyword_attention'
        )(keyword_embedding, keyword_embedding, keyword_embedding)

        keyword_latent = layers.Dense(
            self.latent_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name='keyword_transform'
        )(layers.LayerNormalization()(attn_keywords + keyword_embedding))
        keyword_latent = layers.Dropout(0.2)(keyword_latent)

        # -----------------------------
        # 3. News representation aggregation (global context)
        # -----------------------------
        news_features = layers.Concatenate()([
            layers.GlobalAveragePooling1D()(keyword_latent),
            layers.GlobalMaxPooling1D()(keyword_latent)
        ])

        news_features = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4)
        )(news_features)
        news_features = layers.BatchNormalization()(news_features)
        news_features = layers.Dropout(0.25)(news_features)

        shared_news_rep = layers.Dense(
            self.latent_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name='shared_news_representation'
        )(news_features)
        shared_news_rep = layers.BatchNormalization()(shared_news_rep)
        shared_news_rep = layers.Dropout(0.2)(shared_news_rep)

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
        
        # Get embeddings for all companies
        batch_size = tf.shape(keyword_input)[0]
        all_company_indices = tf.range(self.num_companies, dtype=tf.int32)
        all_company_indices = tf.expand_dims(all_company_indices, 0)
        all_company_indices = tf.tile(all_company_indices, [batch_size, 1])
        all_company_embeddings = company_embedding_layer(all_company_indices)

        # -----------------------------
        # 5. Pairwise correlation features (all company pairs)
        # -----------------------------
        # Create pairwise features: for each pair (i,j), concatenate [company_i, company_j, news]
        
        # Company i: [batch, N, 1, company_dim] -> [batch, N, N, company_dim]
        company_i = layers.Lambda(lambda x: tf.expand_dims(x, 2))(all_company_embeddings)
        company_i = layers.Lambda(lambda x: tf.tile(x, [1, 1, self.num_companies, 1]))(company_i)
        
        # Company j: [batch, 1, N, company_dim] -> [batch, N, N, company_dim]
        company_j = layers.Lambda(lambda x: tf.expand_dims(x, 1))(all_company_embeddings)
        company_j = layers.Lambda(lambda x: tf.tile(x, [1, self.num_companies, 1, 1]))(company_j)
        
        # News: [batch, latent] -> [batch, N, N, latent]
        news_expanded = layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, 1), 1))(shared_news_rep)  # [batch, 1, 1, latent]
        news_for_pairs = layers.Lambda(lambda x: tf.tile(x, [1, self.num_companies, self.num_companies, 1]))(news_expanded)  # [batch, N, N, latent]
        
        # Concatenate all pairwise features
        pairwise_features = layers.Concatenate(axis=-1)([company_i, company_j, news_for_pairs])  # [batch, N, N, 2*company_dim + latent]

        # -----------------------------
        # 6. PROBABILISTIC RESIDUAL LEARNING: Predict correlation CHANGE (delta)
        # -----------------------------
        # Flatten baseline correlation for feature concatenation
        baseline_flat = layers.Reshape((self.num_companies, self.num_companies, 1))(baseline_correlation_input)
        
        # Merge pairwise features with baseline correlation
        merged_features = layers.Concatenate(axis=-1, name='merged_features_with_baseline')([
            pairwise_features, 
            baseline_flat
        ])  # [batch, N, N, 2*company_dim + latent + 1]
        
        # Shared hidden layer for both μ and σ
        delta_hidden = layers.Dense(
            128, 
            activation='tanh',
            kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name='delta_hidden_shared'
        )(merged_features)
        delta_hidden = layers.Dropout(0.2)(delta_hidden)
        
        # TWO SEPARATE DENSE LAYERS for different predictions for Mean (μ) and Log-Variance (log σ²)
        
        # μ: Expected correlation change
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
        
        # log(σ²): Aleatoric uncertainty (data noise)
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
        
        # Ensure symmetry of μ
        delta_mu_symmetric = layers.Lambda(
            lambda m: 0.5 * (m + tf.transpose(m, perm=[0, 2, 1])),
            name='correlation_delta_mu_symmetric'
        )(delta_mu)
        
        # Residual connection: baseline + μ_Δ
        correlation_mean = layers.Add(name='correlation_mean_prediction')([
            baseline_correlation_input,
            delta_mu_symmetric
        ])
        
        # Ensure symmetry of log_var
        delta_log_var_symmetric = layers.Lambda(
            lambda m: 0.5 * (m + tf.transpose(m, perm=[0, 2, 1])),
            name='correlation_delta_log_var_symmetric'
        )(delta_log_var)
        
        # Stack μ and log(σ²) for loss computation: [batch, N, N, 2]
        correlation_changes_prob = layers.Lambda(
            lambda tensors: tf.stack(tensors, axis=-1),
            name='correlation_changes_probabilistic'
        )([correlation_mean, delta_log_var_symmetric])

        # -----------------------------
        # 7. Price deviations (auxiliary task)
        # -----------------------------
        price_dev_input = layers.Concatenate()([shared_news_rep, layers.Flatten()(all_company_embeddings)])
        price_dev_hidden = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(1e-5))(price_dev_input)
        price_dev_hidden = layers.Dropout(0.2)(price_dev_hidden)
        price_deviations = layers.Dense(
            self.num_companies,
            activation='tanh',
            name='price_deviations'
        )(price_dev_hidden)  # [batch, N]

        # -----------------------------
        # 8. News reconstruction (auxiliary task)
        # -----------------------------
        recon_out = layers.Dense(
            self.latent_dim, 
            activation='tanh', 
            name='news_reconstruction'
        )(shared_news_rep)

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
            keyword_sequences = training_data['keyword_sequence']
            
            # Convert TF Tensors to NumPy
            X_keywords = np.array([seq.numpy() for seq in keyword_sequences])
            
            # Shape will be (batch, 1, max_keywords) - squeeze the middle dimension
            if X_keywords.ndim == 3 and X_keywords.shape[1] == 1:
                X_keywords = np.squeeze(X_keywords, axis=1)

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
                val_keyword_sequences = validation_data['keyword_sequence']
                if isinstance(val_keyword_sequences[0], dict):
                    val_X_keywords = np.array([seq['input_ids'][0] if len(seq['input_ids'].shape) > 1 else seq['input_ids'] for seq in val_keyword_sequences])
                else:
                    val_X_keywords = np.array(val_keyword_sequences)
                    if val_X_keywords.ndim == 3 and val_X_keywords.shape[1] == 1:
                        val_X_keywords = np.squeeze(val_X_keywords, axis=1)
                
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
            
            # FIXED: Save model manually to avoid MLflow serialization issues
            try:
                model_path = "./saved_models/news_correlation_model.keras"
                self.model.save(model_path, save_format='keras')
                
                # Log only the path as a parameter (not the model itself)
                mlflow.log_param('model_saved_to', model_path)
                logger.info(f"✅ Model saved successfully to {model_path}")
                
            except Exception as e:
                logger.warning(f"Could not save full model: {e}")
                # Fallback: save weights only
                weights_path = "./saved_models/model_weights.h5"
                self.model.save_weights(weights_path)
                mlflow.log_param('weights_saved_to', weights_path)
                logger.info(f"⚠️ Only weights saved to {weights_path}")
            
            if validation_data:
                mlflow.log_metric('best_val_loss', min(history.history['val_loss']))
            else:
                mlflow.log_metric('best_train_loss', min(history.history['loss']))     
        return history

    def predict_with_uncertainty(self, keywords, baseline_correlation, news_target_embedding):
        """
        Make predictions with uncertainty estimates
        
        Args:
            keywords: Tokenized keywords
            baseline_correlation: Baseline correlation matrix
            news_target_embedding: Ground truth BERT embedding of the news (for reconstruction error)
        
        Returns:
            mu: Predicted correlation change means
            sigma: Predicted correlation change uncertainties
            confidence: Overall prediction confidence (epistemic + aleatoric)
        """

        # Ensure proper shape [batch, max_keywords]
        if keywords.ndim == 3 and keywords.shape[1] == 1:
            keywords = np.squeeze(keywords, axis=1)
        elif keywords.ndim == 1:
            keywords = np.expand_dims(keywords, axis=0)
        
        predictions = self.model.predict([
            keywords, 
            baseline_correlation
        ], verbose=0)
        
        # Calculate REAL reconstruction error
        recon_prediction = predictions[2]  # [batch, latent_dim]
        
        # Ensure shapes match
        if news_target_embedding.ndim == 1:
            news_target_embedding = news_target_embedding.reshape(1, -1)
        
        # Trim to same length if needed
        min_len = min(recon_prediction.shape[-1], news_target_embedding.shape[-1])
        recon_prediction_trimmed = recon_prediction[..., :min_len]
        news_target_trimmed = news_target_embedding[..., :min_len]
        
        # Calculate ACTUAL reconstruction error (epistemic uncertainty)
        recon_error = np.mean(np.abs(recon_prediction_trimmed - news_target_trimmed))
        sigma = np.sqrt(np.exp(predictions[0][..., 1]) + 1e-6)
        
        # Reconstruction quality component (epistemic uncertainty)
        recon_confidence = np.exp(-recon_error)
        # Predicted uncertainty component (aleatoric uncertainty)
        uncertainty_confidence = 1.0 / (1.0 + np.mean(sigma))
        
        # Combined score
        total_confidence = recon_confidence * uncertainty_confidence
        
        return {
            'mean': predictions[0][..., 0],
            'std': sigma,
            'total_confidence': total_confidence,
            'reconstruction_confidence': recon_confidence,
            'uncertainty_confidence': uncertainty_confidence,
            'price_deviations': predictions[1],
            'reconstruction_error': recon_error  # Return for debugging
        }