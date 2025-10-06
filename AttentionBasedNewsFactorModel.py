import logging
import numpy as np
import tensorflow as tf
import mlflow

logger = logging.getLogger("AdvancedNewsFactor.AttentionBasedNewsFactorModel")
layers = tf.keras.layers
models = tf.keras.models
optimizers = tf.keras.optimizers
regularizers = tf.keras.regularizers
callbacks = tf.keras.callbacks
metrics = tf.keras.metrics

class F1Score(metrics.Metric):
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.predicted_positives = self.add_weight(name='pp', initializer='zeros')
        self.actual_positives = self.add_weight(name='ap', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        self.true_positives.assign_add(tf.reduce_sum(y_true * y_pred))
        self.predicted_positives.assign_add(tf.reduce_sum(y_pred))
        self.actual_positives.assign_add(tf.reduce_sum(y_true))
    
    def result(self):
        precision = self.true_positives / (self.predicted_positives + 1e-7)
        recall = self.true_positives / (self.actual_positives + 1e-7)
        return 2 * ((precision * recall) / (precision + recall + 1e-7))
    
    def reset_state(self):
        self.true_positives.assign(0)
        self.predicted_positives.assign(0)
        self.actual_positives.assign(0)

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
        attn_out = self.attention(query, key, value, training=training) # Temperature scaled attention
        attn_out = attn_out * self.temperature
        return self.layer_norm(query + self.dropout(attn_out, training=training)) # Residual connection with layer norm

class CompanyAwareKeywordGating(layers.Layer):
    """Company-specific gating for keyword embeddings"""
    
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        
    def build(self, input_shape):
        # Gate generation from company embeddings
        self.gate_generator = layers.Dense(
            self.latent_dim, 
            activation='sigmoid',
            kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name='company_keyword_gate'
        )
        
        # Importance scoring
        self.importance_scorer = layers.Dense(
            1, 
            activation='sigmoid',
            name='keyword_importance'
        )
        super().build(input_shape)
    
    def call(self, keyword_embeddings, company_embeddings, training=None):
        company_embeddings = tf.squeeze(company_embeddings, axis=1) # [batch, 1, latent_dim] -> [batch, latent_dim]
        # Generate company-specific gates
        company_gates = tf.expand_dims(self.gate_generator(company_embeddings), axis=1) # [batch, 1, latent_dim]
        
        # The more sensitive a company is, the more it needs to "de-noise" the signal
        gated_keywords = keyword_embeddings * (1.0 - company_gates) # [batch, seq_len, latent_dim]
        
        # Calculate keyword importance scores
        importance_scores = self.importance_scorer(keyword_embeddings)
        importance_weights = tf.nn.softmax(importance_scores, axis=1)
        
        # Weighted combination
        weighted_keywords = gated_keywords * importance_weights
        return weighted_keywords

class AttentionBasedNewsFactorModel:
    """Enhanced multi-task learning model focusing on correlation changes"""

    def __init__(self, company_system, tokenizer, keyword_dim=256, company_dim=128, latent_dim=128, max_keywords=100):
        self.company_system = company_system
        self.keyword_dim = keyword_dim
        self.company_dim = company_dim
        self.latent_dim = latent_dim
        self.max_keywords = max_keywords
        self.tokenizer = tokenizer

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("news_correlation_trading")
        
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
                'correlation_changes_symmetric': 'mse',
                'price_deviations': 'mse',
                'news_reconstruction': 'mae'
            },
            loss_weights={
                'correlation_changes_symmetric': 3.0,
                'price_deviations': 0.5,
                'news_reconstruction': 0.5
            },
            metrics={
                'correlation_changes_symmetric': ['mae'],
                'price_deviations': ['mae', 'mse'],
                'news_reconstruction': ['mae']
            }
        )

    def build_model(self):
        """Multi-task model: global baseline + lightweight local corrections for correlation shifts."""

        # -----------------------------
        # 1. Inputs
        # -----------------------------
        keyword_input = layers.Input(shape=(self.max_keywords,), name='keywords')
        company_idx_input = layers.Input(shape=(1,), name='company_idx', dtype='int32')

        # -----------------------------
        # 2. Keyword encoder (impact-aware)
        # -----------------------------
        keyword_embedding = layers.Embedding(
            input_dim=self.tokenizer.vocab_size,
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
        # 3. Company embeddings (single + all companies via same layer)
        # -----------------------------
        company_embedding_layer = layers.Embedding(
            input_dim=self.company_system.num_companies,
            output_dim=self.company_dim,
            name='company_embeddings'
        )
        company_emb = company_embedding_layer(company_idx_input)  # [batch, 1, company_dim]

        # All companies (batch, num_companies)
        all_company_indices = tf.range(self.company_system.num_companies, dtype=tf.int32)
        all_company_indices = tf.expand_dims(all_company_indices, 0)  # [1, N]
        batch_size = tf.shape(company_idx_input)[0]
        all_company_indices = tf.tile(all_company_indices, [batch_size, 1])  # [batch, N]

        all_company_embeddings = company_embedding_layer(all_company_indices)  # [batch, N, company_dim]

        # Process target company embedding into latent
        company_proc = layers.Dense(
            self.latent_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name='company_processed'
        )(company_emb)  # [batch, 1, latent_dim]
        company_proc = layers.BatchNormalization()(company_proc)
        company_proc = layers.Dropout(0.15)(company_proc)
        company_reshaped = layers.Reshape((1, self.latent_dim))(company_proc)  # same shape

        # -----------------------------
        # 4. (Optional) Company-aware keyword gating - keep but regularize hard
        # -----------------------------
        # keep news meaning but allow company-contextualization; we therefore keep gating but apply stronger dropout.
        gated_keywords = CompanyAwareKeywordGating(
            self.latent_dim,
            name='company_keyword_gating'
        )(keyword_latent, company_proc)  # [batch, seq_len, latent_dim]
        gated_keywords = layers.Dropout(0.35)(gated_keywords)

        # small attention: how news modifies company representation (lightweight)
        news_to_comp = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=max(1, self.latent_dim // 4),
            name='news_to_company'
        )(gated_keywords, company_reshaped, company_reshaped)

        # -----------------------------
        # 5. News representation aggregation (keeps the original news meaning for global baseline)
        # -----------------------------
        news_features = layers.Concatenate()([
            layers.Flatten()(news_to_comp),                    # small company-aware signal
            layers.Flatten()(company_proc),                    # target company info
            layers.GlobalAveragePooling1D()(gated_keywords),   # news meaning preserved
            layers.GlobalMaxPooling1D()(gated_keywords)
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
        shared_news_rep = layers.Dropout(0.2)(shared_news_rep)  # final news embedding used by predictors

        # -----------------------------
        # 6. Unified predictor (global baseline + lightweight local corrections)
        # -----------------------------
        unified_predictor = layers.Dense(
            256, activation='relu',
            kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4)
        )(shared_news_rep)
        unified_predictor = layers.Dropout(0.2)(unified_predictor)

        # GLOBAL baseline: predict a price-change vector for all companies (one component per company)
        # This is the raw global "price vector" that encodes the news' average expected effect across companies.
        global_price_vector = layers.Dense(
            self.company_system.num_companies,
            activation='linear',
            name='global_price_vector'
        )(unified_predictor)  # [batch, N]

        # From the global vector we can form an implied correlation-like matrix (outer product)
        # normalized outer product -> gives a symmetric matrix per batch
        def outer_normalized(x):
            # x: [batch, N]
            # outer: [batch, N, N]
            outer = tf.einsum('bi,bj->bij', x, x)
            # normalize by vector norm magnitudes to avoid scale blow-up
            norm = tf.norm(x, axis=1, keepdims=True)  # [batch,1]
            denom = tf.maximum(tf.einsum('bi,bj->bij', norm, norm), 1e-6)
            return outer / denom

        implied_matrix = layers.Lambda(outer_normalized, name='implied_correlation_matrix')(global_price_vector)
        # implied_matrix in [-1,1] roughly (since global_price_vector in [-1,1])

        # LIGHTWEIGHT per-company correction scalars (shared network applied to each company)
        # we compute a per-company factor c_i = f(unified_predictor, company_emb_i) -> [batch, N, 1]
        # implement with a small shared Dense applied to concatenated [unified_predictor, company_emb_i] via time-distributed style
        # first, expand unified_predictor to [batch, N, up_dim]
        up = layers.Dense(self.company_dim, activation='relu')(unified_predictor)  # [batch, company_dim]
        up_expanded = layers.Lambda(lambda t: tf.expand_dims(t, axis=1))(up)      # [batch,1,company_dim]
        up_tiled = layers.Lambda(lambda t: tf.tile(t, [1, self.company_system.num_companies, 1]))(up_expanded)  # [batch,N,company_dim]

        # concatenate per-company: [batch, N, company_dim + company_dim]
        per_company_input = layers.Concatenate(axis=-1)([up_tiled, all_company_embeddings])  # [batch,N, 2*company_dim]

        # shared small network for per-company scalar
        per_company_hidden = layers.TimeDistributed(layers.Dense(32, activation='tanh'))(per_company_input)  # [batch,N,32]
        per_company_scalar = layers.TimeDistributed(layers.Dense(1, activation='linear'), name='per_company_scalar')(per_company_hidden)  # [batch,N,1]
        per_company_scalar = layers.Reshape((self.company_system.num_companies,))(per_company_scalar)  # [batch, N]

        # Build correction matrix as outer product of per_company_scalar (low-parametrization)
        correction_matrix = layers.Lambda(lambda x: tf.einsum('bi,bj->bij', x, x),
                                        name='correction_matrix')(per_company_scalar)  # [batch,N,N]

        # Learned gate/scale between implied_matrix and correction_matrix (scalar per batch)
        scale_logits = layers.Dense(1, activation='sigmoid', name='correction_scale')(unified_predictor)  # [batch,1] in (0,1)
        scale_expanded = layers.Reshape((1, 1))(scale_logits)

        # final predicted correlation change matrix = alpha * implied + (1-alpha) * correction (or sum with learned scaling)
        correlation_changes = layers.Lambda(
            lambda args: args[0] * args[2] + args[1] * (1.0 - args[2]),
            name='correlation_changes_matrix'
        )([implied_matrix, correction_matrix, scale_expanded])

        # ensure symmetry and numerical stability (average with transpose)
        # FONTOS: Linear activation, mert Fisher-z térben tanítunk! A correlation térbe való visszaalakítás (tanh) az inference során történik
        correlation_changes = layers.Lambda(
            lambda m: 0.5 * (m + tf.transpose(m, perm=[0, 2, 1])),
            name='correlation_changes_symmetric'
        )(correlation_changes)

        # -----------------------------
        # 7. Price deviations (per-company residuals) - small, regularized head
        # -----------------------------
        # Predict residual deviations from the global_price_vector per company
        price_dev_input = layers.Concatenate()([unified_predictor, layers.Flatten()(all_company_embeddings)])  # [batch, 256 + N*company_dim]
        price_dev_hidden = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(1e-5))(price_dev_input)
        price_dev_hidden = layers.Dropout(0.2)(price_dev_hidden)
        price_deviations = layers.Dense(
            self.company_system.num_companies,
            activation='tanh',
            name='price_deviations'
        )(price_dev_hidden)  # [batch, N]

        # Optionally combine with global_price_vector to get final price vector prediction:
        # final_price_vector = global_price_vector + 0.1 * price_deviations
        # but we output deviations separately so training can weight them down if desired.

        # -----------------------------
        # 8. News reconstruction (auxiliary, keeps news meaning)
        # -----------------------------
        recon_out = self._create_news_recon(gated_keywords, shared_news_rep, company_emb)

        # -----------------------------
        # 9. Build model
        # -----------------------------
        return models.Model(
            inputs=[keyword_input, company_idx_input],
            outputs=[correlation_changes, price_deviations, recon_out],
            name='NewsCorrelationHybridModel'
        )

    def _create_news_recon(self, keyword_latent, shared, company_emb):
        """Enhanced news reconstruction with company-specific weighting"""
        scale = tf.sqrt(tf.cast(self.latent_dim, tf.float32))
        
        scores = tf.einsum('bcd,btd->bct', company_emb, keyword_latent) / scale
        attn_weights = tf.nn.softmax(scores, axis=-1)
        company_context = tf.einsum('bct,btd->bcd', attn_weights, keyword_latent)

        comp_importance = tf.nn.softmax(
            tf.squeeze(layers.Dense(1)(company_context), -1), axis=-1
        )
        pooled_news = tf.reduce_sum(company_context * tf.expand_dims(comp_importance, -1), axis=1)

        global_hidden = layers.Dense(128, activation='relu')(shared)
        recon_input = layers.Concatenate(axis=-1)([global_hidden, pooled_news])
        recon_input = layers.Dropout(0.2)(recon_input)
        return layers.Dense(self.latent_dim, activation='tanh', name='news_reconstruction')(recon_input)

    def train(self, training_data, validation_data=None, epochs=100, batch_size=32):
        """Train the correlation-focused model"""
        with mlflow.start_run():
            mlflow.log_params({
                'epochs': epochs,
                'batch_size': batch_size,
                'keyword_dim': self.keyword_dim,
                'company_dim': self.company_dim,
                'latent_dim': self.latent_dim,
                'max_keywords': self.max_keywords
            })
            
            class MLflowCallback(callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if logs:
                        for metric, value in logs.items():
                            mlflow.log_metric(metric, value, step=epoch)

            # ---- Train data ----
            X_keywords = np.squeeze(np.array(training_data['keywords']), axis=1)
            X_company_indices = np.array(training_data['company_indices']).reshape(-1, 1)
            
            y_correlation_changes = np.array(training_data['correlation_changes'])  # [batch, N, N]
            y_price_deviations = np.array(training_data['price_deviations'])        # [batch, N]
            y_news_targets = np.array(training_data['news_targets'])                # reconstruction

            # ---- Validation ----
            validation_data_prepared = None
            if validation_data:
                val_X_keywords = np.array(validation_data['keywords'])
                val_X_company_indices = np.array(validation_data['company_indices']).reshape(-1, 1)
                val_y_correlation_changes = np.array(validation_data['correlation_changes'])
                val_y_price_deviations = np.array(validation_data['price_deviations'])
                val_y_news_targets = np.array(validation_data['news_targets'])
                
                validation_data_prepared = (
                    [val_X_keywords, val_X_company_indices],
                    [val_y_correlation_changes, val_y_price_deviations, val_y_news_targets]
                )
            
            # ---- Train ----
            history = self.model.fit(
                [X_keywords, X_company_indices],
                [y_correlation_changes, y_price_deviations, y_news_targets],
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
            
            mlflow.tensorflow.log_model(self.model, "news_correlation_model")
            
            if validation_data:
                mlflow.log_metric('best_val_loss', min(history.history['val_loss']))
        
        return history

    
    def prepare_keyword_sequence(self, text, max_length=None):
        return self.tokenizer.encode(text, max_length or self.max_keywords)

    def analyze_keyword_impact_clusters(self, sample_keywords, similarity_threshold=0.7, return_matrix=False):
        """Analyze how keywords cluster based on their learned impact patterns"""
        if not sample_keywords:
            return {}
        
        try:
            keyword_embedding_layer = self.model.get_layer('keyword_embeddings')
            all_embeddings = keyword_embedding_layer.get_weights()[0]
        except ValueError:
            logger.warning("Keyword embedding layer not found")
            return {}
        
        valid_keywords = []
        embeddings = []
        
        for word in sample_keywords:
            idx = self.tokenizer.word_to_idx.get(word)
            if idx is not None and idx < len(all_embeddings):
                valid_keywords.append(word)
                embeddings.append(all_embeddings[idx])

        if len(valid_keywords) < 2:
            return {}
        
        embeddings = np.array(embeddings)
        normalized_embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8) 
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Find clusters of similar-impact keywords
        clusters = {}
        for i, w1 in enumerate(valid_keywords):
            sims = [(w2, similarity_matrix[i, j]) for j, w2 in enumerate(valid_keywords) 
                   if i != j and similarity_matrix[i, j] > similarity_threshold]
            if sims:
                clusters[w1] = sorted(sims, key=lambda x: x[1], reverse=True)

        return {"clusters": clusters, "similarity_matrix": similarity_matrix, "keywords": valid_keywords} if return_matrix else clusters