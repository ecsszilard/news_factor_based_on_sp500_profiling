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
        # company_embeddings alakja: (batch, 1, latent_dim)
        company_embeddings = tf.squeeze(company_embeddings, axis=1)  # [batch, 1, latent_dim] -> [batch, latent_dim]
        # Generate company-specific gates
        company_gates = tf.expand_dims(self.gate_generator(company_embeddings), axis=1)     # [batch, 1, latent_dim]
        
        # The more sensitive a company is, the more it needs to “de-noise” the signal coming from keywords
        gated_keywords = keyword_embeddings * (1.0 - company_gates)      # [batch, seq_len, latent_dim]
        
        # Calculate keyword importance scores
        importance_scores = self.importance_scorer(keyword_embeddings)  # [batch, seq_len, 1]
        importance_weights = tf.nn.softmax(importance_scores, axis=1)
        
        # Weighted combination
        weighted_keywords = gated_keywords * importance_weights
        return weighted_keywords

class AttentionBasedNewsFactorModel:
    """Enhanced multi-task learning model with company-keyword coupling"""

    def __init__(self, company_system, tokenizer, keyword_dim=256, company_dim=128, latent_dim=128, max_keywords=100):
        self.company_system = company_system
        self.keyword_dim = keyword_dim
        self.company_dim = company_dim
        self.latent_dim = latent_dim
        self.max_keywords = max_keywords
        self.tokenizer = tokenizer

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("news_factor_trading")
        
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
            clipnorm=1.0  # Gradient clipping
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss={
                'price_change_prediction': self._huber_loss,
                'volatility_prediction': 'mse',
                'relevance_prediction': 'binary_crossentropy',
                'news_reconstruction': 'mae',
                'attention_regularization': 'categorical_crossentropy'  # Added back
            },
            loss_weights={
                'price_change_prediction': 3.0,  # Main task
                'volatility_prediction': 2.0,
                'relevance_prediction': 1.5,
                'news_reconstruction': 0.5,
                'attention_regularization': 0.3  # Encourages meaningful attention patterns
            },
            metrics={
                'price_change_prediction': ['mae'],
                'volatility_prediction': ['mae'],
                'relevance_prediction': ['accuracy', F1Score()],
                'news_reconstruction': ['mae'],
                'attention_regularization': ['accuracy']
            }
        )

    def _huber_loss(self, y_true, y_pred, delta=1.0):
        """Robust Huber loss for price predictions"""
        error = y_true - y_pred
        condition = tf.abs(error) <= delta
        squared_loss = 0.5 * tf.square(error)
        linear_loss = delta * tf.abs(error) - 0.5 * tf.square(delta)
        return tf.where(condition, squared_loss, linear_loss)
    
    def build_model(self):
        """Multi-task model with shared company embeddings.
        Combines global context informativeness with local keyword relevance via attention and reconstruction."""

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
        )(keyword_input) # These learn impact patterns

        attn_keywords = OptimizedAttentionLayer(
            num_heads=8,
            key_dim=self.keyword_dim // 8,
            dropout_rate=0.1,
            name='keyword_attention'
        )(keyword_embedding, keyword_embedding, keyword_embedding) # Discovers keyword co-occurrence patterns that predict similar impacts

        keyword_latent = layers.Dense(
            self.latent_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name='keyword_transform'
        )(layers.LayerNormalization()(attn_keywords + keyword_embedding))
        keyword_latent = layers.Dropout(0.2)(keyword_latent)

        # -----------------------------
        # 3. Company embedding
        # -----------------------------
        company_emb = layers.Embedding(
            input_dim=self.company_system.num_companies,
            output_dim=self.company_dim,
            name='company_embeddings'
        )(company_idx_input)

        company_proc = layers.Dense(
            self.latent_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name='company_processed'
        )(company_emb)
        company_proc = layers.BatchNormalization()(company_proc)
        company_proc = layers.Dropout(0.1)(company_proc)
        company_reshaped = layers.Reshape((1, self.latent_dim))(company_proc)

        # Megjegyzés: Itt lehetne MultiHeadAttention a cégek saját embeddingjein, ha többdimenziós cégháló-struktúrát (pl. szektor, beszállítói lánc) akarunk modellezni, ha azonban a cég csak egyedi index és statikus embedding, akkor felesleges.

        # -----------------------------
        # 4. Bi-directional attention
        # -----------------------------
        gated_keywords = CompanyAwareKeywordGating(self.latent_dim, name='company_keyword_gating')(keyword_latent, company_proc)
        
        comp_to_news = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=self.latent_dim // 4,
            name='company_to_news'
        )(company_reshaped, gated_keywords, gated_keywords)

        news_to_comp = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=self.latent_dim // 4,
            name='news_to_company'
        )(gated_keywords, company_reshaped, company_reshaped)

        combined = layers.Concatenate(axis=-1)([
            comp_to_news,
            tf.reduce_mean(news_to_comp, axis=1, keepdims=True)
        ])

        combined_norm = layers.LayerNormalization()(layers.Dense(
            self.latent_dim,
            activation='tanh',
            name='attention_combination'
        )(combined))

        # -----------------------------
        # 5. Shared representation
        # -----------------------------
        combined_features = layers.Concatenate()([
            layers.Flatten()(combined_norm),
            layers.Flatten()(company_proc),
            layers.GlobalAveragePooling1D()(gated_keywords)
        ])

        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4))(combined_features)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        shared = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4), name='shared_rep')(x)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.2)(shared)

        # -----------------------------
        # 6. Prediction heads
        # -----------------------------
        price = layers.Dropout(0.1)(layers.Dense(64, activation='relu')(shared))
        price_out = layers.Dense(3, activation='linear', name='price_change_prediction')(price)

        vol = layers.Dropout(0.1)(layers.Dense(64, activation='relu')(shared))
        vol_out = layers.Dense(2, activation='linear', name='volatility_prediction')(vol)

        rel = layers.Dropout(0.1)(layers.Dense(32, activation='relu')(shared))
        rel_out = layers.Dense(1, activation='sigmoid', name='relevance_prediction')(rel)

        recon_out = self._create_news_recon(gated_keywords, shared, company_emb)

        attn_reg = layers.Dropout(0.1)(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4))(shared))
        attn_reg_out = layers.Dense(self.max_keywords, activation='softmax', name='attention_regularization')(attn_reg)

        return models.Model(
            inputs=[keyword_input, company_idx_input],
            outputs=[price_out, vol_out, rel_out, recon_out, attn_reg_out],
            name='NewsFactorModel'
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
        """Train the multi-task model"""
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                'epochs': epochs,
                'batch_size': batch_size,
                'keyword_dim': self.keyword_dim,
                'company_dim': self.company_dim,
                'latent_dim': self.latent_dim,
                'max_keywords': self.max_keywords
            })
            
            # Custom callback for MLflow logging
            class MLflowCallback(callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if logs:
                        for metric, value in logs.items():
                            mlflow.log_metric(metric, value, step=epoch)

            X_keywords = np.squeeze(np.array(training_data['keywords']), axis=1)
            X_company_indices = np.array(training_data['company_indices']).reshape(-1, 1)
            
            y_price_changes = np.array(training_data['price_changes'])
            y_volatility_changes = np.array(training_data['volatility_changes'])
            y_relevance = np.expand_dims(np.array(training_data['relevance_labels']), -1)  # Shape fix for binary classification
            y_news_targets = np.array(training_data['news_targets'])
            
            # Create attention regularization targets (uniform distribution encourages diverse attention)
            y_attention_reg = np.ones((len(X_keywords), self.max_keywords)) / self.max_keywords
            
            validation_data_prepared = None
            if validation_data:
                val_X_keywords = np.array(validation_data['keywords'])
                val_X_company_indices = np.array(validation_data['company_indices']).reshape(-1, 1)
                val_y_price_changes = np.array(validation_data['price_changes'])
                val_y_volatility_changes = np.array(validation_data['volatility_changes'])
                val_y_relevance = np.expand_dims(np.array(validation_data['relevance_labels']), -1)  # Shape fix for binary classification
                val_y_news_targets = np.array(validation_data['news_targets'])
                val_y_attention_reg = np.ones((len(val_X_keywords), self.max_keywords)) / self.max_keywords
                
                validation_data_prepared = (
                    [val_X_keywords, val_X_company_indices],
                    [val_y_price_changes, val_y_volatility_changes, val_y_relevance, val_y_news_targets, val_y_attention_reg]
                )
            
            history = self.model.fit(
                [X_keywords, X_company_indices],
                [y_price_changes, y_volatility_changes, y_relevance, y_news_targets, y_attention_reg],
                validation_data=validation_data_prepared,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[
                    callbacks.EarlyStopping(
                        monitor='val_loss' if validation_data else 'loss', patience=15, restore_best_weights=True
                    ),
                    callbacks.ReduceLROnPlateau(
                        monitor='val_loss' if validation_data else 'loss', factor=0.5, patience=8, min_lr=1e-6
                    ),
                    MLflowCallback()
                ],
                verbose=1
            )
            
            mlflow.tensorflow.log_model(
                self.model, 
                "news_factor_model"
            )
            
            # Log performance metrics
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
