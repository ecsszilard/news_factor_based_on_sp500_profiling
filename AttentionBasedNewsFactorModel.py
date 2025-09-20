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

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes=2, average='macro', **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.average = average
        self.precision = metrics.Precision()
        self.recall = metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + 1e-7))
    
class OptimizedAttentionLayer(tf.keras.layers.Layer):
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
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate
        )
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        super().build(input_shape)
    
    def call(self, query, key, value, training=None):
        # Temperature scaled attention
        attention_output = self.attention(
            query, key, value, training=training
        )
        attention_output = attention_output * self.temperature
        
        # Residual connection with layer norm
        output = self.layer_norm(query + self.dropout(attention_output, training=training))
        return output

class AttentionBasedNewsFactorModel:
    """Multi-task learning model for news sentiment analysis with shared company embeddings"""

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
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
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
                'relevance_prediction':  ['accuracy', F1Score(num_classes=2, average='macro')],
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
        )(keyword_embedding, keyword_embedding) # Discovers keyword co-occurrence patterns that predict similar impacts

        keyword_latent = layers.Dense(
            self.latent_dim,
            activation='relu',
            kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4),
            name='keyword_transform'
        )(layers.LayerNormalization()(attn_keywords + keyword_embedding))
        keyword_latent = layers.Dropout(0.2)(keyword_latent)

        keyword_impact = layers.Dense(
            self.latent_dim,
            activation='tanh',
            name='impact_regularization'
        )(keyword_latent) # Encourages similar-impact keywords to cluster

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
        comp_to_news = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=self.latent_dim // 4,
            name='company_to_news'
        )(company_reshaped, keyword_impact, keyword_impact)

        news_to_comp = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=self.latent_dim // 4,
            name='news_to_company'
        )(keyword_impact, company_reshaped, company_reshaped)

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
            layers.GlobalAveragePooling1D()(keyword_impact)
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
        price = layers.Dense(64, activation='relu')(shared)
        price = layers.Dropout(0.1)(price)
        price_out = layers.Dense(3, activation='linear', name='price_change_prediction')(price)

        vol = layers.Dense(64, activation='relu')(shared)
        vol = layers.Dropout(0.1)(vol)
        vol_out = layers.Dense(2, activation='linear', name='volatility_prediction')(vol)

        rel = layers.Dense(32, activation='relu')(shared)
        rel = layers.Dropout(0.1)(rel)
        rel_out = layers.Dense(1, activation='sigmoid', name='relevance_prediction')(rel)

        recon_out = self._create_news_recon(keyword_latent, shared, company_emb)

        attn_reg = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(1e-5, 1e-4))(shared)
        attn_reg = layers.Dropout(0.1)(attn_reg)
        attn_reg_out = layers.Dense(self.max_keywords, activation='softmax', name='attention_regularization')(attn_reg)

        return models.Model(
            inputs=[keyword_input, company_idx_input],
            outputs=[price_out, vol_out, rel_out, recon_out, attn_reg_out],
            name='NewsFactorModel'
        )

    def _create_news_recon(self, keyword_latent, shared, company_emb):
        scale = tf.sqrt(tf.cast(self.latent_dim, tf.float32))
        
        scores = tf.einsum('bcd,btd->bct', company_emb, keyword_latent) / scale
        attn_weights = tf.nn.softmax(scores, axis=-1)
        company_context = tf.einsum('bct,btd->bcd', attn_weights, keyword_latent)

        comp_importance = layers.Dense(1)(company_context)
        comp_importance = tf.nn.softmax(tf.squeeze(comp_importance, -1), axis=-1)
        comp_importance = tf.expand_dims(comp_importance, -1)
        pooled_news = tf.reduce_sum(company_context * comp_importance, axis=1)

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
            class MLflowCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if logs:
                        for metric, value in logs.items():
                            mlflow.log_metric(metric, value, step=epoch)
            
            callbacks = [
                MLflowCallback(),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=15, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6
                )
            ]

            X_keywords = np.array(training_data['keywords'])
            X_company_indices = np.array(training_data['company_indices']).reshape(-1, 1)
            
            y_price_changes = np.array(training_data['price_changes'])
            y_volatility_changes = np.array(training_data['volatility_changes'])
            y_relevance = np.array(training_data['relevance_labels'])
            y_news_targets = np.array(training_data['news_targets'])
            
            # Create attention regularization targets (uniform distribution encourages diverse attention)
            y_attention_reg = np.ones((len(X_keywords), self.max_keywords)) / self.max_keywords
            
            validation_data_prepared = None
            if validation_data:
                val_X_keywords = np.array(validation_data['keywords'])
                val_X_company_indices = np.array(validation_data['company_indices']).reshape(-1, 1)
                val_y_price_changes = np.array(validation_data['price_changes'])
                val_y_volatility_changes = np.array(validation_data['volatility_changes'])
                val_y_relevance = np.array(validation_data['relevance_labels'])
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
                    monitor='val_loss' if validation_data else 'loss',
                    patience=15,
                    restore_best_weights=True),
                    callbacks.ReduceLROnPlateau(monitor='val_loss' if validation_data else 'loss', factor=0.5, patience=8, min_lr=1e-6)
                ],
                verbose=1)
        
            mlflow.tensorflow.log_model(
                self.model, 
                "news_factor_model",
                signature=mlflow.models.infer_signature(
                    [training_data['keywords'][:5], training_data['company_indices'][:5]]
                )
            )
            
            # Log performance metrics
            if validation_data:
                val_loss = min(history.history['val_loss'])
                mlflow.log_metric('best_val_loss', val_loss)
        return history
    
    def prepare_keyword_sequence(self, text, max_length=None):
        if max_length is None:
            max_length = self.max_keywords
        return self.tokenizer.encode(text, max_length=max_length)
    
    def analyze_keyword_impact_clusters(self, sample_keywords, return_matrix=False):
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
            if word in self.tokenizer.word_to_idx:
                idx = self.tokenizer.word_to_idx[word]
                if idx < len(all_embeddings):
                    valid_keywords.append(word)
                    embeddings.append(all_embeddings[idx])
        
        if len(valid_keywords) < 2:
            return {}
        
        embeddings = np.array(embeddings)
        
        # Compute similarity matrix
        normalized_embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8) 
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Find clusters of similar-impact keywords
        clusters = {}
        for i, word1 in enumerate(valid_keywords):
            similar_words = []
            for j, word2 in enumerate(valid_keywords):
                if i != j and similarity_matrix[i, j] > 0.7:  # High similarity threshold
                    similar_words.append((word2, similarity_matrix[i, j]))
            
            if similar_words:
                similar_words.sort(key=lambda x: x[1], reverse=True)
                clusters[word1] = similar_words
        
        if return_matrix:
            return {
                'clusters': clusters,
                'similarity_matrix': similarity_matrix,
                'keywords': valid_keywords
            }
        else:
            return clusters