import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Embedding, Dropout, LayerNormalization, MultiHeadAttention, Reshape, Flatten, Concatenate, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import logging

from BiDirectionalAttentionLayer import BiDirectionalAttentionLayer
from ImprovedTokenizer import ImprovedTokenizer
from FinancialImpactTracker import FinancialImpactTracker

logger = logging.getLogger("AdvancedNewsFactor.AttentionBasedNewsFactorModel")

class AttentionBasedNewsFactorModel:
    """Multi-task learning model for news sentiment analysis with shared company embeddings"""

    def __init__(self, company_system, vocab_size=50000, keyword_dim=256, company_dim=256, latent_dim=128, max_keywords=100):
        self.company_system = company_system
        self.vocab_size = vocab_size
        self.keyword_dim = keyword_dim
        self.company_dim = company_dim
        self.latent_dim = latent_dim
        self.max_keywords = max_keywords
        
        self.tokenizer = ImprovedTokenizer(vocab_size)
        self.model = self.build_model()
        
        # Multi-task loss weights
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'price_change_prediction': 'mse',
                'volatility_prediction': 'mse',
                'relevance_prediction': 'binary_crossentropy',
                'news_reconstruction': 'mse',
                'attention_regularization': 'categorical_crossentropy'  # Added back
            },
            loss_weights={
                'price_change_prediction': 2.0,  # Main task
                'volatility_prediction': 1.5,
                'relevance_prediction': 1.0,
                'news_reconstruction': 0.3,
                'attention_regularization': 0.2  # Encourages meaningful attention patterns
            },
            metrics={
                'price_change_prediction': ['mae'],
                'volatility_prediction': ['mae'],
                'relevance_prediction': ['accuracy'],
                'news_reconstruction': ['mae'],
                'attention_regularization': ['accuracy']
            }
        )

    def build_model(self):
        """Multi-task model with shared company embeddings.
        Combines global context informativeness with local keyword relevance via attention and reconstruction."""

        # -----------------------------
        # 1. Inputs
        # -----------------------------
        keyword_input = Input(shape=(self.max_keywords,), name='keywords')
        company_idx_input = Input(shape=(1,), name='company_idx', dtype='int32')

        # -----------------------------
        # 2. Keyword encoder (impact-aware)
        # -----------------------------
        keyword_embedding = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.keyword_dim,
            mask_zero=True,
            embeddings_regularizer=l1_l2(1e-5, 1e-4),
            name='keyword_embeddings'
        )(keyword_input) # These learn impact patterns

        attn_keywords = MultiHeadAttention(
            num_heads=8,
            key_dim=self.keyword_dim // 8,
            dropout=0.1,
            name='keyword_attention'
        )(keyword_embedding, keyword_embedding) # Discovers keyword co-occurrence patterns that predict similar impacts

        keyword_latent = Dense(
            self.latent_dim,
            activation='relu',
            kernel_regularizer=l1_l2(1e-5, 1e-4),
            name='keyword_transform'
        )(LayerNormalization()(attn_keywords + keyword_embedding))
        keyword_latent = Dropout(0.2)(keyword_latent)

        keyword_impact = Dense(
            self.latent_dim,
            activation='tanh',
            name='impact_regularization'
        )(keyword_latent) # Encourages similar-impact keywords to cluster

        # -----------------------------
        # 3. Company embedding
        # -----------------------------
        company_emb = Embedding(
            input_dim=self.company_system.num_companies,
            output_dim=self.company_dim,
            name='company_embeddings'
        )(company_idx_input)

        company_proc = Dense(
            self.latent_dim,
            activation='relu',
            kernel_regularizer=l1_l2(1e-5, 1e-4),
            name='company_processed'
        )(company_emb)
        company_proc = BatchNormalization()(company_proc)
        company_proc = Dropout(0.1)(company_proc)
        company_reshaped = Reshape((1, self.latent_dim))(company_proc)

        # Megjegyzés: Itt lehetne MultiHeadAttention a cégek saját embeddingjein,
        # ha többdimenziós cégháló-struktúrát (pl. szektor, beszállítói lánc) akarunk modellezni.
        # Ha azonban a cég csak egyedi index és statikus embedding, akkor felesleges.

        # -----------------------------
        # 4. Bi-directional attention
        # -----------------------------
        comp_to_news = MultiHeadAttention(
            num_heads=4,
            key_dim=self.latent_dim // 4,
            name='company_to_news'
        )(company_reshaped, keyword_impact, keyword_impact)

        news_to_comp = MultiHeadAttention(
            num_heads=4,
            key_dim=self.latent_dim // 4,
            name='news_to_company'
        )(keyword_impact, company_reshaped, company_reshaped)

        combined = Concatenate(axis=-1)([
            comp_to_news,
            tf.reduce_mean(news_to_comp, axis=1, keepdims=True)
        ])

        combined_norm = LayerNormalization()(Dense(
            self.latent_dim,
            activation='tanh',
            name='attention_combination'
        )(combined))

        # -----------------------------
        # 5. Shared representation
        # -----------------------------
        combined_features = Concatenate()([
            Flatten()(combined_norm),
            Flatten()(company_proc),
            GlobalAveragePooling1D()(keyword_impact)
        ])

        x = Dense(256, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(combined_features)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        shared = Dense(128, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4), name='shared_rep')(x)
        shared = BatchNormalization()(shared)
        shared = Dropout(0.2)(shared)

        # -----------------------------
        # 6. Prediction heads
        # -----------------------------
        price = Dense(64, activation='relu')(shared)
        price = Dropout(0.1)(price)
        price_out = Dense(3, activation='linear', name='price_prediction')(price)

        vol = Dense(64, activation='relu')(shared)
        vol = Dropout(0.1)(vol)
        vol_out = Dense(2, activation='linear', name='volatility_prediction')(vol)

        rel = Dense(32, activation='relu')(shared)
        rel = Dropout(0.1)(rel)
        rel_out = Dense(1, activation='sigmoid', name='relevance_prediction')(rel)

        recon_out = self._create_news_recon(keyword_latent, shared, company_emb)

        attn_reg = Dense(64, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(shared)
        attn_reg = Dropout(0.1)(attn_reg)
        attn_reg_out = Dense(self.max_keywords, activation='softmax', name='attention_regularization')(attn_reg)

        return Model(
            inputs=[keyword_input, company_idx_input],
            outputs=[price_out, vol_out, rel_out, recon_out, attn_reg_out],
            name='NewsFactorModel'
        )

    def _create_news_recon(self, keyword_latent, shared, company_emb):
        d = tf.shape(keyword_latent)[-1]
        scale = tf.sqrt(tf.cast(d, tf.float32))

        scores = tf.einsum('bcd,btd->bct', company_emb, keyword_latent) / scale
        attn_weights = tf.nn.softmax(scores, axis=-1)
        company_context = tf.einsum('bct,btd->bcd', attn_weights, keyword_latent)

        comp_importance = Dense(1)(company_context)
        comp_importance = tf.nn.softmax(tf.squeeze(comp_importance, -1), axis=-1)
        comp_importance = tf.expand_dims(comp_importance, -1)
        pooled_news = tf.reduce_sum(company_context * comp_importance, axis=1)

        global_hidden = Dense(128, activation='relu')(shared)
        recon_input = Concatenate(axis=-1)([global_hidden, pooled_news])
        recon_input = Dropout(0.2)(recon_input)
        return Dense(d, activation='tanh', name='news_reconstruction')(recon_input)
        
        
    def prepare_keyword_sequence(self, text, max_length=None):
        if max_length is None:
            max_length = self.max_keywords
        return self.tokenizer.encode(text, max_length=max_length)
        
    def build_vocabulary(self, news_texts):
        self.tokenizer.build_vocab(news_texts)
    
    def train(self, training_data, validation_data=None, epochs=100, batch_size=32):
        """Train the multi-task model"""
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
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            [X_keywords, X_company_indices],
            [y_price_changes, y_volatility_changes, y_relevance, y_news_targets, y_attention_reg],
            validation_data=validation_data_prepared,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
        
    def predict_impact(self, news_text, company_symbol, return_detailed=False):
        """Predict impact of news on company"""
        keyword_sequence = self.prepare_keyword_sequence(news_text)
        company_idx = self.company_system.get_company_idx(company_symbol)
        
        if keyword_sequence.ndim == 1:
            keyword_sequence = keyword_sequence.reshape(1, -1)
        company_idx_array = np.array([[company_idx]])
        
        predictions = self.model.predict([keyword_sequence, company_idx_array], verbose=0)
        
        price_pred = predictions[0][0]  # [1d, 5d, 20d returns]
        volatility_pred = predictions[1][0]  # [volatility_change, volume_change]
        relevance_pred = predictions[2][0][0]  # relevance score
        reconstruction_pred = predictions[3][0]  # news reconstruction
        
        if return_detailed:
            return {
                'price_changes': {
                    '1d': price_pred[0],
                    '5d': price_pred[1],
                    '20d': price_pred[2]
                },
                'volatility_changes': {
                    'volatility': volatility_pred[0],
                    'volume_proxy': volatility_pred[1]
                },
                'relevance_score': relevance_pred,
                'reconstruction_quality': np.mean(np.abs(reconstruction_pred)),
                'company_idx': company_idx,
                'confidence': relevance_pred  # Use relevance as confidence proxy
            }
        else:
            return price_pred
    
    def get_similar_companies_by_news_response(self, target_company, top_k=5):
        """Find companies with similar news response patterns using learned embeddings"""
        target_idx = self.company_system.get_company_idx(target_company)
        
        # Extract company embeddings from the model
        company_embedding_layer = self.model.get_layer('learnable_company_embeddings')
        all_embeddings = company_embedding_layer.get_weights()[0]  # Shape: (num_companies, company_dim)
        
        target_embedding = all_embeddings[target_idx]
        similarities = []
        
        for i, embedding in enumerate(all_embeddings):
            if i != target_idx:
                similarity = np.dot(target_embedding, embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(embedding) + 1e-8
                )
                company_symbol = self.company_system.idx_to_company[i]
                similarities.append((company_symbol, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
        
    def get_similar_keywords_by_impact(self, target_word, top_k=10):
        """Find keywords with similar market impact patterns using learned embeddings"""
        if target_word not in self.tokenizer.word_to_idx:
            return []
        
        target_idx = self.tokenizer.word_to_idx[target_word]
        
        # Extract keyword embeddings from the model
        keyword_embedding_layer = self.model.get_layer('impact_aware_keyword_embeddings')
        all_embeddings = keyword_embedding_layer.get_weights()[0]  # Shape: (vocab_size, keyword_dim)
        
        if target_idx >= len(all_embeddings):
            return []
        
        target_embedding = all_embeddings[target_idx]
        similarities = []
        
        for word, idx in self.tokenizer.word_to_idx.items():
            if idx != target_idx and idx < len(all_embeddings) and word not in ['[PAD]', '[UNK]', '[CLS]', '[SEP]']:
                embedding = all_embeddings[idx]
                similarity = np.dot(target_embedding, embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(embedding) + 1e-8
                )
                similarities.append((word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def analyze_keyword_impact_clusters(self, sample_keywords, return_matrix=False):
        """Analyze how keywords cluster based on their learned impact patterns"""
        if not sample_keywords:
            return {}
        
        keyword_embedding_layer = self.model.get_layer('impact_aware_keyword_embeddings')
        all_embeddings = keyword_embedding_layer.get_weights()[0]
        
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
        similarity_matrix = np.dot(embeddings, embeddings.T) / (
            np.linalg.norm(embeddings, axis=1)[:, None] * np.linalg.norm(embeddings, axis=1)[None, :] + 1e-8
        )
        
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