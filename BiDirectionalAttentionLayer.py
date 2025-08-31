import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Dense, LayerNormalization, MultiHeadAttention)
import logging

logger = logging.getLogger("AdvancedNewsFactor.BiDirectionalAttentionLayer")

class BiDirectionalAttentionLayer(tf.keras.layers.Layer):
    """
    Bi-directional attention layer: cég→hír és hír→cégek
    """
    
    def __init__(self, latent_dim, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        
        # Cég → Hír attention
        self.company_to_news_attention = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=latent_dim // num_heads,
            name='company_to_news'
        )
        
        # Hír → Cégek attention (ha több cég van)
        self.news_to_company_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=latent_dim // num_heads, 
            name='news_to_company'
        )
        
        # Financial attention súlyozás
        self.financial_weight_layer = Dense(1, activation='sigmoid', name='financial_weight')
        
        # Kombinációs rétegek
        self.combination_layer = Dense(latent_dim, activation='tanh', name='attention_combination')
        self.layer_norm = LayerNormalization()
        
    def call(self, inputs, financial_weights=None):
        """
        inputs: [company_features, keyword_features]
        financial_weights: (batch, max_keywords) - pénzügyi súlyok
        """
        company_features, keyword_features = inputs
        batch_size = tf.shape(company_features)[0]
        
        # 1. Cég → Hír attention: mely szavak fontosak a cégnek
        company_to_news_output, company_attention_scores = self.company_to_news_attention(
            query=company_features,    # (batch, 1, latent_dim)
            key=keyword_features,      # (batch, max_keywords, latent_dim)
            value=keyword_features,    # (batch, max_keywords, latent_dim)
            return_attention_scores=True
        )
        
        # 2. Hír → Cégek attention: ha több cég lenne, mely cégek relevánsak
        # Most 1 cég van, de a struktúra készen áll többre
        news_to_company_output, news_attention_scores = self.news_to_company_attention(
            query=keyword_features,    # (batch, max_keywords, latent_dim) 
            key=company_features,      # (batch, 1, latent_dim)
            value=company_features,    # (batch, 1, latent_dim)
            return_attention_scores=True
        )
        
        # 3. Financial attention súlyok alkalmazása
        if financial_weights is not None:
            # financial_weights: (batch, max_keywords, 1)
            financial_weights_expanded = tf.expand_dims(financial_weights, axis=-1)
            
            # Kombináljuk a szemantikai és pénzügyi figyelem súlyokat
            semantic_weight = 0.6
            financial_weight = 0.4
            
            # Átlagolt attention a company_to_news-ból
            avg_company_attention = tf.reduce_mean(company_attention_scores, axis=1, keepdims=True)  # (batch, 1, max_keywords)
            avg_company_attention = tf.transpose(avg_company_attention, [0, 2, 1])  # (batch, max_keywords, 1)
            
            combined_attention = (
                semantic_weight * avg_company_attention +
                financial_weight * financial_weights_expanded
            )
            
            # Kombinált súlyokkal újrasúlyozzuk a keyword_features-t
            weighted_keywords = keyword_features * combined_attention  # broadcasting
        else:
            weighted_keywords = keyword_features
        
        # 4. Bi-directional információ kombinálása
        # Company-to-news: (batch, 1, latent_dim)
        # News-to-company: (batch, max_keywords, latent_dim) - globálisan összegezzük
        news_to_company_pooled = tf.reduce_mean(news_to_company_output, axis=1, keepdims=True)  # (batch, 1, latent_dim)
        
        # Kombináljuk a két irányt
        combined = tf.concat([company_to_news_output, news_to_company_pooled], axis=-1)  # (batch, 1, 2*latent_dim)
        combined_projected = self.combination_layer(combined)  # (batch, 1, latent_dim)
        combined_normalized = self.layer_norm(combined_projected)
        
        return {
            'combined_output': combined_normalized,
            'company_to_news_scores': company_attention_scores,
            'news_to_company_scores': news_attention_scores,
            'weighted_keywords': weighted_keywords
        }