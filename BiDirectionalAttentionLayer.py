import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Dense, LayerNormalization, MultiHeadAttention)
import logging

logger = logging.getLogger("AdvancedNewsFactor.BiDirectionalAttentionLayer")

class BiDirectionalAttentionLayer(tf.keras.layers.Layer):
    
    def __init__(self, latent_dim, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        
        self.company_to_news_attention = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=latent_dim // num_heads,
            name='company_to_news'
        )

        self.news_to_company_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=latent_dim // num_heads, 
            name='news_to_company'
        )

        self.combination_layer = Dense(latent_dim, activation='tanh', name='attention_combination')
        self.layer_norm = LayerNormalization()
        
    def call(self, inputs):
        company_features, keyword_features = inputs
        
        # Company-to-News attention: what news aspects are relevant for this company
        company_to_news_output, company_attention_scores = self.company_to_news_attention(
            query=company_features,    # (batch, 1, latent_dim)
            key=keyword_features,      # (batch, max_keywords, latent_dim)
            value=keyword_features,    # (batch, max_keywords, latent_dim)
            return_attention_scores=True
        )

        # News-to-Company attention: how should company features be weighted given the news
        news_to_company_output, news_attention_scores = self.news_to_company_attention(
            query=keyword_features,
            key=company_features,
            value=company_features,
            return_attention_scores=True
        )
        
        # Pool the news-to-company attention
        news_to_company_pooled = tf.reduce_mean(news_to_company_output, axis=1, keepdims=True)
        
        # Combine both attention outputs
        combined = tf.concat([company_to_news_output, news_to_company_pooled], axis=-1)
        combined_projected = self.combination_layer(combined)
        combined_normalized = self.layer_norm(combined_projected)
        
        return {
            'combined_output': combined_normalized,
            'company_to_news_scores': company_attention_scores,
            'news_to_company_scores': news_attention_scores,
            'news_to_company_pooled': news_to_company_pooled
        }
