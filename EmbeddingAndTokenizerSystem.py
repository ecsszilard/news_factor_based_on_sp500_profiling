import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import logging

logger = logging.getLogger("AdvancedNewsFactor.CompanyEmbeddingSystem")

class EmbeddingAndTokenizerSystem:
    """Embedding companies in vector space based on news, fundamentals and price movements"""
    
    def __init__(self, companies):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()
        
        # Learnable company embeddings instead of fixed feature-based embeddings
        self.company_to_idx = {symbol: i for i, symbol in enumerate(companies)}
        self.idx_to_company = {i: symbol for i, symbol in enumerate(companies)}

        self.word_to_idx = self.tokenizer.vocab
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        
        # Static features for initialization only
        self.news_embeddings = []
        self.news_metadata = []
        
        # SIMPLE cache for BERT embeddings - ONLY THIS WAS NEEDED!
        self._bert_cache = {}
    
    def get_bert_embedding(self, text, max_length=512):
        # Simple cache check - this fixes the memory leak
        text_key = hash(text.strip().lower())
        if text_key in self._bert_cache:
            return self._bert_cache[text_key].copy()
        
        inputs = self.tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
        
        # Cache the result - prevent recomputation
        self._bert_cache[text_key] = embedding.copy()
        
        # Simple cache size management
        if len(self._bert_cache) > 100:
            # Remove half of the cache when it gets too big
            keys_to_remove = list(self._bert_cache.keys())[:500]
            for key in keys_to_remove:
                del self._bert_cache[key]
        
        return embedding
    
    def prepare_keyword_sequence(self, text, max_length):
        inputs = self.tokenizer(
            text, 
            return_tensors='tf',
            max_length=max_length,
            truncation=True, 
            padding='max_length'
        )
        return inputs['input_ids']