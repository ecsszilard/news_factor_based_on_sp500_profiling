import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import logging

logger = logging.getLogger("AdvancedNewsFactor.CompanyEmbeddingSystem")

class CompanyEmbeddingSystem:
    """Embedding companies in vector space based on news, fundamentals and price movements"""
    
    def __init__(self, companies_file='sp500_companies.csv', embedding_dim=512):
        self.embedding_dim = embedding_dim
        self.companies_df = pd.read_csv(companies_file)
        self.companies = self.companies_df['symbol'].tolist()
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()
        
        # Learnable company embeddings instead of fixed feature-based embeddings
        self.num_companies = len(self.companies)
        self.company_to_idx = {symbol: i for i, symbol in enumerate(self.companies)}
        self.idx_to_company = {i: symbol for i, symbol in enumerate(self.companies)}
        
        # Static features for initialization only
        self.static_features = {}
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
        if len(self._bert_cache) > 1000:
            # Remove half of the cache when it gets too big
            keys_to_remove = list(self._bert_cache.keys())[:500]
            for key in keys_to_remove:
                del self._bert_cache[key]
        
        return embedding
    
    def store_static_features(self, symbol, fundamental_data=None, price_data=None, sector_info=None):
        """Store static features for company (used for embedding layer initialization)"""
        features = {}
        
        if fundamental_data:
            fundamental_features = np.array([
                fundamental_data.get('market_cap', 0),
                fundamental_data.get('pe_ratio', 15),
                fundamental_data.get('revenue_growth', 0),
                fundamental_data.get('profit_margin', 0.1),
                fundamental_data.get('debt_to_equity', 0.5),
                fundamental_data.get('roa', 0.05),
                fundamental_data.get('current_ratio', 1.5),
                fundamental_data.get('book_value', 100),
                fundamental_data.get('dividend_yield', 0.02),
                fundamental_data.get('beta', 1.0)
            ])
            features['fundamental'] = fundamental_features
        
        if price_data:
            price_features = np.array([
                price_data.get('volatility_30d', 0.2),
                price_data.get('return_1d', 0),
                price_data.get('return_5d', 0),
                price_data.get('return_20d', 0),
                price_data.get('return_60d', 0),
                price_data.get('volume_ratio', 1.0),
                price_data.get('momentum_score', 0),
                price_data.get('rsi', 50)
            ])
            features['price'] = price_features
        
        if sector_info:
            sectors = ['Technology', 'Healthcare', 'Financials', 'Energy', 
                      'Consumer Discretionary', 'Industrials', 'Consumer Staples',
                      'Materials', 'Real Estate', 'Utilities', 'Communication Services']
            
            sector_vector = np.zeros(len(sectors))
            if sector_info.get('sector') in sectors:
                sector_idx = sectors.index(sector_info['sector'])
                sector_vector[sector_idx] = 1.0
            features['sector'] = sector_vector
            
        self.static_features[symbol] = features
    
    def get_company_idx(self, symbol):
        return self.company_to_idx.get(symbol, 0)  # Return 0 for unknown companies