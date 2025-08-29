import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Embedding, Attention, Dropout, 
                                   LayerNormalization, MultiHeadAttention, Reshape, 
                                   Flatten, Concatenate, BatchNormalization, Add)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, AutoModel
import torch
import datetime
import time
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import pickle
import os
from collections import defaultdict
import re

logger = logging.getLogger("AdvancedNewsFactor.CompanyEmbeddingSystem")

class CompanyEmbeddingSystem:
    """
    Cégek vektortérbe ágyazása hírek, fundamentumok és árfolyammozgások alapján
    """
    
    def __init__(self, companies_file='sp500_companies.csv', embedding_dim=512):
        """
        Inicializálja a cégek beágyazási rendszerét
        
        Paraméterek:
            companies_file (str): S&P 500 cégek adatait tartalmazó fájl
            embedding_dim (int): A beágyazási dimenzió mérete
        """
        self.embedding_dim = embedding_dim
        self.companies_df = pd.read_csv(companies_file)
        self.companies = self.companies_df['symbol'].tolist()
        
        # BERT tokenizer és model inicializálása
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()
        
        # Cégek beágyazási vektorai
        self.company_embeddings = {}
        
        # Fundamentális adatok normalizálása
        self.fundamental_scaler = StandardScaler()
        
        # Árfolyammozgás adatok
        self.price_movements = {}
        
        # Hírek beágyazása
        self.news_embeddings = []
        self.news_metadata = []
        
        logger.info(f"CompanyEmbeddingSystem inicializálva {len(self.companies)} céggel")
    
    def get_bert_embedding(self, text, max_length=512):
        """
        BERT beágyazást készít szövegből
        
        Paraméterek:
            text (str): A szöveg
            max_length (int): Maximális token hossz
            
        Visszatérési érték:
            numpy.ndarray: BERT beágyazási vektor
        """
        # Tokenizálás
        inputs = self.tokenizer(text, return_tensors='pt', max_length=max_length, 
                              truncation=True, padding=True)
        
        # BERT forward pass
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            
        # [CLS] token beágyazása (első token)
        embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
        
        return embedding
    
    def create_company_embedding(self, symbol, news_texts=None, fundamental_data=None, 
                                price_data=None, sector_info=None):
        """
        Létrehozza egy cég teljes beágyazási vektorát
        
        Paraméterek:
            symbol (str): Cég szimbóluma
            news_texts (list): A céghez kapcsolódó hírek szövegei
            fundamental_data (dict): Fundamentális pénzügyi adatok
            price_data (dict): Árfolyammozgási adatok
            sector_info (dict): Szektoriális információk
            
        Visszatérési érték:
            numpy.ndarray: A cég teljes beágyazási vektora
        """
        embedding_components = []
        
        # 1. Hírek alapú beágyazás (BERT)
        if news_texts:
            # Összes hír összefűzése (truncate if too long)
            combined_news = " ".join(news_texts)[:2000]  # Max 2000 karakter
            news_embedding = self.get_bert_embedding(combined_news)
        else:
            news_embedding = np.zeros(768)  # BERT base embedding size
            
        embedding_components.append(news_embedding)
        
        # 2. Fundamentális adatok beágyazása
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

            fundamental_log = np.log1p(np.abs(fundamental_features)) * np.sign(fundamental_features) # Log transformation nagyságrendi különbségek kezelésére
            # Z-score normalizálás
            fundamental_normalized = (fundamental_log - np.mean(fundamental_log)) / (np.std(fundamental_log) + 1e-8)
            
            # Dimenzió kiterjesztése dense layerrel (10 -> 128)
            fundamental_extended = np.tile(fundamental_normalized, 13)[:128]
        else:
            fundamental_extended = np.zeros(128)
            
        embedding_components.append(fundamental_extended)
        
        # 3. Árfolyammozgás beágyazása
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
            
            # Normalizálás és kiterjesztés (8 -> 64)
            price_normalized = (price_features - np.mean(price_features)) / (np.std(price_features) + 1e-8)
            price_extended = np.tile(price_normalized, 8)[:64]
        else:
            price_extended = np.zeros(64)
            
        embedding_components.append(price_extended)
        
        # 4. Szektoriális információk
        if sector_info:
            # One-hot encoding szektorokhoz (egyszerűsített)
            sectors = ['Technology', 'Healthcare', 'Financials', 'Energy', 
                      'Consumer Discretionary', 'Industrials', 'Consumer Staples',
                      'Materials', 'Real Estate', 'Utilities', 'Communication Services']
            
            sector_vector = np.zeros(len(sectors))
            if sector_info.get('sector') in sectors:
                sector_idx = sectors.index(sector_info['sector'])
                sector_vector[sector_idx] = 1.0
                
            # Kiterjesztés 32 dimenzióra
            sector_extended = np.tile(sector_vector, 3)[:32]
        else:
            sector_extended = np.zeros(32)
            
        embedding_components.append(sector_extended)
        
        # Összes komponens összefűzése
        # 768 (BERT) + 128 (fundamental) + 64 (price) + 32 (sector) = 992
        full_embedding = np.concatenate(embedding_components)
        
        # Finális dimenzióra vágás/kiterjesztés ha szükséges
        if len(full_embedding) > self.embedding_dim:
            full_embedding = full_embedding[:self.embedding_dim]
        elif len(full_embedding) < self.embedding_dim:
            padding = np.zeros(self.embedding_dim - len(full_embedding))
            full_embedding = np.concatenate([full_embedding, padding])
        
        self.company_embeddings[symbol] = full_embedding
        
        logger.info(f"Beágyazás létrehozva: {symbol}")
        return full_embedding
    
    def find_similar_companies(self, symbol, top_k=5):
        """
        Megkeresi a leginkább hasonló cégeket a vektortérben
        
        Paraméterek:
            symbol (str): Keresett cég szimbóluma
            top_k (int): Visszaadandó hasonló cégek száma
            
        Visszatérési érték:
            list: Hasonló cégek listája hasonlósági pontszámmal
        """
        if symbol not in self.company_embeddings:
            return []
        
        target_embedding = self.company_embeddings[symbol]
        similarities = []
        
        for other_symbol, other_embedding in self.company_embeddings.items():
            if other_symbol != symbol:
                # Koszinusz hasonlóság
                similarity = np.dot(target_embedding, other_embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(other_embedding)
                )
                similarities.append((other_symbol, similarity))
        
        # Rendezés hasonlóság szerint
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
