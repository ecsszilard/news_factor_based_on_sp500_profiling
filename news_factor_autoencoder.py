import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dense, Attention, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import logging
from typing import Dict, List, Tuple, Optional, Set
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict

# Logging beállítás
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CefsetEntity:
    """Cefset entitás (cég vagy központi bank)"""
    symbol: str
    market_cap: float  # USD-ben
    sector: str
    embedding: np.ndarray
    fuzzy_membership: Dict[str, float]  # fuzzy halmaztagság különböző kategóriákban
    m2_supply: Optional[float] = None  # csak FED-nél

class CefsetNewsAnalyzer:
    def __init__(self, embedding_dim=768, company_dim=128, latent_dim=64, 
                 keyword_dim=256, fuzzy_dim=32, capitalization_vector_dim=16):
        """
        Cefset-alapú integrált hírfaktor és cégelemző rendszer
        
        Args:
            embedding_dim: Hír embedding dimenzió
            company_dim: Cég profil dimenzió  
            latent_dim: Látens reprezentáció dimenzió
            keyword_dim: Kulcsszó embedding dimenzió
            fuzzy_dim: Fuzzy halmaz dimenzió
            capitalization_vector_dim: Kapitalizációs hatásvektor dimenzió
        """
        self.embedding_dim = embedding_dim
        self.company_dim = company_dim
        self.latent_dim = latent_dim
        self.keyword_dim = keyword_dim
        self.fuzzy_dim = fuzzy_dim
        self.capitalization_vector_dim = capitalization_vector_dim
        
        # Kibővített entitások FED-del
        self.symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'FED']
        
        # Market cap adatok (milliárd USD) - FED esetén M2 money supply
        self.base_market_caps = {
            'AAPL': 3200.0, 'MSFT': 2800.0, 'GOOG': 1600.0, 'AMZN': 1400.0,
            'TSLA': 800.0, 'META': 750.0, 'NVDA': 1800.0,
            'FED': 21000.0  # M2 money supply (trillions -> billions)
        }
        
        # Szektor mapping
        self.sector_mapping = {
            'AAPL': 'tech_hardware', 'MSFT': 'tech_software', 'GOOG': 'tech_internet',
            'AMZN': 'tech_ecommerce', 'TSLA': 'automotive', 'META': 'tech_social',
            'NVDA': 'tech_semiconductor', 'FED': 'monetary_policy'
        }
        
        # Cefset entitások inicializálása
        self.cefsets = self.initialize_cefsets()
        
        # Kulcsszó kategóriák és embeddings
        self.keyword_categories = {
            'earnings': ['revenue', 'profit', 'earnings', 'quarterly', 'financial'],
            'product': ['launch', 'product', 'innovation', 'technology', 'patent'],
            'market': ['market', 'competition', 'share', 'customer', 'demand'],
            'regulation': ['regulation', 'policy', 'government', 'tax', 'compliance'],
            'monetary': ['interest', 'inflation', 'fed', 'monetary', 'rate', 'stimulus'],
            'risk': ['risk', 'volatility', 'uncertainty', 'crisis', 'debt'],
            'growth': ['growth', 'expansion', 'acquisition', 'investment', 'scale'],
            'sentiment': ['bullish', 'bearish', 'optimistic', 'pessimistic', 'confidence']
        }
        
        self.keyword_embeddings = {
            category: np.random.rand(self.keyword_dim) 
            for category in self.keyword_categories
        }
        
        # Fuzzy halmazok definiálása
        self.fuzzy_sets = {
            'tech_giants': {'AAPL': 0.9, 'MSFT': 0.9, 'GOOG': 0.8, 'META': 0.8, 'AMZN': 0.7},
            'growth_stocks': {'TSLA': 0.9, 'NVDA': 0.8, 'META': 0.7, 'AMZN': 0.6},
            'value_stocks': {'AAPL': 0.7, 'MSFT': 0.8, 'GOOG': 0.6},
            'ai_exposed': {'NVDA': 0.95, 'MSFT': 0.8, 'GOOG': 0.7, 'TSLA': 0.6},
            'interest_sensitive': {'TSLA': 0.8, 'META': 0.7, 'FED': 1.0, 'NVDA': 0.6},
            'defensive': {'AAPL': 0.6, 'MSFT': 0.7, 'FED': 0.3},
            'monetary_policy': {'FED': 1.0, 'AAPL': 0.2, 'TSLA': 0.4}
        }
        
        # Neural network komponensek
        self.keyword_encoder = self.build_keyword_encoder()
        self.company_encoder = self.build_company_encoder()
        self.fuzzy_projector = self.build_fuzzy_projector()
        self.cefset_attention = self.build_cefset_attention()
        self.capitalization_predictor = self.build_capitalization_predictor()
        self.reconstruction_decoder = self.build_reconstruction_decoder()
        
        # Történelmi kapitalizációs arányok
        self.historical_ratios = self.initialize_capitalization_ratios()
        
        # Cefset faktortár
        self.cefset_factors = {symbol: [] for symbol in self.symbols}
        
    def initialize_cefsets(self) -> Dict[str, CefsetEntity]:
        """Cefset entitások inicializálása"""
        cefsets = {}
        
        for symbol in self.symbols:
            # Alapvető fuzzy tagságok
            fuzzy_membership = {}
            for fuzzy_set, members in self.fuzzy_sets.items():
                fuzzy_membership[fuzzy_set] = members.get(symbol, 0.0)
            
            # M2 supply csak FED-nél
            m2_supply = self.base_market_caps[symbol] if symbol == 'FED' else None
            
            cefset = CefsetEntity(
                symbol=symbol,
                market_cap=self.base_market_caps[symbol],
                sector=self.sector_mapping[symbol],
                embedding=np.random.rand(self.company_dim),
                fuzzy_membership=fuzzy_membership,
                m2_supply=m2_supply
            )
            
            cefsets[symbol] = cefset
            
        return cefsets
    
    def initialize_capitalization_ratios(self) -> Dict[str, Dict[str, float]]:
        """Kapitalizációs arányok inicializálása"""
        total_market_cap = sum(self.base_market_caps.values())
        
        ratios = {}
        for symbol1 in self.symbols:
            ratios[symbol1] = {}
            for symbol2 in self.symbols:
                if symbol1 == symbol2:
                    ratios[symbol1][symbol2] = 1.0
                else:
                    # Relatív kapitalizációs arány
                    ratio = self.base_market_caps[symbol1] / self.base_market_caps[symbol2]
                    ratios[symbol1][symbol2] = ratio
                    
        return ratios
    
    def build_keyword_encoder(self) -> Model:
        """Kulcsszó mintázat encoder"""
        input_layer = Input(shape=(len(self.keyword_categories),), name="keyword_pattern")
        x = BatchNormalization()(input_layer)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        keyword_latent = Dense(self.latent_dim, activation='relu', name='keyword_latent')(x)
        
        return Model(inputs=input_layer, outputs=keyword_latent, name="KeywordEncoder")
    
    def build_company_encoder(self) -> Model:
        """Cég profil encoder"""
        input_layer = Input(shape=(self.company_dim,), name="company_profile")
        x = BatchNormalization()(input_layer)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        company_latent = Dense(self.latent_dim, activation='relu', name='company_latent')(x)
        
        return Model(inputs=input_layer, outputs=company_latent, name="CompanyEncoder")
    
    def build_fuzzy_projector(self) -> Model:
        """Fuzzy halmaz projektor - cégprofilból fuzzy tagságokba"""
        company_input = Input(shape=(self.latent_dim,), name='company_latent')
        
        x = Dense(64, activation='relu')(company_input)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        
        # Minden fuzzy halmazra tagságot becsülünk
        fuzzy_outputs = []
        for fuzzy_set in self.fuzzy_sets.keys():
            membership = Dense(1, activation='sigmoid', name=f'fuzzy_{fuzzy_set}')(x)
            fuzzy_outputs.append(membership)
        
        fuzzy_vector = Concatenate(name='fuzzy_memberships')(fuzzy_outputs)
        
        return Model(inputs=company_input, outputs=fuzzy_vector, name="FuzzyProjector")
    
    def build_cefset_attention(self) -> Model:
        """Cefset közötti attention mechanizmus"""
        keyword_input = Input(shape=(self.latent_dim,), name='keyword_latent')
        cefset_input = Input(shape=(len(self.symbols), self.latent_dim), name='all_cefsets')
        
        # Kulcsszó query-ként
        keyword_query = layers.Reshape((1, self.latent_dim))(keyword_input)
        
        # Multi-head attention a cefset-ek között
        attention_output = Attention(use_scale=True)([keyword_query, cefset_input, cefset_input])
        
        # Flatten és további feldolgozás
        flattened = layers.Flatten()(attention_output)
        x = Dense(128, activation='relu')(flattened)
        x = Dropout(0.2)(x)
        cefset_context = Dense(64, activation='relu', name='cefset_context')(x)
        
        return Model(inputs=[keyword_input, cefset_input], outputs=cefset_context, name="CefsetAttention")
    
    def build_capitalization_predictor(self) -> Model:
        """Kapitalizációs arányváltozás előrejelző"""
        context_input = Input(shape=(64,), name='cefset_context')
        
        x = Dense(128, activation='relu')(context_input)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        
        # Minden cefset párra kapitalizációs arányváltozást jósolunk
        # Ez egy NxN mátrix lesz, ahol N = len(symbols)
        n_symbols = len(self.symbols)
        
        # Egyszerűsítés: minden symbol-ra relatív változást jósolunk
        ratio_changes = []
        for symbol in self.symbols:
            change = Dense(1, activation='tanh', name=f'ratio_change_{symbol}')(x)
            ratio_changes.append(change)
        
        ratio_vector = Concatenate(name='capitalization_changes')(ratio_changes)
        
        return Model(inputs=context_input, outputs=ratio_vector, name="CapitalizationPredictor")
    
    def build_reconstruction_decoder(self) -> Model:
        """Rekonstrukciós decoder - ellenőrzi a tanulást"""
        combined_input = Input(shape=(64 + len(self.symbols),), name='combined_features')
        
        x = Dense(128, activation='relu')(combined_input)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        
        # Eredeti kulcsszó mintázat rekonstrukciója
        keyword_reconstruction = Dense(
            len(self.keyword_categories), 
            activation='sigmoid', 
            name='keyword_reconstruction'
        )(x)
        
        # Cég fuzzy tagság rekonstrukciója
        fuzzy_reconstruction = Dense(
            len(self.fuzzy_sets), 
            activation='sigmoid', 
            name='fuzzy_reconstruction'
        )(x)
        
        return Model(
            inputs=combined_input, 
            outputs=[keyword_reconstruction, fuzzy_reconstruction], 
            name="ReconstructionDecoder"
        )
    
    def compile_models(self):
        """Modellek kompilálása"""
        self.capitalization_predictor.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.reconstruction_decoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=['mse', 'mse'],
            loss_weights=[0.5, 0.5],
            metrics=['mae']
        )
        
        logger.info("Modellek kompilálva")
    
    def extract_keyword_pattern(self, news_text: str = None, 
                              news_embedding: np.ndarray = None) -> np.ndarray:
        """Kulcsszó mintázat kinyerése hírből"""
        # Szimulált implementáció - valós esetben NLP feldolgozás
        if news_embedding is not None:
            # Embedding alapú pattern extraction
            pattern = np.random.rand(len(self.keyword_categories))
        else:
            # Text alapú (dummy implementation)
            pattern = np.random.rand(len(self.keyword_categories))
        
        # Normalize
        return pattern / np.sum(pattern)
    
    def get_all_cefset_embeddings(self) -> np.ndarray:
        """Összes cefset embedding mátrix"""
        embeddings = []
        for symbol in self.symbols:
            company_latent = self.company_encoder.predict(
                self.cefsets[symbol].embedding.reshape(1, -1), 
                verbose=0
            )
            embeddings.append(company_latent.flatten())
        
        return np.array(embeddings).reshape(1, len(self.symbols), self.latent_dim)
    
    def analyze_news_cefset_impact(self, news_text: str = None, 
                                 news_embedding: np.ndarray = None) -> Dict:
        """
        Komprehenzív cefset hatáselemzés
        """
        results = {
            'timestamp': datetime.now(),
            'keyword_pattern': {},
            'cefset_impacts': {},
            'capitalization_changes': {},
            'fuzzy_set_effects': {},
            'reconstruction_quality': {},
            'summary': {}
        }
        
        # 1. Kulcsszó mintázat kinyerése
        keyword_pattern = self.extract_keyword_pattern(news_text, news_embedding)
        keyword_latent = self.keyword_encoder.predict(
            keyword_pattern.reshape(1, -1), verbose=0
        )
        
        results['keyword_pattern'] = {
            category: float(weight) 
            for category, weight in zip(self.keyword_categories.keys(), keyword_pattern)
        }
        
        # 2. Cefset embeddings
        all_cefsets = self.get_all_cefset_embeddings()
        
        # 3. Cefset attention és kontextus
        cefset_context = self.cefset_attention.predict(
            [keyword_latent, all_cefsets], verbose=0
        )
        
        # 4. Kapitalizációs változások előrejelzése
        cap_changes = self.capitalization_predictor.predict(cefset_context, verbose=0)
        
        results['capitalization_changes'] = {
            symbol: float(change) 
            for symbol, change in zip(self.symbols, cap_changes.flatten())
        }
        
        # 5. Fuzzy halmaz hatások számítása
        fuzzy_effects = self.calculate_fuzzy_set_effects(
            keyword_pattern, cap_changes.flatten()
        )
        results['fuzzy_set_effects'] = fuzzy_effects
        
        # 6. Rekonstrukciós minőség ellenőrzése
        combined_features = np.concatenate([
            cefset_context.flatten(), 
            cap_changes.flatten()
        ]).reshape(1, -1)
        
        keyword_recon, fuzzy_recon = self.reconstruction_decoder.predict(
            combined_features, verbose=0
        )
        
        # Rekonstrukciós hibák
        keyword_error = np.mean(np.abs(keyword_pattern - keyword_recon.flatten()))
        
        # Fuzzy tagságok aktuális értékei
        current_fuzzy = np.array([
            list(self.cefsets[symbol].fuzzy_membership.values())[0:len(self.fuzzy_sets)]
            for symbol in self.symbols
        ]).mean(axis=0)
        
        fuzzy_error = np.mean(np.abs(current_fuzzy - fuzzy_recon.flatten()))
        
        results['reconstruction_quality'] = {
            'keyword_reconstruction_error': float(keyword_error),
            'fuzzy_reconstruction_error': float(fuzzy_error),
            'overall_reconstruction_quality': float(1.0 - (keyword_error + fuzzy_error) / 2)
        }
        
        # 7. Cefset-szintű hatások
        for i, symbol in enumerate(self.symbols):
            cefset = self.cefsets[symbol]
            cap_change = cap_changes.flatten()[i]
            
            # Új kapitalizációs arány számítása
            new_market_cap = cefset.market_cap * (1 + cap_change)
            
            # Relatív hatások más cefset-ekre
            relative_effects = {}
            for j, other_symbol in enumerate(self.symbols):
                if symbol != other_symbol:
                    old_ratio = self.historical_ratios[symbol][other_symbol]
                    new_ratio = new_market_cap / self.cefsets[other_symbol].market_cap
                    ratio_change = (new_ratio - old_ratio) / old_ratio
                    relative_effects[other_symbol] = float(ratio_change)
            
            results['cefset_impacts'][symbol] = {
                'direct_cap_change': float(cap_change),
                'new_market_cap': float(new_market_cap),
                'relative_effects': relative_effects,
                'sector': cefset.sector,
                'fuzzy_exposure': {
                    fs: cefset.fuzzy_membership.get(fs, 0.0) 
                    for fs in self.fuzzy_sets.keys()
                }
            }
        
        # 8. Összefoglaló
        max_cap_change = max(abs(c) for c in cap_changes.flatten())
        most_affected = self.symbols[np.argmax(np.abs(cap_changes.flatten()))]
        
        dominant_keyword = max(
            results['keyword_pattern'].items(), 
            key=lambda x: x[1]
        )[0]
        
        results['summary'] = {
            'max_capitalization_change': float(max_cap_change),
            'most_affected_cefset': most_affected,
            'dominant_keyword_category': dominant_keyword,
            'total_market_disruption': float(np.sum(np.abs(cap_changes.flatten()))),
            'reconstruction_quality': results['reconstruction_quality']['overall_reconstruction_quality'],
            'fuzzy_sets_activated': len([
                fs for fs, effect in fuzzy_effects.items() 
                if effect['total_impact'] > 0.01
            ])
        }
        
        return results
    
    def calculate_fuzzy_set_effects(self, keyword_pattern: np.ndarray, 
                                  cap_changes: np.ndarray) -> Dict:
        """Fuzzy halmazok hatásainak számítása"""
        fuzzy_effects = {}
        
        for fuzzy_set, members in self.fuzzy_sets.items():
            total_impact = 0.0
            weighted_change = 0.0
            member_impacts = {}
            
            for symbol, membership in members.items():
                if symbol in self.symbols:
                    symbol_idx = self.symbols.index(symbol)
                    cap_change = cap_changes[symbol_idx]
                    
                    # Membership-súlyozott hatás
                    weighted_impact = membership * cap_change
                    total_impact += abs(weighted_impact)
                    weighted_change += weighted_impact
                    
                    member_impacts[symbol] = {
                        'membership': float(membership),
                        'cap_change': float(cap_change),
                        'weighted_impact': float(weighted_impact)
                    }
            
            fuzzy_effects[fuzzy_set] = {
                'total_impact': float(total_impact),
                'weighted_change': float(weighted_change),
                'member_impacts': member_impacts,
                'set_coherence': float(1.0 - np.std([
                    m['cap_change'] for m in member_impacts.values()
                ]) if member_impacts else 0.0)
            }
        
        return fuzzy_effects
    
    def update_cefset_factors(self, analysis_result: Dict):
        """Cefset faktorok frissítése elemzés alapján"""
        for symbol in self.symbols:
            if symbol in analysis_result['cefset_impacts']:
                impact_data = analysis_result['cefset_impacts'][symbol]
                
                factor_entry = {
                    'keyword_pattern': analysis_result['keyword_pattern'],
                    'cap_change': impact_data['direct_cap_change'],
                    'relative_effects': impact_data['relative_effects'],
                    'fuzzy_exposure': impact_data['fuzzy_exposure'],
                    'timestamp': analysis_result['timestamp'],
                    'reconstruction_quality': analysis_result['reconstruction_quality']['overall_reconstruction_quality']
                }
                
                self.cefset_factors[symbol].append(factor_entry)
                
                # Történelmi limitálás
                if len(self.cefset_factors[symbol]) > 200:
                    self.cefset_factors[symbol] = self.cefset_factors[symbol][-200:]
    
    def train_on_historical_data(self, historical_data: List[Dict], epochs: int = 10):
        """
        Történelmi adatokon tanítás
        
        Args:
            historical_data: Lista dict-ekről, mindegyik tartalmaz:
                - keyword_pattern: kulcsszó mintázat
                - capitalization_changes: valós kapitalizációs változások
                - cefset_states: cefset állapotok
        """
        if not historical_data:
            logger.warning("Nincs történelmi adat a tanításhoz")
            return
        
        X_context, y_cap_changes = [], []
        X_recon, y_keywords, y_fuzzy = [], [], []
        
        for data_point in historical_data:
            keyword_pattern = np.array(list(data_point['keyword_pattern'].values()))
            cap_changes = np.array(list(data_point['capitalization_changes'].values()))
            
            # Kontextus generálása
            keyword_latent = self.keyword_encoder.predict(
                keyword_pattern.reshape(1, -1), verbose=0
            )
            all_cefsets = self.get_all_cefset_embeddings()
            cefset_context = self.cefset_attention.predict(
                [keyword_latent, all_cefsets], verbose=0
            )
            
            X_context.append(cefset_context.flatten())
            y_cap_changes.append(cap_changes)
            
            # Rekonstrukciós adatok
            combined_features = np.concatenate([
                cefset_context.flatten(), cap_changes
            ])
            X_recon.append(combined_features)
            y_keywords.append(keyword_pattern)
            
            # Fuzzy states generálása (egyszerűsített)
            fuzzy_state = np.random.rand(len(self.fuzzy_sets))
            y_fuzzy.append(fuzzy_state)
        
        if X_context and X_recon:
            X_context = np.array(X_context)
            y_cap_changes = np.array(y_cap_changes)
            X_recon = np.array(X_recon)
            y_keywords = np.array(y_keywords)
            y_fuzzy = np.array(y_fuzzy)
            
            # Kapitalizációs előrejelző tanítása
            logger.info("Kapitalizációs előrejelző tanítása...")
            self.capitalization_predictor.fit(
                X_context, y_cap_changes,
                epochs=epochs, batch_size=16, validation_split=0.2, verbose=1
            )
            
            # Rekonstrukciós decoder tanítása
            logger.info("Rekonstrukciós decoder tanítása...")
            self.reconstruction_decoder.fit(
                X_recon, [y_keywords, y_fuzzy],
                epochs=epochs, batch_size=16, validation_split=0.2, verbose=1
            )
            
            logger.info(f"Tanítás befejezve {len(historical_data)} mintán")
        
    def get_cefset_summary(self) -> Dict:
        """Cefset rendszer összefoglalója"""
        summary = {
            'total_cefsets': len(self.symbols),
            'total_market_cap': sum(cefset.market_cap for cefset in self.cefsets.values()),
            'fuzzy_sets': len(self.fuzzy_sets),
            'keyword_categories': len(self.keyword_categories),
            'cefset_details': {},
            'fuzzy_set_stats': {},
            'recent_activity': {}
        }
        
        # Cefset részletek
        for symbol, cefset in self.cefsets.items():
            summary['cefset_details'][symbol] = {
                'market_cap': cefset.market_cap,
                'sector': cefset.sector,
                'fuzzy_memberships': cefset.fuzzy_membership,
                'recent_factors': len(self.cefset_factors.get(symbol, [])),
                'is_fed': symbol == 'FED',
                'm2_supply': cefset.m2_supply
            }
        
        # Fuzzy halmaz statisztikák
        for fuzzy_set, members in self.fuzzy_sets.items():
            total_cap = sum(
                self.cefsets[symbol].market_cap * membership
                for symbol, membership in members.items()
                if symbol in self.cefsets
            )
            
            summary['fuzzy_set_stats'][fuzzy_set] = {
                'member_count': len(members),
                'total_weighted_cap': total_cap,
                'avg_membership': np.mean(list(members.values())),
                'members': members
            }
        
        return summary


def demo_cefset_usage():
    """Cefset rendszer demonstrációs használat"""
    print("=== Cefset-alapú Hírelemző Rendszer Demo ===\n")
    
    # Rendszer inicializálása
    analyzer = CefsetNewsAnalyzer()
    analyzer.compile_models()
    
    print("1. Cefset rendszer áttekintése:\n")
    summary = analyzer.get_cefset_summary()
    print(f"Összesen {summary['total_cefsets']} cefset")
    print(f"Teljes piaci kapitalizáció: ${summary['total_market_cap']:.1f}B")
    print(f"Fuzzy halmazok száma: {summary['fuzzy_sets']}")
    
    print("\nFuzzy halmazok:")
    for fuzzy_set, stats in summary['fuzzy_set_stats'].items():
        print(f"  {fuzzy_set}: {stats['member_count']} tag, "
              f"${stats['total_weighted_cap']:.1f}B súlyozott kapitalizáció")
    
    print("\n" + "="*60 + "\n")
    
    print("2. Hírek cefset hatáselemzése:\n")
    
    # Szimulált hírek
    test_news = [
        "Federal Reserve raises interest rates by 0.25% to combat inflation",
        "Apple reports record quarterly earnings beating analyst expectations", 
        "Tesla announces new gigafactory expansion in Texas",
        "AI chip demand surges as tech companies increase spending"
    ]
    
    for i, news in enumerate(test_news):
        print(f"--- Hír #{i+1}: {news[:50]}... ---")
        
        # Cefset hatáselemzés
        analysis = analyzer.analyze_news_cefset_impact(news_text=news)
        
        print(f"Domináns kulcsszó