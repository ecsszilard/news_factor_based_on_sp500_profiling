import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dense, Attention, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, List, Tuple, Optional
import random
from datetime import datetime, timedelta

# Logging beállítás
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedNewsAnalyzer:
    def __init__(self, embedding_dim=768, company_dim=128, latent_dim=64, impact_vector_dim=8):
        """
        Integrált hírfaktor és cégelemző rendszer
        
        Args:
            embedding_dim: Hír embedding dimenzió
            company_dim: Cég profil dimenzió  
            latent_dim: Látens reprezentáció dimenzió
            impact_vector_dim: Hatásvektor dimenzió (árfolyam + piaci eltolódások)
        """
        self.embedding_dim = embedding_dim
        self.company_dim = company_dim
        self.latent_dim = latent_dim
        self.impact_vector_dim = impact_vector_dim
        
        # Szimulált cégek és profilok
        self.symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA']
        self.company_profiles = {
            sym: np.random.rand(company_dim) for sym in self.symbols
        }
        
        # Szektor információk (relatív piaci eltolódásokhoz)
        self.sector_mapping = {
            'AAPL': 'tech_hardware', 'MSFT': 'tech_software', 'GOOG': 'tech_internet',
            'AMZN': 'tech_ecommerce', 'TSLA': 'automotive', 'META': 'tech_social',
            'NVDA': 'tech_semiconductor'
        }
        
        # Modell komponensek
        self.news_encoder = self.build_news_encoder()
        self.company_encoder = self.build_company_encoder()
        self.attention_model = self.build_attention_model()
        self.impact_predictor = self.build_impact_predictor()
        self.sector_analyzer = self.build_sector_analyzer()
        
        # Adattároló struktúrák
        self.company_news_factors = {sym: [] for sym in self.symbols}
        self.reliability_scores = {sym: 0.5 for sym in self.symbols}
        self.market_correlations = self.initialize_market_correlations()
        
        # Fuzzy rendszer komponensek (ha skicit-fuzzy használni akarjuk)
        self.fuzzy_weights = {
            'reliability': 0.3,
            'temporal': 0.2,
            'sector_impact': 0.25,
            'market_sentiment': 0.25
        }
        
    def initialize_market_correlations(self) -> Dict:
        """Piaci korrelációs mátrix inicializálása"""
        correlations = {}
        for sym1 in self.symbols:
            correlations[sym1] = {}
            for sym2 in self.symbols:
                if sym1 == sym2:
                    correlations[sym1][sym2] = 1.0
                else:
                    # Szimulált korrelációk szektorok alapján
                    sector1 = self.sector_mapping[sym1]
                    sector2 = self.sector_mapping[sym2]
                    if sector1 == sector2:
                        correlations[sym1][sym2] = np.random.uniform(0.6, 0.9)
                    elif sector1.startswith('tech') and sector2.startswith('tech'):
                        correlations[sym1][sym2] = np.random.uniform(0.3, 0.6)
                    else:
                        correlations[sym1][sym2] = np.random.uniform(0.1, 0.4)
        return correlations
    
    def build_news_encoder(self) -> Model:
        """Hír encoder felépítése"""
        input_layer = Input(shape=(self.embedding_dim,), name="news_embedding")
        x = BatchNormalization()(input_layer)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(self.latent_dim, activation='relu', name='news_latent')(x)
        return Model(inputs=input_layer, outputs=x, name="NewsEncoder")
    
    def build_company_encoder(self) -> Model:
        """Cég encoder felépítése"""
        input_layer = Input(shape=(self.company_dim,), name="company_features")
        x = BatchNormalization()(input_layer)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(self.latent_dim, activation='relu', name='company_latent')(x)
        return Model(inputs=input_layer, outputs=x, name="CompanyEncoder")
    
    def build_attention_model(self) -> Model:
        """Attention-alapú hír-cég kapcsolat modell"""
        news_input = Input(shape=(self.latent_dim,), name='news_latent_input')
        company_input = Input(shape=(self.latent_dim,), name='company_latent_input')
        
        # Kiterjesztés batch dimenzióra az attention-höz
        news_exp = layers.Reshape((1, self.latent_dim))(news_input)
        company_exp = layers.Reshape((1, self.latent_dim))(company_input)
        
        # Multi-head attention
        attn_layer = Attention(use_scale=True)([news_exp, company_exp])
        
        # Kombinálás
        combined = Concatenate(axis=-1)([news_exp, attn_layer])
        flatten = layers.Flatten()(combined)
        
        x = Dense(128, activation='relu')(flatten)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        attention_output = Dense(32, activation='relu', name='attention_features')(x)
        
        return Model(inputs=[news_input, company_input], outputs=attention_output, name="AttentionModel")
    
    def build_impact_predictor(self) -> Model:
        """Többdimenziós hatásvektor előrejelző"""
        attention_input = Input(shape=(32,), name='attention_features')
        
        # Főbb hatáskomponensek előrejelzése
        x = Dense(64, activation='relu')(attention_input)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        
        # Hatásvektor komponensek:
        # [0]: Direkt árfolyamhatás (-1 to 1)
        # [1]: Volatilitás változás (0 to 1) 
        # [2]: Szektoriális spillover (-1 to 1)
        # [3]: Piaci sentiment hatás (-1 to 1)
        # [4]: Időbeli persistencia (0 to 1)
        # [5]: Likviditási hatás (-1 to 1)
        # [6]: Relatív teljesítmény vs market (-1 to 1)
        # [7]: Kockázati prémium változás (-1 to 1)
        
        price_impact = Dense(1, activation='tanh', name='price_impact')(x)
        volatility_impact = Dense(1, activation='sigmoid', name='volatility_impact')(x)
        sector_spillover = Dense(1, activation='tanh', name='sector_spillover')(x)
        sentiment_impact = Dense(1, activation='tanh', name='sentiment_impact')(x)
        temporal_persistence = Dense(1, activation='sigmoid', name='temporal_persistence')(x)
        liquidity_impact = Dense(1, activation='tanh', name='liquidity_impact')(x)
        relative_performance = Dense(1, activation='tanh', name='relative_performance')(x)
        risk_premium = Dense(1, activation='tanh', name='risk_premium')(x)
        
        impact_vector = Concatenate(name='impact_vector')([
            price_impact, volatility_impact, sector_spillover, sentiment_impact,
            temporal_persistence, liquidity_impact, relative_performance, risk_premium
        ])
        
        return Model(inputs=attention_input, outputs=impact_vector, name="ImpactPredictor")
    
    def build_sector_analyzer(self) -> Model:
        """Szektoriális hatás elemző"""
        sector_input = Input(shape=(len(self.symbols),), name='sector_correlations')
        impact_input = Input(shape=(self.impact_vector_dim,), name='base_impact')
        
        x = Dense(32, activation='relu')(sector_input)
        sector_features = Dense(16, activation='relu')(x)
        
        combined = Concatenate()([impact_input, sector_features])
        x = Dense(64, activation='relu')(combined)
        x = Dense(32, activation='relu')(x)
        
        # Relatív piaci eltolódások minden cégre
        market_shifts = Dense(len(self.symbols), activation='tanh', name='market_shifts')(x)
        
        return Model(inputs=[sector_input, impact_input], outputs=market_shifts, name="SectorAnalyzer")
    
    def compile_models(self):
        """Modellek kompilálása"""
        self.impact_predictor.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.sector_analyzer.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Modellek kompilálva")
    
    def encode_news(self, news_emb: np.ndarray) -> np.ndarray:
        """Hír enkódolása"""
        if news_emb.ndim == 1:
            news_emb = news_emb.reshape(1, -1)
        return self.news_encoder.predict(news_emb, verbose=0)
    
    def encode_company(self, symbol: str) -> np.ndarray:
        """Cég enkódolása"""
        profile = self.company_profiles[symbol]
        return self.company_encoder.predict(profile.reshape(1, -1), verbose=0)
    
    def calculate_impact_vector(self, news_emb: np.ndarray, symbol: str) -> np.ndarray:
        """
        Hír-cég pár hatásvektorának kiszámítása
        
        Returns:
            8-dimenziós hatásvektor [price, volatility, sector, sentiment, 
                                   persistence, liquidity, relative, risk]
        """
        # Enkódolás
        news_latent = self.encode_news(news_emb)
        company_latent = self.encode_company(symbol)
        
        # Attention features
        attention_features = self.attention_model.predict(
            [news_latent, company_latent], verbose=0
        )
        
        # Alapvető hatásvektor
        impact_vector = self.impact_predictor.predict(attention_features, verbose=0)
        
        return impact_vector.flatten()
    
    def calculate_market_shifts(self, base_impact: np.ndarray, target_symbol: str) -> Dict[str, float]:
        """
        Relatív piaci eltolódások kiszámítása egy cég hatása alapján
        
        Args:
            base_impact: Alapvető hatásvektor
            target_symbol: Célcég szimbóluma
            
        Returns:
            Dict minden céggel és relatív eltolódásukkal
        """
        # Korrelációs vektor készítése
        correlations = np.array([
            self.market_correlations[target_symbol][sym] for sym in self.symbols
        ])
        
        # Piaci eltolódások számítása
        market_shifts = self.sector_analyzer.predict(
            [correlations.reshape(1, -1), base_impact.reshape(1, -1)], verbose=0
        )
        
        return {
            symbol: float(shift) for symbol, shift in 
            zip(self.symbols, market_shifts.flatten())
        }
    
    def analyze_news_comprehensive(self, news_emb: np.ndarray, 
                                 mentioned_companies: Optional[List[str]] = None) -> Dict:
        """
        Komprehenzív hírelemzés minden releváns cégre
        
        Args:
            news_emb: Hír embedding
            mentioned_companies: Explicitly mentioned companies (if any)
            
        Returns:
            Teljes elemzési eredmény dict
        """
        results = {
            'timestamp': datetime.now(),
            'direct_impacts': {},
            'market_effects': {},
            'sector_analysis': {},
            'summary': {}
        }
        
        # Ha nincsenek megadott cégek, minden céget elemzünk
        if mentioned_companies is None:
            companies_to_analyze = self.symbols
        else:
            companies_to_analyze = [c for c in mentioned_companies if c in self.symbols]
            if not companies_to_analyze:
                companies_to_analyze = self.symbols
        
        total_market_impact = np.zeros(len(self.symbols))
        
        # Minden cégre számítjuk a direkt hatást
        for symbol in companies_to_analyze:
            # Direkt hatásvektor
            impact_vector = self.calculate_impact_vector(news_emb, symbol)
            
            # Piaci eltolódások
            market_shifts = self.calculate_market_shifts(impact_vector, symbol)
            
            results['direct_impacts'][symbol] = {
                'price_impact': float(impact_vector[0]),
                'volatility_impact': float(impact_vector[1]),
                'sector_spillover': float(impact_vector[2]),
                'sentiment_impact': float(impact_vector[3]),
                'temporal_persistence': float(impact_vector[4]),
                'liquidity_impact': float(impact_vector[5]),
                'relative_performance': float(impact_vector[6]),
                'risk_premium_change': float(impact_vector[7]),
                'overall_magnitude': float(np.linalg.norm(impact_vector))
            }
            
            results['market_effects'][symbol] = market_shifts
            
            # Akkumuláljuk a teljes piaci hatást
            shifts_array = np.array([market_shifts[sym] for sym in self.symbols])
            total_market_impact += shifts_array * abs(impact_vector[0])  # súlyozás árhatással
        
        # Szektoriális elemzés
        sector_impacts = {}
        for sector in set(self.sector_mapping.values()):
            sector_companies = [s for s, sec in self.sector_mapping.items() if sec == sector]
            sector_impact = np.mean([
                results['direct_impacts'].get(sym, {}).get('overall_magnitude', 0)
                for sym in sector_companies if sym in results['direct_impacts']
            ])
            sector_impacts[sector] = float(sector_impact)
        
        results['sector_analysis'] = sector_impacts
        
        # Összefoglaló
        all_impacts = [
            data['overall_magnitude'] 
            for data in results['direct_impacts'].values()
        ]
        
        results['summary'] = {
            'total_companies_affected': len(results['direct_impacts']),
            'max_impact_magnitude': float(max(all_impacts)) if all_impacts else 0.0,
            'avg_impact_magnitude': float(np.mean(all_impacts)) if all_impacts else 0.0,
            'market_wide_effect': float(np.linalg.norm(total_market_impact)),
            'most_affected_company': max(
                results['direct_impacts'].items(), 
                key=lambda x: x[1]['overall_magnitude']
            )[0] if results['direct_impacts'] else None,
            'dominant_sector': max(
                sector_impacts.items(), 
                key=lambda x: x[1]
            )[0] if sector_impacts else None
        }
        
        return results
    
    def update_company_factors(self, symbol: str, news_emb: np.ndarray, 
                             impact_vector: np.ndarray):
        """Cég hírfaktorainak frissítése"""
        news_latent = self.encode_news(news_emb)
        
        factor_entry = {
            'news_factor': news_latent.flatten(),
            'impact_vector': impact_vector,
            'timestamp': datetime.now(),
            'reliability': self.reliability_scores.get(symbol, 0.5)
        }
        
        self.company_news_factors[symbol].append(factor_entry)
        
        # Csak az utolsó 100 faktort tartjuk meg
        if len(self.company_news_factors[symbol]) > 100:
            self.company_news_factors[symbol] = self.company_news_factors[symbol][-100:]
    
    def train_on_historical_data(self, news_embeddings: List[np.ndarray],
                               company_symbols: List[str],
                               impact_labels: List[np.ndarray],
                               epochs: int = 10):
        """
        Modell tanítása történelmi adatokon
        
        Args:
            news_embeddings: Lista hír embeddingekről
            company_symbols: Lista cég szimbólumokról
            impact_labels: Lista valós hatásvektorokról
        """
        X_news, X_company, X_attention, y = [], [], [], []
        
        for news_emb, symbol, impact_label in zip(news_embeddings, company_symbols, impact_labels):
            if symbol not in self.symbols:
                continue
                
            news_latent = self.encode_news(news_emb)
            company_latent = self.encode_company(symbol)
            
            attention_features = self.attention_model.predict(
                [news_latent, company_latent], verbose=0
            )
            
            X_attention.append(attention_features.flatten())
            y.append(impact_label)
        
        if X_attention and y:
            X_attention = np.array(X_attention)
            y = np.array(y)
            
            # Impact predictor tanítása
            self.impact_predictor.fit(
                X_attention, y, 
                epochs=epochs, 
                batch_size=32, 
                validation_split=0.2,
                verbose=1
            )
            
            logger.info(f"Modell tanítva {len(y)} mintán")
    
    def get_company_summary(self, symbol: str) -> Dict:
        """Cég összefoglaló információi"""
        factors = self.company_news_factors.get(symbol, [])
        
        if not factors:
            return {'symbol': symbol, 'total_factors': 0}
        
        recent_factors = [
            f for f in factors 
            if (datetime.now() - f['timestamp']).days <= 7
        ]
        
        avg_impact = np.mean([
            np.linalg.norm(f['impact_vector']) for f in recent_factors
        ]) if recent_factors else 0.0
        
        return {
            'symbol': symbol,
            'total_factors': len(factors),
            'recent_factors_7d': len(recent_factors),
            'avg_recent_impact': float(avg_impact),
            'reliability_score': self.reliability_scores.get(symbol, 0.5),
            'sector': self.sector_mapping.get(symbol, 'unknown'),
            'last_update': factors[-1]['timestamp'] if factors else None
        }


def demo_usage():
    """Demonstrációs használat"""
    print("=== Integrált Hírfaktor Elemző Rendszer Demo ===\n")
    
    # Rendszer inicializálása
    analyzer = IntegratedNewsAnalyzer()
    analyzer.compile_models()
    
    # Szimulált hírek (pl. BERT embeddings)
    dummy_news = [
        np.random.rand(768),  # Apple bevétel hír
        np.random.rand(768),  # Microsoft cloud hír  
        np.random.rand(768),  # Tech szektor hír
        np.random.rand(768),  # Piaci volatilitási hír
    ]
    
    print("1. Hírek elemzése:\n")
    
    # Minden hírre komprehenzív elemzés
    for i, news_emb in enumerate(dummy_news):
        print(f"--- Hír #{i+1} Elemzése ---")
        
        # Komprehenzív elemzés
        analysis = analyzer.analyze_news_comprehensive(
            news_emb, 
            mentioned_companies=['AAPL', 'MSFT'] if i < 2 else None
        )
        
        # Eredmények kiírása
        print(f"Érintett cégek száma: {analysis['summary']['total_companies_affected']}")
        print(f"Max hatás nagysága: {analysis['summary']['max_impact_magnitude']:.4f}")
        print(f"Piaci szintű hatás: {analysis['summary']['market_wide_effect']:.4f}")
        print(f"Leginkább érintett cég: {analysis['summary']['most_affected_company']}")
        print(f"Domináns szektor: {analysis['summary']['dominant_sector']}")
        
        # Top 3 direkt hatás
        sorted_impacts = sorted(
            analysis['direct_impacts'].items(),
            key=lambda x: x[1]['overall_magnitude'],
            reverse=True
        )[:3]
        
        print("\nTop 3 Direkt Hatás:")
        for symbol, impact_data in sorted_impacts:
            print(f"  {symbol}: {impact_data['overall_magnitude']:.4f} "
                  f"(ár: {impact_data['price_impact']:.3f}, "
                  f"volatilitás: {impact_data['volatility_impact']:.3f})")
        
        # Faktorok frissítése
        for symbol in analysis['direct_impacts']:
            impact_vector = np.array([
                analysis['direct_impacts'][symbol]['price_impact'],
                analysis['direct_impacts'][symbol]['volatility_impact'],
                analysis['direct_impacts'][symbol]['sector_spillover'],
                analysis['direct_impacts'][symbol]['sentiment_impact'],
                analysis['direct_impacts'][symbol]['temporal_persistence'],
                analysis['direct_impacts'][symbol]['liquidity_impact'],
                analysis['direct_impacts'][symbol]['relative_performance'],
                analysis['direct_impacts'][symbol]['risk_premium_change']
            ])
            analyzer.update_company_factors(symbol, news_emb, impact_vector)
        
        print("\n" + "="*50 + "\n")
    
    print("2. Cégek összefoglalója:\n")
    
    # Cégek jelenlegi állapota
    for symbol in analyzer.symbols:
        summary = analyzer.get_company_summary(symbol)
        print(f"{symbol}: {summary['total_factors']} faktor, "
              f"átlag hatás: {summary['avg_recent_impact']:.4f}, "
              f"megbízhatóság: {summary['reliability_score']:.3f}")
    
    print(f"\n3. Szimulált tanítás történelmi adatokon...")
    
    # Szimulált történelmi adatok
    historical_news = [np.random.rand(768) for _ in range(50)]
    historical_symbols = [random.choice(analyzer.symbols) for _ in range(50)]
    historical_impacts = [np.random.rand(8) * 0.5 - 0.25 for _ in range(50)]  # -0.25 to 0.25
    
    analyzer.train_on_historical_data(
        historical_news, 
        historical_symbols, 
        historical_impacts,
        epochs=3
    )
    
    print("Demo befejezve!")

if __name__ == "__main__":
    demo_usage()