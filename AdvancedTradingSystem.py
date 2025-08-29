import numpy as np
import tensorflow as tf
import datetime
import logging
import pickle
import os

logger = logging.getLogger("AdvancedNewsFactor.AdvancedTradingSystem")

class AdvancedTradingSystem:
    """
    Fejlett kereskedési rendszer az attention-alapú modellel
    """
    
    def __init__(self, company_embedding_system, news_factor_model):
        """
        Inicializálja a fejlett kereskedési rendszert
        
        Paraméterek:
            company_embedding_system (CompanyEmbeddingSystem): Cég beágyazási rendszer
            news_factor_model (AttentionBasedNewsFactorModel): Hírfaktor modell
        """
        self.company_system = company_embedding_system
        self.news_model = news_factor_model
        
        # Kereskedési adatok
        self.positions = {}
        self.portfolio_value = 100000.0
        self.trade_history = []
        
        # Kockázatkezelés
        self.max_position_size = 0.05  # 5% max pozíció
        self.stop_loss_pct = 0.02      # 2% stop loss
        self.take_profit_pct = 0.06    # 6% take profit
        
        logger.info("AdvancedTradingSystem inicializálva")
    
    def analyze_news_impact(self, news_text, target_companies):
        """
        Elemzi egy hír hatását a célcégekre
        
        Paraméterek:
            news_text (str): A hír szövege
            target_companies (list): Célcégek listája
            
        Visszatérési érték:
            dict: Elemzési eredmények cégenként
        """
        # Kulcsszó szekvencia előkészítése
        keyword_sequence = self.news_model.prepare_keyword_sequence(news_text)
        
        results = {}
        
        for company in target_companies:
            if company not in self.company_system.company_embeddings:
                continue
                
            company_embedding = self.company_system.company_embeddings[company]
            
            # Kapitalizációs változás előrejelzése
            prediction = self.news_model.predict_capitalization_change(
                keyword_sequence, company_embedding
            )
            
            # Attention súlyok a magyarázhatóságért
            attention_info = self.news_model.get_attention_weights(
                keyword_sequence, company_embedding
            )
            
            results[company] = {
                'predicted_changes': {
                    '1d': prediction[0],
                    '5d': prediction[1],
                    '20d': prediction[2],
                    'volatility': prediction[3]
                },
                'attention_scores': attention_info['attention_scores'],
                'decoded_keywords': attention_info['decoded_keywords'],
                'confidence': np.abs(prediction).mean()  # Átlagos előrejelzési magabiztosság
            }
        
        return results
    
    def generate_trading_signals(self, news_analysis, risk_threshold=0.6):
        """
        Kereskedési jelzések generálása a hírelemzés alapján
        
        Paraméterek:
            news_analysis (dict): Hírelemzés eredményei
            risk_threshold (float): Kockázati küszöb
            
        Visszatérési érték:
            list: Kereskedési jelzések listája
        """
        signals = []
        
        for company, analysis in news_analysis.items():
            prediction_1d = analysis['predicted_changes']['1d']
            confidence = analysis['confidence']
            
            # Jelzés csak ha elég magabiztos a modell
            if confidence < risk_threshold:
                continue
            
            # Hasonló cégek keresése további kontextusért
            similar_companies = self.company_system.find_similar_companies(company, top_k=3)
            
            # Kereskedési irány meghatározása
            if prediction_1d > 0.01:  # 1% feletti emelkedés várható
                signal_type = 'BUY'
                strength = min(prediction_1d * 10, 1.0)  # Normalizálás 0-1 tartományra
            elif prediction_1d < -0.01:  # 1% feletti esés várható
                signal_type = 'SELL'
                strength = min(abs(prediction_1d) * 10, 1.0)
            else:
                continue  # Semleges, nincs jelzés
            
            # Pozícióméret számítása
            position_size = (
                self.portfolio_value * 
                self.max_position_size * 
                strength * 
                confidence
            )
            
            signal = {
                'company': company,
                'type': signal_type,
                'strength': strength,
                'confidence': confidence,
                'position_size': position_size,
                'predicted_change_1d': prediction_1d,
                'predicted_change_5d': analysis['predicted_changes']['5d'],
                'similar_companies': similar_companies,
                'timestamp': datetime.datetime.now()
            }
            
            signals.append(signal)
        
        # Rendezés erősség szerint
        signals.sort(key=lambda x: x['strength'] * x['confidence'], reverse=True)
        
        return signals
    
    def execute_trades(self, signals, max_trades_per_day=10):
        """
        Kereskedések végrehajtása
        
        Paraméterek:
            signals (list): Kereskedési jelzések
            max_trades_per_day (int): Napi maximum kereskedések száma
            
        Visszatérési érték:
            list: Végrehajtott kereskedések listája
        """
        executed_trades = []
        
        for i, signal in enumerate(signals[:max_trades_per_day]):
            # Portfólió korlátok ellenőrzése
            current_exposure = sum(abs(pos) for pos in self.positions.values())
            
            if current_exposure + signal['position_size'] > self.portfolio_value * 0.8:
                logger.warning(f"Portfolio limit elérve, kereskedés kihagyva: {signal['company']}")
                continue
            
            # Kereskedés végrehajtása (szimuláció)
            trade = {
                'company': signal['company'],
                'type': signal['type'],
                'size': signal['position_size'],
                'confidence': signal['confidence'],
                'predicted_change': signal['predicted_change_1d'],
                'timestamp': signal['timestamp'],
                'executed': True
            }
            
            # Pozíció frissítése
            if signal['company'] not in self.positions:
                self.positions[signal['company']] = 0
            
            position_change = (signal['position_size'] if signal['type'] == 'BUY' 
                             else -signal['position_size'])
            self.positions[signal['company']] += position_change
            
            executed_trades.append(trade)
            self.trade_history.append(trade)
            
            logger.info(f"Kereskedés végrehajtva: {signal['type']} {signal['company']} "
                       f"${signal['position_size']:.2f}")
        
        return executed_trades
    
    def save_model_and_data(self, path='advanced_models'):
        """
        Modellek és adatok mentése
        
        Paraméterek:
            path (str): Mentési útvonal
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Neural network modell mentése
        self.news_model.model.save(os.path.join(path, 'attention_news_model.h5'))
        
        # Company embeddings mentése
        with open(os.path.join(path, 'company_embeddings.pkl'), 'wb') as f:
            pickle.dump(self.company_system.company_embeddings, f)
        
        # Tokenizer teljes objektum mentése, nem csak szótárak
        with open(os.path.join(path, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.news_model.tokenizer, f)
        
        # Trading adatok mentése
        with open(os.path.join(path, 'trading_data.pkl'), 'wb') as f:
            pickle.dump({
                'positions': self.positions,
                'portfolio_value': self.portfolio_value,
                'trade_history': self.trade_history
            }, f)
        
        logger.info(f"Modellek és adatok elmentve: {path}")
    
    def load_model_and_data(self, path='advanced_models'):
        """
        Modellek és adatok betöltése
        
        Paraméterek:
            path (str): Betöltési útvonal
            
        Visszatérési érték:
            bool: Sikerült-e a betöltés
        """
        try:
            # Neural network modell betöltése
            self.news_model.model = tf.keras.models.load_model(
                os.path.join(path, 'attention_news_model.h5')
            )
            
            # Company embeddings betöltése
            with open(os.path.join(path, 'company_embeddings.pkl'), 'rb') as f:
                self.company_system.company_embeddings = pickle.load(f)
            
            # Teljes tokenizer objektum betöltése
            with open(os.path.join(path, 'tokenizer.pkl'), 'rb') as f:
                self.news_model.tokenizer = pickle.load(f)
                # Referenciák frissítése
                self.news_model.word_to_idx = self.news_model.tokenizer.word_to_idx
                self.news_model.idx_to_word = self.news_model.tokenizer.idx_to_word
            
            # Trading adatok betöltése
            with open(os.path.join(path, 'trading_data.pkl'), 'rb') as f:
                trading_data = pickle.load(f)
                self.positions = trading_data['positions']
                self.portfolio_value = trading_data['portfolio_value']
                self.trade_history = trading_data['trade_history']
            
            logger.info(f"Modellek és adatok betöltve: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Hiba a betöltés közben: {str(e)}")
            return False
