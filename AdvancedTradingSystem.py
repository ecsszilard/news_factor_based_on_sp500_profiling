import numpy as np
import tensorflow as tf
import datetime
import logging
import pickle
import os

logger = logging.getLogger("AdvancedNewsFactor.AdvancedTradingSystem")

class AdvancedTradingSystem:
    """Trading system using multi-task learned company embeddings"""

    def __init__(self, company_embedding_system, news_factor_model):
        self.company_system = company_embedding_system
        self.news_model = news_factor_model
        
        # Trading data
        self.positions = {}
        self.portfolio_value = 100000.0
        self.trade_history = []
        
        # Risk management
        self.max_position_size = 0.05  # 5% max pozíció
        self.stop_loss_pct = 0.02      # 2% stop loss
        self.take_profit_pct = 0.06    # 6% take profit
        
        logger.info("AdvancedTradingSystem inicializálva")
    
    def analyze_news_impact(self, news_text, target_companies):
        """Analyze news impact using multi-task predictions"""
        results = {}
        
        for company in target_companies:
            if self.company_system.get_company_idx(company) == 0 and company != self.company_system.companies[0]:
                continue  # Unknown company
                
            prediction_result = self.news_model.predict_impact(news_text, company, return_detailed=True)
            
            # Get similar companies based on learned embeddings
            similar_companies = self.news_model.get_similar_companies_by_news_response(company, top_k=3)
            
            results[company] = {
                'predicted_changes': prediction_result['price_changes'],
                'volatility_impact': prediction_result['volatility_changes'],
                'relevance_score': prediction_result['relevance_score'],
                'confidence': prediction_result['confidence'],
                'similar_companies': similar_companies,
                'reconstruction_quality': prediction_result['reconstruction_quality']
            }
        
        return results
    
    def generate_trading_signals(self, news_analysis, relevance_threshold=0.6, confidence_threshold=0.5):
        """Generate trading signals based on multi-task predictions"""
        signals = []
        
        for company, analysis in news_analysis.items():
            relevance = analysis['relevance_score']
            prediction_1d = analysis['predicted_changes']['1d']
            confidence = analysis['confidence']
            
            # Filter by relevance and confidence
            if relevance < relevance_threshold or confidence < confidence_threshold:
                continue
            
            # Determine signal type and strength
            if prediction_1d > 0.01:  # >1% expected increase
                signal_type = 'BUY'
                strength = min(prediction_1d * 10, 1.0)
            elif prediction_1d < -0.01:  # >1% expected decrease
                signal_type = 'SELL'
                strength = min(abs(prediction_1d) * 10, 1.0)
            else:
                continue  # Neutral, no signal
            
            # Calculate position size based on multiple factors
            position_size = (
                self.portfolio_value * 
                self.max_position_size * 
                strength * 
                confidence * 
                relevance
            )
            
            signal = {
                'company': company,
                'type': signal_type,
                'strength': strength,
                'confidence': confidence,
                'relevance': relevance,
                'position_size': position_size,
                'predicted_change_1d': prediction_1d,
                'predicted_change_5d': analysis['predicted_changes']['5d'],
                'predicted_change_20d': analysis['predicted_changes']['20d'],
                'volatility_impact': analysis['volatility_impact']['volatility'],
                'similar_companies': analysis['similar_companies'],
                'timestamp': datetime.datetime.now()
            }
            
            signals.append(signal)
        
        # Sort by combined score (strength * confidence * relevance)
        signals.sort(key=lambda x: x['strength'] * x['confidence'] * x['relevance'], reverse=True)
        
        return signals
    
    def execute_trades(self, signals, max_trades_per_day=10):
        """Execute trades based on signals"""
        executed_trades = []
        
        for _, signal in enumerate(signals[:max_trades_per_day]):
            current_exposure = sum(abs(pos) for pos in self.positions.values())
            
            if current_exposure + signal['position_size'] > self.portfolio_value * 0.8:
                continue

            trade = {
                'company': signal['company'],
                'type': signal['type'],
                'size': signal['position_size'],
                'confidence': signal['confidence'],
                'relevance': signal['relevance'],
                'predicted_change': signal['predicted_change_1d'],
                'timestamp': signal['timestamp'],
                'executed': True
            }
            
            # Position updating
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
    
    def save_model_and_data(self, path='multi_task_models'):
        """Save the multi-task model and associated data"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.news_model.model.save(os.path.join(path, 'multi_task_news_model.h5'))
        
        # Save company mapping
        with open(os.path.join(path, 'company_system.pkl'), 'wb') as f:
            pickle.dump({
                'company_to_idx': self.company_system.company_to_idx,
                'idx_to_company': self.company_system.idx_to_company,
                'companies': self.company_system.companies,
                'static_features': self.company_system.static_features
            }, f)
        
        with open(os.path.join(path, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.news_model.tokenizer, f)
        
        # Save trading datas
        with open(os.path.join(path, 'trading_data.pkl'), 'wb') as f:
            pickle.dump({
                'positions': self.positions,
                'portfolio_value': self.portfolio_value,
                'trade_history': self.trade_history
            }, f)
        
        logger.info(f"Modellek és adatok elmentve: {path}")
        
    def load_model_and_data(self, path='multi_task_models'):
        """Load the multi-task model and associated data"""
        try:
            self.news_model.model = tf.keras.models.load_model(
                os.path.join(path, 'multi_task_news_model.h5')
            )
            
            with open(os.path.join(path, 'company_system.pkl'), 'rb') as f:
                company_data = pickle.load(f)
                self.company_system.company_to_idx = company_data['company_to_idx']
                self.company_system.idx_to_company = company_data['idx_to_company']
                self.company_system.companies = company_data['companies']
                self.company_system.static_features = company_data['static_features']
            
            with open(os.path.join(path, 'tokenizer.pkl'), 'rb') as f:
                self.news_model.tokenizer = pickle.load(f)

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