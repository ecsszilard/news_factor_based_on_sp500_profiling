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
        self.max_position_size = 0.05  # 5% max position
        self.stop_loss_pct = 0.02      # 2% stop loss
        self.take_profit_pct = 0.06    # 6% take profit
        
        self.correlation_matrix = {}
        logger.info("AdvancedTradingSystem initialized")
    
    def get_similar_companies_by_news_response(self, target_company, top_k=5):
        return self._get_similar_items(
            target_key=target_company,
            idx_lookup={c: i for i, c in self.company_system.idx_to_company.items()},
            name_lookup=self.company_system.idx_to_company,
            embedding_layer_name="company_embeddings",
            top_k=top_k
        )

    def get_similar_keywords_by_impact(self, target_word, top_k=10):
        return self._get_similar_items(
            target_key=target_word,
            idx_lookup=self.tokenizer.word_to_idx,
            name_lookup=self.tokenizer.word_to_idx,
            embedding_layer_name="keyword_embeddings", # Shape: (vocab_size, keyword_dim)
            top_k=top_k,
            invalid_tokens={'[PAD]', '[UNK]', '[CLS]', '[SEP]'}
        )
    
    def _get_similar_items(self, target_key, idx_lookup, name_lookup, embedding_layer_name, top_k=5, invalid_tokens=None):
        """Generic function to find similar items based on embedding cosine similarity."""
        if target_key not in idx_lookup:
            return []
        
        target_idx = idx_lookup[target_key]

        # Extract embeddings
        try:
            embedding_layer = self.news_model.model.get_layer(embedding_layer_name)
            all_embeddings = embedding_layer.get_weights()[0]
        except ValueError:
            logger.warning("%s layer not found, returning empty list", embedding_layer_name)
            return []

        if target_idx >= len(all_embeddings):
            return []

        target_embedding = all_embeddings[target_idx]
        similarities = []
        target_norm = np.linalg.norm(target_embedding)

        if target_norm <= 1e-8:
            return []

        for item, idx in name_lookup.items():
            if idx == target_idx or idx >= len(all_embeddings):
                continue
            if invalid_tokens and item in invalid_tokens:
                continue

            embedding = all_embeddings[idx]
            embedding_norm = np.linalg.norm(embedding)

            if embedding_norm > 1e-8:
                similarity = np.dot(target_embedding, embedding) / (target_norm * embedding_norm)
                similarities.append((item, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def apply_correlation_adjustments(self, signals):
        """Apply correlation-based position size adjustments"""
        if not self.correlation_matrix or not signals:
            return signals
        
        adjusted_signals = []
        
        for signal in signals:
            company = signal['company']
            original_size = signal['position_size']
            
            # Check correlation with existing positions
            correlation_penalty = 1.0
            
            for held_company, position_size in self.positions.items():
                if abs(position_size) > 1000:  # Only consider significant positions
                    correlation = self.get_correlation(company, held_company)
                    
                    if correlation > 0.7:  # High positive correlation
                        # Reduce position size if we already have exposure to correlated asset
                        correlation_penalty *= 0.7
                    elif correlation < -0.3:  # Negative correlation
                        # Slightly increase position size for diversification
                        correlation_penalty *= 1.1
            
            # Apply the correlation adjustment
            signal['position_size'] = original_size * correlation_penalty
            signal['correlation_adjustment'] = correlation_penalty
            adjusted_signals.append(signal)
        
        return adjusted_signals
    
    def get_correlation(self, company1, company2):
        """Get correlation between two companies"""
        if company1 in self.correlation_matrix:
            return self.correlation_matrix[company1].get(company2, 0.0)
        elif company2 in self.correlation_matrix:
            return self.correlation_matrix[company2].get(company1, 0.0)
        return 0.0
    
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
        
        # Apply correlation adjustments
        signals = self.apply_correlation_adjustments(signals)
        
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
                'correlation_adjustment': signal.get('correlation_adjustment', 1.0),
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
            
            logger.info(f"Trade executed: {signal['type']} {signal['company']} "
                       f"${signal['position_size']:.2f} (corr_adj: {signal.get('correlation_adjustment', 1.0):.2f})")
        
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
        
        # Save trading data including correlation matrix
        with open(os.path.join(path, 'trading_data.pkl'), 'wb') as f:
            pickle.dump({
                'positions': self.positions,
                'portfolio_value': self.portfolio_value,
                'trade_history': self.trade_history,
                'correlation_matrix': self.correlation_matrix
            }, f)
        
        logger.info(f"Models and data saved: {path}")
        
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
                # Load correlation matrix if it exists
                self.correlation_matrix = trading_data.get('correlation_matrix', {})
            
            logger.info(f"Models and data loaded: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error during loading: {str(e)}")
            return False
    
    def update_correlation_matrix(self, price_data):
        """Update correlation matrix"""
        if not price_data:
            return
        
        companies = list(price_data.keys())
        for i, company1 in enumerate(companies):
            if company1 not in self.correlation_matrix:
                self.correlation_matrix[company1] = {}
            
            for company2 in companies[i+1:]:
                # Simple correlation calculation
                prices1 = list(price_data[company1].values())[-30:]  # Last 30 days
                prices2 = list(price_data[company2].values())[-30:]
                
                if len(prices1) >= 10 and len(prices2) >= 10:
                    returns1 = np.diff(prices1) / np.array(prices1[:-1])
                    returns2 = np.diff(prices2) / np.array(prices2[:-1])
                    
                    min_len = min(len(returns1), len(returns2))
                    if min_len > 5:
                        correlation = np.corrcoef(returns1[:min_len], returns2[:min_len])[0,1]
                        if not np.isnan(correlation):
                            self.correlation_matrix[company1][company2] = correlation
                            
                            if company2 not in self.correlation_matrix:
                                self.correlation_matrix[company2] = {}
                            self.correlation_matrix[company2][company1] = correlation
        
        logger.info(f"Correlation matrix updated with {len(self.correlation_matrix)} companies")
    
    def get_portfolio_diversification_metrics(self):
        """Calculate portfolio diversification metrics using correlation matrix"""
        if not self.positions or not self.correlation_matrix:
            return {'error': 'Insufficient data for diversification analysis'}
        
        # Get companies with significant positions
        significant_positions = {
            company: position for company, position in self.positions.items() 
            if abs(position) > self.portfolio_value * 0.01  # > 1% of portfolio
        }
        
        if len(significant_positions) < 2:
            return {'diversification_score': 1.0, 'message': 'Insufficient positions for analysis'}
        
        companies = list(significant_positions.keys())
        n_companies = len(companies)
        
        # Calculate average correlation
        correlations = []
        for i, company1 in enumerate(companies):
            for company2 in companies[i+1:]:
                corr = self.get_correlation(company1, company2)
                correlations.append(abs(corr))  # Use absolute correlation
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        # Diversification score: lower correlation = better diversification
        diversification_score = max(0, 1 - avg_correlation)
        
        return {
            'diversification_score': diversification_score,
            'average_correlation': avg_correlation,
            'num_positions': n_companies,
            'correlation_pairs': len(correlations)
        }