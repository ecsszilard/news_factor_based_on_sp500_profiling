import numpy as np
import tensorflow as tf
import datetime
import logging
import pickle
import os

logger = logging.getLogger("AdvancedNewsFactor.AdvancedTradingSystem")
models = tf.keras.models
from NewsDataProcessor import NewsDataProcessor

class AdvancedTradingSystem:
    """Trading system using correlation-based news factor model with Residual Learning"""

    def __init__(self, embeddingAndTokenizerSystem, news_factor_model, companies):
        self.embeddingAndTokenizerSystem = embeddingAndTokenizerSystem
        self.news_model = news_factor_model
        self.companies = companies
        
        # Trading data
        self.positions = {}
        self.portfolio_value = 100000.0
        self.trade_history = []
        
        # Risk management
        self.max_position_size = 0.05  # 5% max position
        self.stop_loss_pct = 0.02      # 2% stop loss
        self.take_profit_pct = 0.06    # 6% take profit
        
        self.correlation_matrix = {}
        logger.info("AdvancedTradingSystem initialized with Residual Learning")
    
    def get_similar_companies_by_news_response(self, target_company, top_k=5):
        return self._get_similar_items(
            target_key=target_company,
            idx_lookup=self.embeddingAndTokenizerSystem.company_to_idx,
            name_lookup=self.embeddingAndTokenizerSystem.idx_to_company,
            embedding_layer_name="company_embeddings",
            top_k=top_k
        )

    def get_similar_keywords_by_impact(self, target_word, top_k=10):
        return self._get_similar_items(
            target_key=target_word,
            idx_lookup=self.embeddingAndTokenizerSystem.word_to_idx,  
            name_lookup=self.embeddingAndTokenizerSystem.idx_to_word,
            embedding_layer_name="keyword_embeddings",
            top_k=top_k,
            invalid_tokens={'[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'} 
        )
    
    def _get_similar_items(self, target_key, idx_lookup, name_lookup, embedding_layer_name, top_k=5, invalid_tokens=None):
        """Generic function to find similar items based on embedding cosine similarity."""
        if target_key not in idx_lookup:
            return []
        
        target_idx = idx_lookup[target_key]

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

        for idx, item in name_lookup.items():
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
    
    def analyze_news_impact(self, news_text, target_companies=None, correlation_threshold=0.05):
        """
        Analyze news impact using RESIDUAL LEARNING model
        
        Key changes:
        1. Now passes baseline correlation as input to the model
        2. Model predicts post-news correlation (baseline + delta)
        3. Calculates and returns the delta explicitly
        """
        if target_companies is None:
            target_companies = self.companies[:10]  # Limit to first 10 for efficiency
        
        # Get current baseline correlation matrix
        baseline_matrix = self._get_baseline_correlation_matrix()
        # Import processor for Fisher-z transforms
        data_processor = NewsDataProcessor(self.embeddingAndTokenizerSystem, self.news_model)
        baseline_z = data_processor.fisher_z_transform(baseline_matrix)
        
        # Prepare keyword sequence
        keyword_sequence = self.embeddingAndTokenizerSystem.prepare_keyword_sequence(
            news_text, 
            self.news_model.max_keywords
        )
        
        results = {}
        
        for company in target_companies:
            company_idx = self.embeddingAndTokenizerSystem.company_to_idx.get(company, 0)
            
            # Predict with baseline as input
            predictions = self.news_model.model.predict([keyword_sequence, np.array([[company_idx]]), np.expand_dims(baseline_z, 0)], verbose=0)
            
            # Extract predictions
            predicted_corr_z = predictions[0][0]  # Fisher-z space [N, N]
            price_deviations = predictions[1][0]  # [N]
            reconstruction = predictions[2][0]     # [latent_dim]
            
            # ⭐ Calculate the DELTA explicitly
            # The model outputs: baseline + delta, so delta = output - baseline
            delta_z = predicted_corr_z - baseline_z
            
            # Convert back to correlation space
            predicted_corr = data_processor.inverse_fisher_z_transform(predicted_corr_z)
            baseline_corr = data_processor.inverse_fisher_z_transform(baseline_z)
            delta_corr = predicted_corr - baseline_corr
            
            # Analyze correlation changes for this company
            company_idx_in_list = self.companies.index(company) if company in self.companies else 0
            
            # Find significant correlation changes
            significant_pairs = []
            for i, other_company in enumerate(self.companies):
                if i != company_idx_in_list:
                    change = delta_corr[company_idx_in_list, i]
                    if abs(change) > correlation_threshold:
                        significant_pairs.append((other_company, float(change)))
            
            significant_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Calculate overall impact metrics
            max_change = float(np.max(np.abs(delta_corr[company_idx_in_list, :])))
            mean_change = float(np.mean(delta_corr[company_idx_in_list, :]))
            
            # Reconstruction quality as confidence
            reconstruction_error = np.mean(np.abs(reconstruction))
            confidence = 1.0 / (1.0 + reconstruction_error)
            
            # Price impact
            price_impact = float(price_deviations[company_idx_in_list])
            
            # Similar companies (based on predicted correlations)
            similar_companies = []
            for i, other_company in enumerate(self.companies):
                if i != company_idx_in_list:
                    similarity = predicted_corr[company_idx_in_list, i]
                    if similarity > 0.3:
                        similar_companies.append((other_company, float(similarity)))
            similar_companies.sort(key=lambda x: x[1], reverse=True)
            
            results[company] = {
                'confidence': float(confidence),
                'correlation_impact': {
                    'max_change': max_change,
                    'mean_change': mean_change,
                    'significant_pairs': significant_pairs[:10],
                    'baseline_avg': float(np.mean(baseline_corr[company_idx_in_list, :])),
                    'predicted_avg': float(np.mean(predicted_corr[company_idx_in_list, :]))
                },
                'price_impact': price_impact,
                'similar_companies': similar_companies[:5],
                # Include all matrices for advanced analysis
                'delta_matrix': delta_corr,
                'baseline_matrix': baseline_corr,
                'predicted_matrix': predicted_corr,
                'delta_z_matrix': delta_z  # Fisher-z space delta
            }
        
        return results
    
    def generate_trading_signals(self, news_analysis, confidence_threshold=0.2):
        """Generate trading signals based on correlation change predictions"""
        signals = []
        
        for company, analysis in news_analysis.items():
            confidence = analysis['confidence']
            correlation_impact = analysis.get('correlation_impact', {})
            
            # Filter by confidence
            if confidence < confidence_threshold:
                continue
            
            # Determine signal type based on correlation changes
            max_correlation_change = correlation_impact.get('max_change', 0)
            mean_correlation_change = correlation_impact.get('mean_change', 0)
            
            # High correlation change suggests volatility opportunity
            if max_correlation_change > 0.15:  # Significant correlation disruption
                if mean_correlation_change > 0.05:
                    # Company becoming more correlated with market - potential momentum play
                    signal_type = 'BUY'
                    strength = min(max_correlation_change * 3, 1.0)
                elif mean_correlation_change < -0.05:
                    # Company becoming less correlated - potential contrarian play
                    signal_type = 'SELL'
                    strength = min(max_correlation_change * 3, 1.0)
                else:
                    # Neutral correlation change - volatility play
                    signal_type = 'BUY'  # Default to buy on volatility
                    strength = min(max_correlation_change * 2, 0.8)
            else:
                continue  # No significant correlation impact
            
            # Calculate position size based on confidence and strength
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
                'volatility_impact': max_correlation_change,
                'correlation_impact': correlation_impact,
                'similar_companies': analysis.get('similar_companies', []),
                'timestamp': datetime.datetime.now()
            }
            
            signals.append(signal)
        
        # Apply correlation adjustments
        signals = self.apply_correlation_adjustments(signals)
        
        # Sort by combined score (strength * confidence)
        signals.sort(key=lambda x: x['strength'] * x['confidence'], reverse=True)
        
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
                'correlation_adjustment': signal.get('correlation_adjustment', 1.0),
                'correlation_impact': signal.get('correlation_impact', {}),
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
            
            logger.info(f"Trade executed: {signal['type']} {signal['company']} " f"${signal['position_size']:.2f} (corr_impact: {signal.get('correlation_impact', {}).get('max_change', 0):.3f})")
        
        return executed_trades
    
    def save_model_and_data(self, path='correlation_models'):
        """Save the correlation model and associated data"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.news_model.model.save(os.path.join(path, 'correlation_news_model.h5'))
        
        # Save company mapping
        with open(os.path.join(path, 'embeddingAndTokenizerSystem.pkl'), 'wb') as f:
            pickle.dump({
                'company_to_idx': self.embeddingAndTokenizerSystem.company_to_idx,
                'idx_to_company': self.embeddingAndTokenizerSystem.idx_to_company,
                'companies': self.companies,
                'static_features': self.embeddingAndTokenizerSystem.static_features
            }, f)
        
        with open(os.path.join(path, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.embeddingAndTokenizerSystem, f)
        
        # Save trading data including correlation matrix
        with open(os.path.join(path, 'trading_data.pkl'), 'wb') as f:
            pickle.dump({
                'positions': self.positions,
                'portfolio_value': self.portfolio_value,
                'trade_history': self.trade_history,
                'correlation_matrix': self.correlation_matrix
            }, f)
        
        logger.info(f"Models and data saved: {path}")
        
    def load_model_and_data(self, path='correlation_models'):
        """Load the correlation model and associated data"""
        try:
            self.news_model.model = models.load_model(
                os.path.join(path, 'correlation_news_model.h5')
            )
            
            with open(os.path.join(path, 'embeddingAndTokenizerSystem.pkl'), 'rb') as f:
                company_data = pickle.load(f)
                self.embeddingAndTokenizerSystem.company_to_idx = company_data['company_to_idx']
                self.embeddingAndTokenizerSystem.idx_to_company = company_data['idx_to_company']
                self.companies = company_data['companies']
                self.embeddingAndTokenizerSystem.static_features = company_data['static_features']
            
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
        
    def _get_baseline_correlation_matrix(self):
        """
        Convert trading system's correlation_matrix dict to numpy matrix
        Returns: [N, N] correlation matrix
        """
        n = len(self.companies)
        matrix = np.eye(n)
        
        for i, company1 in enumerate(self.companies):
            for j, company2 in enumerate(self.companies):
                if i != j:
                    if company1 in self.correlation_matrix and company2 in self.correlation_matrix[company1]:
                        matrix[i, j] = self.correlation_matrix[company1][company2]
                    elif company2 in self.correlation_matrix and company1 in self.correlation_matrix[company2]:
                        matrix[i, j] = self.correlation_matrix[company2][company1]
                    else:
                        matrix[i, j] = 0.0
        
        return matrix