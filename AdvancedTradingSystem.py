import numpy as np
import tensorflow as tf
import datetime
import pickle
import os
import logging

logger = logging.getLogger("AdvancedNewsFactor.AdvancedTradingSystem")
models = tf.keras.models

class AdvancedTradingSystem:
    """Trading system using probabilistic correlation-based news factor model with Residual Learning"""

    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.word_to_idx = data_processor.news_factor_model.tokenizer.vocab
        self.keyword_embedding_layer = self.data_processor.news_factor_model.model.get_layer("keyword_embeddings")
        self.company_embedding_layer = self.data_processor.news_factor_model.model.get_layer("company_embeddings")
        
        # Trading data
        self.positions = {}
        self.portfolio_value = 100000.0
        self.trade_history = []
        
        # Risk management
        self.max_position_size = 0.05  # 5% max position
        self.stop_loss_pct = 0.02      # 2% stop loss
        self.take_profit_pct = 0.06    # 6% take profit
        
        # Uncertainty thresholds
        self.confidence_threshold = 0.6  # Minimum confidence to trade
        self.uncertainty_threshold = 0.4  # Maximum σ to trade
        
        logger.info("AdvancedTradingSystem initialized with Probabilistic Residual Learning")
    
    def analyze_prediction_quality(self, predictions, actuals, baseline):
        """
        Analyze the quality of probabilistic predictions
        
        predictions: dict with 'mean', 'std', 'confidence'
        actuals: actual post-news correlations
        baseline: baseline correlations
        """
        mu = predictions['mean']
        sigma = predictions['std']
        
        # Prediction errors
        errors = np.abs(actuals - mu)
        
        # Check calibration: are high-sigma predictions actually more uncertain?
        high_sigma_mask = sigma > np.median(sigma)
        low_sigma_mask = sigma <= np.median(sigma)
        
        high_sigma_error = np.mean(errors[high_sigma_mask])
        low_sigma_error = np.mean(errors[low_sigma_mask])
        
        # Good calibration: high_sigma_error >> low_sigma_error
        calibration_ratio = high_sigma_error / (low_sigma_error + 1e-6)
        
        # Improvement over baseline
        baseline_errors = np.abs(actuals - baseline)
        model_errors = errors
        
        improvement = np.mean(baseline_errors) - np.mean(model_errors)
        improvement_pct = 100 * improvement / np.mean(baseline_errors)
        
        return {
            'mean_absolute_error': np.mean(errors),
            'high_sigma_error': high_sigma_error,
            'low_sigma_error': low_sigma_error,
            'calibration_ratio': calibration_ratio,
            'improvement_over_baseline': improvement,
            'improvement_percentage': improvement_pct,
            'avg_predicted_sigma': np.mean(sigma),
            'confidence_score': predictions['confidence']['total_confidence']
        }
    
    def get_similar_companies_by_news_response(self, target_company, top_k=5):
        return self._get_similar_items(
            target_key=target_company,
            idx_lookup={symbol: i for i, symbol in enumerate(self.data_processor.companies)},
            name_lookup={i: symbol for i, symbol in enumerate(self.data_processor.companies)},
            embedding_layer=self.company_embedding_layer,
            top_k=top_k
        )

    def get_similar_keywords_by_impact(self, target_word, top_k=10):
        return self._get_similar_items(
            target_key=target_word,
            idx_lookup=self.word_to_idx,  
            name_lookup={v: k for k, v in self.word_to_idx.items()},
            embedding_layer=self.keyword_embedding_layer,
            top_k=top_k,
            invalid_tokens={'[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'} 
        )
    
    def _get_similar_items(self, target_key, idx_lookup, name_lookup, embedding_layer, top_k=5, invalid_tokens=None):
        """Generic function to find similar items based on embedding cosine similarity."""
        if target_key not in idx_lookup:
            return []
        
        target_idx = idx_lookup[target_key]
        all_embeddings = embedding_layer.get_weights()[0]

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
    
    def _classify_news_scope(self, news_text, mentioned_companies):
        """
        Detect if news is global/macro or company-specific
        
        Returns:
            'global': Macro news affecting all companies
            'sector': Sector-specific news
            'company': Company-specific news
        """
        news_lower = news_text.lower()
        
        # Global/macro keywords
        global_keywords = [
            'federal reserve', 'fed', 'interest rate', 'inflation', 'gdp',
            'unemployment', 'central bank', 'monetary policy', 'recession',
            'economic growth', 'trade war', 'pandemic', 'global', 'worldwide',
            'g20', 'g7', 'world bank', 'imf', 'treasury', 'congress',
            'white house', 'president', 'election'
        ]
        
        # Sector keywords
        sector_keywords = {
            'technology': ['tech sector', 'semiconductor', 'chip shortage', 'ai boom', 'cloud computing'],
            'automotive': ['auto industry', 'electric vehicle', 'ev market', 'car sales'],
            'finance': ['banking sector', 'financial sector', 'credit market'],
            'energy': ['oil prices', 'energy sector', 'opec'],
            'healthcare': ['healthcare sector', 'pharmaceutical', 'biotech']
        }
        
        # Check for global keywords
        if any(keyword in news_lower for keyword in global_keywords):
            logger.info(f"Detected GLOBAL news: {news_text[:80]}...")
            return 'global'
        
        # Check for sector keywords
        for sector, keywords in sector_keywords.items():
            if any(keyword in news_lower for keyword in keywords):
                logger.info(f"Detected SECTOR news ({sector}): {news_text[:80]}...")
                return 'sector'
        
        # Company-specific if companies mentioned
        if mentioned_companies and len(mentioned_companies) <= 3:
            logger.info(f"Detected COMPANY-SPECIFIC news: {mentioned_companies}")
            return 'company'
        
        # Default to company-specific
        return 'company'
    
    def apply_correlation_adjustments(self, signals):
        """Apply correlation-based position size adjustments"""
        adjusted_signals = []
        
        for signal in signals:
            company = signal['company']
            original_size = signal['position_size']
            
            # Check correlation with existing positions
            correlation_penalty = 1.0
            
            for held_company, position_size in self.positions.items():
                if abs(position_size) > 1000:  # Only consider significant positions
                    correlation = self.data_processor.get_correlation(company, held_company)
                    
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
    
    def predict_news_impact(self, news_text, target_companies=None, affected_companies=None, correlation_threshold=0.05):
        """
        Analyze news impact using PROBABILISTIC RESIDUAL LEARNING model
        
        Key features:
        1. Detects news scope (global/sector/company)
        2. Uses affected_companies_mask for proper weighting
        3. Returns uncertainty estimates (μ, σ)
        4. Calculates real reconstruction error for epistemic confidence
        
        Args:
            news_text: The news text to analyze
            target_companies: Companies to analyze (default: first 10)
            affected_companies: List of companies mentioned in news (if None, auto-detect)
            correlation_threshold: Minimum correlation change to report
        """
        if target_companies is None:
            target_companies = self.data_processor.companies[:10]  # Limit to first 10 for efficiency
        
        # Auto-detect affected companies if not provided
        if affected_companies is None:
            # Simple detection: check if company names appear in text
            affected_companies = [
                comp for comp in self.data_processor.companies 
                if comp.lower() in news_text.lower()
            ]
        
        # Detect news scope
        news_scope = self._classify_news_scope(news_text, affected_companies)

        # Get news embedding for reconstruction error
        news_target_embedding = self.data_processor.get_bert_embedding(news_text)[:self.data_processor.news_factor_model.latent_dim]
        
        results = {}
        for company in target_companies:            
            # Use predict_with_uncertainty for probabilistic predictions
            predictions = self.data_processor.news_factor_model.predict_with_uncertainty(
                self.data_processor.prepare_keyword_sequence(news_text)['input_ids'],
                np.expand_dims(self.data_processor.baseline_z, 0),
                news_target_embedding  # CRITICAL: Real reconstruction error
            )
            
            # Extract probabilistic predictions
            predicted_corr_z = predictions['mean'][0]  # [N, N] - μ in Fisher-z space
            predicted_sigma = predictions['std'][0]    # [N, N] - σ (uncertainty)
            total_confidence = predictions['total_confidence']
            price_deviations = predictions['price_deviations'][0]
            reconstruction_error = predictions['reconstruction_error']
            
            # Convert back to correlation space
            predicted_corr = self.data_processor.inverse_fisher_z_transform(predicted_corr_z)
            baseline_corr = self.data_processor.inverse_fisher_z_transform(self.data_processor.baseline_z)
            delta_corr = predicted_corr - baseline_corr
            
            # Analyze correlation changes for this company
            company_idx_in_list = self.data_processor.companies.index(company) if company in self.data_processor.companies else 0
            
            # Find significant correlation changes (with low uncertainty)
            significant_pairs = []
            for i, other_company in enumerate(self.data_processor.companies):
                if i != company_idx_in_list:
                    change = delta_corr[company_idx_in_list, i]
                    uncertainty = predicted_sigma[company_idx_in_list, i]
                    
                    # Only report if change is significant AND uncertainty is low
                    if abs(change) > correlation_threshold and uncertainty < self.uncertainty_threshold:
                        significant_pairs.append({
                            'company': other_company,
                            'change': float(change),
                            'uncertainty': float(uncertainty),
                            'confidence_ratio': float(abs(change) / (uncertainty + 1e-6))
                        })
            
            significant_pairs.sort(key=lambda x: x['confidence_ratio'], reverse=True)
            
            # Calculate overall impact metrics
            max_change = float(np.max(np.abs(delta_corr[company_idx_in_list, :])))
            mean_change = float(np.mean(delta_corr[company_idx_in_list, :]))
            avg_uncertainty = float(np.mean(predicted_sigma[company_idx_in_list, :]))
            
            # Price impact
            price_impact = float(price_deviations[company_idx_in_list])
            
            # Similar companies (based on predicted correlations)
            similar_companies = []
            for i, other_company in enumerate(self.data_processor.companies):
                if i != company_idx_in_list:
                    similarity = predicted_corr[company_idx_in_list, i]
                    if similarity > 0.3:
                        similar_companies.append((other_company, float(similarity)))
            similar_companies.sort(key=lambda x: x[1], reverse=True)
            
            # Determine if this company should be traded based on uncertainty
            tradeable = (
                total_confidence > self.confidence_threshold and
                avg_uncertainty < self.uncertainty_threshold and
                max_change > correlation_threshold
            )
            
            results[company] = {
                'total_confidence': total_confidence,
                'reconstruction_error': float(reconstruction_error),
                'news_scope': news_scope,
                'affected_companies': affected_companies,
                'tradeable': tradeable,
                'correlation_impact': {
                    'max_change': max_change,
                    'mean_change': mean_change,
                    'avg_uncertainty': avg_uncertainty,
                    'significant_pairs': significant_pairs[:10],
                    'baseline_avg': float(np.mean(baseline_corr[company_idx_in_list, :])),
                    'predicted_avg': float(np.mean(predicted_corr[company_idx_in_list, :]))
                },
                'price_impact': price_impact,
                'similar_companies': similar_companies[:5],
                # Include matrices for advanced analysis
                'delta_matrix': delta_corr,
                'baseline_matrix': baseline_corr,
                'predicted_matrix': predicted_corr,
                'uncertainty_matrix': predicted_sigma
            }
        
        return results
    
    def generate_trading_signals(self, news_analysis, min_confidence=0.6, max_uncertainty=0.4):
        """
        Generate trading signals based on probabilistic correlation predictions
        
        This method implements uncertainty-aware trading decisions:
        - Filters by confidence threshold (epistemic + aleatoric)
        - Adjusts position size inversely with uncertainty
        - Prioritizes pairs with low σ (high certainty)
        
        Args:
            news_analysis: Output from predict_news_impact
            min_confidence: Minimum total confidence to trade (default: 0.6)
            max_uncertainty: Maximum average σ to trade (default: 0.4)
        """
        signals = []
        
        for company, analysis in news_analysis.items():
            total_confidence = analysis['total_confidence']
            correlation_impact = analysis.get('correlation_impact', {})
            avg_uncertainty = correlation_impact.get('avg_uncertainty', 1.0)
            
            # NEW: Check if tradeable based on uncertainty
            if not analysis.get('tradeable', False):
                logger.info(f"Skipping {company}: Low confidence or high uncertainty")
                continue
            
            # Filter by confidence and uncertainty
            if total_confidence < min_confidence:
                logger.info(f"Skipping {company}: Confidence {total_confidence:.3f} < {min_confidence}")
                continue
            
            if avg_uncertainty > max_uncertainty:
                logger.info(f"Skipping {company}: Uncertainty {avg_uncertainty:.3f} > {max_uncertainty}")
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
                    signal_type = 'BUY'  # Default to buy on volatility case of neutral correlation change
                    strength = min(max_correlation_change * 2, 0.8)
            else:
                continue
            
            # NEW: Adjust position size by uncertainty (inverse relationship)
            uncertainty_adjustment = 1.0 / (avg_uncertainty + 0.1)
            uncertainty_adjustment = np.clip(uncertainty_adjustment, 0.3, 2.0)  # Limit range
            
            # Calculate position size
            position_size = (
                self.portfolio_value * 
                self.max_position_size * 
                strength * 
                total_confidence *
                uncertainty_adjustment  # NEW: Inverse of uncertainty
            )
            
            signal = {
                'company': company,
                'type': signal_type,
                'strength': strength,
                'total_confidence': total_confidence,
                'uncertainty': avg_uncertainty,
                'uncertainty_adjustment': uncertainty_adjustment,
                'position_size': position_size,
                'volatility_impact': max_correlation_change,
                'correlation_impact': correlation_impact,
                'similar_companies': analysis.get('similar_companies', []),
                'news_scope': analysis.get('news_scope', 'unknown'),
                'reconstruction_error': analysis.get('reconstruction_error', 0.0),
                'timestamp': datetime.datetime.now()
            }
            
            signals.append(signal)
        
        # Apply correlation adjustments
        signals = self.apply_correlation_adjustments(signals)
        
        # Sort by risk-adjusted score
        signals.sort(
            key=lambda x: (x['strength'] * x['confidence']) / (x['uncertainty'] + 0.1), 
            reverse=True
        )
        
        return signals
    
    def execute_trading_signals(self, signals, max_trades_per_day=10):
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
                'total_confidence': signal['total_confidence'],
                'uncertainty': signal['uncertainty'],
                'uncertainty_adjustment': signal.get('uncertainty_adjustment', 1.0),
                'correlation_adjustment': signal.get('correlation_adjustment', 1.0),
                'correlation_impact': signal.get('correlation_impact', {}),
                'news_scope': signal.get('news_scope', 'unknown'),
                'reconstruction_error': signal.get('reconstruction_error', 0.0),
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
            
            logger.info(
                f"Trade executed: {signal['type']} {signal['company']} "
                f"${signal['position_size']:.2f} "
                f"(conf: {signal['confidence']:.3f}, σ: {signal['uncertainty']:.3f}, "
                f"Δcorr: {signal.get('correlation_impact', {}).get('max_change', 0):.3f})"
            )
        
        return executed_trades
    
    def save_model_and_data(self, path='probabilistic_correlation_models'):
        """Save the probabilistic correlation model and associated data"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.data_processor.news_factor_model.model.save_weights(os.path.join(path, 'probabilistic_correlation_weights.weights.h5'))
        
        # Save trading data including correlation matrix
        with open(os.path.join(path, 'trading_data.pkl'), 'wb') as f:
            pickle.dump({
                'positions': self.positions,
                'portfolio_value': self.portfolio_value,
                'trade_history': self.trade_history,
                'confidence_threshold': self.confidence_threshold,
                'uncertainty_threshold': self.uncertainty_threshold
            }, f)
        
        logger.info(f"Probabilistic models and data saved: {path}")
        
    def load_model_and_data(self, path='probabilistic_correlation_models'):
        """Load the probabilistic correlation model and associated data"""

        if not self.data_processor.news_factor_model.model:
            self.data_processor.news_factor_model.build_model()

        try:
            self.data_processor.news_factor_model.model.load_weights(os.path.join(path, 'probabilistic_correlation_weights.weights.h5'))

            with open(os.path.join(path, 'trading_data.pkl'), 'rb') as f:
                trading_data = pickle.load(f)
                self.positions = trading_data['positions']
                self.portfolio_value = trading_data['portfolio_value']
                self.trade_history = trading_data['trade_history']
                self.confidence_threshold = trading_data.get('confidence_threshold', 0.6)
                self.uncertainty_threshold = trading_data.get('uncertainty_threshold', 0.4)
            
            logger.info(f"Probabilistic models and data loaded: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error during loading: {str(e)}")
            return False
    
    def get_portfolio_diversification_metrics(self):
        """Calculate portfolio diversification metrics using correlation matrix"""
        if not self.positions:
            return {'error': 'Insufficient data for diversification analysis'}
        
        # Get companies with significant positions
        significant_positions = {
            company: position for company, position in self.positions.items() 
            if abs(position) > self.portfolio_value * 0.01  # > 1% of portfolio
        }
        
        if len(significant_positions) < 2:
            return {'diversification_score': 1.0, 'message': 'Insufficient positions for analysis'}
        
        significant_companies = list(significant_positions.keys())
        correlation_pairs = self.data_processor.get_correlation_pairs(significant_companies)
        avg_correlation = np.mean(correlation_pairs) if correlation_pairs else 0.0 # Calculate average correlation
        diversification_score = max(0, 1 - avg_correlation) # Diversification score: lower correlation = better diversification
        
        return {
            'diversification_score': diversification_score,
            'average_correlation': avg_correlation,
            'num_positions': len(significant_companies),
            'correlation_pairs': len(correlation_pairs)
        }

    def analyze_keyword_impact_clusters(self, sample_keywords, similarity_threshold=0.7, return_matrix=False):
        """Analyze how keywords cluster based on their learned impact patterns"""
        if not sample_keywords:
            return {}
        
        all_embeddings = self.keyword_embedding_layer.get_weights()[0]
        valid_keywords = []
        embeddings = []
        
        for word in sample_keywords:
            idx = self.word_to_idx.get(word)
            if idx is not None and idx < len(all_embeddings):
                valid_keywords.append(word)
                embeddings.append(all_embeddings[idx])

        if len(valid_keywords) < 2:
            return {}
        
        embeddings = np.array(embeddings)
        normalized_embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8) 
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # Find clusters of similar-impact keywords
        clusters = {}
        for i, w1 in enumerate(valid_keywords):
            sims = [(w2, similarity_matrix[i, j]) for j, w2 in enumerate(valid_keywords) 
                   if i != j and similarity_matrix[i, j] > similarity_threshold]
            if sims:
                clusters[w1] = sorted(sims, key=lambda x: x[1], reverse=True)

        return {"clusters": clusters, "similarity_matrix": similarity_matrix, "keywords": valid_keywords} if return_matrix else clusters