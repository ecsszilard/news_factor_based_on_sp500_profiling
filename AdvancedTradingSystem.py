import numpy as np
import datetime
import pickle
import os
import logging

logger = logging.getLogger("AdvancedNewsFactor.AdvancedTradingSystem")

class AdvancedTradingSystem:
    """Trading system using probabilistic correlation-based news factor model with Residual Learning"""

    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.news_factor_model = self.data_processor.news_factor_model
        self.word_to_idx = self.news_factor_model.tokenizer.get_vocab()
        self.idx_to_word = {idx: tok for tok, idx in self.word_to_idx.items()}
        
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
            'avg_sigma_z': np.mean(sigma),
            'confidence_score': predictions['total_confidence']
        }
    
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
            logger.info("Detected GLOBAL news: %s...", news_text[:80])
            return "global"

        # Check for sector keywords
        for sector, keywords in sector_keywords.items():
            if any(keyword in news_lower for keyword in keywords):
                logger.info("Detected SECTOR news (%s): %s...", sector, news_text[:80])
                return "sector"

        # Company-specific if companies mentioned
        if mentioned_companies and len(mentioned_companies) <= 3:
            logger.info("Detected COMPANY-SPECIFIC news: %s", mentioned_companies)
            return "company"
        
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

        # Make probabilistic predictions
        predictions = self.news_factor_model.predict_with_uncertainty(
            text=news_text,  # Shape: (1, seq_len),
            baseline_correlation = self.data_processor.baseline_z,
            n_samples=10  # MC Dropout samples
        )
        
        # Extract probabilistic predictions
        mc_mu = predictions['mc_mu']
        mc_logvar = predictions['mc_logvar']
        recon_error = float(predictions['recon_error'])     

        # Calculate confidence scores
        epistemic_var = np.var(mc_mu, axis=0) # Epistemic uncertainty: variance across MC samples
        aleatoric_var = np.mean(np.exp(mc_logvar) + 1e-6, axis=0) # Aleatoric uncertainty: average of predicted variances
        sigma = np.sqrt(epistemic_var + aleatoric_var) # [N, N] - σ (uncertainty) 
        recon_conf = np.exp(-recon_error)
        unc_conf = 1.0 / (1.0 + sigma.mean())
        total_conf = recon_conf * unc_conf

        # Convert back to correlation space
        mu_z = np.mean(mc_mu, axis=0)[0] # [N, N] - μ in Fisher-z space
        baseline_corr = self.data_processor.inverse_fisher_z_transform(self.data_processor.baseline_z)
        predicted_corr = self.data_processor.inverse_fisher_z_transform(mu_z)
        delta_corr = predicted_corr - baseline_corr
        
        results = {}
        for company in target_companies:
            # Analyze correlation changes for this company
            company_idx_in_list = self.data_processor.companies.index(company) if company in self.data_processor.companies else 0
            
            # Find significant correlation changes (with low uncertainty)
            significant_pairs = []
            for i, other_company in enumerate(self.data_processor.companies):
                if i != company_idx_in_list:
                    change = delta_corr[company_idx_in_list, i]
                    uncertainty = sigma[0][company_idx_in_list, i]
                    
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
            avg_uncertainty = float(np.mean(sigma[0][company_idx_in_list, :]))
            
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
                total_conf > self.confidence_threshold and
                avg_uncertainty < self.uncertainty_threshold and
                max_change > correlation_threshold
            )
            
            results[company] = {
                'mu_z': mu_z,
                'sigma_z': sigma[0],
                'baseline_corr': baseline_corr,
                'predicted_corr': predicted_corr,
                'delta_corr': delta_corr,
                'sigma_corr' : self.data_processor.fisher_z_sigma_to_correlation_sigma(mu_z, sigma[0]),
                'reconstruction_error': recon_error,
                "epistemic_confidence": 1.0 / (1.0 + np.sqrt(epistemic_var).mean()),
                "aleatoric_confidence": 1.0 / (1.0 + np.sqrt(aleatoric_var).mean()),
                "uncertainty_confidence": unc_conf,
                "recon_confidence": recon_conf,
                'total_confidence': total_conf,
                'news_scope': self._classify_news_scope(news_text, affected_companies),
                'affected_companies': affected_companies,
                'tradeable': tradeable,
                'price_deviations': predictions['price_deviations'],
                'correlation_impact': {
                    'max_change': max_change,
                    'mean_change': mean_change,
                    'avg_uncertainty': avg_uncertainty,
                    'significant_pairs': significant_pairs[:10],
                    'baseline_avg': float(np.mean(baseline_corr[company_idx_in_list, :])),
                    'predicted_avg': float(np.mean(predicted_corr[company_idx_in_list, :]))
                },
                'price_impact': float(predictions['price_deviations'][company_idx_in_list]),
                'similar_companies': similar_companies[:5]
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
            
            # Check if tradeable based on uncertainty
            if not analysis.get("tradeable", False):
                logger.info("Skipping %s: Low confidence or high uncertainty", company)
                continue

            if total_confidence < min_confidence:
                logger.info("Skipping %s: Confidence %.3f < %.3f", company, total_confidence, min_confidence)
                continue

            if avg_uncertainty > max_uncertainty:
                logger.info("Skipping %s: Uncertainty %.3f > %.3f", company, avg_uncertainty, max_uncertainty)
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
                uncertainty_adjustment  # Inverse of uncertainty
            )
            
            signal = {
                'company': company,
                'type': signal_type,
                'strength': strength,
                'total_confidence': total_confidence,
                'avg_uncertainty': avg_uncertainty,
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
            key=lambda x: (x['strength'] * x['total_confidence']) / (x['avg_uncertainty'] + 0.1), 
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
                'avg_uncertainty': signal['avg_uncertainty'],
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
                "Trade executed: %s %s $%.2f (conf: %.3f, σ: %.3f, Δcorr: %.3f)",
                signal["type"], signal["company"], signal["position_size"],
                signal["total_confidence"], signal["uncertainty"],
                signal.get("correlation_impact", {}).get("max_change", 0.0),
            )
        return executed_trades
    
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