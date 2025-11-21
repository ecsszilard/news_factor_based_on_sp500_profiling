import numpy as np
import logging
from scipy.stats import pearsonr
import datetime
from typing import Dict, List

logger = logging.getLogger("AdvancedNewsFactor.NewsDataProcessor")

class NewsDataProcessor:
    """Processing news data for probabilistic residual learning correlation model"""
    
    def __init__(self, news_factor_model, sample_prices):
        """Initialize the news data processor"""
        self.news_factor_model = news_factor_model
        self.sample_prices = sample_prices
        self.companies = list(sample_prices.keys())
        
        # Baseline correlation matrices (IMMUTABLE after first computation)
        self.baseline_correlation_matrix = None
        self.baseline_z = None
        self._baseline_computed = False
        self._baseline_computation_period = None
        
        # Dictionary format for backward compatibility
        self.correlation_dict_matrix = {}
        
        logger.info("NewsDataProcessor initialized (Probabilistic Mode)")
    
    def process_news_batch(self, 
                          news_data: List[Dict], 
                          is_training: bool = True, 
                          baseline_lookback_days: int = 30) -> Dict[str, List]:
        """
        Process news batch for probabilistic residual learning model
        Target is now the POST-NEWS correlation (Fisher-z transformed)
        Model learns to predict: baseline + Δ ≈ post_news
        
        Args:
            news_data: List of news dictionaries
            is_training: If True, calculate baseline. If False, use existing one.
            baseline_lookback_days: How many days of data to calculate baseline
        
        Returns:
            Dict with processed training samples
        """
        training_samples = {
            'keyword_sequence': [],
            'baseline_correlation': [],
            'correlation_changes': [],  # Fisher-z transformed post-news correlation (target for NLL loss)
            'price_deviations': [],
            'news_targets': [],
            'affected_companies_mask': [] # Binary mask for weighting
        }
        
        # Baseline calculation ONLY BEFORE TRAINING period
        if is_training and not self._baseline_computed:
            first_train_timestamp = min(n['timestamp'] for n in news_data)
            
            self.compute_baseline_correlation_pre_training(
                end_timestamp=first_train_timestamp,
                lookback_days=baseline_lookback_days
            )
            self._baseline_computed = True
            
            logger.info(
                "✅ Baseline computed from PRE-TRAINING period:\n   %s",
                self._baseline_computation_period
            )
        
        elif not is_training and not self._baseline_computed:
            raise RuntimeError(
                "Baseline correlation must be computed from training period first."
            )
        
        elif not is_training:
            logger.info(
                "ℹ️  Using FROZEN baseline from training period:\n   %s",
                self._baseline_computation_period
            )
        
        # Process each news item
        for news_item in news_data:
            try:
                news_text = news_item['text']
                affected_companies = news_item.get('companies', [])
                news_timestamp = news_item['timestamp']
                
                # Get keeywords and news embedding for reconstruction
                keywords = self.news_factor_model.prepare_keyword_sequence(news_text)
                news_target_embedding = self.news_factor_model.get_bert_embedding(news_text)
                
                # Compute post-news correlation
                post_news_corr = self._compute_post_news_correlation_matrix(news_timestamp, affected_companies)
                post_news_z = self.fisher_z_transform(post_news_corr)
                
                # Compute price deviations
                price_dev = self._compute_price_deviations_from_expected(news_timestamp, affected_companies)
                
                # Create affected companies mask
                affected_mask = self._create_affected_mask(affected_companies, self.companies)
                
                # Store training sample
                training_samples['keyword_sequence'].append(keywords)
                training_samples['baseline_correlation'].append(self.baseline_z)
                training_samples['correlation_changes'].append(post_news_z)
                training_samples['price_deviations'].append(price_dev)
                training_samples['news_targets'].append(news_target_embedding)
                training_samples['affected_companies_mask'].append(affected_mask)
                
            except Exception as e:
                logger.error("Error processing news item: %s", e)
                continue
        
        logger.info(
            "Processed %d news items (%s)",
            len(training_samples['keyword_sequence']),
            "TRAINING" if is_training else "VALIDATION"
        )

        if self.baseline_z is not None:
            logger.info(
                "Baseline Fisher-z range: [%.3f, %.3f]",
                self.baseline_z.min(),
                self.baseline_z.max()
            )
        return training_samples
    
    def compute_baseline_correlation_pre_training(self, 
                                                 end_timestamp: float, 
                                                 lookback_days: int = 30):
        """
        Compute baseline correlation ONLY from data BEFORE the training period
        
        Args:
            end_timestamp: The first news timestamp (we do NOT use this!)
            lookback_days: How many days of historical data
        """
        n = len(self.companies)
        self.baseline_correlation_matrix = np.eye(n, dtype=np.float64)
        
        # Get prices for the [start, end] period
        price_data_in_period = {}
        for company, prices_dict in self.sample_prices.items():
            timestamps, prices = zip(*sorted(prices_dict.items()))
            timestamps = np.array(timestamps)
            prices = np.array(prices)

            # Time-based filtering
            start_time = end_timestamp - lookback_days * 86400
            mask = (timestamps >= start_time) & (timestamps < end_timestamp)
            period_prices = prices[mask]

            # Only if there is enough data
            if len(period_prices) >= 6:  # Need 6 prices for 5 returns
                returns = np.diff(period_prices) / period_prices[:-1]
                price_data_in_period[company] = returns
        
        if not price_data_in_period:
            logger.warning("No price data available for baseline calculation!")
            self.baseline_z = np.zeros((n, n))
            return
        
        # Pairwise correlations
        for i, company1 in enumerate(self.companies):
            returns1 = price_data_in_period.get(company1)
            if returns1 is None:
                continue
            
            for j, company2 in enumerate(self.companies[i+1:], i+1):
                returns2 = price_data_in_period.get(company2)
                if returns2 is None:
                    continue
                
                # Align returns
                min_len = min(len(returns1), len(returns2))
                if min_len < 5:
                    continue
                try:
                    corr_val = np.corrcoef(
                        returns1[-min_len:], 
                        returns2[-min_len:]
                    )[0, 1]
                    if not np.isnan(corr_val):
                        self.baseline_correlation_matrix[i, j] = corr_val
                        self.baseline_correlation_matrix[j, i] = corr_val
                except Exception as e:
                    logger.warning(
                        "Correlation failed for %s-%s: %s", 
                        company1, company2, e
                    )
                    self.baseline_correlation_matrix[i, j] = 0.0
                    self.baseline_correlation_matrix[j, i] = 0.0
        
        # Ensure diagonal is exactly 1.0
        np.fill_diagonal(self.baseline_correlation_matrix, 1.0)
        
        # Fisher-z transform the baseline
        self.baseline_z = self.fisher_z_transform(self.baseline_correlation_matrix)
        
        # Dictionary format (backward compatibility)
        self.correlation_dict_matrix = {
            c1: {
                c2: self.baseline_correlation_matrix[i, j] 
                for j, c2 in enumerate(self.companies) 
                if i != j
            }
            for i, c1 in enumerate(self.companies)
        }
        
        # Store period for validation
        self._baseline_computation_period = (
            f"{datetime.datetime.fromtimestamp(end_timestamp - lookback_days * 86400)} - "
            f"{datetime.datetime.fromtimestamp(end_timestamp)}"
        )
        
        logger.info(
            "✅ Baseline correlation computed\n"
            "   Used %d/%d companies\n"
            "   Mean correlation: %.3f",
            len(price_data_in_period), 
            len(self.companies), 
            np.mean(self.baseline_correlation_matrix[np.triu_indices_from(
                self.baseline_correlation_matrix, k=1
            )])
        )
    
    def fisher_z_transform(self, correlation_matrix: np.ndarray) -> np.ndarray:
        """
        Fisher-z transformation for correlation matrix
        r → z = arctanh(r) = 0.5 * ln((1+r)/(1-r))
        
        Benefits:
        - Linearizes correlations
        - Stabilizes variance
        - Better for learning
        """
        corr_copy = correlation_matrix.copy()
        np.fill_diagonal(corr_copy, 0.0)
        clipped = np.clip(corr_copy, -0.9999, 0.9999) # Clip to avoid numerical issues at ±1
        z_matrix = np.arctanh(clipped) # Fisher-z transform
        np.fill_diagonal(z_matrix, 0.0) # Diagonal should be 0 (no self-correlation change)
        return z_matrix
    
    def inverse_fisher_z_transform(self, z_matrix: np.ndarray) -> np.ndarray:
        """Inverse Fisher-z: z → r = tanh(z)"""
        correlation_matrix = np.tanh(z_matrix)
        
        # Restore self-correlations to 1.0
        np.fill_diagonal(correlation_matrix, 1.0)
        return correlation_matrix
    
    def fisher_z_sigma_to_correlation_sigma(self, 
                                           mu_z: np.ndarray, 
                                           sigma_z: np.ndarray) -> np.ndarray:
        """
        Convert uncertainty from Fisher-z space to correlation space
        Using delta method: σ_r ≈ σ_z * |dr/dz| = σ_z * (1 - tanh²(z))
        """
        tanh_mu = np.tanh(mu_z)
        derivative = 1 - tanh_mu**2  # sech²(z)
        sigma_corr = sigma_z * derivative
        return np.clip(sigma_corr, 0, 1)
    
    def _compute_post_news_correlation_matrix(self, 
                                             news_timestamp: float,
                                             affected_companies: List[str],
                                             window_days: int = 5) -> np.ndarray:
        """
        Compute post-news correlation matrix efficiently
        Only recomputes correlations for affected companies
        """
        n = len(self.companies)
        correlation_matrix = np.eye(n)  # Initialize with 1.0 on diagonal
        company_idx = {c: i for i, c in enumerate(self.companies)}

        # Compute returns after news for affected companies
        company_returns = {}
        for company in affected_companies:
            if company not in self.sample_prices:
                continue
                
            sorted_prices = sorted(self.sample_prices[company].items())
            timestamps, values = zip(*sorted_prices)

            # Find first timestamp >= news_timestamp
            news_idx = np.searchsorted(timestamps, news_timestamp)
            end_idx = min(news_idx + window_days, len(values))

            # Compute percentage returns if enough points exist
            if end_idx - news_idx >= 3:
                post_prices = np.array(values[news_idx:end_idx])
                returns = np.diff(post_prices) / post_prices[:-1]
                company_returns[company] = returns

        # Update correlations only for affected company pairs
        for c1 in affected_companies:
            i = company_idx[c1]
            r1 = company_returns.get(c1)
            if r1 is None:
                continue
            for c2 in self.companies:
                j = company_idx[c2]
                if i == j:
                    continue  # diagonal = 1.0 already set
                r2 = company_returns.get(c2)
                if r2 is None:
                    continue

                # Align to shortest return series length
                min_len = min(len(r1), len(r2))
                if min_len < 3:
                    continue

                try:
                    corr, _ = pearsonr(r1[-min_len:], r2[-min_len:])
                    if not np.isnan(corr):
                        correlation_matrix[i, j] = correlation_matrix[j, i] = corr
                except Exception:
                    correlation_matrix[i, j] = correlation_matrix[j, i] = 0.0

        # Ensure diagonal is exactly 1.0
        np.fill_diagonal(correlation_matrix, 1.0)
        return correlation_matrix
    
    def _compute_price_deviations_from_expected(self, 
                                               news_timestamp: float,
                                               affected_companies: List[str],
                                               window_days: int = 5) -> np.ndarray:
        """Calculate price deviations from correlation-based expectations"""
        deviations = np.zeros(len(self.companies))
        
        # Calculate actual returns
        actual_returns = {}
        for company in self.companies:
            sorted_prices = sorted(self.sample_prices[company].items())
            news_idx = None
            for idx, (timestamp, _) in enumerate(sorted_prices):
                if timestamp >= news_timestamp:
                    news_idx = idx
                    break
            
            # Check if there is enough data AFTER the news
            if news_idx is not None and news_idx + window_days < len(sorted_prices):
                price_before = sorted_prices[news_idx][1]
                price_after = sorted_prices[news_idx + window_days][1]
                
                # Exchange rate check
                if price_before > 1e-6:
                    actual_returns[company] = (price_after - price_before) / price_before
        
        if not actual_returns:
            return deviations
        
        # Determine relevant companies for expectation
        relevant_companies = affected_companies if affected_companies else list(actual_returns.keys())
        
        # Calculate deviations
        for i, company in enumerate(self.companies):
            if company not in actual_returns:
                continue
            
            # Expected return based on correlations
            expected_return = 0.0
            weight_sum = 0.0
            
            for other_company in relevant_companies:
                # We calculate the expected return based on the OTHER company
                if other_company in actual_returns and other_company != company:
                    other_idx = self.companies.index(other_company)
                    
                    # Using baseline correlation as weight
                    correlation = self.baseline_correlation_matrix[i, other_idx]
                    expected_return += correlation * actual_returns[other_company]
                    weight_sum += abs(correlation)
            
            if weight_sum > 0.01:
                expected_return /= weight_sum
            
            deviations[i] = actual_returns[company] - expected_return
        return deviations
    
    def _create_affected_mask(self, 
                             affected_companies_list: List[str],
                             all_companies_list: List[str]) -> np.ndarray:
        """
        Create binary mask for affected company pairs
        
        Args:
            affected_companies_list: List of companies mentioned/affected by news
            all_companies_list: List of all companies in universe
        
        Returns:
            mask [N, N]: Binary mask where 1.0 = pair involves affected company
        
        Examples:
            affected = ['TSLA'] → TSLA pairs marked with 1.0
            affected = [] → All zeros (global news)
            affected = ['TSLA', 'F'] → Both TSLA and F pairs marked
        """
        n = len(all_companies_list)
        mask = np.zeros((n, n))
        
        # Get indices of affected companies
        affected_indices = {
            all_companies_list.index(c) 
            for c in affected_companies_list 
            if c in all_companies_list
        }
        
        if not affected_indices:
            # Global news: return all-zero mask
            return mask
        
        # Mark pairs involving affected companies
        for i in range(n):
            for j in range(n):
                if i in affected_indices or j in affected_indices:
                    mask[i, j] = 1.0
        
        # Don't weight self-correlations
        np.fill_diagonal(mask, 0.0)
        return mask
    
    def get_correlation_pairs(self, significant_companies: List[str]) -> List[float]:
        """Get correlation pairs for given companies"""
        correlation_pairs = []
        for i, company1 in enumerate(significant_companies):
            for company2 in significant_companies[i+1:]:
                if company1 in self.correlation_dict_matrix:
                    corr = self.correlation_dict_matrix[company1].get(company2, 0.0)
                elif company2 in self.correlation_dict_matrix:
                    corr = self.correlation_dict_matrix[company2].get(company1, 0.0)
                else:
                    corr = 0.0
                correlation_pairs.append(abs(corr))  # Use absolute correlation
        return correlation_pairs