import numpy as np
import logging
from scipy.stats import pearsonr

logger = logging.getLogger("AdvancedNewsFactor.NewsDataProcessor")

class NewsDataProcessor:
    """Processing news data for probabilistic residual learning correlation model"""
    
    def __init__(self, news_factor_model, sample_prices, bert_model):
        """Initialize the news data processor"""
        self.news_factor_model = news_factor_model
        self.sample_prices = sample_prices
        self.companies = list(sample_prices.keys())
        self.bert_model = bert_model
        self.baseline_correlation_matrix = np.eye(len(self.companies), dtype=np.float64)
        self.baseline_z = np.zeros_like(self.baseline_correlation_matrix)  # Fisher-z space

        # SIMPLE cache for BERT embeddings
        self.correlation_dict_matrix = {}
        self._bert_cache = {}
        
        logger.info("NewsDataProcessor initialized (Probabilistic Mode)")

    def process_news_batch(self, news_data):
        """
        Process news batch for probabilistic residual learning model
        Target is now the POST-NEWS correlation (Fisher-z transformed)
        Model learns to predict: baseline + Δ ≈ post_news
        
        Returns augmented targets for weighted loss:
        - correlation_changes: actual post-news correlation
        - affected_companies_mask: binary mask for affected company pairs
        - Also stores baseline for loss weighting
        """
        training_samples = {
            'keyword_sequence': [],
            'baseline_correlation': [],
            'correlation_changes': [],  # Fisher-z transformed post-news correlation (target for NLL loss)
            'price_deviations': [],
            'news_targets': [],
            'affected_companies_mask': []  # NEW: Binary mask for weighting
        }
        
        # Calculate baseline correlation ONCE for the entire dataset (this represents the "normal" market state before news)
        self.compute_correlation_matrix()
        
        for news_item in news_data:
            news_text = news_item['text']
            affected_companies = news_item['companies']
            news_timestamp = news_item['timestamp']
            
            # Store training sample - note: ['input_ids'] is a TF Tensor with shape (1, max_keywords)
            training_samples['keyword_sequence'].append(self.prepare_keyword_sequence(news_text)['input_ids'])
            training_samples['baseline_correlation'].append(self.baseline_z)
            training_samples['correlation_changes'].append(self.fisher_z_transform(self.compute_post_news_correlation_matrix(news_timestamp)))  # Target: post-news Fisher-z-transformed
            training_samples['price_deviations'].append(self.compute_price_deviations_from_expected(news_timestamp, affected_companies))
            training_samples['news_targets'].append(self.get_bert_embedding(news_text)[:self.news_factor_model.latent_dim])
            training_samples['affected_companies_mask'].append(self._create_affected_mask(affected_companies, self.companies))
        
        logger.info(f"Processed {len(training_samples['keyword_sequence'])} news items (Probabilistic)")
        logger.info(f"Baseline Fisher-z range: [{self.baseline_z.min():.3f}, {self.baseline_z.max():.3f}]")
        return training_samples
    
    def compute_correlation_matrix(self, lookback_days=30, update_self=False):
        """
        General-purpose correlation matrix calculator.
        Can compute both 'baseline' and 'live' correlation matrices.
        
        Args:
            lookback_days (int): number of trailing days to use
            update_self (bool): if True, updates self.correlation_matrix

        Returns:
            np.ndarray: correlation matrix of shape [N, N]
        """

        price_datas = {}
        for c in self.companies:
            prices = list(self.sample_prices[c].values())
            if len(prices) >= lookback_days + 1:
                p = prices[-(lookback_days + 1):]
                r = np.diff(p) / np.array(p[:-1])
                if len(r) >= 5:
                    price_datas[c] = r

        # Pairwise correlations
        for i, c1 in enumerate(self.companies):
            r1 = price_datas.get(c1)
            for j, c2 in enumerate(self.companies[i+1:], i+1):
                r2 = price_datas.get(c2)
                if r1 is None or r2 is None:
                    continue
                min_len = min(len(r1), len(r2))
                if min_len < 5:
                    continue
                try:
                    corr_val = np.corrcoef(r1[-min_len:], r2[-min_len:])[0, 1]
                    self.baseline_correlation_matrix[i, j] = self.baseline_correlation_matrix[j, i] = corr_val
                except Exception:
                    self.baseline_correlation_matrix[i, j] = self.baseline_correlation_matrix[j, i] = 0.0

        # Ensure diagonal is exactly 1.0
        np.fill_diagonal(self.baseline_correlation_matrix, 1.0)

        if update_self:
            self.correlation_dict_matrix = {
                c1: {c2: self.baseline_correlation_matrix[i, j] for j, c2 in enumerate(self.companies) if i != j}
                for i, c1 in enumerate(self.companies)
            }
            logger.info(f"Correlation matrix updated for {len(self.companies)} companies")

        # Fisher-z transform the baseline
        self.baseline_z = self.fisher_z_transform(self.baseline_correlation_matrix)

    def fisher_z_transform(self, correlation_matrix):
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
    
    def inverse_fisher_z_transform(self, z_matrix):
        """
        Inverse Fisher-z: z → r = tanh(z)
        """
        correlation_matrix = np.tanh(z_matrix)
        
        # Restore self-correlations to 1.0
        np.fill_diagonal(correlation_matrix, 1.0)
        return correlation_matrix
    
    def prepare_keyword_sequence(self, text):
        """
        Tokenize text using the model's tokenizer.
        
        Returns:
            BatchEncoding dict with 'input_ids' as TF Tensor of shape (1, max_keywords)
        """
        return self.news_factor_model.tokenizer(
            text, 
            return_tensors='tf', 
            max_length=self.news_factor_model.max_keywords, 
            truncation=True, 
            padding='max_length'
        )
    
    def compute_post_news_correlation_matrix(self, news_timestamp, window_days=5):
        """
        Calculate correlations in the period AFTER news announcement
        This captures the NEWS-INDUCED correlation structure
        """

        correlation_matrix = np.eye(len(self.companies))
        
        company_returns = {}
        for company in self.companies:
            sorted_prices = sorted(self.sample_prices[company].items())
            
            # Find news timestamp position
            news_idx = None
            for idx, (timestamp, _) in enumerate(sorted_prices):
                if timestamp >= news_timestamp:
                    news_idx = idx
                    break
            
            if news_idx is not None:
                # Extract prices for window_days after news
                end_idx = min(news_idx + window_days, len(sorted_prices))
                if end_idx > news_idx + 1:  # Need at least 2 data points
                    post_news_prices = [price for _, price in sorted_prices[news_idx:end_idx]]
                    if len(post_news_prices) >= 3:  # Minimum for meaningful returns
                        returns = [(post_news_prices[i] - post_news_prices[i-1]) / post_news_prices[i-1] 
                                  for i in range(1, len(post_news_prices))]
                        company_returns[company] = returns
        
        # Calculate pairwise correlations for post-news period
        for i, company1 in enumerate(self.companies):
            for j, company2 in enumerate(self.companies):
                if i == j:
                    correlation_matrix[i, j] = 1.0  # Perfect self-correlation
                elif company1 in company_returns and company2 in company_returns:
                    returns1 = company_returns[company1]
                    returns2 = company_returns[company2]
                    
                    # Align returns to same length
                    min_len = min(len(returns1), len(returns2))
                    if min_len >= 3:  # Minimum for post-news correlation
                        aligned_returns1 = returns1[-min_len:]
                        aligned_returns2 = returns2[-min_len:]
                        
                        try:
                            correlation, _ = pearsonr(aligned_returns1, aligned_returns2)
                            if not np.isnan(correlation):
                                correlation_matrix[i, j] = correlation
                        except:
                            correlation_matrix[i, j] = 0.0
                else:
                    correlation_matrix[i, j] = 0.0
        
        # Ensure diagonal is exactly 1.0
        np.fill_diagonal(correlation_matrix, 1.0)
        return correlation_matrix
    
    def compute_price_deviations_from_expected(self, news_timestamp, affected_companies, window_days=5):
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
            return np.zeros(len(self.companies)) # news too fresh
        
        # If the news is global, we calculate the expected return based on all other companies
        if not affected_companies:
            # The list of relevant companies is all companies for which we have return data
            relevant_companies_for_expectation = list(actual_returns.keys())
        else:
            relevant_companies_for_expectation = affected_companies
        
        # Calculate deviations
        for i, company in enumerate(self.companies):
            if company not in actual_returns:
                continue
            
            # Expected return based on correlations
            expected_return = 0.0
            weight_sum = 0.0
            
            for other_company in relevant_companies_for_expectation:
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
    
    def get_bert_embedding(self, text):
        # Simple cache check - this fixes the memory leak
        text_key = hash(text.strip().lower())
        if text_key in self._bert_cache:
            return self._bert_cache[text_key].copy()
        
        inputs = self.prepare_keyword_sequence(text)
        outputs = self.bert_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
                
        # Cache the result - prevent recomputation
        self._bert_cache[text_key] = embedding.copy()
        
        # Simple cache size management
        if len(self._bert_cache) > 100:
            # Remove half of the cache when it gets too big
            keys_to_remove = list(self._bert_cache.keys())[:100]
            for key in keys_to_remove:
                del self._bert_cache[key]
        return embedding
    
    def _create_affected_mask(self, affected_companies_list, all_companies_list):
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

    def get_correlation_pairs(self, significant_companies):
        """Get correlation pairs"""

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