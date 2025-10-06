import numpy as np
import logging
from scipy.stats import pearsonr

logger = logging.getLogger("AdvancedNewsFactor.NewsDataProcessor")

class NewsDataProcessor:
    """Processing news data to train the correlation-focused model"""
    
    def __init__(self, company_system, news_model):
        """Initialize the news data processor"""
        self.company_system = company_system
        self.news_model = news_model
        self.processed_news = []
        
        logger.info("NewsDataProcessor initialized")

    def fisher_z_transform(self, correlation_matrix):
        """
        Fisher-z transzformáció korrelációs mátrixra
        r → z = arctanh(r) = 0.5 * ln((1+r)/(1-r))
        
        Előnyök:
        - Linearizálja a korrelációkat
        - Stabilabb varianciát ad
        - Jobb a tanuláshoz
        """
        # Clip to avoid numerical issues at ±1
        clipped = np.clip(correlation_matrix, -0.9999, 0.9999)
        # Fisher-z transform
        z_matrix = np.arctanh(clipped)
        # Diagonal should be 0 (no self-correlation change)
        np.fill_diagonal(z_matrix, 0.0)
        return z_matrix
    
    def inverse_fisher_z_transform(self, z_matrix):
        """
        Inverse Fisher-z: z → r = tanh(z)
        """
        correlation_matrix = np.tanh(z_matrix)
        np.fill_diagonal(correlation_matrix, 0.0)
        return correlation_matrix

    def process_news_batch(self, news_data, price_data):
        training_samples = {
            'keywords': [],
            'company_indices': [],
            'correlation_changes': [],  # Fisher-z transformed
            'price_deviations': [],
            'news_targets': []
        }
        
        baseline_correlations = self.calculate_baseline_correlations(price_data)
        
        for news_item in news_data:
            news_text = news_item['text']
            affected_companies = news_item['companies']
            news_timestamp = news_item['timestamp']
            
            keyword_sequence = self.news_model.prepare_keyword_sequence(news_text)
            # Generate news target embedding
            news_targets = self.company_system.get_bert_embedding(news_text)[:self.news_model.latent_dim]
            
            price_deviations = self.calculate_price_deviations(
                price_data, news_timestamp, baseline_correlations, affected_companies
            )
            
            if price_deviations is None:
                continue
            
            post_news_correlations = self.calculate_post_news_correlations(price_data, news_timestamp)
            correlation_changes_raw = self.calculate_correlation_change_matrix(
                baseline_correlations, post_news_correlations
            )
            
            if correlation_changes_raw is None:
                continue
            
            # FISHER-Z TRANSZFORMÁCIÓ - Ez a kulcs!
            correlation_changes = self.fisher_z_transform(correlation_changes_raw)
            
            # Use the first affected company as the primary context
            primary_company = affected_companies[0] if affected_companies else self.company_system.companies[0]
            company_idx = self.company_system.get_company_idx(primary_company)
            
            training_samples['keywords'].append(keyword_sequence)
            training_samples['company_indices'].append(company_idx)
            training_samples['correlation_changes'].append(correlation_changes)
            training_samples['price_deviations'].append(price_deviations)
            training_samples['news_targets'].append(news_targets)
        
        logger.info(f"Processed {len(training_samples['keywords'])} news items (Fisher-z transformed)")
        return training_samples

    def calculate_price_deviations(self, price_data, news_timestamp, baseline_correlations, affected_companies, window_days=5):
        """Számítja az árfolyamváltozások eltérését a korrelációk alapján várttól"""
        companies = list(price_data.keys())
        num_companies = len(companies)
        deviations = np.zeros(num_companies)
        
        # Tényleges hozamok minden cégre
        actual_returns = {}
        for company in companies:
            sorted_prices = sorted(price_data[company].items())
            news_idx = None
            for idx, (timestamp, _) in enumerate(sorted_prices):
                if timestamp >= news_timestamp:
                    news_idx = idx
                    break
            
            if news_idx is not None and news_idx + window_days < len(sorted_prices):
                price_before = sorted_prices[news_idx][1]
                price_after = sorted_prices[news_idx + window_days][1]
                actual_returns[company] = (price_after - price_before) / price_before
        
        if not actual_returns:
            return None
        
        # Deviációk számítása
        for i, company in enumerate(companies):
            if company not in actual_returns:
                continue
            
            # Korreláció alapján várt hozam
            expected_return = 0.0
            weight_sum = 0.0
            
            for affected_company in affected_companies:
                if affected_company in actual_returns and affected_company != company:
                    affected_idx = companies.index(affected_company)
                    if affected_idx < len(baseline_correlations):
                        correlation = baseline_correlations[i, affected_idx]
                        expected_return += correlation * actual_returns[affected_company]
                        weight_sum += abs(correlation)
            
            if weight_sum > 0.01:
                expected_return /= weight_sum
            
            # Deviáció = tényleges - várt
            deviations[i] = actual_returns[company] - expected_return
        
        return deviations
    
    def calculate_baseline_correlations(self, price_data, lookback_days=30):
        """Calculate baseline correlation matrix using historical price data"""
        companies = list(price_data.keys())
        num_companies = len(companies)
        correlation_matrix = np.zeros((num_companies, num_companies))
        
        # Get historical returns for all companies
        company_returns = {}
        for i, company in enumerate(companies):
            prices = list(price_data[company].values())
            if len(prices) >= lookback_days + 1:
                # Use the last lookback_days for baseline calculation
                recent_prices = prices[-lookback_days-1:]
                returns = [(recent_prices[j] - recent_prices[j-1]) / recent_prices[j-1] 
                          for j in range(1, len(recent_prices))]
                company_returns[company] = returns
        
        # Calculate pairwise correlations
        for i, company1 in enumerate(companies):
            for j, company2 in enumerate(companies):
                if i == j:
                    correlation_matrix[i, j] = 1.0  # Perfect self-correlation
                elif company1 in company_returns and company2 in company_returns:
                    returns1 = company_returns[company1]
                    returns2 = company_returns[company2]
                    
                    # Align returns to same length
                    min_len = min(len(returns1), len(returns2))
                    if min_len >= 10:  # Minimum data requirement
                        aligned_returns1 = returns1[-min_len:]
                        aligned_returns2 = returns2[-min_len:]
                        
                        try:
                            correlation, p_value = pearsonr(aligned_returns1, aligned_returns2)
                            if not np.isnan(correlation):
                                correlation_matrix[i, j] = correlation
                        except:
                            correlation_matrix[i, j] = 0.0
                else:
                    correlation_matrix[i, j] = 0.0  # No data available
        
        return correlation_matrix
    
    def calculate_post_news_correlations(self, price_data, news_timestamp, window_days=5):
        """Calculate correlations in the period after news announcement"""
        companies = list(price_data.keys())
        num_companies = len(companies)
        correlation_matrix = np.zeros((num_companies, num_companies))
        
        # Get post-news returns
        company_returns = {}
        for company in companies:
            sorted_prices = sorted(price_data[company].items())
            
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
        for i, company1 in enumerate(companies):
            for j, company2 in enumerate(companies):
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
                            correlation, p_value = pearsonr(aligned_returns1, aligned_returns2)
                            if not np.isnan(correlation):
                                correlation_matrix[i, j] = correlation
                        except:
                            correlation_matrix[i, j] = 0.0
                else:
                    correlation_matrix[i, j] = 0.0
        
        return correlation_matrix
    
    def calculate_correlation_change_matrix(self, baseline_correlations, post_news_correlations):
        """Calculate the change in correlation matrix after news"""
        if baseline_correlations is None or post_news_correlations is None:
            return None
        
        # Ensure matrices have the same shape
        if baseline_correlations.shape != post_news_correlations.shape:
            logger.warning("Correlation matrices have different shapes")
            return None
        
        # Calculate the difference (még NEM Fisher-z transformed)
        correlation_changes = post_news_correlations - baseline_correlations
        
        # Apply smoothing to reduce noise in correlation changes
        # Cap extreme changes to prevent training instability
        correlation_changes = np.clip(correlation_changes, -0.8, 0.8)
        
        # Ensure diagonal remains zero (no self-correlation change)
        np.fill_diagonal(correlation_changes, 0.0)
        
        # Ensure symmetry
        correlation_changes = (correlation_changes + correlation_changes.T) / 2.0
        
        return correlation_changes