import numpy as np
import logging

logger = logging.getLogger("AdvancedNewsFactor.NewsDataProcessor")

class NewsDataProcessor:
    """Processing news data to train the model"""
    
    def __init__(self, company_system, news_model):
        """Initialize the news data processor"""
        self.company_system = company_system
        self.news_model = news_model
        self.processed_news = []
        
        logger.info("NewsDataProcessor initialized")
    
    def process_news_batch(self, news_data, price_data):
        """Prepare multi-task training data"""
        training_samples = {
            'keywords': [],
            'company_indices': [],
            'price_changes': [],
            'volatility_changes': [],
            'relevance_labels': [],
            'news_targets': []
        }
        
        for news_item in news_data:
            news_text = news_item['text']
            affected_companies = news_item['companies']
            news_timestamp = news_item['timestamp']
            
            keyword_sequence = self.news_model.prepare_keyword_sequence(news_text)
            
            news_targets = self.company_system.get_bert_embedding(news_text)[:getattr(self.news_model, 'latent_dim', 128)]
            
            for company in affected_companies:
                company_idx = self.company_system.get_company_idx(company)
                
                if company in price_data:
                    # Calculate price changes
                    price_changes = self.calculate_price_changes(price_data[company], news_timestamp)
                    
                    # Calculate volatility changes
                    volatility_changes = self.calculate_volatility_changes(price_data[company], news_timestamp)
                    
                    if price_changes is not None and volatility_changes is not None:
                        # Determine relevance based on significant price movement
                        relevance = 1.0 if abs(price_changes[0]) > 0.01 else 0.0  # >1% change = relevant
                        
                        training_samples['keywords'].append(keyword_sequence)
                        training_samples['company_indices'].append(company_idx)
                        training_samples['price_changes'].append(price_changes)
                        training_samples['volatility_changes'].append(volatility_changes)
                        training_samples['relevance_labels'].append(relevance)
                        training_samples['news_targets'].append(news_targets)
        
        return training_samples
    
    def calculate_price_changes(self, company_prices, news_timestamp):
        """Calculate price changes for 1d, 5d, 20d periods"""
        sorted_prices = sorted(company_prices.items())
        
        # Find the date of a news item
        base_price = None
        base_idx = None
        
        for i, (timestamp, price) in enumerate(sorted_prices):
            if timestamp >= news_timestamp:
                if i > 0:
                    base_price = sorted_prices[i-1][1]  # Previous price
                    base_idx = i-1
                else:
                    base_price = price
                    base_idx = i
                break
        
        if base_price is None or base_idx is None:
            return None
        
        changes = []
        periods = [1, 5, 20]  # days
        
        for period in periods:
            # We are looking for a price for the period days later.
            target_timestamp = news_timestamp + (period * 24 * 3600)  # seconds
            
            # Find closest price
            closest_price = None
            min_time_diff = float('inf')
            
            for timestamp, price in sorted_prices[base_idx:]:
                time_diff = abs(timestamp - target_timestamp)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_price = price
                
                # If we go too far back in time, we stop.
                if timestamp > target_timestamp + (2 * 24 * 3600):  # 2 days tolerance
                    break
            
            if closest_price:
                price_change = (closest_price - base_price) / base_price
                changes.append(price_change)
            else:
                changes.append(0.0)  # No data
        
        return np.array(changes)
    
    def calculate_volatility_changes(self, company_prices, news_timestamp):
        """Calculate volatility and volume changes"""
        sorted_prices = sorted(company_prices.items())
        
        base_idx = None
        for i, (timestamp, price) in enumerate(sorted_prices):
            if timestamp >= news_timestamp:
                base_idx = i-1 if i > 0 else i
                break
        
        if base_idx is None:
            return None
        
        try:
            # Pre-news volatility (5 days before)
            pre_prices = []
            for i in range(max(0, base_idx-5), base_idx):
                if i < len(sorted_prices):
                    pre_prices.append(sorted_prices[i][1])
            
            # Post-news volatility (5 days after)
            post_prices = []
            for i in range(base_idx, min(len(sorted_prices), base_idx+5)):
                post_prices.append(sorted_prices[i][1])
            
            if len(pre_prices) < 3 or len(post_prices) < 3:
                return np.array([0.0, 0.0])
            
            # Calculate returns
            pre_returns = [(pre_prices[i] - pre_prices[i-1]) / pre_prices[i-1] 
                          for i in range(1, len(pre_prices))]
            post_returns = [(post_prices[i] - post_prices[i-1]) / post_prices[i-1] 
                           for i in range(1, len(post_prices))]
            
            # Volatility change
            pre_vol = np.std(pre_returns) if pre_returns else 0.0
            post_vol = np.std(post_returns) if post_returns else 0.0
            vol_change = (post_vol - pre_vol) / (pre_vol + 1e-8)
            
            # Volume change proxy (using price range)
            pre_range = (max(pre_prices) - min(pre_prices)) / np.mean(pre_prices) if pre_prices else 0.0
            post_range = (max(post_prices) - min(post_prices)) / np.mean(post_prices) if post_prices else 0.0
            volume_change = (post_range - pre_range) / (pre_range + 1e-8)
            
            return np.array([vol_change, volume_change])
            
        except Exception as e:
            logger.warning(f"Volatility calculation error: {str(e)}")
            return np.array([0.0, 0.0])