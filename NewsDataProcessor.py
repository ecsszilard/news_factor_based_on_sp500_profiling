import numpy as np
import logging

logger = logging.getLogger("AdvancedNewsFactor.NewsDataProcessor")

class NewsDataProcessor:
    """
    Híradatok feldolgozása a modell tanításához
    """
    
    def __init__(self, company_system, news_model):
        """
        Inicializálja a híradatok feldolgozót
        
        Paraméterek:
            company_system (CompanyEmbeddingSystem): Cég beágyazási rendszer
            news_model (AttentionBasedNewsFactorModel): Hírfaktor modell
        """
        self.company_system = company_system
        self.news_model = news_model
        self.processed_news = []
        
        logger.info("NewsDataProcessor inicializálva")
    
    def process_news_batch(self, news_data, price_data):
        """
        Hírek batch-feldolgozása tanítási adatok létrehozásához
        
        Paraméterek:
            news_data (list): Hírek listája [{'text': ..., 'companies': [...], 'timestamp': ...}]
            price_data (dict): Árfolyamadatok {company: {timestamp: price}}
            
        Visszatérési érték:
            dict: Feldolgozott tanítási adatok
        """
        training_samples = {
            'keywords': [],
            'company_embeddings': [],
            'capitalization_changes': [],
            'keyword_targets': []
        }
        
        for news_item in news_data:
            news_text = news_item['text']
            affected_companies = news_item['companies']
            news_timestamp = news_item['timestamp']
            
            # Kulcsszó szekvencia készítése
            keyword_sequence = self.news_model.prepare_keyword_sequence(news_text)
            
            # Target keyword embedding (BERT-ből)
            target_embedding = self.company_system.get_bert_embedding(news_text)
            
            # Minden érintett céghez egy tanítási példa
            for company in affected_companies:
                if company not in self.company_system.company_embeddings:
                    continue
                
                company_embedding = self.company_system.company_embeddings[company]
                
                # Tényleges árfolyamváltozás számítása
                if company in price_data:
                    price_changes = self.calculate_price_changes(
                        price_data[company], news_timestamp
                    )
                    
                    if price_changes is not None:
                        training_samples['keywords'].append(keyword_sequence)
                        training_samples['company_embeddings'].append(company_embedding)
                        training_samples['capitalization_changes'].append(price_changes)
                        training_samples['keyword_targets'].append(target_embedding[:self.news_model.latent_dim])
        
        logger.info(f"Feldolgozva {len(training_samples['keywords'])} tanítási példa")
        return training_samples
    
    def calculate_price_changes(self, company_prices, news_timestamp):
        """
        Számítja az árfolyamváltozásokat a hír után
        
        Paraméterek:
            company_prices (dict): Árfolyamadatok {timestamp: price}
            news_timestamp (float): Hír időbélyege
            
        Visszatérési érték:
            numpy.ndarray: [1d_change, 5d_change, 20d_change, volatility_change]
        """
        # Időrendbe rakott árak
        sorted_prices = sorted(company_prices.items())
        
        # Hír időpontjának megkeresése
        base_price = None
        base_idx = None
        
        for i, (timestamp, price) in enumerate(sorted_prices):
            if timestamp >= news_timestamp:
                if i > 0:
                    base_price = sorted_prices[i-1][1]  # Az előző ár
                    base_idx = i-1
                else:
                    base_price = price
                    base_idx = i
                break
        
        if base_price is None or base_idx is None:
            return None
        
        changes = []
        periods = [1, 5, 20]  # 1, 5, 20 napos időszakok
        
        for period in periods:
            # Keresünk árat a period nappal később
            target_timestamp = news_timestamp + (period * 24 * 3600)  # seconds
            
            # Legközelebbi ár keresése
            closest_price = None
            min_time_diff = float('inf')
            
            for timestamp, price in sorted_prices[base_idx:]:
                time_diff = abs(timestamp - target_timestamp)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_price = price
                
                # Ha túl messze megyünk időben, abbahagyjuk
                if timestamp > target_timestamp + (2 * 24 * 3600):  # 2 nap tolerancia
                    break
            
            if closest_price:
                price_change = (closest_price - base_price) / base_price
                changes.append(price_change)
            else:
                changes.append(0.0)  # Nincs adat
        
        # Volatilitás változás számítása (egyszerűsített)
        # Az utolsó 5 nap volatilitása vs. az előző 5 nap volatilitása
        volatility_change = self.calculate_volatility_change(
            sorted_prices, base_idx, news_timestamp
        )
        changes.append(volatility_change)
        
        return np.array(changes)
    
    def calculate_volatility_change(self, sorted_prices, base_idx, news_timestamp):
        """
        Volatilitás változásának számítása
        
        Paraméterek:
            sorted_prices (list): Rendezett árfolyamok
            base_idx (int): Alapindex
            news_timestamp (float): Hír időbélyege
            
        Visszatérési érték:
            float: Volatilitás változás
        """
        try:
            # Előző 5 nap volatilitása
            pre_prices = []
            for i in range(max(0, base_idx-5), base_idx):
                pre_prices.append(sorted_prices[i][1])
            
            # Következő 5 nap volatilitása  
            post_prices = []
            for i in range(base_idx, min(len(sorted_prices), base_idx+5)):
                post_prices.append(sorted_prices[i][1])
            
            if len(pre_prices) < 3 or len(post_prices) < 3:
                return 0.0
            
            # Napi hozamok számítása
            pre_returns = []
            for i in range(1, len(pre_prices)):
                ret = (pre_prices[i] - pre_prices[i-1]) / pre_prices[i-1]
                pre_returns.append(ret)
            
            post_returns = []
            for i in range(1, len(post_prices)):
                ret = (post_prices[i] - post_prices[i-1]) / post_prices[i-1]
                post_returns.append(ret)
            
            if not pre_returns or not post_returns:
                return 0.0
            
            pre_vol = np.std(pre_returns)
            post_vol = np.std(post_returns)
            
            if pre_vol > 0:
                vol_change = (post_vol - pre_vol) / pre_vol
            else:
                vol_change = 0.0
                
            return vol_change
            
        except Exception as e:
            logger.warning(f"Volatilitás számítási hiba: {str(e)}")
            return 0.0
