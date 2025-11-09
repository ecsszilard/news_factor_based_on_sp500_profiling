import numpy as np
import pandas as pd
import time

class Utils:
    
    def __init__(self):
        pass

    def create_sample_data(self):
        """Creating enhanced sample data"""

        sample_companies = pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'F', 'GM', 'NFLX', 'NIKE'],
            'name': ['Apple Inc.', 'Microsoft Corporation', 'Tesla Inc.', 'NVIDIA Corporation', 
                     'Ford Motor Company', 'General Motors Company', 'Netflix Inc.','Nike Inc.'],
            'sector': ['Technology', 'Technology', 'Consumer Discretionary', 'Technology', 
                       'Consumer Discretionary','Consumer Discretionary', 'Communication Services', 'Consumer Discretionary']
        })
        
        sample_news = [
            {
                'text': 'Apple Inc. delivered stunning quarterly results with record-breaking iPhone sales that exceeded all analyst expectations. The company reported remarkable 28% year-over-year growth in its services division, including App Store, iCloud, and Apple TV+. CEO Tim Cook highlighted unprecedented demand across the entire ecosystem, with particularly strong performance in emerging international markets where Apple has been strategically investing.',
                'companies': ['AAPL'],
                'timestamp': time.time() - 86400
            },
            {
                'text': 'Microsoft Corporation announced Azure cloud platform achieved 35% year-over-year revenue increase, significantly surpassing Wall Street forecasts. The growth has been primarily driven by unprecedented surge in AI-powered services demand. CEO Satya Nadella emphasized strategic OpenAI partnership proving particularly lucrative, with GPT model integration attracting major corporate clients seeking cutting-edge AI capabilities.',
                'companies': ['MSFT'],
                'timestamp': time.time() - 3600
            },
            {
                'text': 'Tesla Inc. announced record-breaking quarterly delivery numbers across all model lines, with Model Y becoming one of the best-selling vehicles globally. The achievement is remarkable given ongoing supply chain challenges. CEO Elon Musk emphasized vertical integration strategy and battery production capabilities as crucial competitive advantages, enabling premium pricing while scaling production.',
                'companies': ['TSLA'],
                'timestamp': time.time() - 7200
            },
            {
                'text': 'NVIDIA Corporation unveiled breakthrough AI semiconductor technology representing quantum leap in processing efficiency. The new architecture offers unprecedented performance improvements for machine learning training and real-time inference. CEO Jensen Huang described it as dawn of new computing era. Strategic partnerships with major cloud providers ensure global availability, with early benchmarks showing 10x faster performance while consuming significantly less power.',
                'companies': ['NVDA'],
                'timestamp': time.time() - 14400
            },
            {
                'text': 'Ford Motor Company and General Motors announced significant production delays due to unprecedented semiconductor shortage crisis. Both automakers implementing temporary shutdowns at multiple North American facilities. Ford Executive Chairman described challenges as unlike anything in company 120-year history. The situation highlights strategic importance of supply chain resilience and domestic semiconductor manufacturing.',
                'companies': ['F', 'GM'],
                'timestamp': time.time() - 21600
            },
            {
                'text': 'Netflix Inc. reported subscriber growth significantly exceeding analyst expectations, with particularly strong international market performance. The streaming giant added millions of new subscribers driven by exceptional original content slate. Co-CEO Reed Hastings emphasized data-driven content creation approach continues yielding impressive results. Recent price adjustments being well-received by subscribers who recognize platform value.',
                'companies': ['NFLX'],
                'timestamp': time.time() - 10800
            },
            {
                'text': 'Nike Inc. unveiled revolutionary sustainable athletic footwear program combining cutting-edge performance with environmental responsibility. The comprehensive circular economy initiative reimagines entire product lifecycle from design through recycling. CEO John Donahoe emphasized largest sustainability investment in Nike history, positioning company ahead of competitors in rapidly growing sustainable products market favored by younger demographics.',
                'companies': ['NIKE'],
                'timestamp': time.time() - 18000
            },
            {
                'text': 'Technology sector rally powered by strong earnings from Apple and Microsoft, lifting both companies to new all-time highs. Cloud computing adoption and AI integration accelerating across multiple industries. Analysts noting synchronized strength across tech megacaps suggesting broad-based demand recovery. Both companies benefiting from enterprise digital transformation trends.',
                'companies': ['AAPL', 'MSFT'],
                'timestamp': time.time() - 5000
            },
            {
                'text': 'Electric vehicle market experiencing unprecedented demand surge as Tesla, Ford, and GM all report strong quarterly deliveries. Battery technology improvements and expanding charging infrastructure driving mainstream adoption. Tesla maintaining market leadership while traditional automakers accelerating EV programs. Industry analysts predict sustained high growth as consumers increasingly prioritize sustainability.',
                'companies': ['TSLA', 'F', 'GM'],
                'timestamp': time.time() - 12000
            },
            {
                'text': 'Semiconductor supply constraints showing signs of gradual improvement, particularly benefiting Tesla and technology manufacturers. NVIDIA capacity expansion initiatives helping ease shortage situation. However, automotive sector still facing significant challenges. Industry experts warn recovery timeline remains uncertain, with structural changes to supply chain management likely permanent.',
                'companies': ['TSLA', 'NVDA'],
                'timestamp': time.time() - 16000
            },
            {
                'text': 'Consumer discretionary sector showing mixed performance with Nike reporting exceptionally strong brand demand and premium pricing power, while Ford grapples with inventory management challenges due to production constraints. Divergence highlights importance of supply chain resilience and brand strength in current environment.',
                'companies': ['NIKE', 'F'],
                'timestamp': time.time() - 20000
            }
        ]
        
        base_correlation_matrix = {
            ('AAPL', 'MSFT'): 0.6, ('AAPL', 'NVDA'): 0.4, ('MSFT', 'NVDA'): 0.5,
            ('F', 'GM'): 0.8, ('TSLA', 'F'): 0.3, ('TSLA', 'GM'): 0.3,
            ('AAPL', 'TSLA'): 0.2, ('MSFT', 'TSLA'): 0.2,
            ('NIKE', 'NFLX'): 0.3, ('NIKE', 'AAPL'): 0.25
        }

        sample_prices = self.generate_realistic_prices(sample_companies, sample_news, base_correlation_matrix)
        return sample_companies, sample_news, sample_prices

    def generate_realistic_prices(self, companies, news_events, base_correlation_matrix):
        n_days, n_companies = 35, len(companies)

        # Correlation matrix construction
        corr = np.eye(n_companies)
        company_list = companies['symbol'].tolist()
        for i, c1 in enumerate(company_list):
            for j, c2 in enumerate(company_list):
                if i != j:
                    corr[i, j] = base_correlation_matrix.get((c1, c2), base_correlation_matrix.get((c2, c1), 0.1))

        # Covariance matrix and returns
        vols = np.random.uniform(0.015, 0.035, size=n_companies)
        cov = np.outer(vols, vols) * corr
        returns = np.random.multivariate_normal(np.zeros(n_companies), cov, size=n_days)

        # Price generation
        price_data = {}
        for i, company in enumerate(company_list):
            price, prices = np.random.uniform(50, 300), {}
            for day in range(n_days):
                r = returns[day, i]
                for news in news_events:
                    if company in news['companies']:
                        news_day = int((news['timestamp'] - (time.time() - 30*24*3600)) / 86400)
                        dt = day - news_day
                        if dt >= 0:
                            r += news.get("impact", 0.02) * np.exp(-dt/2)
                price *= (1 + r)
                prices[time.time() + ((day - 30) * 86400)] = max(price, 1.0)
            price_data[company] = prices
        return price_data