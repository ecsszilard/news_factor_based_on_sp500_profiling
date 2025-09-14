import numpy as np
import pandas as pd
import time
import logging

from CompanyEmbeddingSystem import CompanyEmbeddingSystem
from AttentionBasedNewsFactorModel import AttentionBasedNewsFactorModel
from NewsDataProcessor import NewsDataProcessor
from AdvancedTradingSystem import AdvancedTradingSystem
from PerformanceAnalyzer import PerformanceAnalyzer

# Logging beállítások
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_newsfactor_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AdvancedNewsFactor")

def create_sample_data():
    """
    Minta adatok létrehozása a rendszer tesztelésére
    """
    # Minta cégadatok
    companies_data = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        'name': ['Apple Inc.', 'Microsoft Corporation', 'Alphabet Inc.', 
                'Amazon.com Inc.', 'Tesla Inc.'],
        'sector': ['Technology', 'Technology', 'Technology', 
                  'Consumer Discretionary', 'Consumer Discretionary']
    })
    
    # Minta hírek
    sample_news = [
        {
            'text': 'Apple announces record quarterly earnings with strong iPhone sales and services growth',
            'companies': ['AAPL'],
            'timestamp': time.time() - 86400  # 1 nap ezelőtt
        },
        {
            'text': 'Microsoft Azure cloud revenue grows 30% year-over-year, beating expectations',
            'companies': ['MSFT'],
            'timestamp': time.time() - 3600  # 1 óra ezelőtt
        },
        {
            'text': 'Tesla delivers record number of vehicles in Q3, stock surges on production milestone',
            'companies': ['TSLA'],
            'timestamp': time.time() - 7200  # 2 óra ezelőtt
        }
    ]
    
    # Minta árfolyamadatok
    sample_prices = {}
    for company in companies_data['symbol']:
        prices = {}
        base_price = 100 + np.random.rand() * 200  # 100-300 közötti alapár
        
        for i in range(-30, 5):  # 30 nappal ezelőttől 5 nappal előre
            timestamp = time.time() + (i * 24 * 3600)
            # Random walk árfolyam szimuláció
            price_change = np.random.normal(0, 0.02)  # 2% átlagos napi volatilitás
            base_price *= (1 + price_change)
            prices[timestamp] = max(base_price, 1.0)  # Pozitív ár
        
        sample_prices[company] = prices
    
    return companies_data, sample_news, sample_prices

if __name__ == "__main__":
    logger.info("Fejlett hírfaktor-elemző kereskedési rendszer indítása...")
    
    # Create sample data
    companies_df, sample_news, sample_prices = create_sample_data()
    companies_df.to_csv('sp500_companies.csv', index=False)
    
    company_system = CompanyEmbeddingSystem('sp500_companies.csv')
    news_model = AttentionBasedNewsFactorModel(company_system)
    
    # Build vocabulary
    all_news_texts = [news['text'] for news in sample_news]
    news_model.build_vocabulary(all_news_texts)

    # Store static company features (used for initialization/analysis)
    for _, company_row in companies_df.iterrows():
        symbol = company_row['symbol']
        
        fundamental_data = {
            'market_cap': np.random.uniform(10e9, 1e12),
            'pe_ratio': np.random.uniform(10, 40),
            'revenue_growth': np.random.uniform(-0.1, 0.3),
            'profit_margin': np.random.uniform(0.01, 0.3),
            'debt_to_equity': np.random.uniform(0.1, 2.0),
            'roa': np.random.uniform(-0.05, 0.2),
            'current_ratio': np.random.uniform(0.5, 3.0),
            'book_value': np.random.uniform(10, 500),
            'dividend_yield': np.random.uniform(0, 0.08),
            'beta': np.random.uniform(0.5, 2.0)
        }
        
        price_data = {
            'volatility_30d': np.random.uniform(0.15, 0.35),
            'return_1d': np.random.uniform(-0.05, 0.05),
            'return_5d': np.random.uniform(-0.1, 0.1),
            'return_20d': np.random.uniform(-0.2, 0.2),
            'return_60d': np.random.uniform(-0.3, 0.3),
            'volume_ratio': np.random.uniform(0.5, 2.0),
            'momentum_score': np.random.uniform(-1, 1),
            'rsi': np.random.uniform(20, 80)
        }
        
        # Sectoral information
        sector_info = {'sector': company_row['sector']}
        
        company_system.store_static_features(
            symbol=symbol,
            fundamental_data=fundamental_data,
            price_data=price_data,
            sector_info=sector_info
        )
    
    # Process training data
    data_processor = NewsDataProcessor(company_system, news_model)
    training_data = data_processor.process_news_batch(sample_news, sample_prices)
    
    # Test keyword impact clustering
    if len(training_data['keywords']) >= 5:
        print("Training multi-task model...")
        history = news_model.train(training_data=training_data, epochs=20, batch_size=4)
        print("Training completed!")
        
        # Analyze keyword impact patterns
        test_keywords = ['breakthrough', 'revenue', 'profit', 'loss', 'acquisition', 'bankruptcy', 'innovation', 'decline']
        keyword_clusters = news_model.analyze_keyword_impact_clusters(test_keywords)
        
        print("\nKeyword Impact Clusters (words with similar market effects):")
        for keyword, similar_words in keyword_clusters.items():
            if similar_words:  # Only show keywords that have similar ones
                similar_names = [word for word, sim in similar_words[:3]]
                print(f"  '{keyword}' clusters with: {similar_names}")
        
        # Test specific keyword similarity
        test_word = 'breakthrough'
        if test_word in news_model.tokenizer.word_to_idx:
            similar_keywords = news_model.get_similar_keywords_by_impact(test_word, top_k=5)
            print(f"\nKeywords with similar impact to '{test_word}':")
            for word, similarity in similar_keywords:
                print(f"  {word}: {similarity:.3f}")
    
    
    # Create trading system
    trading_system = AdvancedTradingSystem(company_system, news_model)
    
    # Test the system
    test_news = "Tesla reports breakthrough in battery technology, expects 50% cost reduction and improved range"
    target_companies = ['TSLA', 'AAPL', 'F', 'GM']
    
    print(f"\nAnalyzing news: {test_news}")
    print(f"Target companies: {target_companies}")
    
    news_impact = trading_system.analyze_news_impact(test_news, target_companies)
    
    for company, analysis in news_impact.items():
        print(f"\n{company}:")
        print(f"  Relevance Score: {analysis['relevance_score']:.3f}")
        print(f"  Confidence: {analysis['confidence']:.3f}")
        print(f"  Price Changes: 1d={analysis['predicted_changes']['1d']:.3f}, "
              f"5d={analysis['predicted_changes']['5d']:.3f}, "
              f"20d={analysis['predicted_changes']['20d']:.3f}")
        print(f"  Volatility Impact: {analysis['volatility_impact']['volatility']:.3f}")
        print(f"  Similar Companies: {[comp[0] for comp in analysis['similar_companies'][:2]]}")
    
    # Generate and execute trading signals
    trading_signals = trading_system.generate_trading_signals(news_impact, 
                                                            relevance_threshold=0.3, 
                                                            confidence_threshold=0.3)
    executed_trades = trading_system.execute_trades(trading_signals)
    
    # 11. Teljesítmény elemzése
    performance_analyzer = PerformanceAnalyzer(trading_system)
    performance_report = performance_analyzer.generate_performance_report('improved_performance_report.json')
    
    logger.info("Teljesítményjelentés:")
    logger.info(f"Portfolio érték: ${performance_report['portfolio_value']:.2f}")
    logger.info(f"Aktív pozíciók: {performance_report['active_positions']}")
    logger.info(f"Összes kereskedés: {performance_report['total_trades']}")
    
    # 12. Modellek mentése
    trading_system.save_model_and_data('improved_models')
    
    logger.info("Rendszer futása sikeres!")
    
    print("✅ Fejlett hírfaktor-elemző kereskedési rendszer sikeresen futott!")
    print(f"📊 Portfolio érték: ${performance_report['portfolio_value']:.2f}")
    print(f"📈 Aktív pozíciók: {performance_report['active_positions']}")