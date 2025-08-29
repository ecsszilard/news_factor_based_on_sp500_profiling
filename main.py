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
    
    # Minta adatok létrehozása
    companies_df, sample_news, sample_prices = create_sample_data()
    companies_df.to_csv('sp500_companies.csv', index=False)
    
    # 1. Company Embedding System inicializálása
    company_system = CompanyEmbeddingSystem('sp500_companies.csv')
    
    # 2. Attention-alapú modell inicializálása
    news_model = AttentionBasedNewsFactorModel()
    
    # 3. FONTOS: Szótár építése a hírekből
    all_news_texts = [news['text'] for news in sample_news]
    news_model.build_vocabulary(all_news_texts)
    
    # 4. Cégek beágyazásainak létrehozása
    for _, company_row in companies_df.iterrows():
        symbol = company_row['symbol']
        
        # Minta adatok generálása (ugyanaz mint előtte)
        fundamental_data = {
            'market_cap': np.random.uniform(10e9, 1e12),
            'pe_ratio': np.random.uniform(10, 40),
            'revenue_growth': np.random.uniform(-0.1, 0.3),
            'profit_margin': np.random.uniform(0.05, 0.25),
            'debt_to_equity': np.random.uniform(0.1, 2.0),
            'roa': np.random.uniform(0.01, 0.15),
            'current_ratio': np.random.uniform(0.8, 3.0),
            'book_value': np.random.uniform(50, 200),
            'dividend_yield': np.random.uniform(0, 0.05),
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
        
        # Szektoriális információk
        sector_info = {'sector': company_row['sector']}
        
        # Minta hírek a cégről
        company_news = [news['text'] for news in sample_news if symbol in news['companies']]
        
        company_system.create_company_embedding(
            symbol=symbol,
            news_texts=company_news,
            fundamental_data=fundamental_data,
            price_data=price_data,
            sector_info=sector_info
        )
    
    # 5. Híradatok feldolgozása tanításhoz
    data_processor = NewsDataProcessor(company_system, news_model)
    training_data = data_processor.process_news_batch(sample_news, sample_prices)
    
    # 6. Modell tanítása
    if len(training_data['keywords']) >= 5:
        logger.info("Modell tanítása...")
        history = news_model.train(
            training_data=training_data,
            epochs=10,
            batch_size=2
        )
        logger.info("Modell tanítása befejezve")
    else:
        logger.warning("Túl kevés tanítási adat, modell nem tanítva")
    
    # 7. Kereskedési rendszer inicializálása
    trading_system = AdvancedTradingSystem(company_system, news_model)
    
    # 8. Új hír elemzése fejlett tokenizálással
    test_news = "Tesla reports breakthrough in battery technology, expects 50% cost reduction"
    target_companies = ['TSLA', 'AAPL']
    
    logger.info("Fejlett tokenizálás teszt:")
    test_tokens = news_model.prepare_keyword_sequence(test_news)
    logger.info(f"Tokenizált hír: {test_tokens[:10]}...")  # Első 10 token
    
    news_impact = trading_system.analyze_news_impact(test_news, target_companies)
    logger.info(f"Hírelemzés eredmények: {news_impact}")
    
    # 9. Attention weights elemzése
    if 'TSLA' in news_impact:
        test_company_embedding = company_system.company_embeddings['TSLA']
        attention_info = news_model.get_attention_weights(test_tokens, test_company_embedding)
        logger.info(f"Kulcsszavak: {attention_info['decoded_keywords'][:10]}")
        logger.info(f"Szekvencia hossz: {attention_info['sequence_length']}")
    
    # 10. Kereskedési folyamat
    trading_signals = trading_system.generate_trading_signals(news_impact)
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