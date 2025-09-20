import numpy as np
import pandas as pd
import time
import logging

from CompanyEmbeddingSystem import CompanyEmbeddingSystem
from AttentionBasedNewsFactorModel import AttentionBasedNewsFactorModel
from NewsDataProcessor import NewsDataProcessor
from AdvancedTradingSystem import AdvancedTradingSystem
from PerformanceAnalyzer import PerformanceAnalyzer
from ImprovedTokenizer import ImprovedTokenizer

# Logging settings
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
    """Creating sample data to test the system"""

    sample_companies = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'F', 'GM', 'JPM', 'NFLX'],
        'name': ['Apple Inc.', 'Microsoft Corporation', 'Alphabet Inc.', 'Amazon.com Inc.', 
                'Tesla Inc.', 'NVIDIA Corporation', 'Meta Platforms Inc.', 'Ford Motor Company',
                'General Motors Company', 'JPMorgan Chase & Co.', 'Netflix Inc.'],
        'sector': ['Technology', 'Technology', 'Technology', 'Consumer Discretionary', 
                  'Consumer Discretionary', 'Technology', 'Technology', 'Consumer Discretionary',
                  'Consumer Discretionary', 'Financials', 'Communication Services']
    })
    
    sample_news = [
        {
            'text': 'Apple announces record quarterly earnings with strong iPhone sales and services growth in its services division. The results exceeded analyst expectations, highlighting Apple\'s ability to maintain strong demand despite global supply chain pressures. Investors responded positively, sending the stock higher in after-hours trading.',
            'companies': ['AAPL'],
            'timestamp': time.time() - 86400  # 1 day ago
        },
        {
            'text': 'Microsoft\'s Azure cloud business posted a 30% year-over-year revenue increase, surpassing Wall Street forecasts. Strong enterprise demand for AI-driven services and hybrid cloud adoption were cited as major contributors to growth. This continued momentum strengthens Microsoft\'s position in the competitive cloud computing market.',
            'companies': ['MSFT'],
            'timestamp': time.time() - 3600  # 1 hour ago
        },
        {
            'text': 'Tesla announced it had delivered a record number of vehicles in the third quarter, reinforcing its lead in the EV sector. The achievement was supported by improved production efficiency and strong demand across multiple markets. Shares of Tesla surged as analysts raised their price targets following the news.',
            'companies': ['TSLA'],
            'timestamp': time.time() - 7200  # 2 hours ago
        },
        {
            'text': 'NVIDIA unveiled a major breakthrough in AI-focused semiconductor technology, designed to accelerate training and inference workloads. The company also announced strategic partnerships with leading cloud providers to integrate the new chips, underscoring its dominance in the AI hardware space',
            'companies': ['NVDA'],
            'timestamp': time.time() - 14400  # 4 hours ago
        },
        {
            'text': 'Global supply chain challenges continue to impact the auto industry, with Ford and General Motors reporting production delays. The shortages, particularly in semiconductor availability, have led to temporary shutdowns at several plants. Industry experts warn that continued disruptions may weigh on profitability into the next quarter.',
            'companies': ['F', 'GM'],
            'timestamp': time.time() - 21600  # 6 hours ago
        },
        {
            'text': 'Netflix reported an increase in global subscribers, fueled by strong performance in international markets and successful original content launches. Despite intensifying competition from other streaming services, Netflix\'s strategy of localized content and global expansion helped boost its growth trajectory, reassuring investors of its long-term prospects.',
            'companies': ['NFLX'],
            'timestamp': time.time() - 10800,
        }
    ]
    
    # Generate more realistic correlated price data
    sample_prices = {}
    base_correlation_matrix = {
        # Tech stocks tend to be correlated
        ('AAPL', 'MSFT'): 0.6, ('AAPL', 'GOOGL'): 0.5, ('MSFT', 'GOOGL'): 0.7,
        ('NVDA', 'AAPL'): 0.4, ('NVDA', 'MSFT'): 0.5, ('META', 'GOOGL'): 0.6,
        # Auto stocks are highly correlated
        ('F', 'GM'): 0.8, ('TSLA', 'F'): 0.3, ('TSLA', 'GM'): 0.3,
        # Some negative correlations
        ('JPM', 'TSLA'): -0.2, ('JPM', 'NVDA'): -0.1
    }
    
    # Generate correlated returns
    companies = sample_companies['symbol'].tolist()
    n_days = 35
    n_companies = len(companies)
    
    # Create correlation matrix
    corr_matrix = np.eye(n_companies)
    for i, comp1 in enumerate(companies):
        for j, comp2 in enumerate(companies):
            if i != j:
                key = (comp1, comp2) if (comp1, comp2) in base_correlation_matrix else (comp2, comp1)
                corr_matrix[i, j] = base_correlation_matrix.get(key, 0.1)
    
    # Generate correlated returns using multivariate normal
    returns = np.random.multivariate_normal(
        mean=np.zeros(n_companies),
        cov=corr_matrix * 0.02**2,  # 2% daily volatility
        size=n_days
    )
    
    # Convert returns to prices
    for i, company in enumerate(companies):
        prices = {}
        base_price = 100 + np.random.rand() * 200  # 100-300 range
        
        for day in range(n_days):
            timestamp = time.time() + ((day - 30) * 24 * 3600)  # 30 days ago to 5 days future
            base_price *= (1 + returns[day, i])
            prices[timestamp] = max(base_price, 1.0)  # Ensure positive price
        
        sample_prices[company] = prices
    
    return sample_companies, sample_news, sample_prices

if __name__ == "__main__":
    logger.info("Starting advanced news factor trading system...")

    # --- Sample data & system setup ---
    companies_df, sample_news, sample_prices = create_sample_data()
    companies_df.to_csv("sp500_companies.csv", index=False)

    company_system = CompanyEmbeddingSystem("sp500_companies.csv")
    tokenizer = ImprovedTokenizer([n["text"] for n in sample_news], vocab_size=50_000)
    news_model = AttentionBasedNewsFactorModel(company_system, tokenizer)

    # --- Store static company features ---
    for _, row in companies_df.iterrows():
        company_system.store_static_features(
            symbol=row["symbol"],
            fundamental_data={
                "market_cap": np.random.uniform(1e10, 1e12),
                "pe_ratio": np.random.uniform(10, 40),
                "revenue_growth": np.random.uniform(-0.1, 0.3),
                "profit_margin": np.random.uniform(0.01, 0.3),
                "debt_to_equity": np.random.uniform(0.1, 2.0),
                "roa": np.random.uniform(-0.05, 0.2),
                "current_ratio": np.random.uniform(0.5, 3.0),
                "book_value": np.random.uniform(10, 500),
                "dividend_yield": np.random.uniform(0, 0.08),
                "beta": np.random.uniform(0.5, 2.0),
            },
            price_data={
                "volatility_30d": np.random.uniform(0.15, 0.35),
                "return_1d": np.random.uniform(-0.05, 0.05),
                "return_5d": np.random.uniform(-0.1, 0.1),
                "return_20d": np.random.uniform(-0.2, 0.2),
                "return_60d": np.random.uniform(-0.3, 0.3),
                "volume_ratio": np.random.uniform(0.5, 2.0),
                "momentum_score": np.random.uniform(-1, 1),
                "rsi": np.random.uniform(20, 80),
            },
            sector_info={"sector": row["sector"]},
        )

    # --- Trading system setup ---
    trading_system = AdvancedTradingSystem(company_system, news_model)

    print("Updating correlation matrix...")
    trading_system.update_correlation_matrix(sample_prices)
    print(f"Correlation matrix: {len(trading_system.correlation_matrix)} companies")

    # Show sample correlations
    if trading_system.correlation_matrix:
        print("\nSample correlations:")
        for c1 in list(trading_system.correlation_matrix)[:3]:
            for c2, corr in list(trading_system.correlation_matrix[c1].items())[:3]:
                print(f"  {c1} <-> {c2}: {corr:.3f}")

    # --- Training data ---
    data_processor = NewsDataProcessor(company_system, news_model)
    training_data = data_processor.process_news_batch(sample_news, sample_prices)

    if len(training_data["keywords"]) >= 5:
        print("Training multi-task model...")
        news_model.train(training_data=training_data, epochs=20, batch_size=4)
        print("Training completed!")

    # --- Test news impact ---
    test_news = "Tesla reports breakthrough in battery technology, expects 50% cost reduction"
    target_companies = ["TSLA", "AAPL", "F", "GM", "NVDA"]

    print(f"\nAnalyzing news: {test_news}")
    news_impact = {}
    for company in target_companies:
        idx = company_system.get_company_idx(company)
        if idx == 0 and company != company_system.companies[0]:
            continue

        predictions = news_model.model.predict([news_model.prepare_keyword_sequence(test_news), np.array([[idx]])], verbose=0)

        news_impact[company] = {
            "predicted_changes": {"1d": predictions[0][0][0], "5d": predictions[0][0][1], "20d": predictions[0][0][2]},
            "volatility_impact": {"volatility": predictions[1][0][0], "volume_proxy": predictions[1][0][1]},
            "relevance_score": predictions[2][0][0],
            "confidence": predictions[2][0][0], # Use relevance as confidence proxy
            "similar_companies": trading_system.get_similar_companies_by_news_response(company, 3),
            "reconstruction_quality": np.mean(np.abs(predictions[3][0])),
        }

    for company, analysis in news_impact.items():
        print(f"\n{company}:")
        print(f"  Relevance Score: {analysis['relevance_score']:.3f}")
        print(f"  Confidence: {analysis['confidence']:.3f}")
        print(f"  Price Changes: 1d={analysis['predicted_changes']['1d']:.3f}, "
              f"5d={analysis['predicted_changes']['5d']:.3f}, "
              f"20d={analysis['predicted_changes']['20d']:.3f}")
        print(f"  Volatility Impact: {analysis['volatility_impact']['volatility']:.3f}")
        print(f"  Similar Companies: {[comp[0] for comp in analysis['similar_companies'][:2]]}")

    # --- Trading signals & execution ---
    signals = trading_system.generate_trading_signals(news_impact, 0.3, 0.3)
    print(f"\nGenerated {len(signals)} trading signals")
    for i, s in enumerate(signals[:3], 1):
        print(f"  Signal {i}: {s['type']} {s['company']} "
              f"(strength={s['strength']:.3f}, corr_adj={s.get('correlation_adjustment',1.0):.3f})")
    trading_system.execute_trades(signals)

    # --- Keyword clustering ---
    if len(training_data["keywords"]) >= 5:
        print("\nKeyword Impact Clusters:")
        test_keywords = ["breakthrough", "revenue", "profit", "loss", "acquisition", "bankruptcy", "innovation", "decline"]
        for keyword, similar_words in news_model.analyze_keyword_impact_clusters(test_keywords).items():
            if similar_words:  # Only show keywords that have similar ones
                similar_names = [word for word, sim in similar_words[:3]]
                print(f"  '{keyword}' clusters with: {similar_names}")

        test_word = "breakthrough"
        if test_word in tokenizer.word_to_idx:
            print(f"\nSimilar keywords to '{test_word}':")
            for w, sim in trading_system.get_similar_keywords_by_impact(test_word, 5):
                print(f"  {w}: {sim:.3f}")

    # --- Portfolio & performance ---
    div = trading_system.get_portfolio_diversification_metrics()
    print(f"\nPortfolio Diversification:\n  Score={div.get('diversification_score',0):.3f}, "
          f"AvgCorr={div.get('average_correlation',0):.3f}, "
          f"Positions={div.get('num_positions',0)}")

    report = PerformanceAnalyzer(trading_system).generate_performance_report("improved_performance_report.json")
    logger.info(f"Portfolio Value: ${report['portfolio_value']:.2f}")
    logger.info(f"Active Positions: {report['active_positions']}")
    logger.info(f"Total Trades: {report['total_trades']}")
    
    trading_system.save_model_and_data("improved_models")

    print("✅ System executed successfully!")
    print(f"📊 Portfolio Value: ${report['portfolio_value']:.2f}")
    print(f"📈 Active Positions: {report['active_positions']}")
    print(f"🔗 Correlation Matrix: {len(trading_system.correlation_matrix)} companies tracked")
