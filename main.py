import numpy as np
import pandas as pd
import time
import logging

from EmbeddingAndTokenizerSystem import EmbeddingAndTokenizerSystem
from AttentionBasedNewsFactorModel import AttentionBasedNewsFactorModel
from NewsDataProcessor import NewsDataProcessor
from AdvancedTradingSystem import AdvancedTradingSystem
from PerformanceAnalyzer import PerformanceAnalyzer

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

    sample_prices = generate_realistic_prices(sample_companies, sample_news, base_correlation_matrix)
    
    return sample_companies, sample_news, sample_prices

def generate_realistic_prices(companies, news_events, base_correlation_matrix):
    n_days, n_companies = 35, len(companies)

    # Correlation matrix construction
    corr = np.eye(n_companies)
    company_list = companies['symbol'].tolist()
    for i, c1 in enumerate(company_list):
        for j, c2 in enumerate(company_list):
            if i != j:
                corr[i, j] = base_correlation_matrix.get((c1, c2), base_correlation_matrix.get((c2, c1), 0.1))

    # Kovariancia mátrix és hozamok
    vols = np.random.uniform(0.015, 0.035, size=n_companies)
    cov = np.outer(vols, vols) * corr
    returns = np.random.multivariate_normal(np.zeros(n_companies), cov, size=n_days)

    # Árfolyamok generálása
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

if __name__ == "__main__":
    logger.info("Starting RESIDUAL LEARNING correlation-based news factor trading system...")

    # --- Sample data & system setup ---
    companies_df, sample_news, sample_prices = create_sample_data()
    companies = list(sample_prices.keys())
    max_keywords = 100

    embeddingAndTokenizerSystem = EmbeddingAndTokenizerSystem(companies)
    news_model = AttentionBasedNewsFactorModel(embeddingAndTokenizerSystem, max_keywords)

    # --- Trading system setup ---
    trading_system = AdvancedTradingSystem(embeddingAndTokenizerSystem, news_model, companies)
    
    print("Updating correlation matrix...")
    trading_system.update_correlation_matrix(sample_prices)
    print(f"Correlation matrix: {len(trading_system.correlation_matrix)} companies")

    if trading_system.correlation_matrix:
        print("\nSample correlations:")
        for c1 in list(trading_system.correlation_matrix)[:3]:
            for c2, corr in list(trading_system.correlation_matrix[c1].items())[:3]:
                print(f"  {c1} <-> {c2}: {corr:.3f}")

    # --- TRAINING DATA WITH BASELINE CORRELATION ---
    data_processor = NewsDataProcessor(embeddingAndTokenizerSystem, news_model)
    
    # ✅ The processor AUTOMATICALLY calculates baseline correlation from price_data
    training_data = data_processor.process_news_batch(sample_news, sample_prices)
    
    # ✅ Check what's in training_data
    print(f"\n📊 Training data prepared:")
    print(f"  Keys: {list(training_data.keys())}")
    print(f"  Samples: {len(training_data['keywords'])}")
    if len(training_data['keywords']) > 0:
        print(f"  Baseline correlation shape: {training_data['baseline_correlation'][0].shape}")
        print(f"  Target correlation shape: {training_data['correlation_changes'][0].shape}")
        print(f"  Baseline range (Fisher-z): [{np.min(training_data['baseline_correlation'][0]):.3f}, {np.max(training_data['baseline_correlation'][0]):.3f}]")

    # --- TRAIN THE MODEL ---
    if len(training_data["keywords"]) >= 5:
        print("\n🔥 Training residual correlation model...")
        news_model.train(training_data=training_data, epochs=20, batch_size=4)
        print("✅ Training completed!")

    # --- TEST NEWS IMPACT ---
    test_news = "Tesla reports breakthrough in battery technology, expects 50% cost reduction"
    target_companies = ["TSLA", "AAPL", "F", "GM", "NVDA"]

    print(f"\n📰 Analyzing news: {test_news}")
    
    # ⭐ Use the UPDATED analyze_news_impact (with baseline)
    news_impact = trading_system.analyze_news_impact(test_news, target_companies)

    for company, analysis in news_impact.items():
        print(f"\n{company}:")
        print(f"  Confidence: {analysis['confidence']:.3f}")
        print(f"  Max Correlation Change (Δ): {analysis['correlation_impact']['max_change']:.3f}")
        print(f"  Mean Correlation Change (Δ): {analysis['correlation_impact']['mean_change']:.3f}")
        print(f"  Baseline Avg Corr: {analysis['correlation_impact']['baseline_avg']:.3f}")
        print(f"  Predicted Avg Corr: {analysis['correlation_impact']['predicted_avg']:.3f}")
        print(f"  Significant Pairs: {len(analysis['correlation_impact']['significant_pairs'])}")
        if analysis['correlation_impact']['significant_pairs']:
            for other_company, corr_change in analysis['correlation_impact']['significant_pairs'][:2]:
                print(f"    {company} <-> {other_company}: Δ={corr_change:+.3f}")
        print(f"  Similar Companies: {[comp[0] for comp in analysis['similar_companies'][:2]]}")

    # --- Trading signals & execution ---
    signals = trading_system.generate_trading_signals(news_impact, 0.2)  # Lower thresholds for correlation model
    print(f"\nGenerated {len(signals)} trading signals")
    for i, s in enumerate(signals[:3], 1):
        corr_impact = s.get('correlation_impact', {})
        print(f"  Signal {i}: {s['type']} {s['company']} "
              f"(strength={s['strength']:.3f}, Δcorr={corr_impact.get('max_change', 0):.3f}, "
              f"adj={s.get('correlation_adjustment',1.0):.3f})")
    
    trading_system.execute_trades(signals)

    # --- Keyword clustering ---
    if len(training_data["keywords"]) >= 5:
        print("\n🔤 Keyword Impact Clusters:")
        test_keywords = ["breakthrough", "revenue", "profit", "loss", "acquisition", "bankruptcy", "innovation", "decline"]
        for keyword, similar_words in news_model.analyze_keyword_impact_clusters(test_keywords).items():
            if similar_words:  # Only show keywords that have similar ones
                similar_names = [word for word, sim in similar_words[:3]]
                print(f"  '{keyword}' clusters with: {similar_names}")

        test_word = "breakthrough"
        if test_word in embeddingAndTokenizerSystem.word_to_idx:
            print(f"\nSimilar keywords to '{test_word}':")
            for w, sim in trading_system.get_similar_keywords_by_impact(test_word, 5):
                print(f"  {w}: {sim:.3f}")

    # --- Portfolio & performance ---
    div = trading_system.get_portfolio_diversification_metrics()
    print(f"\n📈 Portfolio Diversification:\n  Score={div.get('diversification_score',0):.3f}, "
          f"AvgCorr={div.get('average_correlation',0):.3f}, "
          f"Positions={div.get('num_positions',0)}")

    print(f"🔗 Correlation Matrix: {len(trading_system.correlation_matrix)} companies tracked")    
    # Get baseline for analysis
    baseline_matrix = trading_system._get_baseline_correlation_matrix()
    baseline_z = data_processor.fisher_z_transform(baseline_matrix)
    
    company_idx = embeddingAndTokenizerSystem.company_to_idx.get(target_companies[0], 0)
    keywords = embeddingAndTokenizerSystem.prepare_keyword_sequence(test_news, max_keywords)
    
    predictions = news_model.model.predict([keywords, np.array([[company_idx]]), np.expand_dims(baseline_z, 0)], verbose=0)
    predicted_corr_z = predictions[0][0]
    delta_z = predicted_corr_z - baseline_z
    
    # Convert to correlation space for interpretation
    predicted_corr = data_processor.inverse_fisher_z_transform(predicted_corr_z)
    baseline_corr = data_processor.inverse_fisher_z_transform(baseline_z)
    delta_corr = predicted_corr - baseline_corr
    
    print(f"  Baseline correlation (Fisher-z) range: [{baseline_z.min():.3f}, {baseline_z.max():.3f}]")
    print(f"  Predicted correlation (Fisher-z) range: [{predicted_corr_z.min():.3f}, {predicted_corr_z.max():.3f}]")
    print(f"  Learned delta (Fisher-z) range: [{delta_z.min():.3f}, {delta_z.max():.3f}]")
    print(f"  Mean absolute delta: {np.abs(delta_corr).mean():.3f}")
    
    # Show top changes
    significant_changes = []
    for i in range(len(companies)):
        for j in range(i+1, len(companies)):
            if abs(delta_corr[i, j]) > 0.05:
                significant_changes.append((
                    companies[i], 
                    companies[j], 
                    delta_corr[i, j],
                    baseline_corr[i, j],
                    predicted_corr[i, j]
                ))
    
    significant_changes.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if significant_changes:
        print(f"\n  Top predicted correlation changes (Δ):")
        for c1, c2, delta, baseline, predicted in significant_changes[:5]:
            print(f"    {c1} <-> {c2}:")
            print(f"      Baseline: {baseline:+.3f}, Predicted: {predicted:+.3f}, Δ: {delta:+.3f}")
    else:
        print("  No significant correlation changes predicted")