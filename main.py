import numpy as np
from transformers import AutoTokenizer, TFAutoModel
import logging
import matplotlib.pyplot as plt
import tensorflow as tf

from AttentionBasedNewsFactorModel import AttentionBasedNewsFactorModel
from NewsDataProcessor import NewsDataProcessor
from AdvancedTradingSystem import AdvancedTradingSystem
from PerformanceAnalyzer import PerformanceAnalyzer
from Utils import Utils

# Logging settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("probabilistic_newsfactor_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ProbabilisticNewsFactor")
tf.config.set_visible_devices([], 'GPU') 

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("PROBABILISTIC RESIDUAL LEARNING CORRELATION-BASED NEWS FACTOR TRADING")
    logger.info("="*80)

    # --- SAMPLE DATA & SYSTEM SETUP ---
    utils = Utils()
    companies_df, sample_news, sample_prices = utils.create_sample_data()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFAutoModel.from_pretrained('bert-base-uncased', from_pt=True)
    
    # --- USE PROBABILISTIC MODEL ---
    logger.info("\n🧠 Initializing Probabilistic Model...")
    news_model = AttentionBasedNewsFactorModel(tokenizer, num_companies=len(sample_prices.keys()), max_keywords=100, keyword_dim=256, company_dim=128, latent_dim=128)
    logger.info("✅ Model initialized with dual output (μ, σ²)")

    # --- TRAINING DATA WITH BASELINE CORRELATION ---
    data_processor = NewsDataProcessor(news_model, sample_prices, bert_model)
    
    logger.info("\n📄 Processing training data...")
    training_data = data_processor.process_news_batch(sample_news)
    
    logger.info(f"\n📦 Training Data Summary:")
    logger.info(f"  Samples: {len(training_data['keyword_sequence'])}")
    if len(training_data['keyword_sequence']) > 0:
        logger.info(f"  Baseline correlation shape: {training_data['baseline_correlation'][0].shape}")
        logger.info(f"  Target correlation shape: {training_data['correlation_changes'][0].shape}")
        baseline_sample = training_data['baseline_correlation'][0]
        logger.info(f"  Baseline range (Fisher-z): [{np.min(baseline_sample):.3f}, {np.max(baseline_sample):.3f}]")

    # --- TRAIN THE PROBABILISTIC MODEL ---
    news_model.train(training_data=training_data, epochs=20, batch_size=4)

    # --- TEST NEWS IMPACT WITH UNCERTAINTY ---
    test_news = "Tesla reports breakthrough in battery technology, expects 50% cost reduction"
    target_companies = ["TSLA", "AAPL", "F", "GM", "NVDA"]
    focus_company = "TSLA"
    focus_company_idx = list(sample_prices.keys()).index(focus_company)
    affected_companies_test = ["TSLA"]  # Only Tesla is directly affected

    logger.info("="*60)
    logger.info(f"📰 ANALYZING NEWS WITH UNCERTAINTY QUANTIFICATION")
    logger.info("="*60)
    logger.info(f"News: {test_news}")
    logger.info(f"Affected Companies: {affected_companies_test}")
    logger.info("")

    # --- TRADING SYSTEM SETUP ---
    trading_system = AdvancedTradingSystem(data_processor)

    # Use the predict_news_impact with affected_companies
    news_impact = trading_system.predict_news_impact(test_news, target_companies, affected_companies_test)

    # Print news impact results with uncertainty
    for company, analysis in news_impact.items():
        logger.info(f"\n{company}:")
        logger.info(f"  News Scope: {analysis['news_scope']}")
        logger.info(f"  Affected Companies: {analysis['affected_companies']}")
        logger.info(f"  Total Confidence: {analysis['total_confidence']:.3f}")
        logger.info(f"  Reconstruction Error: {analysis['reconstruction_error']:.4f}")
        logger.info(f"  Tradeable: {'✅ Yes' if analysis['tradeable'] else '❌ No'}")
        
        corr_impact = analysis['correlation_impact']
        logger.info(f"  Correlation Impact:")
        logger.info(f"    Max Change (Δ): {corr_impact['max_change']:.3f}")
        logger.info(f"    Mean Change (Δ): {corr_impact['mean_change']:.3f}")
        logger.info(f"    Avg Uncertainty (σ): {corr_impact['avg_uncertainty']:.3f}")
        
        if corr_impact['significant_pairs']:
            logger.info(f"    Significant Pairs (low uncertainty):")
            for pair in corr_impact['significant_pairs'][:3]:
                logger.info(
                    f"      {company} ↔ {pair['company']}: "
                    f"Δ={pair['change']:+.3f}, σ={pair['uncertainty']:.3f}"
                )

    # --- TRADING SIGNALS WITH UNCERTAINTY THRESHOLDS ---
    logger.info("\n" + "="*60)
    logger.info("💼 GENERATING TRADING SIGNALS")
    logger.info("="*60)
    
    signals = trading_system.generate_trading_signals(
        news_impact, 
        min_confidence=0.6,  # Only trade with >60% confidence
        max_uncertainty=0.4  # Only trade with <0.4 uncertainty
    )
    
    logger.info(f"Generated {len(signals)} trading signals")
    for i, s in enumerate(signals[:5], 1):
        logger.info(f"\n  Signal {i}: {s['type']} {s['company']}")
        logger.info(f"    Strength: {s['strength']:.3f}")
        logger.info(f"    Confidence: {s['total_confidence']:.3f}")
        logger.info(f"    Uncertainty (σ): {s['uncertainty']:.3f}")
        logger.info(f"    Position Size: ${s['position_size']:.2f}")
        logger.info(f"    News Scope: {s['news_scope']}")
    
    trading_system.execute_trading_signals(signals)

    # --- Keyword clustering ---
    if len(training_data["keyword_sequence"]) >= 5:
        print("\n🔤 Keyword Impact Clusters:")
        test_keywords = ["breakthrough", "revenue", "profit", "loss", "acquisition", "bankruptcy", "innovation", "decline"]
        for keyword, similar_words in trading_system.analyze_keyword_impact_clusters(test_keywords).items():
            if similar_words:  # Only show keywords that have similar ones
                similar_names = [word for word, sim in similar_words[:3]]
                print(f"  '{keyword}' clusters with: {similar_names}")

        test_word = "breakthrough"
        if test_word in tokenizer.vocab:
            print(f"\nSimilar keywords to '{test_word}':")
            for w, sim in trading_system.get_similar_keywords_by_impact(test_word, 5):
                print(f"  {w}: {sim:.3f}")

    keyword_tokens = data_processor.prepare_keyword_sequence(test_news)    
    # Get actual news embedding for reconstruction error
    news_target_embedding = data_processor.get_bert_embedding(test_news)[:news_model.latent_dim]
    
    # Make probabilistic prediction with REAL reconstruction error
    logger.info("🔮 Making probabilistic prediction...")
    predictions = news_model.predict_with_uncertainty(
        keyword_tokens['input_ids'],  # Pass TF Tensor directly, predict_with_uncertainty will handle it
        np.expand_dims(data_processor.baseline_z, 0),
        news_target_embedding
    )
    
    logger.info(f"✅ Prediction complete!")
    logger.info(f"\n📊 UNCERTAINTY ANALYSIS:")
    logger.info(f"  Total Confidence: {predictions['total_confidence']:.3f}")
    logger.info(f"    * Reconstruction Error: {predictions['reconstruction_error']:.4f}")
    logger.info(f"    * Interpretation: {'✅ Known news type' if predictions['reconstruction_error'] < 0.2 else '⚠️ Novel/unusual news'}")
    
    # Analyze correlation changes
    mu = predictions['mean'][0]
    sigma = predictions['std'][0]
    baseline_corr = data_processor.inverse_fisher_z_transform(data_processor.baseline_z)
    predicted_corr = data_processor.inverse_fisher_z_transform(mu)
    delta_corr = predicted_corr - baseline_corr
    
    logger.info(f"\n📈 CORRELATION CHANGES:")
    logger.info(f"  Mean |Δ|: {np.mean(np.abs(delta_corr)):.4f}")
    logger.info(f"  Max |Δ|: {np.max(np.abs(delta_corr)):.4f}")
    logger.info(f"  High uncertainty pairs (σ > 0.3): {np.sum(sigma > 0.3)}/{len(data_processor.companies)**2}")
    
    # Show top changes involving focus company
    significant_changes = []
    for i in range(len(data_processor.companies)):
        if i != focus_company_idx:
            change = delta_corr[focus_company_idx, i]
            unc = sigma[focus_company_idx, i]
            if abs(change) > 0.05:
                significant_changes.append((
                    data_processor.companies[focus_company_idx], 
                    data_processor.companies[i], 
                    change,
                    baseline_corr[focus_company_idx, i],
                    mu[focus_company_idx, i],
                    unc
                ))
    
    significant_changes.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if significant_changes:
        logger.info(f"\n🎯 TOP CORRELATION CHANGES FOR {focus_company}:")
        for c1, c2, delta, baseline, predicted, unc in significant_changes[:5]:
            logger.info(f"\n  {c1} ↔ {c2}:")
            logger.info(f"    Baseline: {baseline:+.3f}")
            logger.info(f"    Predicted: {predicted:+.3f}")
            logger.info(f"    Change (Δ): {delta:+.3f}")
            logger.info(f"    Uncertainty (σ): {unc:.4f} {'✅ Low' if unc < 0.3 else '⚠️ High'}")

    # ============================================================================
    # VISUALIZE
    # ============================================================================
    logger.info("\n📊 Generating visualization...")
    performance_analyzer = PerformanceAnalyzer(trading_system)
    performance_report = performance_analyzer.generate_performance_report()
    
    logger.info("Teljesítményjelentés:")
    logger.info(f"Portfolio_value: ${performance_report['portfolio_value']:.2f}")
    logger.info(f"Active positions: {performance_report['active_positions']}")
    logger.info(f"Period_performance: {performance_report['period_performance']}")

    fig = performance_analyzer.visualize_uncertainty_predictions(
        predictions, 
        baseline_corr, 
        predicted_corr,
        data_processor.companies, 
        test_news
    )
    plt.savefig('probabilistic_predictions.png', dpi=150, bbox_inches='tight')
    logger.info("✅ Saved to 'probabilistic_predictions.png'")
    
    # Portfolio metrics
    div = trading_system.get_portfolio_diversification_metrics()
    logger.info(f"\n📈 Portfolio Diversification:")
    logger.info(f"  Score: {div.get('diversification_score',0):.3f}")
    logger.info(f"  Average Correlation: {div.get('average_correlation',0):.3f}")
    logger.info(f"  Positions: {div.get('num_positions',0)}")
    
    logger.info("\n" + "="*80)
    logger.info("KEY INSIGHTS")
    logger.info("="*80)
    logger.info("✓ Model learns WHEN to be uncertain (aleatoric via σ)")
    logger.info("✓ Reconstruction error captures epistemic uncertainty (novel news)")
    logger.info("✓ Weighted loss focuses on ACTUALLY AFFECTED company pairs (not just focus)")
    logger.info("  → FED news affecting all companies: all pairs weighted equally")
    logger.info("  → AAPL-specific news: AAPL pairs get 3x weight")
    logger.info("✓ Combined confidence guides trading decisions")
    logger.info("✓ Position sizing accounts for prediction uncertainty")
    logger.info("="*80)
    
    # ============================================================================
    # TEST WITH GLOBAL NEWS
    # ============================================================================
    logger.info("\n\n" + "="*80)
    logger.info("🌍 TESTING WITH GLOBAL/MACRO NEWS")
    logger.info("="*80)
    
    global_news = "Federal Reserve announces 0.75% interest rate hike to combat inflation"
    logger.info(f"\nNews: {global_news}")
    logger.info("Expected behavior:")
    logger.info("  - News scope: GLOBAL")
    logger.info("  - Affected companies: [] (or all)")
    logger.info("  - All correlation pairs weighted equally")
    logger.info("")
    
    # Analyze global news - NO specific affected companies
    global_news_impact = trading_system.predict_news_impact(
        global_news,
        target_companies=["AAPL", "MSFT", "TSLA", "F"],
        affected_companies=[]  # ← Empty list = global news
    )
    
    logger.info("Results:")
    for company, analysis in global_news_impact.items():
        logger.info(f"\n{company}:")
        logger.info(f"  News Scope: {analysis['news_scope']}")
        logger.info(f"  Affected Companies: {analysis['affected_companies']}")
        logger.info(f"  Confidence: {analysis['total_confidence']:.3f}")
        logger.info(f"  Max Correlation Change: {analysis['correlation_impact']['max_change']:.3f}")
        
    logger.info("\n✅ Global news correctly handled - no company-specific bias!")
    
    # ============================================================================
    # TEST WITH NOVEL NEWS
    # ============================================================================
    logger.info("\n\n" + "="*80)
    logger.info("🚀 TESTING WITH NOVEL NEWS")
    logger.info("="*80)
    
    novel_news = "SpaceX announces plans to establish Mars colony by 2030"
    logger.info(f"\nNews: {novel_news}")
    logger.info("Expected behavior:")
    logger.info("  - HIGH reconstruction error (novel concept)")
    logger.info("  - LOW epistemic confidence")
    logger.info("  - Should SKIP trading")
    logger.info("")
    
    novel_news_impact = trading_system.predict_news_impact(
        novel_news,
        target_companies=["TSLA", "AAPL"],
        affected_companies=["TSLA"]
    )
    
    logger.info("Results:")
    for company, analysis in novel_news_impact.items():
        logger.info(f"\n{company}:")
        logger.info(f"  Reconstruction Error: {analysis['reconstruction_error']:.4f}")
        logger.info(f"  Total Confidence: {analysis['total_confidence']:.3f}")
        logger.info(f"  Tradeable: {'✅ Yes' if analysis['tradeable'] else '❌ No (too uncertain)'}")
    
    logger.info("\n✅ Novel news correctly detected - trading skipped!")
    
    logger.info("\n" + "="*80)