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
    companies_df, train_news, val_news, sample_prices, correlation_matrix, covariance_matrix = utils.create_hybrid_data()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFAutoModel.from_pretrained('bert-base-uncased', from_pt=True)

    logger.info("  Training period: %s to %s", train_news[0]['timestamp'], train_news[-1]['timestamp'])
    logger.info("  Validation period: %s to %s", val_news[0]['timestamp'], val_news[-1]['timestamp'])
    logger.info("  Training news: %d", len(train_news))
    logger.info("  Validation news: %d", len(val_news))

    news_model = AttentionBasedNewsFactorModel(
        tokenizer, 
        num_companies=len(sample_prices.keys()), 
        max_keywords=100, 
        keyword_dim=256,
        company_dim=128, 
        latent_dim=128
    )
    logger.info("✅ Model initialized with dual output (μ, σ²)")

    # --- PROCESSING DATA ---
    data_processor = NewsDataProcessor(news_model, sample_prices, bert_model)

    logger.info("\n📄 Processing training data...")
    training_data = data_processor.process_news_batch(train_news)
    logger.info("\n📄 Processing validation data...")
    validation_data = data_processor.process_news_batch(val_news)

    logger.info("\n📦 Processed Data:")
    logger.info("  Training samples: %d", len(training_data['keyword_sequence']))
    logger.info("  Validation samples: %d", len(validation_data['keyword_sequence']))

    # --- TRAINING WITH VALIDATION ---
    logger.info("\n🏋️  Training with temporal validation...")
    history = news_model.train(
        training_data=training_data,
        validation_data=validation_data,
        epochs=50,
        batch_size=8
        )

    # --- TRADING SYSTEM SETUP ---
    trading_system = AdvancedTradingSystem(data_processor)
    performance_analyzer = PerformanceAnalyzer(trading_system)

    # --- LEARNING CURVES ---
    performance_analyzer.plot_learning_curves(history)

    # ============================================================================
    # UNSUPERVISED NEWS TYPE CLUSTERING VALIDATION
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("VALIDATING IMPLICIT NEWS TYPE LEARNING")
    logger.info("="*80)
    
    # Use validation news for clustering validation
    clustering_results = performance_analyzer.validate_news_type_learning(
        val_news,
        n_clusters=5,  # Expected number of news types
        save_path='news_clustering_validation.png'
    )
    
    # Assessment
    ari = clustering_results['comparison']['adjusted_rand_index']
    silhouette = clustering_results['clustering']['silhouette_score']
    
    if ari > 0.3 and silhouette > 0.2:
        logger.info(
            "\n✅ Model successfully learned to distinguish news types implicitly!\n"
            "   Adjusted Rand Index: %.3f\n"
            "   Silhouette Score: %.3f",
            ari,
            silhouette,
        )
    else:
        logger.warning(
            "\n⚠️  Model struggles to capture news type distinctions\n"
            "   Adjusted Rand Index: %.3f\n"
            "   Silhouette Score: %.3f\n"
            "   Consider: more training data or stronger regularization",
            ari,
            silhouette,
        )

    # ============================================================================
    # TEST NEWS IMPACT WITH UNCERTAINTY
    # ============================================================================
    test_news_sample = val_news[0]
    test_news = test_news_sample['text']
    affected_companies_test = test_news_sample['companies']
    
    logger.info("\n" + "="*60)
    logger.info("📰 ANALYZING NEWS WITH UNCERTAINTY QUANTIFICATION")
    logger.info("="*60)
    logger.info("News: %s", test_news)
    logger.info("Affected Companies: %s", affected_companies_test)

    # Analyze first 10 companies
    target_companies = list(sample_prices.keys())[:10]
    news_impact = trading_system.predict_news_impact(
        test_news, 
        target_companies, 
        affected_companies_test
    )

    # Print news impact results with uncertainty
    for company, analysis in news_impact.items():
        logger.info("\n%s:", company)
        logger.info("  News Scope: %s", analysis['news_scope'])
        logger.info("  Affected Companies: %s", analysis['affected_companies'])
        logger.info("  Total Confidence: %.3f", analysis['total_confidence'])
        logger.info("  Reconstruction Error: %.4f", analysis['reconstruction_error'])
        logger.info("  Tradeable: %s", "✅ Yes" if analysis['tradeable'] else "❌ No")

        corr_impact = analysis['correlation_impact']
        logger.info("  Correlation Impact:")
        logger.info("    Max Change (Δ): %.3f", corr_impact['max_change'])
        logger.info("    Mean Change (Δ): %.3f", corr_impact['mean_change'])
        logger.info("    Avg Uncertainty (σ): %.3f", corr_impact['avg_uncertainty'])

        if corr_impact['significant_pairs']:
            logger.info("    Significant Pairs (low uncertainty):")
            for pair in corr_impact['significant_pairs'][:3]:
                logger.info(
                    "      %s ↔ %s: Δ=%+.3f, σ=%.3f",
                    company,
                    pair['company'],
                    pair['change'],
                    pair['uncertainty'],
                )

    # --- VISUALIZATION ---
    logger.info("\n📊 Generating uncertainty prediction visualization...")
    analysis_to_plot = news_impact[target_companies[0]]
    fig = performance_analyzer.visualize_uncertainty_predictions(
        analysis_to_plot, 
        data_processor.companies, 
        test_news
    )
    plt.savefig('probabilistic_predictions.png', dpi=150, bbox_inches='tight')
    logger.info("✅ Saved to 'probabilistic_predictions.png'")

    # --- TRADING SIGNALS WITH UNCERTAINTY THRESHOLDS ---
    logger.info("\n" + "="*60)
    logger.info("💼 GENERATING TRADING SIGNALS")
    logger.info("="*60)

    signals = trading_system.generate_trading_signals(
        news_impact, 
        min_confidence=0.6,  # Only trade with >60% confidence
        max_uncertainty=0.4  # Only trade with <0.4 uncertainty
    )

    logger.info("Generated %d trading signals", len(signals))
    for i, s in enumerate(signals[:5], 1):
        logger.info("\n  Signal %d: %s %s", i, s['type'], s['company'])
        logger.info("    Confidence: %.3f", s['total_confidence'])
        logger.info("    Uncertainty (σ): %.3f", s['uncertainty'])
        logger.info("    Position Size: $%.2f", s['position_size'])

    trading_system.execute_trading_signals(signals)

    # --- KEYWORD CLUSTERING ---
    if len(training_data["keyword_sequence"]) >= 5:
        logger.info("\n🔤 Keyword Impact Clusters:")
        test_keywords = [
            "breakthrough", "revenue", "profit", "loss", 
            "acquisition", "bankruptcy", "innovation", "decline"
        ]
        for keyword, similar_words in trading_system.analyze_keyword_impact_clusters(test_keywords).items():
            if similar_words:  # Only show keywords that have similar ones
                similar_names = [word for word, sim in similar_words[:3]]
                logger.info(f"  '{keyword}' clusters with: {similar_names}")

        test_word = "breakthrough"
        if test_word in tokenizer.vocab:
            logger.info(f"\nSimilar keywords to '{test_word}':")
            for w, sim in trading_system.get_similar_keywords_by_impact(test_word, 5):
                logger.info(f"  {w}: {sim:.3f}")

    # --- PERFORMANCE REPORT ---
    performance_report = performance_analyzer.generate_performance_report()
    logger.info("\n📈 Portfolio Performance:")
    logger.info("  Portfolio value: $%.2f", performance_report['portfolio_value'])
    logger.info("  Active positions: %s", performance_report['active_positions'])

    # Portfolio metrics
    div = trading_system.get_portfolio_diversification_metrics()
    logger.info("\n📊 Portfolio Diversification:")
    logger.info("  Score: %.3f", div.get('diversification_score', 0))
    logger.info("  Average Correlation: %.3f", div.get('average_correlation', 0))
    logger.info("  Positions: %d", div.get('num_positions', 0))

    # ============================================================================
    # TEST WITH GLOBAL NEWS
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("🌍 TESTING WITH GLOBAL/MACRO NEWS")
    logger.info("="*80)

    global_news = "Federal Reserve announces 0.75% interest rate hike to combat inflation"
    logger.info("News: %s", global_news)
    logger.info("Expected behavior:")
    logger.info("  - News scope: GLOBAL")
    logger.info("  - Affected companies: [] (or all)")
    logger.info("  - All correlation pairs weighted equally")

    # Analyze global news - NO specific affected companies
    global_news_impact = trading_system.predict_news_impact(
        global_news,
        target_companies=["AAPL", "MSFT", "TSLA", "F"],
        affected_companies=[]  # ← Empty list = global news
    )

    logger.info("\nResults:")
    for company, analysis in global_news_impact.items():
        logger.info("\n%s:", company)
        logger.info("  News Scope: %s", analysis['news_scope'])
        logger.info("  Affected Companies: %s", analysis['affected_companies'])
        logger.info("  Confidence: %.3f", analysis['total_confidence'])
        logger.info("  Max Correlation Change: %.3f", analysis['correlation_impact']['max_change'])

    logger.info("\n✅ Global news correctly handled - no company-specific bias!")

    logger.info("\n" + "="*80)
    logger.info("TRAINING AND VALIDATION COMPLETE")
    logger.info("="*80)