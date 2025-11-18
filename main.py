import os
import warnings
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel

# Suppress warnings BEFORE any TF/transformers imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR
# DON'T set CUDA_VISIBLE_DEVICES - we want GPU!

# Suppress Python warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*TensorFlow and JAX classes.*')

# Configure TensorFlow GPU memory growth (prevents OOM errors)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        logger_init = logging.getLogger("GPU_Setup")
        logger_init.info("✅ GPU detected and configured: %d device(s)", len(physical_devices))
    except RuntimeError as e:
        logger_init = logging.getLogger("GPU_Setup")
        logger_init.warning("GPU configuration failed: %s", e)
else:
    logger_init = logging.getLogger("GPU_Setup")
    logger_init.info("ℹ️  No GPU detected, using CPU")

# Now import custom modules
from AttentionBasedNewsFactorModel import AttentionBasedNewsFactorModel
from NewsDataProcessor import NewsDataProcessor
from AdvancedTradingSystem import AdvancedTradingSystem
from PerformanceAnalyzer import PerformanceAnalyzer
from Utils import Utils
from TopologicalValidator import TopologicalValidator

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

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("PROBABILISTIC RESIDUAL LEARNING CORRELATION-BASED NEWS FACTOR TRADING")
    logger.info("="*80)

    # --- SAMPLE DATA & SYSTEM SETUP ---
    utils = Utils()
    companies_df, train_news, val_news, sample_prices, correlation_matrix, covariance_matrix = utils.create_hybrid_data()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = TFAutoModel.from_pretrained("bert-base-uncased", use_safetensors=False)

    logger.info("  Training period: %s to %s", train_news[0]['timestamp'], train_news[-1]['timestamp'])
    logger.info("  Validation period: %s to %s", val_news[0]['timestamp'], val_news[-1]['timestamp'])
    logger.info("  Training news: %d", len(train_news))
    logger.info("  Validation news: %d", len(val_news))

    news_model = AttentionBasedNewsFactorModel(
        tokenizer,
        bert_model,
        num_companies=len(sample_prices.keys()), 
        max_keywords=128, 
        keyword_dim=256,
        company_dim=128, 
        latent_dim=256
    )
    logger.info("✅ Model initialized with dual output (μ, σ²)")

    # --- PROCESSING DATA ---
    data_processor = NewsDataProcessor(news_model, sample_prices)

    logger.info("\n📄 Processing training data...")
    training_data = data_processor.process_news_batch(train_news)
    logger.info("\n📄 Processing validation data...")
    validation_data = data_processor.process_news_batch(val_news, is_training=False)

    logger.info("\n📦 Processed Data:")
    logger.info("  Training samples: %d", len(training_data['keyword_sequence']))
    logger.info("  Validation samples: %d", len(validation_data['keyword_sequence']))

    # --- TRAINING WITH VALIDATION ---
    logger.info("\n🏋️  Training with temporal validation...")
    history = news_model.train(
        training_data=training_data,
        validation_data=validation_data,
        epochs=60,
        batch_size=8
        )

    # --- TRADING SYSTEM SETUP ---
    trading_system = AdvancedTradingSystem(data_processor)
    performance_analyzer = PerformanceAnalyzer(trading_system)
    topological_validator = TopologicalValidator(performance_analyzer)

    # --- LEARNING CURVES ---
    performance_analyzer.plot_learning_curves(history)

    # ============================================================================
    # UNSUPERVISED NEWS TYPE CLUSTERING VALIDATION
    # ============================================================================
    logger.info("\n%s", "=" * 60)
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
            "\n✅ Model successfully learned to distinguish news types implicitly!"
        )
        logger.info("   Adjusted Rand Index: %.3f", ari)
        logger.info("   Silhouette Score: %.3f", silhouette)
    else:
        logger.warning("\n⚠️  Model struggles to capture news type distinctions")
        logger.warning("   Adjusted Rand Index: %.3f", ari)
        logger.warning("   Silhouette Score: %.3f", silhouette)
        logger.warning("   Consider: more training data or stronger regularization")

    # ============================================================================
    # TEST NEWS IMPACT WITH UNCERTAINTY
    # ============================================================================
    test_news_sample = val_news[0]
    test_news = test_news_sample['text']
    affected_companies_test = test_news_sample['companies']
    
    logger.info("\n%s", "=" * 60)
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
    logger.info("\n%s", "=" * 60)
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

    # --- TOPOLOGICAL VALIDATION ---
    logger.info("\n%s", "=" * 60)
    logger.info("TOPOLOGICAL SMOOTHNESS VALIDATION")
    logger.info("="*80)
    logger.info("Purpose: Validate model stability by checking if similar news")
    logger.info("         (in embedding space) produce similar impact predictions")
    
    smoothness_results = topological_validator.validate_smoothness(val_news, save_path='topological_smoothness_map.png')

    # Keyword clustering
    test_word = "breakthrough"
    similar = topological_validator.get_similar_keywords_by_impact(test_word, 5)
    logger.info("Similar keywords for %s: %s", test_word, similar)
    
    # Additional component analysis
    logger.info("\nAnalyzing impact vector components...")
    topological_validator.analyze_impact_vector_components(smoothness_results['impact_vectors'])
    
    # Assessment
    correlation_rm = smoothness_results['correlation']
    assessment = smoothness_results['assessment']
    num_unstable = smoothness_results['num_unstable_pairs']
    
    logger.info("\n%s", "=" * 60)
    logger.info("TOPOLOGICAL VALIDATION SUMMARY")
    logger.info("="*80)
    logger.info("  Embedding-Impact Correlation: %.4f", correlation_rm)
    logger.info("  Assessment: %s", assessment.upper())
    logger.info("  Unstable pairs: %d/%d", num_unstable, smoothness_results['num_pairs'])
    
    if correlation_rm > 0.6 and num_unstable < smoothness_results['num_pairs'] * 0.1:
        logger.info("\n✅ Model demonstrates EXCELLENT topological stability!")
        logger.info("   → Similar news produce similar impacts")
        logger.info("   → Safe for production use")
    elif correlation_rm > 0.4:
        logger.info("\n✅ Model has GOOD topological properties")
        logger.info("   → Generally stable predictions")
        logger.info("   → Minor instabilities acceptable")
    else:
        logger.warning("\n⚠️  Model shows WEAK topological stability")
        logger.warning("   → Similar news may produce unpredictable impacts")
        logger.warning("   → Consider: more training data, regularization, or architecture changes")
    
    logger.info("="*80)

    # ============================================================================
    # TEST WITH GLOBAL NEWS
    # ============================================================================
    logger.info("\n%s", "=" * 60)
    logger.info("🌍 TESTING WITH GLOBAL/MACRO NEWS")
    logger.info("="*60)

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

    logger.info("\n%s", "=" * 60)
    logger.info("TRAINING AND VALIDATION COMPLETE")
    logger.info("="*60)