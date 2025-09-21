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
    """Creating enhanced sample data to test the system with more companies and immersive news"""

    sample_companies = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'F', 'GM', 'JPM', 'NFLX', 'NIKE', 'HD', 'MCD', 'SBUX', 'DIS', 'ADBE', 'CRM', 'ORCL', 'IBM', 'INTC', 'AMD'],
        'name': ['Apple Inc.', 'Microsoft Corporation', 'Alphabet Inc.', 'Amazon.com Inc.', 'Tesla Inc.', 'NVIDIA Corporation', 'Meta Platforms Inc.', 'Ford Motor Company', 'General Motors Company', 'JPMorgan Chase & Co.', 'Netflix Inc.','Nike Inc.', 'The Home Depot Inc.', 'McDonald\'s Corporation', 'Starbucks Corporation', 'The Walt Disney Company', 'Adobe Inc.', 'Salesforce Inc.', 'Oracle Corporation', 'International Business Machines', 'Intel Corporation', 'Advanced Micro Devices'],
        'sector': ['Technology', 'Technology', 'Technology', 'Consumer Discretionary', 'Consumer Discretionary', 'Technology', 'Technology', 'Consumer Discretionary','Consumer Discretionary', 'Financials', 'Communication Services', 'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary', 'Communication Services', 'Technology', 'Technology','Technology', 'Technology', 'Technology', 'Technology']
    })
    
    sample_news = [
        {
            'text': '''Apple Inc. delivered a stunning performance in its latest quarterly earnings report, with the tech giant announcing record-breaking results that sent shockwaves through Wall Street and beyond. The Cupertino-based company reported iPhone sales that not only exceeded all analyst expectations but also demonstrated remarkable resilience in an increasingly challenging global economic environment.
            CEO Tim Cook, speaking during the earnings call, highlighted the extraordinary growth in Apple's services division, which has become a cornerstone of the company's revenue strategy. "We're seeing unprecedented demand across our entire ecosystem," Cook stated, emphasizing how the integration between hardware and services continues to drive customer loyalty and engagement.
            The services segment, which includes the App Store, iCloud, Apple Music, and the rapidly expanding Apple TV+ streaming service, posted a remarkable 28% year-over-year growth. This performance particularly impressed analysts who had been concerned about potential saturation in the services market. The company's commitment to privacy and seamless user experience appears to be paying dividends as consumers increasingly value these features in their digital lives.
            Perhaps most notably, Apple's international markets showed extraordinary strength, with particular growth in emerging economies where the company has been strategically investing in retail presence and localized services. The results have prompted several major investment firms to raise their price targets for Apple stock, with some analysts now projecting the company could reach new all-time highs within the coming quarters.''',
            'companies': ['AAPL'],
            'timestamp': time.time() - 86400  # 1 day ago
        },
        {
            'text': '''Microsoft Corporation's Azure cloud computing platform has achieved a milestone that industry experts are calling a "game-changing moment" for the enterprise technology sector. The platform reported an astounding 35% year-over-year revenue increase, significantly surpassing even the most optimistic Wall Street forecasts and cementing Microsoft's position as a formidable competitor to Amazon Web Services in the cloud infrastructure space.
            The growth has been primarily driven by an unprecedented surge in demand for artificial intelligence-powered services, with enterprises across various industries rushing to integrate AI capabilities into their operations. Microsoft's strategic partnership with OpenAI has proven to be particularly lucrative, with the integration of GPT models into Azure services attracting major corporate clients who are eager to leverage cutting-edge AI technology.
            Satya Nadella, Microsoft's CEO, expressed his enthusiasm during a recent investor meeting: "We're witnessing a fundamental transformation in how businesses operate, and Azure is at the forefront of this revolution. Our comprehensive suite of AI tools, combined with our robust cloud infrastructure, is enabling organizations to achieve levels of efficiency and innovation that were previously unimaginable."
            The hybrid cloud adoption rates have also exceeded projections, with many enterprises choosing Microsoft's solutions due to their seamless integration with existing on-premises systems. Industry analysts are now predicting that Microsoft could capture an even larger share of the global cloud market, potentially challenging Amazon's long-standing dominance in the sector. The company's focus on security, compliance, and industry-specific solutions has particularly resonated with large enterprise customers who prioritize reliability and regulatory compliance.''',
            'companies': ['MSFT'],
            'timestamp': time.time() - 3600  # 1 hour ago
        },
        {
            'text': '''Tesla Inc. has once again redefined expectations in the electric vehicle industry by announcing record-breaking quarterly delivery numbers that have left competitors scrambling to keep pace. The Austin-based EV pioneer delivered an unprecedented number of vehicles across all model lines, with the Model Y leading the charge as one of the best-selling vehicles globally, not just among electric cars.
            The achievement becomes even more remarkable when considered against the backdrop of ongoing global supply chain challenges that have plagued the automotive industry for over two years. Tesla's vertical integration strategy, including its ownership of battery production facilities and semiconductor manufacturing capabilities, has proven to be a crucial competitive advantage during these turbulent times.
            Elon Musk, Tesla's CEO, took to social media to celebrate the milestone, emphasizing how the company's innovative manufacturing processes and continuous improvement philosophy have enabled this success. "This isn't just about building cars," Musk stated, "we're revolutionizing transportation while simultaneously advancing sustainable energy solutions for the entire planet."
            The delivery numbers have particularly impressed analysts due to strong performance across diverse geographic markets. Tesla's expansion into emerging markets, coupled with its ability to maintain premium pricing while scaling production, demonstrates the brand's exceptional global appeal. The company's Supercharger network expansion has also played a crucial role, alleviating range anxiety concerns that previously hindered EV adoption.
            Wall Street responded enthusiastically to the news, with Tesla's stock price surging in after-hours trading as multiple analysts raised their price targets. Several investment firms are now projecting that Tesla could maintain its market leadership position even as traditional automakers accelerate their own electric vehicle programs.''',
            'companies': ['TSLA'],
            'timestamp': time.time() - 7200  # 2 hours ago
        },
        {
            'text': '''NVIDIA Corporation has unveiled what industry experts are calling the most significant advancement in artificial intelligence semiconductor technology in over a decade. The company's latest breakthrough, developed in collaboration with leading research institutions and technology partners, promises to revolutionize how AI systems are trained, deployed, and scaled across various applications.
            The new semiconductor architecture represents a quantum leap in processing efficiency, offering unprecedented performance improvements for both machine learning training workloads and real-time inference applications. Jensen Huang, NVIDIA's CEO, described the innovation as "the dawn of a new era in computing," emphasizing how these advances will accelerate AI adoption across industries ranging from healthcare and finance to autonomous vehicles and scientific research.
            What makes this announcement particularly significant is NVIDIA's simultaneous reveal of strategic partnerships with major cloud computing providers, including comprehensive integration plans that will make these cutting-edge capabilities accessible to organizations of all sizes. The partnerships span multiple continents and include agreements with both established tech giants and emerging cloud platforms, ensuring global availability of the new technology.
            The semiconductor breakthrough addresses several critical challenges that have historically limited AI deployment, including energy efficiency, processing speed, and cost-effectiveness. Early benchmarks suggest that the new chips can perform complex AI workloads up to 10 times faster than previous generations while consuming significantly less power, a combination that could dramatically reduce the operational costs of AI-driven applications.
            Industry analysts are already predicting that this development will further solidify NVIDIA's dominance in the AI hardware market, potentially creating new revenue streams worth billions of dollars. The announcement has also sparked increased investor interest in AI-focused companies, as the improved hardware capabilities are expected to enable previously impossible applications and business models.''',
            'companies': ['NVDA'],
            'timestamp': time.time() - 14400  # 4 hours ago
        },
        {
            'text': '''The global automotive industry finds itself grappling with an unprecedented supply chain crisis that continues to reshape production strategies and market dynamics across the sector. Ford Motor Company and General Motors Corporation, two of America's most storied automakers, have announced significant production delays that underscore the severity of ongoing semiconductor shortages and raw material constraints.
            The situation has become particularly acute in recent weeks, with both companies forced to implement temporary shutdowns at multiple manufacturing facilities across North America. Ford's Executive Chairman Bill Ford Jr. described the challenges as "unlike anything we've experienced in our company's 120-year history," while GM's leadership has acknowledged that the disruptions may persist well into the next fiscal quarter.
            The semiconductor shortage, which initially emerged during the pandemic as a temporary disruption, has evolved into a structural challenge that is fundamentally altering how automakers approach supply chain management. Both Ford and GM are now accelerating their efforts to establish more resilient supplier networks, including investments in domestic semiconductor manufacturing capabilities and strategic partnerships with technology companies.
            Beyond semiconductors, the companies are also contending with shortages of critical raw materials including lithium, nickel, and rare earth elements essential for electric vehicle battery production. This has created a complex web of interdependent supply challenges that require sophisticated coordination and long-term strategic planning.
            Industry experts warn that these disruptions could have far-reaching implications for vehicle pricing, product availability, and the pace of electric vehicle adoption. The situation has also highlighted the strategic importance of supply chain resilience, prompting both companies to reconsider their global sourcing strategies and invest heavily in supply chain visibility technologies. Wall Street analysts are closely monitoring how these challenges will impact profitability and market share as the automotive industry continues its historic transition toward electrification.''',
            'companies': ['F', 'GM'],
            'timestamp': time.time() - 21600  # 6 hours ago
        },
        {
            'text': '''Netflix Inc. has delivered a performance that has reinvigorated confidence in the streaming entertainment sector, reporting subscriber growth numbers that significantly exceeded analyst expectations and demonstrated the company's continued ability to compete effectively in an increasingly crowded marketplace. The Los Gatos-based streaming giant added millions of new subscribers globally, with particularly strong performance in international markets that have become crucial to the company's long-term growth strategy.
            The growth surge has been fueled by an exceptional slate of original content that has resonated with diverse global audiences. Netflix's investment in localized content production has proven particularly successful, with several non-English language series achieving worldwide popularity and critical acclaim. This strategy of creating culturally authentic content for specific regions while maintaining global appeal has become a key differentiator in the competitive streaming landscape.
            Co-CEO Reed Hastings emphasized during the earnings call how the company's data-driven approach to content creation continues to yield impressive results. "We're not just creating entertainment; we're building cultural bridges and telling stories that matter to communities around the world," Hastings explained, highlighting how Netflix's global perspective sets it apart from competitors who may focus primarily on domestic markets.
            The company's technological innovations have also contributed to its success, with improvements in streaming quality, personalized recommendations, and user interface design enhancing the overall subscriber experience. Netflix's investment in advanced analytics and machine learning capabilities has enabled more precise content targeting and improved customer retention rates.
            Perhaps most significantly, Netflix has demonstrated its ability to maintain pricing power even as competition intensifies, with recent price adjustments being well-received by subscribers who recognize the value of the platform's extensive content library and original programming. This pricing resilience, combined with robust subscriber growth, has prompted several Wall Street analysts to raise their long-term outlook for the company, viewing Netflix as well-positioned to navigate the evolving entertainment landscape.''',
            'companies': ['NFLX'],
            'timestamp': time.time() - 10800  # 3 hours ago
        },
        {
            'text': '''Nike Inc. has unveiled a revolutionary approach to athletic footwear and apparel that combines cutting-edge sustainable materials with breakthrough performance technologies, setting new industry standards for both environmental responsibility and athletic excellence. The Beaverton-based sportswear giant announced its most ambitious sustainability initiative to date, promising to transform how athletic products are designed, manufactured, and recycled.
            The centerpiece of Nike's announcement is a comprehensive circular economy program that reimagines the entire product lifecycle, from initial design through end-of-life recycling. The company has developed proprietary materials that deliver superior performance characteristics while being fully recyclable, addressing long-standing concerns about waste in the athletic apparel industry.
            CEO John Donahoe emphasized Nike's commitment to innovation during a press conference at the company's headquarters: "We're not just changing how we make products; we're revolutionizing what it means to be a responsible global brand. This initiative represents the largest investment in sustainable technology in Nike's history, and we believe it will define the future of athletic performance."
            The program includes partnerships with leading material science companies and research institutions to develop next-generation fabrics and manufacturing processes. These collaborations have already yielded promising results, including new materials that offer enhanced breathability, durability, and moisture management while maintaining Nike's signature style and performance standards.
            Market analysts are viewing this announcement as a strategic masterstroke that positions Nike ahead of competitors in the rapidly growing sustainable products market. Consumer research indicates that younger demographics, which represent Nike's core growth opportunity, increasingly prioritize environmental sustainability when making purchasing decisions. The company's ability to deliver both performance and sustainability could provide a significant competitive advantage in key markets worldwide.''',
            'companies': ['NIKE'],
            'timestamp': time.time() - 18000  # 5 hours ago
        },
        {
            'text': '''The Home Depot Inc. has reported extraordinary quarterly results that reflect the company's successful adaptation to changing consumer behavior and market dynamics in the home improvement sector. The Atlanta-based retailer announced revenue figures that significantly exceeded analyst expectations, driven by robust performance across both professional contractor and do-it-yourself customer segments.
            The company's digital transformation initiatives have proven particularly successful, with online sales growing at an unprecedented rate while maintaining strong margins. Home Depot's investment in omnichannel capabilities, including buy-online-pickup-in-store services and enhanced mobile applications, has resonated strongly with customers who increasingly value convenience and flexibility in their shopping experience.
            Chairman and CEO Craig Menear highlighted the company's strategic focus during the earnings call: "We've successfully evolved our business model to meet customers wherever they are, whether that's in our stores, online, or through our extensive network of professional services. This comprehensive approach has enabled us to capture market share while building deeper customer relationships."
            The professional contractor business has been a particular bright spot, with specialized services and bulk purchasing programs driving significant revenue growth. Home Depot's Pro Xtra loyalty program has attracted hundreds of thousands of professional customers who value the company's extensive inventory, competitive pricing, and reliable supply chain capabilities.
            The company's supply chain resilience has also been a key competitive advantage during recent global disruptions, with Home Depot's sophisticated inventory management systems and supplier relationships enabling consistent product availability when competitors have struggled with stock shortages. This operational excellence has translated into market share gains and improved customer satisfaction scores across all major product categories.''',
            'companies': ['HD'],
            'timestamp': time.time() - 25200  # 7 hours ago
        },
        {
            'text': '''Adobe Inc. has announced a groundbreaking expansion of its Creative Cloud platform that integrates advanced artificial intelligence capabilities across its entire suite of professional creative applications. The San Jose-based software company revealed that its latest AI-powered features will revolutionize how creative professionals approach design, video editing, and digital content creation.
            The new AI integration, built on Adobe's proprietary Sensei platform, offers unprecedented automation capabilities while maintaining the precision and creative control that professional users demand. Features include intelligent background removal, automated color grading, advanced object recognition, and predictive design suggestions that can significantly accelerate creative workflows.
            CEO Shantanu Narayen emphasized the transformative potential of these innovations during Adobe's annual MAX conference: "We're not replacing human creativity; we're amplifying it. These AI tools enable our users to focus on what they do best - creating amazing content - while our technology handles the repetitive and time-consuming tasks that previously slowed down the creative process."
            The announcement has generated significant excitement within the creative community, with beta testing programs already showing dramatic improvements in productivity and project turnaround times. Professional designers, video editors, and digital marketers have reported that the new AI features can reduce project completion times by up to 60% while maintaining or improving output quality.
            Industry analysts view this development as potentially game-changing for Adobe's competitive position, particularly as the company faces increasing competition from both established software companies and emerging AI-driven startups. The integration of AI capabilities into Adobe's already dominant creative software suite could further solidify the company's market leadership while creating new revenue opportunities through premium feature tiers and expanded subscription offerings.''',
            'companies': ['ADBE'],
            'timestamp': time.time() - 28800  # 8 hours ago
        },
        {
            'text': '''Intel Corporation has unveiled its most ambitious semiconductor manufacturing roadmap in over a decade, announcing breakthrough technologies and massive capital investments that could reshape the global chip industry landscape. The Santa Clara-based processor giant revealed plans for next-generation manufacturing facilities and revolutionary chip architectures that promise to restore Intel's technology leadership position.
            The comprehensive roadmap includes the development of advanced 2-nanometer and 1.8-nanometer manufacturing processes, representing significant improvements in both performance and energy efficiency compared to current industry standards. CEO Pat Gelsinger described the announcement as "Intel's return to undisputed technology leadership," emphasizing how these advances will benefit everything from personal computers to data center infrastructure.
            Central to Intel's strategy is the expansion of its foundry services business, which will allow the company to manufacture chips for other technology companies while leveraging its advanced manufacturing capabilities. This approach directly challenges the dominance of Taiwan Semiconductor Manufacturing Company (TSMC) and could provide Intel with significant new revenue streams.
            The company has also announced strategic partnerships with major technology firms and government agencies, including substantial investments from the U.S. CHIPS Act that will support domestic semiconductor manufacturing. These collaborations underscore the strategic importance of semiconductor independence and supply chain resilience in an increasingly complex geopolitical environment.
            Perhaps most significantly, Intel's roadmap includes revolutionary packaging technologies that will enable the integration of multiple chip types into single, high-performance modules. This approach could unlock new levels of computing performance while addressing the physical limitations that have traditionally constrained chip design. Industry experts are already speculating that these innovations could trigger a new wave of technological advancement across multiple sectors, from artificial intelligence to quantum computing.''',
            'companies': ['INTC'],
            'timestamp': time.time() - 32400  # 9 hours ago
        },
        {
            'text': '''McDonald's Corporation has reported exceptional quarterly performance that demonstrates the global fast-food giant's remarkable ability to adapt to changing consumer preferences while maintaining operational excellence across its vast restaurant network. The Chicago-based company announced revenue and profit figures that significantly exceeded Wall Street expectations, driven by innovative menu offerings and strategic technology investments.
            The company's digital transformation initiatives have been particularly successful, with mobile ordering and delivery services now representing a substantial portion of total sales. McDonald's comprehensive digital ecosystem, including its mobile app, loyalty program, and partnership with leading delivery platforms, has created new customer engagement opportunities while improving operational efficiency.
            CEO Chris Kempczinski highlighted the company's strategic focus during the quarterly earnings call: "We've successfully modernized the McDonald's experience while staying true to our core values of quality, service, and value. Our customers can now interact with our brand in ways that were unimaginable just a few years ago, and this technological integration has driven both customer satisfaction and business performance."
            The company's menu innovation strategy has also contributed significantly to its success, with new product offerings designed to appeal to health-conscious consumers while maintaining the taste and affordability that McDonald's customers expect. Limited-time offerings and regional menu variations have helped drive customer traffic and increase average transaction values.
            International expansion remains a key growth driver, with McDonald's continuing to open new locations in emerging markets while adapting its menu and service model to local preferences and cultural norms. This balanced approach of global consistency and local adaptation has enabled McDonald's to maintain its position as the world's leading fast-food chain while capturing market share in diverse geographic regions.''',
            'companies': ['MCD'],
            'timestamp': time.time() - 36000  # 10 hours ago
        }
    ]
    
    # Generate more realistic correlated price data with extended company list
    sample_prices = {}
    base_correlation_matrix = {
        # Tech stocks tend to be correlated
        ('AAPL', 'MSFT'): 0.6, ('AAPL', 'GOOGL'): 0.5, ('MSFT', 'GOOGL'): 0.7,
        ('NVDA', 'AAPL'): 0.4, ('NVDA', 'MSFT'): 0.5, ('META', 'GOOGL'): 0.6,
        ('ADBE', 'MSFT'): 0.5, ('CRM', 'MSFT'): 0.4, ('ORCL', 'MSFT'): 0.3,
        ('IBM', 'ORCL'): 0.4, ('INTC', 'NVDA'): 0.3, ('AMD', 'NVDA'): 0.5,
        ('INTC', 'AMD'): 0.4,
        # Auto stocks are highly correlated
        ('F', 'GM'): 0.8, ('TSLA', 'F'): 0.3, ('TSLA', 'GM'): 0.3,
        # Consumer discretionary correlations
        ('NIKE', 'SBUX'): 0.4, ('HD', 'NIKE'): 0.3, ('MCD', 'SBUX'): 0.5,
        ('DIS', 'NFLX'): 0.4, ('AMZN', 'HD'): 0.3,
        # Some negative correlations
        ('JPM', 'TSLA'): -0.2, ('JPM', 'NVDA'): -0.1, ('JPM', 'NFLX'): -0.1
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
