import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from typing import Dict, List

logger = logging.getLogger("AdvancedNewsFactor.TopologicalValidator")

class TopologicalValidator:
    """
    Topological validation: measuring model stability and robustness
    
    Goal: Verify that the "smoothness" of the news representation space 
    (where textually similar news are close) corresponds to the "smoothness" 
    of the impact prediction space (where news causing similar correlation 
    changes and price deviations are close).
    
    If the model is stable:
    - Textually similar news → similar impact predictions
    - Small embedding distance → small impact distance
    
    If unstable:
    - Small textual difference → drastically different impact
    """
    
    def __init__(self, performance_analyzer):
        """
        Initialize the topological validator
        
        Args:
            performance_analyzer: PerformanceAnalyzer instance
            trading_system: AdvancedTradingSystem instance
        """
        self.performance_analyzer = performance_analyzer
        self.trading_system = performance_analyzer.trading_system
        self.data_processor = performance_analyzer.trading_system.data_processor
        self.news_factor_model = performance_analyzer.trading_system.data_processor.news_factor_model

        self.keyword_embedding_layer = self.news_factor_model.model.get_layer("keyword_embeddings")
        self.company_embedding_layer = self.news_factor_model.model.get_layer("company_embeddings")
        
        logger.info("TopologicalValidator initialized")
    
    def _compute_pairwise_distances(self,
                                   vectors: np.ndarray,
                                   metric: str,
                                   normalize: bool = False,
                                   space_name: str = "vector") -> np.ndarray:
        """
        Compute pairwise distances between vectors using scipy.pdist (vectorized)
        
        This unified method handles both embedding and impact distance computation
        following clean code principles (DRY - Don't Repeat Yourself).
        
        Args:
            vectors: [K, D] array where K=number of items, D=dimensionality
            metric: Distance metric ('cosine' for embeddings, 'euclidean' for impact)
            normalize: Whether to normalize vectors before distance computation
            space_name: Name of the space for logging (e.g., "embedding", "impact")
            
        Returns:
            distances: [K*(K-1)/2] condensed distance matrix
        """
        K = len(vectors)
        logger.info("Computing %s distances for %d items (vectorized)...", 
                   space_name, K)
        
        # Optional normalization (for impact vectors)
        if normalize:
            logger.info("Normalizing %s vectors...", space_name)
            scaler = StandardScaler()
            vectors = scaler.fit_transform(vectors)
        
        # Vectorized pairwise distance computation
        # pdist computes distances for all (i,j) pairs where i < j
        distances = pdist(vectors, metric)
        
        logger.info("✅ Computed %d pairwise %s distances", len(distances), space_name)
        logger.info("   Range: [%.4f, %.4f]", distances.min(), distances.max())
        
        return distances
    
    def compute_embedding_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """
        SPACE 1: Embedding distances (R_ij)
        
        Cosine distance for all news pairs:
        R_ij = 1 - cos(E_news_i, E_news_j)
        
        This measures "how similar are the news textually?"
        
        Args:
            embeddings: [K, latent_dim] - K news embedding representations
            
        Returns:
            distances: [K*(K-1)/2] - pairwise distances in condensed form
        """
        return self._compute_pairwise_distances(
            vectors=embeddings,
            metric='cosine',
            normalize=False,
            space_name='embedding'
        )
    
    def compute_impact_distances(self, impact_vectors: np.ndarray) -> np.ndarray:
        """
        SPACE 2: Impact distances (M_ij)
        
        Euclidean distance in normalized impact space:
        M_ij = Euclidean(V_impact_i', V_impact_j')
        
        This measures "how similar are the predicted impacts?"
        
        Args:
            impact_vectors: [K, 6] - unnormalized impact vectors
            
        Returns:
            distances: [K*(K-1)/2] - pairwise distances in condensed form
        """
        return self._compute_pairwise_distances(
            vectors=impact_vectors,
            metric='euclidean',
            normalize=True,
            space_name='impact'
        )
    
    def compute_impact_vectors(self, news_data: List[Dict]) -> np.ndarray:
        """
        Construct impact vector (V_impact) for each news item
        
        For each news i:
        - Prediction: μ_i (corr. change), σ_i (uncertainty), P_i (price deviation)
        - V_impact_i = [mean(|Δ_i|), std(Δ_i), mean(σ_i), std(σ_i), mean(|P_i|), std(P_i)]
        
        This 6D vector captures the complete predicted impact profile.
        
        Args:
            news_data: List of news dictionaries
            
        Returns:
            impact_vectors: [K, 6] - one 6D impact vector per news
        """
        K = len(news_data)
        impact_vectors = []
        
        logger.info("Computing impact vectors for %d news items...", K)
        
        for idx, news_item in enumerate(news_data):
            news_text = news_item.get('text', '')
            
            # Predict with uncertainty
            predictions = self.news_factor_model.predict_with_uncertainty(news_text, self.data_processor.baseline_z, 10)
            
            # Extract predictions
            mu = predictions['mean'][0]  # [N, N]
            sigma = predictions['std'][0]  # [N, N]
            P = predictions['price_deviations'][0]  # [N]
            
            # Compute delta: Δ = μ - baseline
            delta = mu - self.data_processor.baseline_z
            
            # Create 6D impact vector
            impact_vector = np.array([
                np.mean(np.abs(delta)),  # Mean absolute correlation change
                np.std(delta),           # Std of correlation change
                np.mean(sigma),          # Mean uncertainty
                np.std(sigma),           # Std of uncertainty
                np.mean(np.abs(P)),      # Mean absolute price deviation
                np.std(P)                # Std of price deviation
            ])
            
            impact_vectors.append(impact_vector)
            
            if (idx + 1) % 10 == 0:
                logger.info("  Processed %d/%d news items...", idx + 1, K)
        
        impact_vectors = np.array(impact_vectors)
        logger.info("✅ Impact vectors computed: shape %s", str(impact_vectors.shape))
        
        return impact_vectors
    
    def validate_smoothness(self, 
                           val_news: List[Dict],
                           save_path: str = 'smoothness_validation.png') -> Dict:
        """
        COMPLETE TOPOLOGICAL VALIDATION PIPELINE
        
        1. Extract embeddings
        2. Compute embedding distances (R_ij)
        3. Compute impact vectors
        4. Compute impact distances (M_ij)
        5. Generate 2D density heatmap visualization
        6. Perform statistical analysis
        
        Args:
            val_news: Validation news data
            save_path: Path to save visualization
            
        Returns:
            Dict with validation results including:
            - embedding_distances (R)
            - impact_distances (M)
            - correlation between R and M
            - stability assessment
        """
        logger.info("\n%s", "=" * 60)
        logger.info("TOPOLOGICAL SMOOTHNESS VALIDATION")
        logger.info("="*80)
        
        K = len(val_news)
        logger.info("Validating on %d news items...", K)
        
        # 1. Extract embeddings
        logger.info("\n[1/5] Extracting news embeddings...")
        embeddings, _ = self.performance_analyzer.extract_news_embeddings(val_news)
        
        # 2. Compute embedding distances (R_ij)
        logger.info("\n[2/5] Computing embedding distances (R_ij)...")
        R = self.compute_embedding_distances(embeddings)
        
        # 3. Compute impact vectors
        logger.info("\n[3/5] Computing impact vectors (V_impact)...")
        impact_vectors = self.compute_impact_vectors(val_news)
        
        # 4. Compute impact distances (M_ij)
        logger.info("\n[4/5] Computing impact distances (M_ij)...")
        M = self.compute_impact_distances(impact_vectors)
        
        # 5. Visualize smoothness map
        logger.info("\n[5/5] Generating smoothness heatmap...")
        fig = self.plot_smoothness_map(R, M, save_path)
        
        # 6. Statistical analysis
        logger.info("\n%s", "=" * 60)
        logger.info("SMOOTHNESS ANALYSIS")
        logger.info("="*80)
        
        # Correlation between R and M
        correlation = np.corrcoef(R, M)[0, 1]
        
        logger.info("\n📊 Key Metrics:")
        logger.info("  Correlation(R, M): %.4f", correlation)
        
        if correlation > 0.7:
            logger.info("  ✅ EXCELLENT: Strong smoothness - stable model!")
            assessment = "excellent"
        elif correlation > 0.5:
            logger.info("  ✅ GOOD: Moderate smoothness - acceptable stability")
            assessment = "good"
        elif correlation > 0.3:
            logger.info("  ⚠️  MODERATE: Weak smoothness - some instability")
            assessment = "moderate"
        else:
            logger.info("  ❌ POOR: No smoothness - unstable predictions!")
            assessment = "poor"
        
        # Binned analysis
        logger.info("\n📈 Binned Analysis:")
        bins = [0, 0.3, 0.6, 1.0, np.inf]
        labels = ['Very Similar', 'Similar', 'Different', 'Very Different']
        
        for i in range(len(bins) - 1):
            mask = (R >= bins[i]) & (R < bins[i+1])
            if mask.sum() > 0:
                avg_impact_dist = M[mask].mean()
                std_impact_dist = M[mask].std()
                logger.info(
                    "  %s (R ∈ [%.1f, %.1f)):",
                    labels[i].ljust(15), bins[i], bins[i+1]
                )
                logger.info(
                    "    → Avg impact distance: %.4f ± %.4f",
                    avg_impact_dist, std_impact_dist
                )
        
        # Identify unstable pairs
        logger.info("\n⚠️  Potential Instabilities:")
        unstable_threshold = 0.3  # Low R (similar news)
        high_impact_threshold = np.percentile(M, 75)  # High M (different impact)
        
        unstable_mask = (R < unstable_threshold) & (M > high_impact_threshold)
        num_unstable = unstable_mask.sum()
        
        if num_unstable > 0:
            logger.info("  Found %d news pairs with:", num_unstable)
            logger.info("    - Low embedding distance (R < %.2f)", unstable_threshold)
            logger.info("    - High impact distance (M > %.2f)", high_impact_threshold)
            logger.info(
                "  This represents %.1f%% of all pairs",
                100 * num_unstable / len(R)
            )
        else:
            logger.info("  ✅ No significant instabilities detected!")
        
        logger.info("\n%s", "=" * 60)
        
        return {
            'embedding_distances': R,
            'impact_distances': M,
            'correlation': correlation,
            'assessment': assessment,
            'num_news': K,
            'num_pairs': len(R),
            'num_unstable_pairs': num_unstable,
            'impact_vectors': impact_vectors,
            'embeddings': embeddings
        }
    
    def plot_smoothness_map(self, 
                           R: np.ndarray, 
                           M: np.ndarray,
                           save_path: str) -> plt.Figure:
        """
        2D Density Heatmap: Smoothness Map
        
        X-axis: Embedding distance (R_ij) - "Similar News ↔ Different News"
        Y-axis: Impact distance (M_ij) - "Similar Impact ↔ Different Impact"
        
        Ideal case: positive correlation (diagonal band)
        - Small R → small M (similar news, similar impact)
        - Large R → large M (different news, different impact)
        
        Args:
            R: Embedding distances (condensed)
            M: Impact distances (condensed)
            save_path: Output file path
            
        Returns:
            Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: 2D Hexbin (density)
        ax1 = axes[0]
        hexbin = ax1.hexbin(R, M, gridsize=30, cmap='YlOrRd', mincnt=1)
        ax1.set_xlabel('Embedding Distance (R) →\n← Similar News | Different News →', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Impact Distance (M) →\n← Similar Impact | Different Impact →', fontsize=12, fontweight='bold')
        ax1.set_title('Smoothness Map: Embedding Space vs Impact Space', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add diagonal reference line (perfect smoothness)
        ax1.plot([0, R.max()], [0, M.max()], 'b--', linewidth=2, label='Perfect Smoothness', alpha=0.7)
        ax1.legend(loc='upper left', fontsize=10)
        
        cbar1 = plt.colorbar(hexbin, ax=ax1)
        cbar1.set_label('Number of News Pairs', fontsize=11)
        
        # Plot 2: Scatter with trend
        ax2 = axes[1]
        
        # Scatter plot with transparency
        ax2.scatter(R, M, alpha=0.3, s=20, c='steelblue', edgecolors='none')
        
        # Add polynomial fit
        z = np.polyfit(R, M, 2)
        p = np.poly1d(z)
        R_smooth = np.linspace(R.min(), R.max(), 100)
        corr_val = np.corrcoef(R, M)[0, 1]
        ax2.plot(R_smooth, p(R_smooth), 'r-', linewidth=3, label='Trend (correlation: %.3f)' % corr_val)
        ax2.set_xlabel('Embedding Distance (R)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Impact Distance (M)', fontsize=12, fontweight='bold')
        ax2.set_title('Scatter Plot with Trend Line', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize=10)
        
        # Add text box with statistics
        stats_text = (
            f"N pairs: {len(R)}\n"
            f"R: μ={R.mean():.3f}, σ={R.std():.3f}\n"
            f"M: μ={M.mean():.3f}, σ={M.std():.3f}\n"
            f"Correlation: {corr_val:.4f}"
        )
        
        ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
                 horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("✅ Smoothness map saved to '%s'", save_path)
        return fig
    
    def analyze_impact_vector_components(self, impact_vectors: np.ndarray):
        """
        Analyze impact vector components
        
        Shows which components dominate the impact distance:
        - Δ mean/std (correlation change)
        - σ mean/std (uncertainty)
        - P mean/std (price deviation)
        
        Args:
            impact_vectors: [K, 6] array of impact vectors
            
        Returns:
            Figure object
        """
        logger.info("\n%s", "=" * 60)
        logger.info("IMPACT VECTOR COMPONENT ANALYSIS")
        logger.info("="*80)
        
        component_names = [
            'mean(|Δ|)',
            'std(Δ)',
            'mean(σ)',
            'std(σ)',
            'mean(|P|)',
            'std(P)'
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for i, (name, ax) in enumerate(zip(component_names, axes)):
            values = impact_vectors[:, i]
            
            ax.hist(values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel(name, fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('%s Distribution' % name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = "μ=%.4f\nσ=%.4f" % (values.mean(), values.std())
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            logger.info(
                "  %s: μ=%.4f, σ=%.4f, range=[%.4f, %.4f]",
                name.ljust(12), values.mean(), values.std(), 
                values.min(), values.max()
            )
        
        plt.tight_layout()
        plt.savefig('impact_vector_components.png', dpi=150, bbox_inches='tight')
        logger.info("\n✅ Component analysis saved to 'impact_vector_components.png'")
        logger.info("\n%s", "=" * 60)
        return fig
    
    def get_similar_companies_by_news_response(self, target_company, top_k=5):
        return self._get_similar_items(
            target_key=target_company,
            idx_lookup={symbol: i for i, symbol in enumerate(self.data_processor.companies)},
            name_lookup={i: symbol for i, symbol in enumerate(self.data_processor.companies)},
            embedding_layer=self.company_embedding_layer,
            top_k=top_k
        )

    def get_similar_keywords_by_impact(self, target_word, top_k=10):
        if target_word not in self.trading_system.word_to_idx:
            logger.info("Target word %s not found in vocab", target_word)
            return {}
        
        similar_list = self._get_similar_items(
            target_key=target_word,
            idx_lookup=self.trading_system.word_to_idx,
            name_lookup=self.trading_system.idx_to_word,
            embedding_layer=self.keyword_embedding_layer,
            top_k=top_k,
            invalid_tokens={"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"},
        )

        similar_dict = {word: float(sim) for word, sim in similar_list}
        return similar_dict
    
    def _get_similar_items(self, target_key, idx_lookup, name_lookup, embedding_layer, top_k=5, invalid_tokens=None):
        """Generic function to find similar items based on embedding cosine similarity."""
        if target_key not in idx_lookup:
            return []
        
        target_idx = idx_lookup[target_key]
        all_embeddings = embedding_layer.get_weights()[0]

        if target_idx >= len(all_embeddings):
            return []

        target_embedding = all_embeddings[target_idx]
        similarities = []
        target_norm = np.linalg.norm(target_embedding)

        if target_norm <= 1e-8:
            return []

        for idx, item in name_lookup.items():
            if idx == target_idx or idx >= len(all_embeddings):
                continue
            if invalid_tokens and item in invalid_tokens:
                continue

            embedding = all_embeddings[idx]
            embedding_norm = np.linalg.norm(embedding)

            if embedding_norm > 1e-8:
                similarity = np.dot(target_embedding, embedding) / (target_norm * embedding_norm)
                similarities.append((item, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]