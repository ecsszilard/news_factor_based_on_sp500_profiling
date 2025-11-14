import numpy as np
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score,
    adjusted_rand_score, 
    normalized_mutual_info_score
)
from typing import List, Dict, Tuple, Optional
from collections import Counter

logger = logging.getLogger("AdvancedNewsFactor.PerformanceAnalyzer")
models = tf.keras.models

class PerformanceAnalyzer:
    """
    Unified performance analysis and visualization system
    Includes clustering validation, learning curves, and prediction analysis
    """
    
    def __init__(self, trading_system):
        """Initialize the performance analyzer"""
        self.trading_system = trading_system
        self.data_processor = trading_system.data_processor
        self.news_model = trading_system.data_processor.news_factor_model
        logger.info("PerformanceAnalyzer initialized with clustering capabilities")

    # ========================================================================
    # TRADING PERFORMANCE METRICS
    # ========================================================================
    
    def generate_performance_report(self):
        """Generate comprehensive trading performance report"""
        periods = [7, 30, 90]
        period_results = {}
        
        for days in periods:
            start_date = datetime.date.today() - datetime.timedelta(days=days)
            period_results[f'{days}d'] = self.calculate_returns(start_date)

        portfolio_composition = {
            company: position for company, position 
            in self.trading_system.positions.items() 
            if abs(position) > 100
        }

        report = {
            'generated_at': datetime.datetime.now().isoformat(),
            'portfolio_value': self.trading_system.portfolio_value,
            'active_positions': len(portfolio_composition),
            'period_performance': period_results,
            'top_positions': dict(sorted(
                portfolio_composition.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:10])
        }
        return report

    def calculate_returns(self, start_date=None, end_date=None):
        """Calculate returns for a given period"""
        if start_date is None:
            start_date = datetime.date.today() - datetime.timedelta(days=30)
        if end_date is None:
            end_date = datetime.date.today()
        
        daily_returns = []
        
        # Calculation of performance indicators
        if daily_returns:
            avg_return = np.mean(daily_returns)
            volatility = np.std(daily_returns)
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            max_drawdown = self.calculate_max_drawdown(daily_returns)
        else:
            avg_return = volatility = sharpe_ratio = max_drawdown = 0
        
        return {
            'period': f"{start_date} to {end_date}",
            'average_daily_return': avg_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': len([r for r in daily_returns if r > 0]) / len(daily_returns) if daily_returns else 0
        }
    
    def calculate_max_drawdown(self, returns):
        """Maximum drawdown calculation"""
        if not returns:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(min(drawdown))

    # ========================================================================
    # NEWS EMBEDDING EXTRACTION & CLUSTERING
    # ========================================================================
    
    def extract_news_embeddings(self, news_data: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """
        Extract learned news representations from the model
        
        Args:
            news_data: List of news dictionaries with 'text' and optional 'news_type'
            
        Returns:
            embeddings: [num_news, latent_dim] - learned representations
            true_types: Ground truth news types (for validation)
        """
        logger.info(f"Extracting embeddings for {len(news_data)} news items...")
        
        # Get the shared news representation layer
        try:
            embedding_layer = self.news_model.model.get_layer('shared_news_representation')
        except ValueError:
            logger.error("Model does not have 'shared_news_representation' layer")
            raise
        
        # Build extraction model
        keyword_input = self.news_model.model.input[0]
        embedding_output = embedding_layer.output
        extractor_model = models.Model(inputs=keyword_input, outputs=embedding_output)
        
        embeddings = []
        true_types = []
        
        for news_item in news_data:
            # Tokenize
            text = news_item.get('text', '')
            if not text:
                logger.warning("Skipping news with empty text")
                continue
                
            keywords = self.data_processor.prepare_keyword_sequence(text)['input_ids']
            
            # Ensure proper shape: [batch_size, seq_length]
            if keywords.ndim == 3 and keywords.shape[1] == 1:
                keywords = np.squeeze(keywords, axis=1)
            elif keywords.ndim == 1:
                keywords = np.expand_dims(keywords, axis=0)
            
            # Extract embedding
            embedding = extractor_model.predict(keywords, verbose=0)[0]
            embeddings.append(embedding)
            
            # Ground truth type
            if 'news_type' in news_item:
                true_types.append(news_item['news_type'])
            else:
                true_types.append('unknown')
        
        embeddings = np.array(embeddings)
        logger.info(f"✅ Extracted embeddings: shape {embeddings.shape}")
        
        return embeddings, true_types
    
    def cluster_news_embeddings(self, 
                                embeddings: np.ndarray,
                                n_clusters: int = 5,
                                method: str = 'kmeans') -> Dict:
        """
        Perform unsupervised clustering on learned news embeddings
        
        Args:
            embeddings: [num_news, latent_dim]
            n_clusters: Expected number of clusters
            method: 'kmeans' or 'dbscan'
            
        Returns:
            Dict with clustering results and quality metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"UNSUPERVISED NEWS TYPE CLUSTERING ({method.upper()})")
        logger.info(f"{'='*60}")
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            predicted_clusters = clusterer.fit_predict(embeddings)
            cluster_centers = clusterer.cluster_centers_
            
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=3)
            predicted_clusters = clusterer.fit_predict(embeddings)
            n_clusters = len(set(predicted_clusters)) - (1 if -1 in predicted_clusters else 0)
            cluster_centers = None
            logger.info(f"DBSCAN found {n_clusters} clusters")
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Clustering quality metrics
        if len(set(predicted_clusters)) > 1:
            silhouette = silhouette_score(embeddings, predicted_clusters)
            calinski = calinski_harabasz_score(embeddings, predicted_clusters)
        else:
            silhouette = calinski = -1
        
        logger.info(f"\nClustering Quality:")
        logger.info(f"  Silhouette Score: {silhouette:.3f} (higher = better, max=1)")
        logger.info(f"  Calinski-Harabasz: {calinski:.1f} (higher = better)")
        
        # Cluster distribution
        unique, counts = np.unique(predicted_clusters, return_counts=True)
        logger.info(f"\nCluster Distribution:")
        for cluster_id, count in zip(unique, counts):
            logger.info(f"  Cluster {cluster_id}: {count} news ({100*count/len(embeddings):.1f}%)")
        
        return {
            'predicted_clusters': predicted_clusters,
            'cluster_centers': cluster_centers,
            'n_clusters': n_clusters,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'cluster_sizes': dict(zip(unique.tolist(), counts.tolist()))
        }
    
    def compare_clusters_with_ground_truth(self, 
                                          predicted_clusters: np.ndarray,
                                          true_types: List[str]) -> Dict:
        """
        Compare clustering results with ground truth news types
        
        Args:
            predicted_clusters: Predicted cluster IDs
            true_types: Ground truth news types
            
        Returns:
            Dict with comparison metrics
        """
        # Convert string types to numeric
        type_to_id = {t: i for i, t in enumerate(sorted(set(true_types)))}
        true_labels = np.array([type_to_id[t] for t in true_types])
        
        # Comparison metrics
        ari = adjusted_rand_score(true_labels, predicted_clusters)
        nmi = normalized_mutual_info_score(true_labels, predicted_clusters)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"GROUND TRUTH COMPARISON")
        logger.info(f"{'='*60}")
        logger.info(f"  Adjusted Rand Index: {ari:.3f} (1.0 = perfect match)")
        logger.info(f"  Normalized Mutual Info: {nmi:.3f} (1.0 = perfect match)")
        
        # Cluster → News Type mapping
        logger.info(f"\nCluster → News Type Mapping:")
        for cluster_id in sorted(set(predicted_clusters)):
            cluster_mask = predicted_clusters == cluster_id
            types_in_cluster = [true_types[i] for i in range(len(true_types)) if cluster_mask[i]]
            
            if types_in_cluster:
                type_counts = Counter(types_in_cluster)
                dominant_type = type_counts.most_common(1)[0]
                purity = dominant_type[1] / len(types_in_cluster)
                
                logger.info(f"\n  Cluster {cluster_id} (n={len(types_in_cluster)}):")
                logger.info(f"    Dominant type: {dominant_type[0]} ({purity:.1%} purity)")
                logger.info(f"    Distribution: {dict(type_counts)}")
        
        return {
            'adjusted_rand_index': ari,
            'normalized_mutual_info': nmi,
            'true_type_to_id': type_to_id
        }
    
    def analyze_cluster_characteristics(self, 
                                       predicted_clusters: np.ndarray,
                                       news_data: List[Dict]) -> Dict:
        """
        Analyze characteristics of each cluster
        
        Args:
            predicted_clusters: Cluster assignments
            news_data: Original news data with metadata
            
        Returns:
            Dict with cluster statistics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"CLUSTER CHARACTERISTICS ANALYSIS")
        logger.info(f"{'='*60}")
        
        cluster_stats = {}
        
        for cluster_id in sorted(set(predicted_clusters)):
            cluster_mask = predicted_clusters == cluster_id
            cluster_news = [news_data[i] for i in range(len(news_data)) if cluster_mask[i]]
            
            # Extract properties
            impacts = []
            scopes = []
            keywords_all = []
            
            for news in cluster_news:
                if 'impact_magnitude' in news:
                    impacts.append(abs(news['impact_magnitude']))
                
                if 'news_scope' in news:
                    scopes.append(news['news_scope'])
                
                if 'keywords' in news:
                    keywords_all.extend(news['keywords'])
            
            # Calculate statistics
            cluster_stats[cluster_id] = {
                'size': len(cluster_news),
                'avg_impact': np.mean(impacts) if impacts else 0,
                'std_impact': np.std(impacts) if impacts else 0,
                'dominant_scope': max(set(scopes), key=scopes.count) if scopes else 'unknown',
                'top_keywords': self._get_top_keywords(keywords_all, top_k=5)
            }
            
            logger.info(f"\nCluster {cluster_id}:")
            logger.info(f"  Size: {cluster_stats[cluster_id]['size']}")
            logger.info(f"  Avg Impact: {cluster_stats[cluster_id]['avg_impact']:.4f}")
            logger.info(f"  Dominant Scope: {cluster_stats[cluster_id]['dominant_scope']}")
            logger.info(f"  Top Keywords: {cluster_stats[cluster_id]['top_keywords']}")
        
        return cluster_stats
    
    def _get_top_keywords(self, keywords_list: List[str], top_k: int = 5) -> List[str]:
        """Get top-k most frequent keywords"""
        if not keywords_list:
            return []
        counter = Counter(keywords_list)
        return [word for word, count in counter.most_common(top_k)]

    # ========================================================================
    # COMPREHENSIVE CLUSTERING VALIDATION
    # ========================================================================
    
    def validate_news_type_learning(self, 
                                   news_data: List[Dict],
                                   n_clusters: int = 5,
                                   save_path: str = 'news_clustering_validation.png') -> Dict:
        """
        Complete validation pipeline: extract → cluster → compare → visualize
        
        Args:
            news_data: List of news dicts with 'text' and optionally 'news_type'
            n_clusters: Expected number of news types
            save_path: Path to save visualization
            
        Returns:
            Comprehensive validation results
        """
        logger.info("\n" + "="*80)
        logger.info("UNSUPERVISED NEWS TYPE DISCOVERY VALIDATION")
        logger.info("="*80)
        
        # 1. Extract embeddings
        embeddings, true_types = self.extract_news_embeddings(news_data)
        
        # 2. Cluster
        clustering_result = self.cluster_news_embeddings(
            embeddings, 
            n_clusters=n_clusters,
            method='kmeans'
        )
        
        # 3. Compare with ground truth
        comparison_result = self.compare_clusters_with_ground_truth(
            clustering_result['predicted_clusters'],
            true_types
        )
        
        # 4. Characterize clusters
        cluster_chars = self.analyze_cluster_characteristics(
            clustering_result['predicted_clusters'],
            news_data
        )
        
        # 5. Visualize
        self.visualize_news_clustering(
            embeddings,
            predicted_clusters=clustering_result['predicted_clusters'],
            true_types=true_types,
            save_path=save_path
        )
        
        # Overall assessment
        logger.info("\n" + "="*80)
        logger.info("OVERALL ASSESSMENT")
        logger.info("="*80)
        
        ari = comparison_result['adjusted_rand_index']
        silhouette = clustering_result['silhouette_score']
        
        if ari > 0.5 and silhouette > 0.3:
            logger.info("✅ EXCELLENT: Model learned meaningful news type distinctions!")
        elif ari > 0.3 and silhouette > 0.2:
            logger.info("✅ GOOD: Model captures some news type structure")
        elif ari > 0.1:
            logger.info("⚠️  MODERATE: Some structure learned, but not aligned with types")
        else:
            logger.info("❌ WEAK: Model does not distinguish news types well")
        
        logger.info(f"  → Adjusted Rand Index: {ari:.3f}")
        logger.info(f"  → Silhouette Score: {silhouette:.3f}")
        logger.info("="*80 + "\n")
        
        return {
            'embeddings': embeddings,
            'clustering': clustering_result,
            'comparison': comparison_result,
            'cluster_characteristics': cluster_chars
        }

    # ========================================================================
    # VISUALIZATION METHODS
    # ========================================================================
    
    def visualize_news_clustering(self, 
                                  embeddings: np.ndarray,
                                  predicted_clusters: Optional[np.ndarray] = None,
                                  true_types: Optional[List[str]] = None,
                                  save_path: str = 'news_clustering.png'):
        """
        Visualize news embeddings with optional cluster/type coloring
        
        Args:
            embeddings: [num_news, latent_dim]
            predicted_clusters: Optional cluster assignments
            true_types: Optional ground truth types
            save_path: Output file path
        """
        logger.info(f"\n📊 Generating clustering visualization...")
        
        # t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Determine layout
        if predicted_clusters is not None and true_types is not None:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            plot_clusters = True
            plot_types = True
        elif predicted_clusters is not None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            axes = [ax]
            plot_clusters = True
            plot_types = False
        elif true_types is not None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            axes = [ax]
            plot_clusters = False
            plot_types = True
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            axes = [ax]
            plot_clusters = False
            plot_types = False
        
        # Plot 1: Predicted Clusters
        if plot_clusters:
            ax = axes[0]
            scatter = ax.scatter(
                embeddings_2d[:, 0], 
                embeddings_2d[:, 1],
                c=predicted_clusters,
                cmap='tab10',
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=1
            )
            ax.set_title('Predicted Clusters (Unsupervised)', fontsize=14, fontweight='bold')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.grid(True, alpha=0.3)
            
            # Add cluster labels
            for cluster_id in set(predicted_clusters):
                cluster_mask = predicted_clusters == cluster_id
                cluster_center = embeddings_2d[cluster_mask].mean(axis=0)
                ax.annotate(
                    f'C{cluster_id}',
                    xy=cluster_center,
                    fontsize=12,
                    fontweight='bold',
                    color='red',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
                )
            
            plt.colorbar(scatter, ax=ax, label='Cluster ID')
        
        # Plot 2: Ground Truth Types
        if plot_types:
            ax = axes[1] if len(axes) > 1 else axes[0]
            unique_types = sorted(set(true_types))
            type_colors = {t: i for i, t in enumerate(unique_types)}
            colors = [type_colors[t] for t in true_types]
            
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=colors,
                cmap='tab10',
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=1
            )
            
            ax.set_title('Ground Truth News Types', fontsize=14, fontweight='bold')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.grid(True, alpha=0.3)
            
            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=plt.cm.tab10(type_colors[t]), 
                     edgecolor='black', label=t)
                for t in unique_types
            ]
            ax.legend(handles=legend_elements, loc='best', title='News Type')
        
        # Simple embedding space (no colors)
        if not plot_clusters and not plot_types:
            ax = axes[0]
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c='steelblue',
                s=100,
                alpha=0.6,
                edgecolors='black',
                linewidths=1
            )
            ax.set_title('News Embedding Space (t-SNE)', fontsize=14, fontweight='bold')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Saved to '{save_path}'")
        
        return fig

    def visualize_uncertainty_predictions(self, analysis, companies, news_text):
        """
        Visualize probabilistic predictions with uncertainty bands
        
        Args:
            analysis: Dict with confidence metrics from predict_news_impact
            companies: List of company names
            news_text: News text for title
        """
        predicted_corr = analysis['predicted_corr']
        sigma_corr = analysis['sigma_corr']
        delta_corr = analysis['delta_corr']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Probabilistic Prediction Analysis\nNews: "{news_text[:80]}..."', 
                    fontsize=14, fontweight='bold')
        
        # 1. Correlation Heatmap with Uncertainty
        ax1 = axes[0, 0]
        im1 = ax1.imshow(predicted_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax1.set_xticks(range(len(companies)))
        ax1.set_yticks(range(len(companies)))
        ax1.set_xticklabels(companies, rotation=45, ha='right')
        ax1.set_yticklabels(companies)
        ax1.set_title('Predicted Correlations (μ)')
        plt.colorbar(im1, ax=ax1)
        
        # Add uncertainty overlays (circle size = uncertainty)
        for i in range(len(companies)):
            for j in range(len(companies)):
                if i != j:
                    circle_size = sigma_corr[i, j] * 500  # Scale for visibility
                    ax1.scatter(j, i, s=circle_size, c='none', 
                            edgecolors='yellow', linewidths=1, alpha=0.6)
        
        # 2. Uncertainty Heatmap
        ax2 = axes[0, 1]
        im2 = ax2.imshow(sigma_corr, cmap='YlOrRd', aspect='auto')
        ax2.set_xticks(range(len(companies)))
        ax2.set_yticks(range(len(companies)))
        ax2.set_xticklabels(companies, rotation=45, ha='right')
        ax2.set_yticklabels(companies)
        ax2.set_title('Predicted Uncertainty (σ)')
        plt.colorbar(im2, ax=ax2, label='Standard Deviation')
        
        # 3. Correlation Changes (Δ = predicted - baseline)
        ax3 = axes[1, 0]
        max_abs_delta = max(abs(delta_corr.min()), abs(delta_corr.max()))
        im3 = ax3.imshow(delta_corr, cmap='RdBu_r', vmin=-max_abs_delta, vmax=max_abs_delta, aspect='auto')
        ax3.set_xticks(range(len(companies)))
        ax3.set_yticks(range(len(companies)))
        ax3.set_xticklabels(companies, rotation=45, ha='right')
        ax3.set_yticklabels(companies)
        ax3.set_title('Correlation Change (Δ = μ - baseline)')
        plt.colorbar(im3, ax=ax3, label='Change')
        
        # 4. Confidence Breakdown
        ax4 = axes[1, 1]
        components = ['Total\nConfidence', 'Reconstruction\n(Epistemic)', 'Uncertainty\n(Total)', 'Epistemic\n(MC Dropout)', 'Aleatoric\n(Data)']
        values = [
            analysis['total_confidence'],
            analysis['recon_confidence'],
            analysis['uncertainty_confidence'],
            analysis.get('epistemic_confidence', 0.5),  # New: MC Dropout
            1.0 / (1.0 + analysis.get('aleatoric_uncertainty', 0.3))  # New: Data noise
        ]
        colors = ['green', 'blue', 'orange', 'purple', 'red']
        bars = ax4.bar(components, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax4.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax4.set_title('Confidence Score Breakdown', fontsize=12, fontweight='bold')
        ax4.set_ylim([0, 1.1])
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Trading Threshold (0.7)')
        ax4.legend()
        
        # Add values on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Statistics
        uncertainty_matrix = analysis.get('uncertainty_matrix', sigma_corr)
        epistemic_unc = analysis.get('epistemic_uncertainty', 0)
        aleatoric_unc = analysis.get('aleatoric_uncertainty', 0)
        
        stats_text = f"Avg σ_total: {np.mean(uncertainty_matrix):.4f}\n"
        stats_text += f"σ_epistemic: {epistemic_unc:.4f}\n"
        stats_text += f"σ_aleatoric: {aleatoric_unc:.4f}\n"
        stats_text += f"Max σ: {np.max(uncertainty_matrix):.4f}\n"
        stats_text += f"High uncertainty pairs: {np.sum(uncertainty_matrix > 0.3)}/{len(companies)**2}"
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curves(self, history):
        """Plot training learning curves"""
        logger.info("\n📊 Generating learning curves...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training vs Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Directional Accuracy
        train_acc = history.history['correlation_changes_probabilistic_directional_accuracy']
        val_acc = history.history['val_correlation_changes_probabilistic_directional_accuracy']
        axes[0, 1].plot(train_acc, label='Train', linewidth=2)
        axes[0, 1].plot(val_acc, label='Validation', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Directional Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Calibration
        train_cal = history.history['correlation_changes_probabilistic_calibration_metric']
        val_cal = history.history['val_correlation_changes_probabilistic_calibration_metric']
        axes[1, 0].plot(train_cal, label='Train', linewidth=2)
        axes[1, 0].plot(val_cal, label='Validation', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Calibration Error')
        axes[1, 0].set_title('Calibration Quality (lower = better)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Price MAE
        train_mae = history.history['price_deviations_mae']
        val_mae = history.history['val_price_deviations_mae']
        axes[1, 1].plot(train_mae, label='Train', linewidth=2)
        axes[1, 1].plot(val_mae, label='Validation', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('Price Deviation MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
        logger.info("✅ Saved to 'learning_curves.png'")

        # Check for overfitting
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        if final_val_loss > final_train_loss * 1.5:
            logger.warning("⚠️  Possible overfitting detected!")
            logger.warning(f"  Train loss: {final_train_loss:.4f}")
            logger.warning(f"  Val loss: {final_val_loss:.4f}")
        else:
            logger.info("✅ Model generalizes well to validation data")
        
        return fig
    
    def plot_calibration_curve(self, predictions, actuals):
        """
        Reliability diagram: predicted σ vs actual error
        
        Args:
            predictions: Dict with 'mean' and 'std'
            actuals: Actual correlation values
        """
        logger.info("\n📊 Generating calibration curve...")
        
        sigma = predictions['std'].flatten()
        errors = np.abs(actuals - predictions['mean']).flatten()
        
        # Bin by σ
        bins = np.linspace(0, sigma.max(), 10)
        bin_indices = np.digitize(sigma, bins)
        
        expected_errors = []
        observed_errors = []
        
        for i in range(1, len(bins)):
            mask = bin_indices == i
            if mask.sum() > 0:
                expected_errors.append(bins[i-1])
                observed_errors.append(errors[mask].mean())
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(expected_errors, observed_errors, 'o-', linewidth=2, markersize=8, label='Observed')
        ax.plot([0, max(expected_errors)], [0, max(expected_errors)], 'r--', linewidth=2, label='Perfect calibration')
        ax.set_xlabel('Predicted σ', fontsize=12)
        ax.set_ylabel('Observed MAE', fontsize=12)
        ax.set_title('Calibration Curve (Reliability Diagram)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('calibration_curve.png', dpi=150, bbox_inches='tight')
        logger.info("✅ Saved to 'calibration_curve.png'")
        
        return fig