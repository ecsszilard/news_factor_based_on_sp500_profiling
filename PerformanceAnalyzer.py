import numpy as np
import datetime
import json
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger("AdvancedNewsFactor.PerformanceAnalyzer")

class PerformanceAnalyzer:
    """Performance analysis system"""
    
    def __init__(self, trading_system):
        """Initialize the performance analyzer"""
        self.trading_system = trading_system
        logger.info("PerformanceAnalyzer inicializálva")

    def generate_performance_report(self):
        periods = [7, 30, 90]
        period_results = {}
        
        for days in periods:
            start_date = datetime.date.today() - datetime.timedelta(days=days)
            period_results[f'{days}d'] = self.calculate_returns(start_date)

        portfolio_composition = {
            company: position for company, position 
            in self.trading_system.positions.items() 
            if abs(position) > 100  # Csak jelentős pozíciók
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
        """Calculation of yields"""

        if start_date is None:
            start_date = datetime.date.today() - datetime.timedelta(days=30)
        if end_date is None:
            end_date = datetime.date.today()
        
        # Simulation of daily returns (market data is needed in real implementation)
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

    def visualize_uncertainty_predictions(self, predictions, baseline_corr, predicted_corr, companies, news_text):
        """
        Visualize predictions with uncertainty bands
        
        Note: This is a standalone visualization function.
        The trading decision logic is now integrated into
        AdvancedTradingSystem.generate_trading_signals()
        """
        sigma = predictions['std'][0]  # [N, N]
        
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
                    circle_size = sigma[i, j] * 500  # Scale for visibility
                    ax1.scatter(j, i, s=circle_size, c='none', 
                            edgecolors='yellow', linewidths=1, alpha=0.6)
        
        # 2. Uncertainty Heatmap
        ax2 = axes[0, 1]
        im2 = ax2.imshow(sigma, cmap='YlOrRd', aspect='auto')
        ax2.set_xticks(range(len(companies)))
        ax2.set_yticks(range(len(companies)))
        ax2.set_xticklabels(companies, rotation=45, ha='right')
        ax2.set_yticklabels(companies)
        ax2.set_title('Predicted Uncertainty (σ)')
        plt.colorbar(im2, ax=ax2, label='Standard Deviation')
        
        # 3. Correlation Changes (Δ = predicted - baseline)
        ax3 = axes[1, 0]
        delta = predicted_corr - baseline_corr
        max_abs_delta = max(abs(delta.min()), abs(delta.max()))
        im3 = ax3.imshow(delta, cmap='RdBu_r', 
                        vmin=-max_abs_delta, vmax=max_abs_delta, aspect='auto')
        ax3.set_xticks(range(len(companies)))
        ax3.set_yticks(range(len(companies)))
        ax3.set_xticklabels(companies, rotation=45, ha='right')
        ax3.set_yticklabels(companies)
        ax3.set_title('Correlation Change (Δ = μ - baseline)')
        plt.colorbar(im3, ax=ax3, label='Change')
        
        # 4. Confidence Breakdown
        ax4 = axes[1, 1]
        components = ['Total\nConfidence', 'Reconstruction\n(Epistemic)', 'Uncertainty\n(Aleatoric)']
        values = [
            predictions['total_confidence'],
            predictions['reconstruction_confidence'],
            predictions['uncertainty_confidence']
        ]
        colors = ['green', 'blue', 'orange']
        bars = ax4.bar(components, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax4.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax4.set_title('Confidence Score Breakdown', fontsize=12, fontweight='bold')
        ax4.set_ylim([0, 1.1])
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=0.7, color='red', linestyle='--', linewidth=2, 
                    label='Trading Threshold (0.7)')
        ax4.legend()
        
        # Add values on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)
        
        # Add statistics text
        stats_text = f"Avg σ: {np.mean(sigma):.4f}\n"
        stats_text += f"Max σ: {np.max(sigma):.4f}\n"
        stats_text += f"High uncertainty pairs: {np.sum(sigma > 0.3)}/{len(companies)**2}"
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig