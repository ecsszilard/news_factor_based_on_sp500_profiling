import numpy as np
import datetime
import json
import logging

logger = logging.getLogger("AdvancedNewsFactor.PerformanceAnalyzer")

class PerformanceAnalyzer:
    """Performance analysis system"""
    
    def __init__(self, trading_system):
        """Initialize the performance analyzer"""
        self.trading_system = trading_system
        
        logger.info("PerformanceAnalyzer inicializálva")
    
    def calculate_returns(self, start_date=None, end_date=None):
        """Calculation of yields"""

        if start_date is None:
            start_date = datetime.date.today() - datetime.timedelta(days=30)
        if end_date is None:
            end_date = datetime.date.today()
        
        # Filter trades for the specified period
        relevant_trades = [
            trade for trade in self.trading_system.trade_history
            if start_date <= trade['timestamp'].date() <= end_date
        ]
        
        if not relevant_trades:
            return {'error': 'Nincs kereskedés a megadott időszakban'}
        
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
            'total_trades': len(relevant_trades),
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
