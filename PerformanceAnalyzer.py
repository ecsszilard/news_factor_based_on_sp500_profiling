import numpy as np
import datetime
import logging

logger = logging.getLogger("AdvancedNewsFactor.PerformanceAnalyzer")

class PerformanceAnalyzer:
    """
    Teljesítményelemző rendszer
    """
    
    def __init__(self, trading_system):
        """
        Inicializálja a teljesítményelemzőt
        
        Paraméterek:
            trading_system (AdvancedTradingSystem): Kereskedési rendszer
        """
        self.trading_system = trading_system
        
        logger.info("PerformanceAnalyzer inicializálva")
    
    def calculate_returns(self, start_date=None, end_date=None):
        """
        Hozamok számítása
        
        Paraméterek:
            start_date (datetime.date): Kezdő dátum
            end_date (datetime.date): Záró dátum
            
        Visszatérési érték:
            dict: Hozammutatók
        """
        if start_date is None:
            start_date = datetime.date.today() - datetime.timedelta(days=30)
        if end_date is None:
            end_date = datetime.date.today()
        
        # Szűrjük a kereskedéseket a megadott időszakra
        relevant_trades = [
            trade for trade in self.trading_system.trade_history
            if start_date <= trade['timestamp'].date() <= end_date
        ]
        
        if not relevant_trades:
            return {'error': 'Nincs kereskedés a megadott időszakban'}
        
        # Napi hozamok szimulációja (valódi implementációban piaci adatok kellenek)
        daily_returns = []
        cumulative_return = 0.0
        
        # Egyszerűsített hozamszámítás a előrejelzések alapján
        for trade in relevant_trades:
            predicted_return = trade['predicted_change']
            # Feltételezzük 70%-os pontosságot
            actual_return = predicted_return * np.random.choice([1, -0.3], p=[0.7, 0.3])
            daily_returns.append(actual_return)
            cumulative_return += actual_return
        
        # Teljesítménymutatók számítása
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
            'cumulative_return': cumulative_return,
            'average_daily_return': avg_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': len([r for r in daily_returns if r > 0]) / len(daily_returns) if daily_returns else 0
        }
    
    def calculate_max_drawdown(self, returns):
        """
        Maximum drawdown számítása
        
        Paraméterek:
            returns (list): Napi hozamok listája
            
        Visszatérési érték:
            float: Maximum drawdown
        """
        if not returns:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(min(drawdown))
    
    def analyze_prediction_accuracy(self, actual_price_data):
        """
        Előrejelzési pontosság elemzése
        
        Paraméterek:
            actual_price_data (dict): Tényleges árfolyamadatok
            
        Visszatérési érték:
            dict: Pontossági mutatók
        """
        predictions = []
        actuals = []
        
        for trade in self.trading_system.trade_history[-100:]:  # Utolsó 100 kereskedés
            company = trade['company']
            predicted_change = trade['predicted_change']
            
            # Itt a tényleges változást kellene lekérni
            # Most szimulált adatokkal dolgozunk
            actual_change = predicted_change * np.random.normal(0.8, 0.3)  # Szimulált pontosság
            
            predictions.append(predicted_change)
            actuals.append(actual_change)
        
        if predictions and actuals:
            # Korrelációs együttható
            correlation = np.corrcoef(predictions, actuals)[0, 1]
            
            # Átlagos abszolút hiba
            mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
            
            # Irány pontossága (felfelé/lefelé)
            direction_correct = sum(
                (p > 0) == (a > 0) for p, a in zip(predictions, actuals)
            ) / len(predictions)
            
            return {
                'correlation': correlation,
                'mean_absolute_error': mae,
                'direction_accuracy': direction_correct,
                'sample_size': len(predictions)
            }
        
        return {'error': 'Nincs elég adat az elemzéshez'}
    
    def generate_performance_report(self, save_path=None):
        """
        Teljes teljesítményjelentés generálása
        
        Paraméterek:
            save_path (str): Mentési útvonal (opcionális)
            
        Visszatérési érték:
            dict: Teljes teljesítményjelentés
        """
        # Különböző időszakok elemzése
        periods = [7, 30, 90]
        period_results = {}
        
        for days in periods:
            start_date = datetime.date.today() - datetime.timedelta(days=days)
            period_results[f'{days}d'] = self.calculate_returns(start_date)
        
        # Előrejelzési pontosság
        prediction_accuracy = self.analyze_prediction_accuracy({})
        
        # Portfolio összetétel
        portfolio_composition = {
            company: position for company, position 
            in self.trading_system.positions.items() 
            if abs(position) > 100  # Csak jelentős pozíciók
        }
        
        # Teljes jelentés összeállítása
        report = {
            'generated_at': datetime.datetime.now().isoformat(),
            'portfolio_value': self.trading_system.portfolio_value,
            'active_positions': len(portfolio_composition),
            'total_trades': len(self.trading_system.trade_history),
            'period_performance': period_results,
            'prediction_accuracy': prediction_accuracy,
            'top_positions': dict(sorted(
                portfolio_composition.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:10])
        }
        
        # Mentés ha kért
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Teljesítményjelentés mentve: {save_path}")
        
        return report
