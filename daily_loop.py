#!/usr/bin/env python3
"""Daily loop for strategy evaluation and portfolio optimization"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from data import DataLoader
from market_analysis import MarketAnalyzer
from portfolio_manager import PortfolioManager, StrategySelectorModel
from historian import Historian
from reporting import ReportGenerator
from strategy import StrategyFactory, StrategyConfig
from trading_types import Config, Trade, ValidationError, DataError

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DailyLoop:
    """Daily loop for strategy evaluation and portfolio optimization"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize daily loop components
        
        Args:
            config: Optional system configuration
        """
        self.config = config or Config()
        self.data_loader = DataLoader(self.config)
        self.market_analyzer = MarketAnalyzer()
        self.portfolio_mgr = PortfolioManager(self.config)
        self.historian = Historian()
        self.reporter = ReportGenerator(self.config)
        self.strategy_selector = StrategySelectorModel()
        self.logger = logging.getLogger(__name__)

    def run_single_iteration(self) -> Dict:
        """Execute a single iteration of the daily loop
        
        Returns:
            Dictionary containing iteration results
        """
        try:
            self.logger.info("Starting daily evaluation...")
            results = {
                'market_conditions_analyzed': False,
                'model_updated': False,
                'strategies_selected': False,
                'portfolio_rebalanced': False,
                'report_generated': False,
                'timestamp': datetime.now().isoformat()
            }
            
            # 1. Get market data
            symbols = self.data_loader.get_high_volume_symbols(N=10)  # Top 10 by volume
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Training window
            
            market_data = self.data_loader.fetch_historical_data(
                symbols=symbols,
                start=start_date,
                end=end_date,
                interval="1d"
            )
            
            # 2. Analyze market conditions
            market_conditions = self.market_analyzer.analyze_market_conditions(market_data)
            results['market_conditions_analyzed'] = True
            
            # 3. Prepare strategy configurations
            strategy_configs = [
                StrategyConfig(
                    name="mean_rev_1",
                    type="mean_reversion",
                    parameters={
                        'entry_threshold': 2.0,
                        'exit_threshold': 0.5
                    },
                    symbols=symbols
                ),
                StrategyConfig(
                    name="momentum_1",
                    type="momentum",
                    parameters={
                        'lookback_period': 20,
                        'entry_threshold': 0.02
                    },
                    symbols=symbols
                ),
                StrategyConfig(
                    name="pca_1",
                    type="pca",
                    parameters={
                        'n_components': 3,
                        'entry_threshold': 1.0
                    },
                    symbols=symbols
                )
            ]
            
            # 4. Evaluate strategies
            strategy_metrics = []
            for config in strategy_configs:
                try:
                    strategy = StrategyFactory.create(config, self.config)
                    signals = strategy.generate_signals(market_data)
                    
                    # Calculate strategy metrics
                    returns = pd.Series(0.0, index=market_data[symbols[0]].index)
                    for symbol, signal in signals.items():
                        symbol_returns = market_data[symbol]['close'].pct_change()
                        returns += signal * symbol_returns
                    
                    metrics = {
                        'strategy_id': config.name,
                        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
                        'volatility': returns.std() * np.sqrt(252),
                        'total_return': (1 + returns).prod() - 1
                    }
                    strategy_metrics.append(metrics)
                    
                except Exception as e:
                    self.logger.error(f"Strategy evaluation failed for {config.name}: {str(e)}")
            
            # 5. Train strategy selector
            if strategy_metrics:
                features = pd.DataFrame([
                    {
                        'volatility': market_conditions.volatility,
                        'skewness': market_conditions.skewness,
                        'mean_return': market_conditions.mean_return,
                        'strategy_id': metric['strategy_id'],
                        'sharpe_ratio': metric['sharpe_ratio']
                    }
                    for metric in strategy_metrics
                ])
                
                self.strategy_selector.train(
                    market_features=features.drop(['strategy_id', 'sharpe_ratio'], axis=1),
                    strategy_performance=features['sharpe_ratio']
                )
                results['model_updated'] = True
            
            # 6. Select strategies for live trading
            current_features = pd.DataFrame([{
                'volatility': market_conditions.volatility,
                'skewness': market_conditions.skewness,
                'mean_return': market_conditions.mean_return
            }])
            
            predictions = self.strategy_selector.predict(current_features)
            
            # Select top strategies
            top_strategies = sorted(
                predictions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.config.max_strategies]
            
            selected_strategies = [strat[0] for strat in top_strategies]
            results['strategies_selected'] = True
            results['selected_strategies'] = selected_strategies
            
            # 7. Update portfolio allocations
            total_score = sum(strat[1] for strat in top_strategies)
            allocations = {
                strat[0]: min(strat[1] / total_score, self.config.max_strategy_allocation)
                for strat in top_strategies
            }
            
            # Normalize allocations
            total_allocation = sum(allocations.values())
            allocations = {k: v/total_allocation for k, v in allocations.items()}
            
            self.portfolio_mgr.update_allocations(allocations)
            results['portfolio_rebalanced'] = True
            
            # 8. Generate and send daily report
            portfolio_state = self.portfolio_mgr.get_portfolio_state()
            trades = self.historian.load_trades(start_date, end_date)
            
            report = self.reporter.generate_portfolio_report(
                portfolio_state=portfolio_state,
                trades=trades,
                market_conditions=market_conditions
            )
            
            results['report_generated'] = True
            results['report'] = report
            
            self.logger.info("Daily evaluation completed successfully")
            return results
            
        except Exception as e:
            error_msg = f"Daily evaluation failed: {str(e)}"
            self.logger.error(error_msg)
            results['error'] = error_msg
            return results

    def run(self) -> None:
        """Run the daily loop once"""
        try:
            self.run_single_iteration()
        except Exception as e:
            self.logger.error(f"Critical error in daily loop: {str(e)}")
            raise

def main():
    """Main entry point"""
    loop = DailyLoop()
    loop.run()

if __name__ == "__main__":
    main()