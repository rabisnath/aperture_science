#!/usr/bin/env python3
"""5-minute trading loop for real-time execution"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from data import DataLoader
from strategy import BaseStrategy, StrategyFactory, StrategyConfig
from portfolio_manager import PortfolioManager
from back_of_house import LiveTrader
from historian import Historian
from reporting import ReportGenerator
from market_analysis import MarketAnalyzer
from trading_types import Config, Trade, ValidationError, DataError

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FiveMinuteLoop:
    """Main trading loop that executes every 5 minutes"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize trading loop components
        
        Args:
            config: Optional system configuration
        """
        self.config = config or Config()
        self.data_loader = DataLoader(self.config)
        self.market_analyzer = MarketAnalyzer()
        self.portfolio_mgr = PortfolioManager(self.config)
        self.live_trader = LiveTrader()
        self.historian = Historian()
        self.reporter = ReportGenerator(self.config)
        self.interval = 300  # 5 minutes in seconds
        self.next_run = time.time()
        self.logger = logging.getLogger(__name__)

    def run_single_iteration(self) -> Dict:
        """Execute a single iteration of the trading loop
        
        Returns:
            Dictionary containing iteration results
        """
        try:
            self.logger.info("Starting new cycle...")
            cycle_start = datetime.now()
            results = {
                'market_data_updated': False,
                'signals_generated': False,
                'trades_executed': False,
                'data_recorded': False,
                'timestamp': cycle_start.isoformat()
            }
            
            # 1. Get latest market data
            symbols = self.data_loader.get_high_volume_symbols(N=5)  # Top 5 by volume
            market_data = self.data_loader.get_market_data(
                symbols=symbols,
                interval="5m",
                lookback=100  # Need enough data for indicators
            )
            results['market_data_updated'] = True
            
            # 2. Analyze market conditions
            market_conditions = self.market_analyzer.analyze_market_conditions(market_data)
            
            # 3. Generate signals for each active strategy
            signals = {}
            for strategy in self.portfolio_mgr.get_active_strategies():
                try:
                    strategy_signals = strategy.generate_signals(market_data)
                    signals[strategy.name] = strategy_signals
                except Exception as e:
                    self.logger.error(f"Signal generation failed for {strategy.name}: {str(e)}")
            
            results['signals_generated'] = True
            results['signal_count'] = len(signals)
            
            # 4. Execute trades with safety checks
            executed_trades = []
            for strategy_name, strategy_signals in signals.items():
                for symbol, signal in strategy_signals.items():
                    if signal != 0:  # Only trade on non-zero signals
                        try:
                            trade = Trade(
                                symbol=symbol,
                                strategy_id=strategy_name,
                                direction="BUY" if signal > 0 else "SELL",
                                size=1.0,  # Base size, will be adjusted by position sizing
                                entry_time=datetime.now(),
                                entry_price=float(market_data[symbol]['close'].iloc[-1])
                            )
                            
                            if self.live_trader.validate_trade(trade):
                                executed = self.live_trader.execute_trade(trade)
                                if executed:
                                    self.historian.save_trade(trade)
                                    executed_trades.append(trade)
                                    
                        except Exception as e:
                            self.logger.error(f"Trade execution failed for {symbol}: {str(e)}")
            
            results['trades_executed'] = True
            results['executed_trades_count'] = len(executed_trades)
            
            # 5. Update analytics and record data
            self.portfolio_mgr.update_positions(market_data)
            self.historian.save_market_data(market_data)
            self.historian.save_market_conditions(market_conditions)
            results['data_recorded'] = True
            
            # 6. Generate status report
            portfolio_state = self.portfolio_mgr.get_portfolio_state()
            report = self.reporter.generate_portfolio_report(
                portfolio_state=portfolio_state,
                trades=executed_trades,
                market_conditions=market_conditions
            )
            
            execution_time = (datetime.now() - cycle_start).total_seconds()
            self.logger.info(f"Cycle completed in {execution_time:.2f} seconds")
            
            results['execution_time'] = execution_time
            results['report'] = report
            return results
            
        except Exception as e:
            error_msg = f"Cycle failed: {str(e)}"
            self.logger.error(error_msg)
            self.live_trader.emergency_stop()
            results['error'] = error_msg
            return results

    def wait_for_next(self) -> None:
        """Sleep until next scheduled interval"""
        now = time.time()
        sleep_time = self.next_run - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.next_run += self.interval

    def run(self) -> None:
        """Run the trading loop indefinitely"""
        try:
            while True:
                self.run_single_iteration()
                self.wait_for_next()
        except KeyboardInterrupt:
            self.logger.info("Shutting down gracefully...")
            self.live_trader.emergency_stop()

def main():
    """Main entry point"""
    loop = FiveMinuteLoop()
    loop.run()

if __name__ == "__main__":
    main()