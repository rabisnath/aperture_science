"""
Backtesting and validation module for trading strategies.
Provides frameworks for testing strategy performance and reliability.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from trading_types import (
    BacktestConfiguration, BacktestResult, Trade, 
    Candle_Bundle, MarketConditions
)
from strategy import BaseStrategy
from market_analysis import MarketAnalyzer
from portfolio_manager import PortfolioManager
from back_of_house import RiskEngine

class Backtester:
    """Comprehensive strategy backtesting engine"""
    
    def __init__(
        self, 
        strategy: BaseStrategy,
        config: BacktestConfiguration
    ):
        """Initialize backtesting environment
        
        Args:
            strategy: Trading strategy to evaluate
            config: Backtesting configuration parameters
        """
        self.strategy = strategy
        self.config = config
        self.portfolio_manager = PortfolioManager(
            initial_capital=config.initial_capital
        )
        self.risk_engine = RiskEngine()
        self.market_analyzer = MarketAnalyzer()

    def run_backtest(self, historical_data: Candle_Bundle) -> BacktestResult:
        """Execute full backtest on historical data
        
        Args:
            historical_data: Historical market data
        
        Returns:
            Comprehensive backtest results
        """
        try:
            # Filter data to match configuration dates
            filtered_data = {
                symbol: df[
                    (df.index >= self.config.start_date) & 
                    (df.index <= self.config.end_date)
                ] 
                for symbol, df in historical_data.items()
            }

            current_capital = self.config.initial_capital
            trade_log = []
            equity_curve = []
            
            # Iterate through historical data
            timestamps = filtered_data[self.strategy.symbols[0]].index[
                self.strategy.lookback:
            ]
            
            for timestamp in timestamps:
                # Get historical window of data
                window_data = {
                    symbol: df[:timestamp].iloc[-self.strategy.lookback:]
                    for symbol, df in filtered_data.items()
                }
                
                # Generate trading signals
                signals = self.strategy.generate_signals(window_data)
                
                # Execute trades based on signals
                for symbol, signal in signals.items():
                    if signal != 0:
                        trade = self._execute_simulated_trade(
                            symbol=symbol,
                            signal=signal,
                            timestamp=timestamp,
                            current_price=filtered_data[symbol].loc[timestamp, 'close'],
                            available_capital=current_capital
                        )
                        
                        if trade:
                            trade_log.append(trade)
                            current_capital += trade.pnl
                
                equity_curve.append({
                    'timestamp': timestamp,
                    'equity': current_capital
                })
            
            # Create equity curve series
            equity_df = pd.DataFrame(equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            # Calculate final performance metrics
            total_trades = len(trade_log)
            winning_trades = sum(1 for trade in trade_log if trade.pnl > 0)
            
            return BacktestResult(
                strategy_name=self.strategy.name,
                total_return=(current_capital - self.config.initial_capital) 
                           / self.config.initial_capital,
                sharpe_ratio=self._calculate_sharpe_ratio(equity_df),
                max_drawdown=self._calculate_max_drawdown(equity_df),
                total_trades=total_trades,
                win_rate=winning_trades / total_trades if total_trades > 0 else 0,
                trades=trade_log,
                equity_curve=equity_df['equity']
            )
            
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            raise

    def _execute_simulated_trade(
        self,
        symbol: str,
        signal: float,
        timestamp: datetime,
        current_price: float,
        available_capital: float
    ) -> Optional[Trade]:
        """Simulate trade execution with risk management
        
        Args:
            symbol: Trading symbol
            signal: Trading signal (-1, 0, 1)
            timestamp: Current timestamp
            current_price: Current market price
            available_capital: Current portfolio capital
        
        Returns:
            Simulated trade or None
        """
        # Apply position sizing
        position_size = self.risk_engine.calculate_position_size(
            symbol_price=current_price,
            account_balance=available_capital,
            risk_percentage=0.01  # 1% risk per trade
        )
        
        # Create trade record
        direction = "LONG" if signal > 0 else "SHORT"
        entry_price = current_price
        
        # Simulate exit (simplified)
        exit_price = entry_price * (1 + (0.02 * signal))  # Simplified 2% move
        pnl = (exit_price - entry_price) * position_size
        if direction == "SHORT":
            pnl = -pnl
        
        return Trade(
            symbol=symbol,
            direction=direction,
            size=position_size,
            entry_time=timestamp,
            entry_price=entry_price,
            exit_time=timestamp + timedelta(hours=1),  # Simplified exit
            exit_price=exit_price,
            strategy_id=self.strategy.name,
            pnl=pnl,
            status="EXECUTED"
        )

    def _calculate_sharpe_ratio(self, equity_df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio from equity curve
        
        Args:
            equity_df: DataFrame with equity curve
            
        Returns:
            Annualized Sharpe ratio
        """
        returns = equity_df['equity'].pct_change().dropna()
        
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (self.config.risk_free_rate / 252)
        
        if excess_returns.std() == 0:
            return 0.0
            
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

    def _calculate_max_drawdown(self, equity_df: pd.DataFrame) -> float:
        """Calculate maximum drawdown from equity curve
        
        Args:
            equity_df: DataFrame with equity curve
            
        Returns:
            Maximum drawdown percentage
        """
        rolling_max = equity_df['equity'].expanding(min_periods=1).max()
        drawdowns = equity_df['equity'] / rolling_max - 1
        return abs(float(drawdowns.min()))

class WalkForwardAnalyzer:
    """Walk-forward testing framework"""
    
    def __init__(
        self,
        strategy: BaseStrategy,
        n_splits: int = 5,
        train_size: int = 180,
        test_size: int = 30
    ):
        """Initialize walk-forward analysis
        
        Args:
            strategy: Strategy to analyze
            n_splits: Number of train/test splits
            train_size: Training window in days
            test_size: Testing window in days
        """
        self.strategy = strategy
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        
    def run_analysis(
        self,
        historical_data: Candle_Bundle,
        initial_capital: float = 100000
    ) -> Dict[str, Any]:
        """Perform walk-forward analysis
        
        Args:
            historical_data: Historical market data
            initial_capital: Starting capital for each test
            
        Returns:
            Analysis results dictionary
        """
        results = []
        
        # Create time series splits
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=self.test_size
        )
        
        # Get common timestamps
        timestamps = historical_data[self.strategy.symbols[0]].index
        
        for train_idx, test_idx in tscv.split(timestamps):
            # Split data
            train_data = {
                symbol: df.iloc[train_idx]
                for symbol, df in historical_data.items()
            }
            test_data = {
                symbol: df.iloc[test_idx]
                for symbol, df in historical_data.items()
            }
            
            # Create backtester instance
            config = BacktestConfiguration(
                strategy_name=self.strategy.name,
                initial_capital=initial_capital,
                start_date=timestamps[test_idx[0]],
                end_date=timestamps[test_idx[-1]],
                symbols=self.strategy.symbols
            )
            
            backtester = Backtester(self.strategy, config)
            
            # Run backtest on out-of-sample data
            backtest_result = backtester.run_backtest(test_data)
            results.append(backtest_result)
        
        # Aggregate results
        return {
            'splits': len(results),
            'mean_return': np.mean([r.total_return for r in results]),
            'std_return': np.std([r.total_return for r in results]),
            'mean_sharpe': np.mean([r.sharpe_ratio for r in results]),
            'mean_drawdown': np.mean([r.max_drawdown for r in results]),
            'win_rate': np.mean([r.win_rate for r in results]),
            'results': results
        }

class MonteCarloAnalyzer:
    """Monte Carlo simulation for strategy analysis"""
    
    def __init__(
        self,
        n_simulations: int = 1000,
        confidence_level: float = 0.95
    ):
        """Initialize Monte Carlo analyzer
        
        Args:
            n_simulations: Number of simulations to run
            confidence_level: Confidence level for metrics
        """
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
    
    def analyze_returns(
        self,
        returns: pd.Series,
        initial_capital: float
    ) -> Dict[str, float]:
        """Run Monte Carlo analysis on return series
        
        Args:
            returns: Historical returns
            initial_capital: Starting capital
            
        Returns:
            Analysis metrics dictionary
        """
        # Generate random paths
        paths = np.random.choice(
            returns,
            size=(self.n_simulations, len(returns)),
            replace=True
        )
        
        # Calculate cumulative paths
        cum_paths = np.cumprod(1 + paths, axis=1)
        final_values = cum_paths[:, -1] * initial_capital
        
        # Calculate metrics
        conf_level = int(self.confidence_level * 100)
        var = np.percentile(final_values, 100 - conf_level)
        es = np.mean(final_values[final_values <= var])
        
        return {
            'mean_final': float(np.mean(final_values)),
            'median_final': float(np.median(final_values)),
            f'var_{conf_level}': float(var),
            'expected_shortfall': float(es),
            'upside_potential': float(np.percentile(final_values, 95)),
            'downside_risk': float(np.percentile(final_values, 5))
        }