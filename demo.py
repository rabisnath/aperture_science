#!/usr/bin/env python3
"""
Interactive demo of the algorithmic trading package.
This script walks through key features while setting up PCA-based paper trading.
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required packages are installed."""
    try:
        from data import DataLoader, DataPreprocessor
        from market_analysis import MarketAnalyzer
        from strategy import StrategyConfig, StrategyFactory
        from portfolio_manager import PortfolioManager
        from back_of_house import LiveTrader
        from risk import RiskEngine
        from trading_types import Config
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        logger.error("Please install required packages using: pip install -r requirements.txt")
        return False

def print_section(title):
    """Print a section header."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def pause_for_user():
    """Pause for user to read output."""
    input("\nPress Enter to continue...\n")

def demo_data_acquisition():
    """Demonstrate data acquisition and preprocessing."""
    print_section("1. Data Acquisition and Preprocessing")
    
    try:
        from data import DataLoader, DataPreprocessor
        from trading_types import Config
    except ImportError as e:
        logger.error(f"Failed to import data modules: {str(e)}")
        raise
    
    print("Initializing data components...")
    config = Config()
    loader = DataLoader(config)
    preprocessor = DataPreprocessor()
    
    # Fetch some market data
    symbols = ["BTCUSDT", "ETHUSDT"]
    print(f"\nFetching 5-minute data for {symbols}...")
    try:
        market_data = loader.get_market_data(
            symbols=symbols,
            interval="5m",
            lookback=100
        )
    except Exception as e:
        logger.error(f"Failed to fetch market data: {str(e)}")
        raise
    
    if not all(symbol in market_data for symbol in symbols):
        missing = [s for s in symbols if s not in market_data]
        raise ValueError(f"Missing data for symbols: {missing}")
    
    print("\nSample of BTC market data:")
    print(market_data["BTCUSDT"].head())
    
    print("\nCalculating returns and creating features...")
    try:
        returns = preprocessor.calculate_returns(market_data["BTCUSDT"])
        print("\nSample returns:")
        print(returns.head())
    except Exception as e:
        logger.error(f"Failed to calculate returns: {str(e)}")
        raise
    
    pause_for_user()
    return market_data

def demo_market_analysis(market_data):
    """Demonstrate market analysis tools."""
    print_section("2. Market Analysis")
    
    try:
        from market_analysis import MarketAnalyzer
    except ImportError as e:
        logger.error(f"Failed to import market analysis module: {str(e)}")
        raise
    
    if not isinstance(market_data, dict) or not market_data:
        raise ValueError("Invalid market data provided")
    
    print("Analyzing market conditions...")
    analyzer = MarketAnalyzer()
    
    try:
        conditions = analyzer.analyze_market_conditions(market_data)
    except Exception as e:
        logger.error(f"Failed to analyze market conditions: {str(e)}")
        raise
    
    try:
        print("\nCurrent Market Statistics:")
        for stat, value in conditions.basic_stats.items():
            print(f"{stat.title()}: {value:.4f}")
        
        print("\nCorrelation Matrix:")
        corr_matrix = pd.DataFrame(
            conditions.correlation_matrix,
            index=market_data.keys(),
            columns=market_data.keys()
        )
        print(corr_matrix.round(4))
        
        # Validate correlation matrix
        if not np.allclose(corr_matrix, corr_matrix.T):
            logger.warning("Correlation matrix is not symmetric")
        if not np.allclose(np.diagonal(corr_matrix), 1.0):
            logger.warning("Correlation matrix diagonal is not 1.0")
            
    except Exception as e:
        logger.error(f"Failed to display market analysis results: {str(e)}")
        raise
    
    pause_for_user()
    return conditions

def demo_pca_strategy(market_data):
    """Demonstrate PCA strategy setup and signal generation."""
    print_section("3. PCA Strategy Implementation")
    
    try:
        from strategy import StrategyConfig, StrategyFactory
        from trading_types import Config
    except ImportError as e:
        logger.error(f"Failed to import strategy modules: {str(e)}")
        raise
    
    if not isinstance(market_data, dict) or not market_data:
        raise ValueError("Invalid market data provided")
    
    print("Creating PCA strategy configuration...")
    config = Config()
    
    # Validate number of components
    n_symbols = len(market_data)
    n_components = min(2, n_symbols - 1)  # Must be less than number of symbols
    
    try:
        pca_config = StrategyConfig(
            name="demo_pca",
            type="pca",
            parameters={
                'n_components': n_components,
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'lookback_period': 20
            },
            symbols=list(market_data.keys())
        )
    except Exception as e:
        logger.error(f"Failed to create strategy configuration: {str(e)}")
        raise
    
    print("\nStrategy Configuration:")
    print(f"- Type: {pca_config.type}")
    print(f"- Symbols: {', '.join(pca_config.symbols)}")
    print("- Parameters:")
    for key, value in pca_config.parameters.items():
        print(f"  - {key}: {value}")
    
    print("\nInitializing PCA strategy...")
    try:
        strategy = StrategyFactory.create(pca_config, config)
    except Exception as e:
        logger.error(f"Failed to create strategy: {str(e)}")
        raise
    
    print("\nGenerating trading signals...")
    try:
        signals = strategy.generate_signals(market_data)
        
        # Validate signals
        if not isinstance(signals, dict):
            raise ValueError("Invalid signal format")
        if set(signals.keys()) != set(market_data.keys()):
            raise ValueError("Signal symbols don't match market data")
            
        print("\nLatest signals:")
        for symbol, signal in signals.items():
            if not -1.0 <= signal <= 1.0:
                logger.warning(f"Unexpected signal value for {symbol}: {signal}")
            print(f"{symbol}: {signal:.2f}")
            
    except Exception as e:
        logger.error(f"Failed to generate signals: {str(e)}")
        raise
    
    pause_for_user()
    return strategy

def demo_portfolio_management(strategy):
    """Demonstrate portfolio management."""
    print_section("4. Portfolio Management")
    
    from portfolio_manager import PortfolioManager
    from trading_types import Config
    
    print("Initializing portfolio manager...")
    config = Config(initial_capital=100000.0)  # Start with $100k
    portfolio_mgr = PortfolioManager(config)
    
    print("\nAdding PCA strategy to portfolio...")
    portfolio_mgr.add_strategy(strategy)
    
    print("\nUpdating strategy allocations...")
    portfolio_mgr.update_allocations({"demo_pca": 1.0})  # Allocate 100% to PCA
    
    print("\nCurrent portfolio state:")
    state = portfolio_mgr.get_portfolio_state()
    print(f"Capital: ${state.capital:,.2f}")
    print(f"Positions: {state.positions}")
    print(f"Strategy Allocations: {state.strategy_allocations}")
    
    pause_for_user()
    return portfolio_mgr

def setup_paper_trading(strategy, portfolio_mgr):
    """Set up and start paper trading."""
    print_section("5. Paper Trading Setup")
    
    from back_of_house import LiveTrader
    from risk import RiskEngine
    
    print("Initializing trading components...")
    risk_engine = RiskEngine()
    live_trader = LiveTrader(portfolio_mgr.config, risk_engine)
    
    print("\nStarting paper trading...")
    live_trader.start_paper_trading()
    
    print("\nPaper trading is now active!")
    print("- PCA strategy is monitoring the market")
    print("- Risk engine is checking all trades")
    print("- Portfolio manager is tracking positions")
    
    return live_trader

def monitor_trading(portfolio_mgr, live_trader, duration_seconds=60):
    """Monitor paper trading for a specified duration."""
    print_section("6. Trading Monitor")
    
    print(f"Monitoring trading activity for {duration_seconds} seconds...")
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration_seconds:
            # Get current state
            state = portfolio_mgr.get_portfolio_state()
            active_trades = live_trader.get_active_trades()
            
            # Display status
            print("\nCurrent Status:")
            print(f"Portfolio Value: ${state.capital:,.2f}")
            print(f"Active Trades: {len(active_trades)}")
            print(f"Open Positions: {state.positions}")
            
            # Wait before next update
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    
    print("\nDemo complete! Your paper trading session is still running.")
    print("You can continue monitoring it using the live_trader and portfolio_mgr objects.")

def main():
    """Run the complete demo."""
    print_section("Algorithmic Trading Package Demo")
    print("This demo will walk you through the key features of the package")
    print("while setting up live paper trading with a PCA-based strategy.")
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    pause_for_user()
    
    try:
        # Run through each component
        market_data = demo_data_acquisition()
        conditions = demo_market_analysis(market_data)
        strategy = demo_pca_strategy(market_data)
        portfolio_mgr = demo_portfolio_management(strategy)
        live_trader = setup_paper_trading(strategy, portfolio_mgr)
        
        # Monitor trading
        monitor_trading(portfolio_mgr, live_trader)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise
    finally:
        # Cleanup
        if 'live_trader' in locals():
            try:
                live_trader.stop_paper_trading()
            except Exception as e:
                logger.error(f"Failed to stop paper trading: {str(e)}")
    
    print("\nDemo objects are available in the global scope:")
    print("- market_data: Latest market data")
    print("- strategy: Configured PCA strategy")
    print("- portfolio_mgr: Portfolio manager")
    print("- live_trader: Paper trading interface")

if __name__ == "__main__":
    main() 