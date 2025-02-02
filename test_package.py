#!/usr/bin/env python3
"""Comprehensive test suite for algorithmic trading package"""

import sys
import os
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import logging
from fpdf import FPDF
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import unittest
from unittest.mock import Mock, patch
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

# Import all package modules
from data import DataLoader, DataPreprocessor, DataError
from indicators import (
    IndicatorEngine, MovingAverage, RSI, MACD, BollingerBands,
    IndicatorRegistry
)
from market_analysis import MarketAnalyzer, MarketConditions
from historian import Historian
from modeling import BaseModel, PCAModel, RollingPCA, DynamicPCA
from strategy import (
    BaseStrategy, MeanReversionStrategy, MomentumStrategy,
    StrategyConfig, StrategyFactory
)
from portfolio_manager import (
    PortfolioManager, StrategySelectorModel, StrategyAllocation
)
from back_of_house import (
    RiskEngine, LiveTrader, BrokerInterface, TradeOrder
)
from reporting import ReportGenerator
from data_viz import TradingVisualizer, MarketVisualizer
from risk import RiskManager, RiskMetrics
from strategy_zoo import StrategyZoo, StrategyRange

# Import workflow scripts
from five_min_loop import FiveMinuteLoop
from daily_loop import DailyLoop

# Import common types
from trading_types import (
    Symbol, Trade, PortfolioState, Config, BrokerType,
    TradeDirection, TradingError, ValidationError,
    BrokerCredentials, TradeOrder, TradeStatus
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure plotting - use a simple built-in style
plt.style.use('classic')

class TestResults:
    """Container for test results and report generation"""
    
    def __init__(self):
        """Initialize test results container"""
        self.results = {}
        self.figures = {}
        self.output_dir = Path("test_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def add_result(self, module_name: str, passed: bool, error: str = None):
        """Record test result for a module"""
        self.results[module_name] = {
            'passed': passed,
            'error': error
        }
        
    def add_figure(self, name: str, fig: plt.Figure):
        """Save figure for report"""
        figure_path = self.output_dir / f"{name}.png"
        fig.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        self.figures[name] = figure_path
        
    def generate_report(self):
        """Generate PDF report of test results"""
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Algorithmic Trading Package Test Report', ln=True, align='C')
        pdf.ln(10)
        
        # Test Results Summary
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Test Results Summary', ln=True)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 12)
        for module, result in self.results.items():
            status = "✓ PASSED" if result['passed'] else "✗ FAILED"
            pdf.cell(0, 8, f"{module}: {status}", ln=True)
            if not result['passed'] and result['error']:
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 6, f"Error: {result['error']}", ln=True)
                pdf.set_font('Arial', '', 12)
        pdf.ln(10)
        
        # Visualizations
        if self.figures:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Test Visualizations', ln=True)
            pdf.ln(5)
            
            for name, path in self.figures.items():
                pdf.image(str(path), x=10, w=190)
                pdf.ln(5)
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 6, name, ln=True, align='C')
                pdf.ln(10)
        
        # Save report
        report_path = self.output_dir / "test_report.pdf"
        pdf.output(str(report_path))
        logger.info(f"Test report generated: {report_path}")

def assert_dataframe_structure(df, required_columns):
    """Verify DataFrame has required columns and proper index"""
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in required_columns)
    assert isinstance(df.index, pd.DatetimeIndex)

def assert_market_conditions_equal(cond1, cond2):
    """Compare two market condition objects"""
    assert cond1.timestamp == cond2.timestamp
    assert np.allclose(cond1.volatility, cond2.volatility)
    assert np.allclose(cond1.skewness, cond2.skewness)
    assert np.allclose(cond1.mean_return, cond2.mean_return)

def assert_valid_predictions(predictions):
    """Verify prediction output format and ranges"""
    assert isinstance(predictions, dict)
    for strategy, score in predictions.items():
        assert isinstance(strategy, str)
        assert 0 <= score <= 1

def assert_valid_report(report):
    """Verify report structure and content"""
    required_sections = ['portfolio_summary', 'strategy_performance', 'market_conditions']
    assert all(section in report for section in required_sections)

def test_data_module(results: TestResults):
    """Test Data module functionality"""
    try:
        # Initialize with environment variables
        loader = DataLoader()
        preprocessor = DataPreprocessor()
        
        # Test fetch_latest_data
        symbols = ["BTCUSDT", "ETHUSDT"]  # Use exchange format
        latest_data = loader.fetch_latest_data(symbols)
        assert isinstance(latest_data, dict)
        for symbol, df in latest_data.items():
            assert_dataframe_structure(df, ['open', 'high', 'low', 'close', 'volume'])
        
        # Test fetch_historical_data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        historical_data = loader.fetch_historical_data(symbols, start_date, end_date)
        for symbol, df in historical_data.items():
            assert_dataframe_structure(df, ['open', 'high', 'low', 'close', 'volume'])
            assert len(df) > 0
        
        # Test data preprocessing
        price_matrix = preprocessor.create_price_matrix(historical_data)
        return_matrix = preprocessor.create_return_matrix(price_matrix)
        assert isinstance(price_matrix, np.ndarray)
        assert isinstance(return_matrix, np.ndarray)
        assert return_matrix.shape[0] == price_matrix.shape[0] - 1  # Check return calculation
        
        results.add_result("Data Module", True)
        logger.info("✓ Data module tests passed")
        
    except Exception as e:
        results.add_result("Data Module", False, str(e))
        logger.error(f"✗ Data module tests failed: {e}")

def test_indicators_module(results: TestResults):
    """Test Indicators module functionality"""
    try:
        engine = IndicatorEngine()
        
        # Generate sample price data
        prices = pd.Series(
            np.random.randn(100).cumsum(),
            index=pd.date_range('2024-01-01', periods=100)
        )
        df = pd.DataFrame({
            'close': prices,
            'open': prices + np.random.randn(100) * 0.1,
            'high': prices + np.random.randn(100) * 0.2,
            'low': prices - np.random.randn(100) * 0.2,
            'volume': np.abs(np.random.randn(100)) * 1000
        })
        
        # Add and test indicators
        engine.add_indicator("sma", window=20)
        engine.add_indicator("rsi", period=14)
        engine.add_indicator("macd", fast=12, slow=26, signal=9)
        engine.add_indicator("bollinger", window=20, std_dev=2.0)
        
        result_df = engine.calculate(df)
        
        # Verify indicator outputs
        assert 'sma_20' in result_df.columns
        assert 'rsi_14' in result_df.columns
        assert 'macd_12_26_9_line' in result_df.columns
        assert 'bb_20_middle' in result_df.columns
        
        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        
        # Plot price and SMA
        result_df['close'].plot(ax=axes[0], label='Price')
        result_df['sma_20'].plot(ax=axes[0], label='SMA(20)')
        axes[0].set_title('Price and SMA')
        axes[0].legend()
        
        # Plot RSI
        result_df['rsi_14'].plot(ax=axes[1])
        axes[1].set_title('RSI(14)')
        axes[1].axhline(y=70, color='r', linestyle='--')
        axes[1].axhline(y=30, color='g', linestyle='--')
        
        # Plot MACD
        result_df['macd_12_26_9_line'].plot(ax=axes[2], label='MACD')
        result_df['macd_12_26_9_signal'].plot(ax=axes[2], label='Signal')
        axes[2].set_title('MACD')
        axes[2].legend()
        
        # Plot Bollinger Bands
        result_df['close'].plot(ax=axes[3], label='Price')
        result_df['bb_20_upper'].plot(ax=axes[3], label='Upper BB')
        result_df['bb_20_middle'].plot(ax=axes[3], label='Middle BB')
        result_df['bb_20_lower'].plot(ax=axes[3], label='Lower BB')
        axes[3].set_title('Bollinger Bands')
        axes[3].legend()
        
        plt.tight_layout()
        results.add_figure('indicators_test', fig)
        
        results.add_result("Indicators Module", True)
        logger.info("✓ Indicators module tests passed")
        
    except Exception as e:
        results.add_result("Indicators Module", False, str(e))
        logger.error(f"✗ Indicators module tests failed: {e}")

def test_market_analysis_module(results: TestResults):
    """Test Market Analysis module functionality"""
    try:
        analyzer = MarketAnalyzer()
        loader = DataLoader()
        
        # Get test data
        symbols = ["BTCUSDT", "ETHUSDT"]
        data = loader.fetch_historical_data(
            symbols,
            start=datetime.now() - timedelta(days=30),
            end=datetime.now()
        )
        
        # Test market conditions analysis
        conditions = analyzer.analyze_market_conditions(data)
        assert isinstance(conditions, MarketConditions)
        
        # Check basic statistics
        assert hasattr(conditions, 'basic_stats')
        assert all(stat in conditions.basic_stats for stat in ['volatility', 'skewness', 'mean_return'])

        # Check correlation matrix
        assert hasattr(conditions, 'correlation_matrix')
        assert isinstance(conditions.correlation_matrix, np.ndarray)
        assert conditions.correlation_matrix.shape == (len(symbols), len(symbols))
        
        # Check regression statistics
        assert hasattr(conditions, 'regression_stats')
        assert all(stat in conditions.regression_stats for stat in ['beta', 'r_squared'])

        # Visualize results
        fig = plot_market_conditions(conditions)
        assert isinstance(fig, go.Figure)
        
        results.add_result("Market Analysis Module", True)
        logger.info("✓ Market analysis tests passed")
        
    except Exception as e:
        results.add_result("Market Analysis Module", False, str(e))
        logger.error(f"✗ Market analysis tests failed: {e}")

def test_historian_module(results: TestResults):
    """Test Historian module functionality"""
    try:
        # Initialize historian with test storage path
        storage_path = Path("test_storage")
        historian = Historian(storage_path)
        
        # Test market conditions storage
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
        
        test_conditions = MarketConditions(
            timestamp=timestamp,
            basic_stats={
                'volatility': 0.15,
                'skewness': 0.05,
                'mean_return': 0.02
            },
            regression_stats={
                'beta': 1.2,
                'r_squared': 0.85
            },
            correlation_matrix=np.array([[1.0, 0.5], [0.5, 1.0]])
        )
        
        historian.save_market_conditions(test_conditions)
        loaded_data = historian.load_market_conditions(
            start=timestamp - timedelta(minutes=1),
            end=timestamp + timedelta(minutes=1)
        )
        
        assert 'basic_stats' in loaded_data
        assert 'regression_stats' in loaded_data
        assert 'correlation_matrices' in loaded_data
        assert len(loaded_data['correlation_matrices']) > 0
        
        # Test trade record storage
        test_trade = Trade(
            symbol="BTC-USDT",
            strategy_id="test_strategy",
            direction="BUY",
            size=1.0,
            entry_time=timestamp,
            entry_price=50000.0,
            exit_time=timestamp + timedelta(minutes=5),
            exit_price=51000.0,
            pnl=1000.0,
            status="CLOSED"
        )
        
        historian.save_trade(test_trade)
        loaded_trades = historian.load_trades(
            start=timestamp - timedelta(minutes=1),
            end=timestamp + timedelta(minutes=10)
        )
        
        assert len(loaded_trades) > 0
        assert loaded_trades.iloc[0]['symbol'] == test_trade.symbol
        assert loaded_trades.iloc[0]['pnl'] == test_trade.pnl
        
        # Clean up test data
        historian.clean_old_data(retention_days=0)
        
        results.add_result("Historian Module", True)
        logger.info("✓ Historian module tests passed")
        
    except Exception as e:
        results.add_result("Historian Module", False, str(e))
        logger.error(f"✗ Historian module tests failed: {e}")

def test_modeling_module(results: TestResults):
    """Test Modeling module functionality"""
    try:
        # Test PCA model
        n_components = 3
        pca_model = PCAModel(n_components=n_components)
        
        # Generate sample return data
        n_assets = 5
        n_periods = 100
        returns = np.random.randn(n_periods, n_assets) * 0.02
        
        # Fit and transform
        pca_model.fit(returns)
        transformed = pca_model.transform(returns)
        
        # Verify PCA results
        assert transformed.shape[1] <= returns.shape[1]
        assert pca_model.components is not None
        assert pca_model.components.shape == (n_components, n_assets)
        
        # Visualize results
        fig = make_subplots(rows=1, cols=1)
        
        # Plot transformed data
        for i in range(transformed.shape[1]):
            fig.add_trace(
                go.Scatter(y=transformed[:, i], name=f'PC{i+1}'),
                row=1, col=1
            )
        
        fig.update_layout(height=400, title='PCA Results')
        
        results.add_result("Modeling Module", True)
        logger.info("✓ Modeling module tests passed")
        
    except Exception as e:
        results.add_result("Modeling Module", False, str(e))
        logger.error(f"✗ Modeling module tests failed: {e}")

def test_strategy_module(results: TestResults):
    """Test Strategy module functionality"""
    try:
        # Test Mean Reversion Strategy
        mean_rev_config = StrategyConfig(
            name="test_mean_rev",
            type="mean_reversion",
            parameters={
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'lookback_period': 20
            },
            symbols=["BTC-USDT", "ETH-USDT"]
        )
        mean_rev = MeanReversionStrategy(
            name=mean_rev_config.name,
            symbols=mean_rev_config.symbols,
            entry_threshold=mean_rev_config.parameters['entry_threshold'],
            exit_threshold=mean_rev_config.parameters['exit_threshold']
        )
        
        # Test PCA Strategy
        pca_model = PCAModel(n_components=3)
        pca_config = StrategyConfig(
            name="test_pca",
            type="pca",
            parameters={
                'entry_threshold': 1.0,
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            },
            symbols=["BTC-USDT", "ETH-USDT"]
        )
        pca_strategy = PCAStrategy(
            name=pca_config.name,
            symbols=pca_config.symbols,
            model=pca_model,
            entry_threshold=pca_config.parameters['entry_threshold']
        )
        
        # Generate sample market data
        dates = pd.date_range('2024-01-01', periods=100)
        sample_data = {
            "BTC-USDT": pd.DataFrame({
                'open': np.random.randn(100).cumsum(),
                'high': np.random.randn(100).cumsum(),
                'low': np.random.randn(100).cumsum(),
                'close': np.random.randn(100).cumsum(),
                'volume': np.abs(np.random.randn(100))
            }, index=dates),
            "ETH-USDT": pd.DataFrame({
                'open': np.random.randn(100).cumsum(),
                'high': np.random.randn(100).cumsum(),
                'low': np.random.randn(100).cumsum(),
                'close': np.random.randn(100).cumsum(),
                'volume': np.abs(np.random.randn(100))
            }, index=dates)
        }
        
        # Test signal generation
        mean_rev_signals = mean_rev.generate_signals(sample_data)
        pca_signals = pca_strategy.generate_signals(sample_data)
        
        assert isinstance(mean_rev_signals, dict)
        assert isinstance(pca_signals, dict)
        assert all(signal in [-1.0, 0.0, 1.0] for signal in mean_rev_signals.values())
        assert all(signal in [-1.0, 0.0, 1.0] for signal in pca_signals.values())
        
        # Test performance tracking
        test_trade = Trade(
            symbol="BTC-USDT",
            strategy_id=mean_rev.name,
            direction="BUY",
            size=1.0,
            entry_time=dates[0],
            entry_price=50000.0,
            exit_time=dates[1],
            exit_price=51000.0,
            pnl=1000.0,
            status="CLOSED"
        )
        
        mean_rev.update_performance(test_trade)
        assert 'sharpe_ratio' in mean_rev.performance_metrics
        assert 'win_rate' in mean_rev.performance_metrics
        assert 'total_pnl' in mean_rev.performance_metrics
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot signals
        pd.Series(mean_rev_signals).plot(ax=ax1, label='Mean Reversion')
        pd.Series(pca_signals).plot(ax=ax1, label='PCA')
        ax1.set_title('Strategy Signals')
        ax1.legend()
        
        # Plot performance metrics
        pd.Series(mean_rev.performance_metrics).plot(
            kind='bar', ax=ax2, title='Strategy Performance'
        )
        
        plt.tight_layout()
        results.add_figure('strategy_test', fig)
        
        results.add_result("Strategy Module", True)
        logger.info("✓ Strategy module tests passed")
        
    except Exception as e:
        results.add_result("Strategy Module", False, str(e))
        logger.error(f"✗ Strategy module tests failed: {e}")

def test_testing_module(results: TestResults):
    """Test Testing module functionality"""
    try:
        # Create test strategy
        strategy = MeanReversionStrategy(
            name="test_strategy",
            symbols=["BTC-USDT", "ETH-USDT"],
            entry_threshold=2.0,
            exit_threshold=0.5
        )
        
        # Create backtest configuration
        config = BacktestConfiguration(
            strategy_name=strategy.name,
            initial_capital=100000.0,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 31),
            symbols=strategy.symbols,
            transaction_costs=0.001,
            risk_free_rate=0.02
        )
        
        # Initialize backtester
        backtester = Backtester(strategy, config)
        
        # Generate sample historical data
        dates = pd.date_range(config.start_date, config.end_date, freq='H')
        historical_data = {
            "BTC-USDT": pd.DataFrame({
                'open': np.random.randn(len(dates)).cumsum(),
                'high': np.random.randn(len(dates)).cumsum(),
                'low': np.random.randn(len(dates)).cumsum(),
                'close': np.random.randn(len(dates)).cumsum(),
                'volume': np.abs(np.random.randn(len(dates)))
            }, index=dates),
            "ETH-USDT": pd.DataFrame({
                'open': np.random.randn(len(dates)).cumsum(),
                'high': np.random.randn(len(dates)).cumsum(),
                'low': np.random.randn(len(dates)).cumsum(),
                'close': np.random.randn(len(dates)).cumsum(),
                'volume': np.abs(np.random.randn(len(dates)))
            }, index=dates)
        }
        
        # Run backtest
        backtest_result = backtester.run_backtest(historical_data)
        
        # Verify results
        assert isinstance(backtest_result, BacktestResult)
        assert backtest_result.strategy_name == strategy.name
        assert isinstance(backtest_result.total_return, float)
        assert isinstance(backtest_result.sharpe_ratio, float)
        assert isinstance(backtest_result.max_drawdown, float)
        assert isinstance(backtest_result.total_trades, int)
        assert isinstance(backtest_result.win_rate, float)
        assert isinstance(backtest_result.trades, list)
        assert isinstance(backtest_result.equity_curve, pd.Series)
        
        # Test walk-forward analysis
        walk_forward = WalkForwardAnalyzer(
            strategy=strategy,
            n_splits=5,
            train_size=180,
            test_size=30
        )
        
        wf_results = walk_forward.run_analysis(
            historical_data=historical_data,
            initial_capital=config.initial_capital
        )
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot equity curve
        backtest_result.equity_curve.plot(
            ax=ax1, title='Backtest Equity Curve'
        )
        
        # Plot trade results
        trade_results = pd.DataFrame([
            {
                'entry_time': t.entry_time,
                'pnl': t.pnl,
                'symbol': t.symbol
            }
            for t in backtest_result.trades
        ]).set_index('entry_time')
        
        trade_results['pnl'].plot(
            ax=ax2, kind='bar', title='Trade Results'
        )
        
        # Plot walk-forward results
        pd.DataFrame(wf_results['performance']).plot(
            ax=ax3, title='Walk-Forward Performance'
        )
        
        plt.tight_layout()
        results.add_figure('testing_test', fig)
        
        results.add_result("Testing Module", True)
        logger.info("✓ Testing module tests passed")
        
    except Exception as e:
        results.add_result("Testing Module", False, str(e))
        logger.error(f"✗ Testing module tests failed: {e}")

def test_portfolio_manager_module(results: TestResults):
    """Test Portfolio Manager module functionality"""
    try:
        # Initialize portfolio manager with configuration
        config = PortfolioConfiguration(
            initial_capital=100000.0,
            max_position_size=0.1,
            risk_free_rate=0.02,
            rebalancing_frequency='daily',
            max_strategies=10,
            max_strategy_allocation=0.2
        )
        portfolio_mgr = PortfolioManager(config)
        
        # Test initial portfolio state
        state = portfolio_mgr.get_portfolio_state()
        assert isinstance(state, PortfolioState)
        assert state.capital == config.initial_capital
        assert isinstance(state.positions, dict)
        assert isinstance(state.strategy_allocations, dict)
        
        # Test position tracking
        portfolio_mgr.update_position("BTC-USDT", 1.0, 50000.0)
        positions = portfolio_mgr.get_positions()
        assert "BTC-USDT" in positions
        assert positions["BTC-USDT"]["quantity"] == 1.0
        
        # Test strategy management
        strategy = MeanReversionStrategy(
            name="test_strategy",
            symbols=["BTC-USDT"],
            entry_threshold=2.0,
            exit_threshold=0.5
        )
        portfolio_mgr.add_strategy(strategy)
        assert strategy.name in portfolio_mgr.get_active_strategies()
        
        # Test allocation updates
        allocations = {"test_strategy": 0.5}
        portfolio_mgr.update_allocations(allocations)
        current_allocations = portfolio_mgr.get_allocations()
        assert current_allocations["test_strategy"] == 0.5
        assert current_allocations["test_strategy"] <= config.max_strategy_allocation
        
        # Test strategy selection
        historical_data = pd.DataFrame({
            'volatility': np.random.randn(100),
            'trend': np.random.randn(100),
            'momentum': np.random.randn(100)
        })
        selected_strategies = portfolio_mgr.select_live_strategies(historical_data)
        assert isinstance(selected_strategies, list)
        assert len(selected_strategies) <= config.max_strategies
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot positions
        position_data = pd.Series({k: v['quantity'] for k, v in positions.items()})
        position_data.plot(kind='bar', ax=ax1, title='Current Positions')
        
        # Plot strategy allocations
        pd.Series(current_allocations).plot(
            kind='pie', ax=ax2, title='Strategy Allocations'
        )
        
        plt.tight_layout()
        results.add_figure('portfolio_test', fig)
        
        results.add_result("Portfolio Manager Module", True)
        logger.info("✓ Portfolio Manager module tests passed")
        
    except Exception as e:
        results.add_result("Portfolio Manager Module", False, str(e))
        logger.error(f"✗ Portfolio Manager module tests failed: {e}")

def test_back_office_operations(results: TestResults):
    """Test back office operations including trade execution and risk management"""
    try:
        # Create mock broker implementation
        class MockBroker(BrokerInterface):
            def _validate_credentials(self) -> bool:
                return True
                
            def get_account_balance(self) -> Dict[str, float]:
                return {"USDT": 10000.0}
                
            def place_order(self, order: TradeOrder) -> TradeOrder:
                order.status = "EXECUTED"
                order.execution_price = order.price
                return order
        
        # Test broker interface
        credentials = BrokerCredentials(
            api_key="test_key",
            api_secret="test_secret",
            passphrase="test_pass"
        )
        broker = MockBroker(credentials)
        assert broker.connect(), "Broker connection should succeed"
        
        # Test risk engine
        risk_engine = RiskEngine(
            max_portfolio_risk=0.02,
            max_single_position_risk=0.005
        )
        
        # Test position size calculation
        position_size = risk_engine.calculate_position_size(
            symbol_price=100.0,
            account_balance=10000.0,
            risk_percentage=0.01
        )
        assert position_size > 0, "Position size should be positive"
        assert position_size <= 100.0, "Position size should not exceed account balance / price"
        
        # Test trade safety checks
        test_order = TradeOrder(
            symbol="BTCUSDT",
            direction="BUY",
            order_type="MARKET",
            quantity=1.0,
            price=50000.0,
            strategy_id="test_strategy"
        )
        assert risk_engine.check_trade_safety(test_order), "Trade should pass safety checks"
        
        # Test live trader
        live_trader = LiveTrader(broker, risk_engine)
        
        # Test trade execution
        test_signals = {
            "test_strategy": {
                "BTCUSDT": 1.0,  # Buy signal
                "ETHUSDT": -1.0  # Sell signal
            }
        }
        executed_trades = live_trader.execute_trades(test_signals)
        assert len(executed_trades) == 2, "Should execute both trades"
        assert all(trade.status == "EXECUTED" for trade in executed_trades.values())
        
        # Test emergency stop
        live_trader.emergency_stop()
        assert len(live_trader._active_trades) == 0, "Should clear all active trades"
        
        results.add_result("Back Office Operations", True)
        logger.info("✓ Back office operations tests passed")
        
    except Exception as e:
        results.add_result("Back Office Operations", False, str(e))
        logger.error(f"✗ Back office operations tests failed: {e}")

def test_workflows(results: TestResults):
    """Test trading workflows"""
    try:
        # Initialize components with Alpaca
        config = Config(default_broker=BrokerType.ALPACA)
        data_loader = DataLoader(config)
        market_analyzer = MarketAnalyzer()
        historian = Historian(Path("test_storage"))
        portfolio_config = PortfolioConfiguration(
            initial_capital=100000.0,
            max_strategies=2,
            max_strategy_allocation=0.5
        )
        portfolio_mgr = PortfolioManager(portfolio_config)
        
        # Create mock Alpaca broker
        class MockAlpacaBroker(BrokerInterface):
            def _validate_credentials(self) -> bool:
                return True
            def get_account_balance(self) -> Dict[str, float]:
                return {"USD": 10000.0}
            def place_order(self, order: TradeOrder) -> TradeOrder:
                order.status = "EXECUTED"
                order.execution_price = order.price
                return order
        
        broker = MockAlpacaBroker(BrokerCredentials("test", "test"))
        risk_engine = RiskEngine()
        live_trader = LiveTrader(broker, risk_engine)
        
        # Test Five Minute Loop
        five_min_loop = FiveMinuteLoop()
        five_min_loop.data_loader = data_loader
        five_min_loop.market_analyzer = market_analyzer
        five_min_loop.historian = historian
        five_min_loop.portfolio_mgr = portfolio_mgr
        five_min_loop.live_trader = live_trader
        
        iteration_result = five_min_loop.run_single_iteration()
        assert isinstance(iteration_result, dict)
        assert 'market_data_updated' in iteration_result
        assert 'signals_generated' in iteration_result
        assert 'trades_executed' in iteration_result
        assert 'data_recorded' in iteration_result
        
        results.add_result("Five Minute Loop", True)
        logger.info("✓ Five minute loop test passed")
        
        # Test Daily Loop
        daily_loop = DailyLoop(initial_capital=portfolio_config.initial_capital)
        daily_loop.data_loader = data_loader
        daily_loop.market_analyzer = market_analyzer
        daily_loop.portfolio_mgr = portfolio_mgr
        daily_loop.historian = historian
        
        # Add some test strategies
        mean_rev = MeanReversionStrategy(
            name="test_mean_rev",
            symbols=["BTC/USD"],
            entry_threshold=2.0,
            exit_threshold=0.5
        )
        pca_strat = PCAStrategy(
            name="test_pca",
            symbols=["BTC/USD"],
            model=PCAModel(n_components=2),
            entry_threshold=1.0
        )
        portfolio_mgr.add_strategy(mean_rev)
        portfolio_mgr.add_strategy(pca_strat)
        
        iteration_result = daily_loop.run_single_iteration()
        assert isinstance(iteration_result, dict)
        assert 'market_conditions_analyzed' in iteration_result
        assert 'model_updated' in iteration_result
        assert 'strategies_selected' in iteration_result
        assert 'portfolio_rebalanced' in iteration_result
        assert 'report_generated' in iteration_result
        
        results.add_result("Daily Loop", True)
        logger.info("✓ Daily loop test passed")
        
    except Exception as e:
        if 'Five Minute Loop' not in results.results:
            results.add_result("Five Minute Loop", False, str(e))
            logger.error(f"✗ Five minute loop test failed: {e}")
        if 'Daily Loop' not in results.results:
            results.add_result("Daily Loop", False, str(e))
            logger.error(f"✗ Daily loop test failed: {e}")
        
    finally:
        # Clean up test data
        if 'historian' in locals():
            historian.clean_old_data(retention_days=0)

class MockAlpacaClient:
    """Mock Alpaca client for testing"""
    
    def __init__(self):
        """Initialize mock client"""
        self.positions = []
        self.orders = []
        
    def get_crypto_bars(self, request: CryptoBarsRequest):
        """Mock crypto bars data"""
        # Convert timeframe to pandas frequency string
        if isinstance(request.timeframe, TimeFrame):
            if request.timeframe == TimeFrame.Minute:
                freq = '1min'
            elif request.timeframe == TimeFrame.Hour:
                freq = '1H'
            elif request.timeframe == TimeFrame.Day:
                freq = '1D'
            else:
                # Handle custom timeframes
                value = request.timeframe.value
                unit = request.timeframe.unit
                if unit == TimeFrameUnit.Minute:
                    freq = f'{value}min'
                elif unit == TimeFrameUnit.Hour:
                    freq = f'{value}H'
                elif unit == TimeFrameUnit.Day:
                    freq = f'{value}D'
                else:
                    freq = '5min'  # Default to 5min if unknown
        else:
            freq = '5min'  # Default frequency
            
        dates = pd.date_range(request.start, request.end, freq=freq)
        
        # Create mock data for each requested symbol
        if isinstance(request.symbol_or_symbols, list):
            symbols = request.symbol_or_symbols
        else:
            symbols = [request.symbol_or_symbols]
            
        data_frames = []
        for symbol in symbols:
            df = pd.DataFrame({
                'symbol': symbol,
                'Open': np.random.randn(len(dates)).cumsum() + 100,
                'High': np.random.randn(len(dates)).cumsum() + 102,
                'Low': np.random.randn(len(dates)).cumsum() + 98,
                'Close': np.random.randn(len(dates)).cumsum() + 100,
                'Volume': np.abs(np.random.randn(len(dates))) * 1000,
                'Trades': np.random.randint(100, 1000, len(dates)),
                'VWAP': np.random.randn(len(dates)).cumsum() + 100
            }, index=dates)
            data_frames.append(df)
            
        # Combine all data frames
        if data_frames:
            result = pd.concat(data_frames)
            result.index.name = 'timestamp'
            return type('Bars', (), {'df': result})()
        return type('Bars', (), {'df': pd.DataFrame()})()
        
    def get_all_positions(self):
        """Mock positions data"""
        return self.positions
        
    def get_all_assets(self, request: GetAssetsRequest = None):
        """Mock assets data"""
        return [
            {
                'symbol': 'BTC/USD',
                'name': 'Bitcoin',
                'status': 'active',
                'tradable': True,
                'marginable': False,
                'maintenance_margin_requirement': 0,
                'shortable': False,
                'easy_to_borrow': False,
                'fractionable': True,
                'min_order_size': '0.0001',
                'min_trade_increment': '0.0001',
                'price_increment': '0.01'
            },
            {
                'symbol': 'ETH/USD',
                'name': 'Ethereum',
                'status': 'active',
                'tradable': True,
                'marginable': False,
                'maintenance_margin_requirement': 0,
                'shortable': False,
                'easy_to_borrow': False,
                'fractionable': True,
                'min_order_size': '0.001',
                'min_trade_increment': '0.001',
                'price_increment': '0.01'
            }
        ]

class TestDataModule(unittest.TestCase):
    """Test data acquisition and preprocessing"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = Config()
        self.data_loader = DataLoader(self.config)
        self.preprocessor = DataPreprocessor()
        self.test_symbols = ["BTC/USD", "ETH/USD"]  # Alpaca crypto symbols
        
        # Initialize strategy configurations
        self.mean_rev_config = StrategyConfig(
            name="test_mean_rev",
            type="mean_reversion",
            parameters={
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'lookback_period': 20
            },
            symbols=self.test_symbols
        )
        
        self.momentum_config = StrategyConfig(
            name="test_momentum",
            type="momentum",
            parameters={
                'lookback_period': 20,
                'entry_threshold': 0.02,
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            },
            symbols=self.test_symbols
        )
        
        # Skip tests if API keys not configured
        if not os.environ.get('ALPACA_API_KEY') or not os.environ.get('ALPACA_API_SECRET'):
            self.skipTest("Alpaca API credentials not configured")
            
        # Set up mock clients for testing
        self.mock_trading_client = MockAlpacaClient()
        self.mock_data_client = MockAlpacaClient()
        
        # Patch the Alpaca clients
        self.trading_client_patcher = patch('alpaca.trading.client.TradingClient')
        self.data_client_patcher = patch('alpaca.data.historical.StockHistoricalDataClient')
        
        self.mock_trading_client = self.trading_client_patcher.start()
        self.mock_data_client = self.data_client_patcher.start()
    
    def tearDown(self):
        """Clean up after tests"""
        self.trading_client_patcher.stop()
        self.data_client_patcher.stop()
    
    def test_alpaca_historical_data(self):
        """Test fetching historical data from Alpaca"""
        try:
            # Test single symbol data fetch
            df = self.data_loader.get_historical_data(
                symbol="BTC/USD",
                interval="5m",
                start_time=datetime.now() - timedelta(days=1),
                end_time=datetime.now()
            )
            
            self.assertIsInstance(df, pd.DataFrame)
            self.assertTrue(len(df) > 0)
            self.assertTrue(all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']))
            
        except DataError as e:
            self.skipTest(f"Alpaca API error: {str(e)}")
    
    def test_market_data_fetch(self):
        """Test fetching current market data"""
        try:
            data = self.data_loader.get_market_data(
                symbols=self.test_symbols,
                interval="5m",
                lookback=100
            )
            
            self.assertIsInstance(data, dict)
            self.assertEqual(len(data), len(self.test_symbols))
            
            for symbol in self.test_symbols:
                df = data[symbol]
                self.assertIsInstance(df, pd.DataFrame)
                self.assertTrue(len(df) > 0)
                self.assertTrue(all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']))
                
        except DataError as e:
            self.skipTest(f"Market data fetch error: {str(e)}")
    
    def test_data_preprocessing(self):
        """Test data preprocessing functions"""
        # Create sample data
        data = {
            'Open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'Close': [102.0, 103.0, 104.0, 105.0, 106.0],
            'Volume': [1000, 1100, 1200, 1300, 1400]
        }
        df = pd.DataFrame(data)
        
        # Test returns calculation
        returns = self.preprocessor.calculate_returns(df)
        self.assertIsInstance(returns, pd.Series)
        self.assertEqual(len(returns), len(df))
        
        # Test technical indicators
        df_indicators = self.preprocessor.add_technical_indicators(df)
        self.assertTrue(all(col in df_indicators.columns for col in ['sma_20', 'rsi', 'macd']))
        
    def test_high_volume_symbols(self):
        """Test getting high volume symbols"""
        try:
            symbols = self.data_loader.get_high_volume_symbols(N=5, base_currency='USD')
            
            self.assertIsInstance(symbols, list)
            self.assertTrue(len(symbols) > 0)
            self.assertTrue(all(isinstance(s, str) for s in symbols))
            self.assertTrue(all('USD' in s for s in symbols))
            
        except DataError as e:
            self.skipTest(f"High volume symbols fetch error: {str(e)}")

def main():
    """Run all tests and generate report"""
    results = TestResults()
    
    # Run module tests
    test_data_module(results)
    test_indicators_module(results)
    test_market_analysis_module(results)
    test_historian_module(results)
    test_modeling_module(results)
    test_strategy_module(results)
    test_testing_module(results)
    test_portfolio_manager_module(results)
    test_back_office_operations(results)
    test_workflows(results)
    
    # Generate test report
    results.generate_report()
    
    # Print summary
    print("\nTest Summary:")
    passed = sum(1 for result in results.results.values() if result['passed'])
    total = len(results.results)
    print(f"Passed: {passed}/{total} tests")
    
    if passed < total:
        print("\nFailed Tests:")
        for module, result in results.results.items():
            if not result['passed']:
                print(f"- {module}: {result['error']}")
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    unittest.main() 