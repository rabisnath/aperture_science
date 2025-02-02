"""
Strategy module for algorithmic trading.
Provides base strategy class and implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
from datetime import datetime
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from trading_types import (
    Config, Trade, TradeOrder, ValidationError
)

from modeling import PCAModel

@dataclass
class StrategyConfig:
    """Configuration for a trading strategy"""
    name: str
    type: str
    parameters: Dict[str, float] = field(default_factory=dict)
    symbols: List[str] = field(default_factory=list)

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, symbols: List[str], **kwargs):
        """Initialize strategy
        
        Args:
            name: Strategy identifier
            symbols: List of trading symbols
            **kwargs: Additional strategy parameters
        """
        self.name = name
        self.symbols = symbols
        self.trades: List[Trade] = []
        self.performance_metrics = {
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0
        }
        self.is_active = True
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{name}")
        
        # Store any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    @abstractmethod
    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Generate trading signals
        
        Args:
            market_data: Dictionary of OHLCV DataFrames by symbol
            
        Returns:
            Dictionary of signals by symbol (-1.0 to 1.0)
        """
        pass
        
    def update_performance(self, trade: Trade) -> None:
        """Update strategy performance metrics
        
        Args:
            trade: Completed trade
        """
        if trade.pnl is not None:
            # Update total PnL
            self.performance_metrics['total_pnl'] += trade.pnl
            
            # Update win rate
            trades_won = self.performance_metrics.get('trades_won', 0)
            total_trades = self.performance_metrics.get('total_trades', 0)
            
            if trade.pnl > 0:
                trades_won += 1
            total_trades += 1
            
            self.performance_metrics['trades_won'] = trades_won
            self.performance_metrics['total_trades'] = total_trades
            self.performance_metrics['win_rate'] = trades_won / total_trades
            
    def validate_data(self, market_data: Dict[str, pd.DataFrame]) -> None:
        """Validate market data
        
        Args:
            market_data: Market data to validate
            
        Raises:
            ValidationError: If data is invalid
        """
        if not market_data:
            raise ValidationError("Empty market data")
            
        for symbol in self.symbols:
            if symbol not in market_data:
                raise ValidationError(f"Missing data for symbol: {symbol}")
                
            df = market_data[symbol]
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValidationError(
                    f"Missing columns for {symbol}: {missing_columns}"
                )

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get strategy performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        return self.performance_metrics.copy()

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy"""
    
    def __init__(
        self,
        name: str,
        symbols: List[str],
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        lookback_period: int = 20,
        **kwargs
    ):
        """Initialize strategy
        
        Args:
            name: Strategy name
            symbols: Trading symbols
            entry_threshold: Entry z-score threshold
            exit_threshold: Exit z-score threshold
            lookback_period: Lookback period for mean/std
        """
        super().__init__(name, symbols, **kwargs)
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.lookback_period = lookback_period
        
    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Generate mean reversion signals
        
        Args:
            market_data: Dictionary of OHLCV DataFrames by symbol
            
        Returns:
            Dictionary of signals by symbol
        """
        self.validate_data(market_data)
        signals = {}
        
        for symbol in self.symbols:
            df = market_data[symbol]
            
            # Calculate z-score
            returns = df['close'].pct_change()
            mean = returns.rolling(self.lookback_period).mean()
            std = returns.rolling(self.lookback_period).std()
            z_score = (returns - mean) / std
            
            current_z = z_score.iloc[-1]
            
            # Generate signal
            if abs(current_z) > self.entry_threshold:
                # Mean reversion signal (opposite to z-score direction)
                signals[symbol] = -np.sign(current_z)
            elif abs(current_z) < self.exit_threshold:
                # Exit signal
                signals[symbol] = 0.0
                
        return signals

class MomentumStrategy(BaseStrategy):
    """Momentum trading strategy"""
    
    def __init__(
        self,
        name: str,
        symbols: List[str],
        fast_period: int = 10,
        slow_period: int = 30,
        **kwargs
    ):
        """Initialize strategy
        
        Args:
            name: Strategy name
            symbols: Trading symbols
            fast_period: Fast moving average period
            slow_period: Slow moving average period
        """
        super().__init__(name, symbols, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Generate momentum signals
        
        Args:
            market_data: Dictionary of OHLCV DataFrames by symbol
            
        Returns:
            Dictionary of signals by symbol
        """
        self.validate_data(market_data)
        signals = {}
        
        for symbol in self.symbols:
            df = market_data[symbol]
            
            # Calculate moving averages
            fast_ma = df['close'].rolling(self.fast_period).mean()
            slow_ma = df['close'].rolling(self.slow_period).mean()
            
            # Generate signal based on MA crossover
            if fast_ma.iloc[-1] > slow_ma.iloc[-1]:
                signals[symbol] = 1.0  # Buy signal
            elif fast_ma.iloc[-1] < slow_ma.iloc[-1]:
                signals[symbol] = -1.0  # Sell signal
            else:
                signals[symbol] = 0.0  # No signal
                
        return signals

class PCAStrategy(BaseStrategy):
    """PCA-based trading strategy"""
    
    def __init__(
        self,
        name: str,
        symbols: List[str],
        model: PCAModel,
        entry_threshold: float = 1.0,
        **kwargs
    ):
        """Initialize strategy
        
        Args:
            name: Strategy name
            symbols: Trading symbols
            model: PCA model
            entry_threshold: Entry threshold for signals
        """
        super().__init__(name, symbols, **kwargs)
        self.model = model
        self.entry_threshold = entry_threshold
        
    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        """Generate PCA-based signals
        
        Args:
            market_data: Dictionary of OHLCV DataFrames by symbol
            
        Returns:
            Dictionary of signals by symbol
        """
        self.validate_data(market_data)
        signals = {}
        
        # Create return matrix
        returns = pd.DataFrame({
            symbol: data['close'].pct_change()
            for symbol, data in market_data.items()
        })
        
        # Transform returns using PCA
        transformed = self.model.transform(returns.values)
        
        # Generate signals based on first principal component
        pc1 = transformed[:, 0]
        current_score = pc1[-1]
        
        if abs(current_score) > self.entry_threshold:
            signal_value = -np.sign(current_score)  # Mean reversion on PC1
            for symbol in self.symbols:
                # Weight signal by component loading
                loading = self.model.components[0, self.symbols.index(symbol)]
                signals[symbol] = signal_value * abs(loading)
                
        return signals

class StrategyFactory:
    """Factory for creating trading strategies"""
    
    _strategies: Dict[str, Type[BaseStrategy]] = {
        'mean_reversion': MeanReversionStrategy,
        'momentum': MomentumStrategy,
        'pca': PCAStrategy
    }
    
    @classmethod
    def register(cls, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """Register new strategy
        
        Args:
            name: Strategy name
            strategy_class: Strategy class
        """
        cls._strategies[name] = strategy_class
        
    @classmethod
    def create(cls, config: StrategyConfig, system_config: Optional[Config] = None) -> BaseStrategy:
        """Create strategy instance
        
        Args:
            config: Strategy configuration
            system_config: Optional system configuration
            
        Returns:
            Strategy instance
            
        Raises:
            ValidationError: If strategy type is invalid
        """
        if config.type not in cls._strategies:
            raise ValidationError(f"Invalid strategy type: {config.type}")
            
        strategy_class = cls._strategies[config.type]
        
        # Extract strategy parameters
        kwargs = {
            'name': config.name,
            'symbols': config.symbols,
            'config': system_config
        }
        kwargs.update(config.parameters)
        
        return strategy_class(**kwargs)