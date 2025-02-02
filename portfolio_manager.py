"""
Portfolio management module for handling asset allocation and risk management.
Includes ML-based strategy selection and dynamic capital allocation.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb

from trading_types import (
    Symbol, Trade, PortfolioState, Config,
    TradingError, ValidationError, TradeDirection
)
from strategy import BaseStrategy

@dataclass
class StrategyAllocation:
    """Represents allocation details for a specific strategy"""
    strategy_id: str
    active: bool = True
    capital_allocation: float = 0.0
    current_exposure: float = 0.0
    performance_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

class StrategySelectorModel:
    """Machine learning model for strategy selection and performance prediction"""
    
    def __init__(
        self, 
        feature_columns: Optional[List[str]] = None,
        target_column: str = 'sharpe_ratio'
    ):
        """Initialize strategy selection model
        
        Args:
            feature_columns: Input features for prediction
            target_column: Performance metric to predict
        """
        self.feature_columns = feature_columns or [
            'market_volatility', 
            'correlation', 
            'previous_return', 
            'strategy_win_rate'
        ]
        self.target_column = target_column
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1
        )
        self.is_trained = False
    
    def train(self, historical_data: pd.DataFrame) -> None:
        """Train model on historical strategy performance data
        
        Args:
            historical_data: DataFrame with strategy performance metrics
            
        Raises:
            ValidationError: If training data is invalid
        """
        try:
            if not all(col in historical_data.columns for col in self.feature_columns):
                raise ValidationError(f"Missing required features: {self.feature_columns}")
                
            X = historical_data[self.feature_columns]
            y = historical_data[self.target_column]
            
            self.model.fit(X, y)
            self.is_trained = True
            
        except Exception as e:
            raise ValidationError(f"Training failed: {str(e)}")
    
    def predict_performance(self, features: pd.DataFrame) -> Dict[str, float]:
        """Predict performance scores for strategies
        
        Args:
            features: Current market features for prediction
            
        Returns:
            Dictionary of predicted performance scores by strategy
            
        Raises:
            ValidationError: If model is not trained or features are invalid
        """
        if not self.is_trained:
            raise ValidationError("Model must be trained before prediction")
            
        if not all(col in features.columns for col in self.feature_columns):
            raise ValidationError(f"Missing required features: {self.feature_columns}")
            
        try:
            predictions = self.model.predict(features[self.feature_columns])
            return {
                strategy: float(score)
                for strategy, score in zip(features.index, predictions)
            }
        except Exception as e:
            raise ValidationError(f"Prediction failed: {str(e)}")

class PortfolioManager:
    """Manages portfolio allocation and risk with ML-based strategy selection"""
    
    def __init__(self, config: Config):
        """Initialize portfolio manager
        
        Args:
            config: System configuration
        """
        self.config = config
        self.capital = config.initial_capital
        self.positions: Dict[str, float] = {}
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_allocations: Dict[str, StrategyAllocation] = {}
        self.strategy_selector = StrategySelectorModel()
        
    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state
        
        Returns:
            Current portfolio state
        """
        return PortfolioState(
            capital=self.capital,
            positions=self.positions.copy(),
            strategy_allocations={
                k: v.capital_allocation 
                for k, v in self.strategy_allocations.items()
            },
            timestamp=datetime.now()
        )
    
    def select_strategies(self, market_data: pd.DataFrame) -> List[str]:
        """Select strategies based on market conditions using ML model
        
        Args:
            market_data: Current market features and conditions
            
        Returns:
            List of selected strategy IDs
            
        Raises:
            ValidationError: If strategy selection fails
        """
        try:
            # Predict strategy performance
            performance_scores = self.strategy_selector.predict_performance(market_data)
            
            # Select top N strategies
            selected_strategies = sorted(
                performance_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.config.max_strategies]
            
            return [strategy for strategy, _ in selected_strategies]
            
        except Exception as e:
            raise ValidationError(f"Strategy selection failed: {str(e)}")
    
    def allocate_capital(self, selected_strategies: List[str]) -> None:
        """Allocate capital among selected strategies
        
        Args:
            selected_strategies: List of strategies to allocate capital to
            
        Raises:
            ValidationError: If allocation is invalid
        """
        if not selected_strategies:
            raise ValidationError("No strategies selected for allocation")
            
        # Equal allocation for now - could be made more sophisticated
        allocation_per_strategy = 1.0 / len(selected_strategies)
        
        # Update allocations
        for strategy_id in self.strategy_allocations.keys():
            if strategy_id in selected_strategies:
                self.strategy_allocations[strategy_id].active = True
                self.strategy_allocations[strategy_id].capital_allocation = allocation_per_strategy
            else:
                self.strategy_allocations[strategy_id].active = False
                self.strategy_allocations[strategy_id].capital_allocation = 0.0
    
    def execute_trade(self, trade: Trade) -> None:
        """Execute a trade and update portfolio state
        
        Args:
            trade: Trade to execute
            
        Raises:
            ValidationError: If trade is invalid
            TradingError: If trade execution fails
        """
        try:
            # Validate trade
            self._validate_trade(trade)
            
            # Calculate trade impact
            position_value = trade.size * trade.entry_price
            
            # Update positions
            if trade.direction == "BUY":
                self.positions[trade.symbol] = self.positions.get(trade.symbol, 0) + trade.size
                self.capital -= position_value
            else:
                self.positions[trade.symbol] = self.positions.get(trade.symbol, 0) - trade.size
                self.capital += position_value
                
            # Update strategy exposure
            allocation = self.strategy_allocations[trade.strategy_id]
            allocation.current_exposure = position_value
            allocation.last_updated = datetime.now()
                
            # Clean up empty positions
            if abs(self.positions[trade.symbol]) < 1e-8:
                del self.positions[trade.symbol]
                
        except Exception as e:
            raise TradingError(f"Failed to execute trade: {str(e)}")
    
    def close_trade(self, trade: Trade) -> None:
        """Close an existing trade
        
        Args:
            trade: Trade to close
            
        Raises:
            ValidationError: If trade closure is invalid
            TradingError: If trade closure fails
        """
        try:
            # Validate trade exists
            current_position = self.positions.get(trade.symbol, 0)
            if current_position == 0:
                raise ValidationError(f"No position exists for {trade.symbol}")
                
            # Calculate PnL
            if trade.direction == "BUY":
                pnl = (trade.exit_price - trade.entry_price) * trade.size
            else:
                pnl = (trade.entry_price - trade.exit_price) * trade.size
                
            # Update capital and positions
            self.capital += pnl
            self.positions[trade.symbol] = 0
            
            # Update strategy allocation
            allocation = self.strategy_allocations[trade.strategy_id]
            allocation.current_exposure = 0
            allocation.performance_score += pnl / self.config.initial_capital
            allocation.last_updated = datetime.now()
            
            # Clean up position
            del self.positions[trade.symbol]
            
        except Exception as e:
            raise TradingError(f"Failed to close trade: {str(e)}")
    
    def _validate_trade(self, trade: Trade) -> None:
        """Validate trade parameters
        
        Args:
            trade: Trade to validate
            
        Raises:
            ValidationError: If trade is invalid
        """
        # Check strategy is active
        allocation = self.strategy_allocations.get(trade.strategy_id)
        if not allocation or not allocation.active:
            raise ValidationError(f"Strategy {trade.strategy_id} is not active")
        
        # Check position size
        position_value = trade.size * trade.entry_price
        max_position = self.capital * self.config.max_position_size
        if position_value > max_position:
            raise ValidationError(
                f"Position size {position_value} exceeds maximum allowed {max_position}"
            )
        
        # Check strategy allocation
        strategy_max = self.capital * allocation.capital_allocation
        if position_value > strategy_max:
            raise ValidationError(
                f"Position size {position_value} exceeds strategy allocation {strategy_max}"
            )
    
    def rebalance_portfolio(self) -> List[Trade]:
        """Rebalance portfolio to target allocations
        
        Returns:
            List of rebalancing trades
        """
        rebalancing_trades = []
        
        # Calculate current portfolio value
        total_value = self.capital + sum(
            size * self._get_current_price(symbol) 
            for symbol, size in self.positions.items()
        )
        
        # Check each strategy allocation
        for strategy_id, allocation in self.strategy_allocations.items():
            if not allocation.active:
                continue
                
            target_value = total_value * allocation.capital_allocation
            current_value = allocation.current_exposure
            
            # Generate rebalancing trade if deviation is significant
            if abs(target_value - current_value) > total_value * 0.01:  # 1% threshold
                trade = Trade(
                    symbol=self._get_symbol_for_strategy(strategy_id),
                    strategy_id=strategy_id,
                    direction="BUY" if target_value > current_value else "SELL",
                    size=abs(target_value - current_value),
                    entry_time=datetime.now(),
                    entry_price=self._get_current_price(symbol)
                )
                rebalancing_trades.append(trade)
        
        return rebalancing_trades
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol
        
        This is a placeholder - in a real implementation, this would
        fetch the current market price from a data source
        """
        raise NotImplementedError("Price fetching not implemented")
    
    def _get_symbol_for_strategy(self, strategy_id: str) -> str:
        """Get primary trading symbol for a strategy
        
        This is a placeholder - in a real implementation, this would
        look up the primary symbol for the strategy
        """
        raise NotImplementedError("Symbol lookup not implemented")

    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Add a strategy to the portfolio
        
        Args:
            strategy: Strategy instance to add
        """
        if strategy.name in self.strategies:
            raise ValidationError(f"Strategy {strategy.name} already exists")
            
        self.strategies[strategy.name] = strategy
        self.strategy_allocations[strategy.name] = StrategyAllocation(
            strategy_id=strategy.name,
            active=True,
            capital_allocation=0.0,
            current_exposure=0.0,
            performance_score=0.0
        )

    def update_allocations(self, allocations: Dict[str, float]) -> None:
        """Update strategy allocations
        
        Args:
            allocations: Dictionary mapping strategy names to allocation percentages
        """
        # Validate allocations
        if not allocations:
            raise ValidationError("No allocations provided")
            
        total_allocation = sum(allocations.values())
        if not np.isclose(total_allocation, 1.0, atol=0.0001):
            raise ValidationError(f"Total allocation {total_allocation} must sum to 1.0")
            
        # Validate strategies exist
        for strategy_id in allocations:
            if strategy_id not in self.strategies:
                raise ValidationError(f"Strategy {strategy_id} not found")
                
        # Update allocations
        for strategy_id, allocation in allocations.items():
            self.strategy_allocations[strategy_id].capital_allocation = allocation
            self.strategy_allocations[strategy_id].active = allocation > 0