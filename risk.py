"""
Risk management module for monitoring and controlling trading risks.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats

from trading_types import (
    Symbol, Trade, RiskMetrics, Config,
    TradingError, ValidationError, TradeDirection
)

class RiskManager:
    """Manages trading risk at portfolio and strategy levels"""
    
    def __init__(self, config: Config):
        """Initialize risk manager
        
        Args:
            config: System configuration
        """
        self.config = config
        self.position_history: List[Dict[str, float]] = []
        self.trade_history: List[Trade] = []
    
    def calculate_risk_metrics(self, returns: Union[pd.Series, pd.DataFrame], positions: Dict[str, float]) -> RiskMetrics:
        """Calculate risk metrics for the portfolio
        
        Args:
            returns: Historical returns series or dataframe
            positions: Current positions {symbol: size}
            
        Returns:
            RiskMetrics object with calculated metrics
        """
        try:
            # Convert Series to DataFrame if needed
            if isinstance(returns, pd.Series):
                returns = pd.DataFrame({'returns': returns})
            
            # Calculate portfolio returns
            portfolio_returns = pd.Series(0.0, index=returns.index)
            if 'returns' in returns.columns:
                # Single asset case
                portfolio_returns = returns['returns'] * list(positions.values())[0]
            else:
                # Multiple assets case
                for symbol, size in positions.items():
                    if symbol in returns.columns:
                        portfolio_returns += returns[symbol] * size
                    
            # Calculate metrics
            volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            var = -np.percentile(portfolio_returns, 5)  # 95% VaR
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            sortino = self._calculate_sortino_ratio(portfolio_returns)
            
            return RiskMetrics(
                volatility=float(volatility),
                var_95=float(var),
                max_drawdown=float(max_drawdown),
                sortino_ratio=float(sortino)
            )
            
        except Exception as e:
            raise ValidationError(f"Risk calculation failed: {str(e)}")
    
    def validate_trade(self, trade: Trade, portfolio_value: float) -> None:
        """Validate trade against risk limits
        
        Args:
            trade: Proposed trade
            portfolio_value: Current portfolio value
            
        Raises:
            ValidationError: If trade violates risk limits
        """
        # Calculate position value
        position_value = trade.size * trade.entry_price
        
        # Check absolute position size
        if position_value > portfolio_value * self.config.max_position_size:
            raise ValidationError(
                f"Position size {position_value} exceeds maximum allowed "
                f"{portfolio_value * self.config.max_position_size}"
            )
        
        # Check total exposure
        total_exposure = sum(
            abs(size * self._get_current_price(symbol))
            for symbol, size in self._get_total_positions(trade).items()
        )
        
        if total_exposure > portfolio_value:
            raise ValidationError(
                f"Total exposure {total_exposure} exceeds portfolio value {portfolio_value}"
            )
    
    def update_position_history(self, positions: Dict[str, float]) -> None:
        """Update position history for risk tracking
        
        Args:
            positions: Current positions
        """
        self.position_history.append(positions.copy())
        
        # Keep last 252 days of history
        if len(self.position_history) > 252:
            self.position_history.pop(0)
    
    def record_trade(self, trade: Trade) -> None:
        """Record trade for risk analysis
        
        Args:
            trade: Completed trade
        """
        self.trade_history.append(trade)
        
        # Keep last 1000 trades
        if len(self.trade_history) > 1000:
            self.trade_history.pop(0)
    
    def get_position_concentration(self) -> Dict[str, float]:
        """Calculate position concentration metrics
        
        Returns:
            Dictionary of position concentrations
        """
        if not self.position_history:
            return {}
            
        latest_positions = self.position_history[-1]
        total_value = sum(
            abs(size * self._get_current_price(symbol))
            for symbol, size in latest_positions.items()
        )
        
        if total_value == 0:
            return {symbol: 0.0 for symbol in latest_positions}
            
        return {
            symbol: abs(size * self._get_current_price(symbol)) / total_value
            for symbol, size in latest_positions.items()
        }
    
    def get_correlation_matrix(self, returns_dict: Dict[str, pd.Series]) -> np.ndarray:
        """Calculate correlation matrix for positions
        
        Args:
            returns_dict: Dictionary of return series by symbol
            
        Returns:
            Correlation matrix
            
        Raises:
            ValidationError: If return data is invalid
        """
        try:
            # Convert to DataFrame
            returns_df = pd.DataFrame(returns_dict)
            
            # Calculate correlation matrix
            return returns_df.corr().values
            
        except Exception as e:
            raise ValidationError(f"Correlation calculation failed: {str(e)}")
    
    def _get_total_positions(self, new_trade: Trade) -> Dict[str, float]:
        """Get total positions including new trade
        
        Args:
            new_trade: Proposed new trade
            
        Returns:
            Combined positions dictionary
        """
        if not self.position_history:
            return {new_trade.symbol: new_trade.size}
            
        positions = self.position_history[-1].copy()
        
        # Add new trade
        if new_trade.direction == "BUY":
            positions[new_trade.symbol] = (
                positions.get(new_trade.symbol, 0) + new_trade.size
            )
        else:
            positions[new_trade.symbol] = (
                positions.get(new_trade.symbol, 0) - new_trade.size
            )
            
        return positions
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price
        """
        # For testing purposes, return a mock price
        # In production, this would fetch real-time prices
        mock_prices = {
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3000.0
        }
        return mock_prices.get(symbol, 1000.0)  # Default price for unknown symbols

    def calculate_var(self, returns: pd.Series, positions: Dict[str, float], confidence_level: float) -> float:
        """Calculate Value at Risk for the portfolio
        
        Args:
            returns: Historical returns series
            positions: Current positions {symbol: size}
            confidence_level: VaR confidence level (e.g. 0.95 for 95% VaR)
            
        Returns:
            Value at Risk as a percentage of portfolio value
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = pd.Series(0.0, index=returns.index)
            for symbol, size in positions.items():
                if symbol in returns:
                    portfolio_returns += returns[symbol] * size
            
            # Calculate VaR (ensure positive value)
            var = abs(np.percentile(portfolio_returns, (1 - confidence_level) * 100))
            return float(var)
            
        except Exception as e:
            raise ValidationError(f"VaR calculation failed: {str(e)}")

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series
        
        Args:
            returns: Historical returns series
            
        Returns:
            Maximum drawdown as a percentage
        """
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        return float(drawdowns.min())
        
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio
        
        Args:
            returns: Historical returns series
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        if len(excess_returns) < 2:
            return 0.0
        return float(excess_returns.mean() / excess_returns.std() * np.sqrt(252))
        
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio
        
        Args:
            returns: Historical returns series
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) < 2:
            return 0.0
        return float(excess_returns.mean() / downside_returns.std() * np.sqrt(252))