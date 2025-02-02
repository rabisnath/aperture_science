"""
Common types and configurations for trading system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
import numpy as np

@dataclass
class Config:
    """System configuration"""
    initial_capital: float = 100000.0
    max_position_size: float = 0.1
    risk_free_rate: float = 0.02
    rebalancing_frequency: str = 'daily'
    max_strategies: int = 10
    max_strategy_allocation: float = 0.2

@dataclass
class Symbol:
    """Trading symbol"""
    name: str
    exchange: str
    quote_currency: str
    min_size: float
    price_decimals: int
    size_decimals: int

@dataclass
class Trade:
    """Trading record"""
    symbol: str
    strategy_id: str
    direction: str
    size: float
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "OPEN"

@dataclass
class PortfolioState:
    """Current portfolio state"""
    capital: float
    positions: Dict[str, Dict[str, float]]
    strategy_allocations: Dict[str, float]
    timestamp: datetime

@dataclass
class MarketConditions:
    """Market conditions analysis"""
    timestamp: datetime
    volatility: float
    skewness: float
    mean_return: float
    correlation_matrix: np.ndarray
    basic_stats: Dict[str, Dict[str, float]]
    regression_stats: Dict[str, Dict[str, float]]

@dataclass
class BrokerCredentials:
    """Broker API credentials"""
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None

@dataclass
class TradeOrder:
    """Trade order details"""
    symbol: str
    direction: str
    order_type: str
    quantity: float
    price: Optional[float]
    strategy_id: str
    status: str = "PENDING"
    execution_price: Optional[float] = None

class TradeDirection:
    """Trade direction constants"""
    BUY = "BUY"
    SELL = "SELL"

class TradeStatus:
    """Trade status constants"""
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class TradingError(Exception):
    """Base class for trading errors"""
    pass

class ValidationError(TradingError):
    """Validation error"""
    pass

class DataError(TradingError):
    """Data acquisition or processing error"""
    pass

@dataclass
class RiskMetrics:
    """Risk metrics for portfolio and strategies"""
    volatility: float
    var_95: float  # 95% Value at Risk
    max_drawdown: float
    beta: Optional[float] = None
    correlation: Optional[float] = None
    tail_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None