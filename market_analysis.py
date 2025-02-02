"""
Market analysis module for analyzing market conditions and trends.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

from trading_types import (
    Symbol, MarketConditions, Config,
    TradingError, ValidationError
)

class MarketAnalyzer:
    """Analyzes market conditions and trends"""
    
    def __init__(self, config: Config):
        """Initialize market analyzer
        
        Args:
            config: System configuration
        """
        self.config = config
        
    def analyze_market_conditions(
        self,
        market_data: Dict[str, pd.DataFrame],
        window: int = 252
    ) -> MarketConditions:
        """Analyze current market conditions
        
        Args:
            market_data: Dictionary of OHLCV DataFrames by symbol
            window: Analysis window in days
            
        Returns:
            Market conditions analysis
            
        Raises:
            ValidationError: If input data is invalid
        """
        try:
            # Calculate returns
            returns_dict = {
                symbol: df['close'].pct_change().dropna()
                for symbol, df in market_data.items()
            }
            
            # Convert to DataFrame for analysis
            returns_df = pd.DataFrame(returns_dict).tail(window)
            
            if len(returns_df) < 2:
                raise ValidationError("Insufficient data for analysis")
            
            # Calculate basic statistics
            volatility = returns_df.std() * np.sqrt(252)
            skewness = returns_df.skew()
            mean_return = returns_df.mean() * 252
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr().values
            
            # Calculate regression statistics
            regression_stats = self._calculate_regression_stats(returns_df)
            
            # Create market conditions object
            return MarketConditions(
                timestamp=datetime.now(),
                volatility=float(volatility.mean()),
                skewness=float(skewness.mean()),
                mean_return=float(mean_return.mean()),
                correlation_matrix=correlation_matrix,
                basic_stats={
                    'volatility': volatility.to_dict(),
                    'skewness': skewness.to_dict(),
                    'mean_return': mean_return.to_dict()
                },
                regression_stats=regression_stats
            )
            
        except Exception as e:
            raise ValidationError(f"Market analysis failed: {str(e)}")
    
    def detect_regime_change(
        self,
        market_data: Dict[str, pd.DataFrame],
        window: int = 63
    ) -> Dict[str, str]:
        """Detect market regime changes
        
        Args:
            market_data: Dictionary of OHLCV DataFrames by symbol
            window: Detection window in days
            
        Returns:
            Dictionary of market regimes by symbol
            
        Raises:
            ValidationError: If input data is invalid
        """
        try:
            regimes = {}
            
            for symbol, df in market_data.items():
                # Calculate indicators
                returns = df['close'].pct_change().dropna()
                volatility = returns.rolling(window).std() * np.sqrt(252)
                trend = df['close'].rolling(window).mean()
                
                # Current values
                current_return = returns.iloc[-1]
                current_vol = volatility.iloc[-1]
                current_trend = trend.iloc[-1] > trend.iloc[-2]
                
                # Determine regime
                if current_vol > volatility.mean() + volatility.std():
                    regime = "HIGH_VOLATILITY"
                elif current_return > returns.mean() + returns.std() and current_trend:
                    regime = "BULLISH"
                elif current_return < returns.mean() - returns.std() and not current_trend:
                    regime = "BEARISH"
                else:
                    regime = "NORMAL"
                    
                regimes[symbol] = regime
                
            return regimes
            
        except Exception as e:
            raise ValidationError(f"Regime detection failed: {str(e)}")
    
    def calculate_market_impact(
        self,
        symbol: str,
        size: float,
        market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Estimate market impact of a trade
        
        Args:
            symbol: Trading symbol
            size: Trade size
            market_data: Market data dictionary
            
        Returns:
            Estimated price impact percentage
            
        Raises:
            ValidationError: If input data is invalid
        """
        try:
            if symbol not in market_data:
                raise ValidationError(f"No data available for {symbol}")
                
            df = market_data[symbol]
            
            # Calculate average daily volume
            avg_volume = df['volume'].mean()
            
            # Simple square root model for market impact
            impact = 0.1 * np.sqrt(size / avg_volume)
            
            return min(impact, 0.01)  # Cap at 1%
            
        except Exception as e:
            raise ValidationError(f"Market impact calculation failed: {str(e)}")
    
    def _calculate_regression_stats(self, returns_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate regression statistics
        
        Args:
            returns_df: Returns DataFrame
            
        Returns:
            Dictionary of regression statistics
        """
        stats_dict = {}
        
        # Use first column as market proxy
        market_returns = returns_df.iloc[:, 0]
        
        for column in returns_df.columns[1:]:
            asset_returns = returns_df[column]
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                market_returns, asset_returns
            )
            
            stats_dict[column] = {
                'beta': slope,
                'alpha': intercept * 252,  # Annualize
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'std_error': std_err
            }
            
        return stats_dict