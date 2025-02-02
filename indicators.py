"""
Technical indicators module.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class IndicatorRegistry:
    """Registry of available indicators"""
    
    _indicators = {}
    
    @classmethod
    def register(cls, name: str, indicator_class):
        """Register an indicator"""
        cls._indicators[name] = indicator_class
        
    @classmethod
    def get(cls, name: str):
        """Get indicator by name"""
        return cls._indicators.get(name)
        
    @classmethod
    def list_indicators(cls) -> List[str]:
        """List available indicators"""
        return list(cls._indicators.keys())

class MovingAverage:
    """Moving average indicator"""
    
    def __init__(self, window: int = 20):
        """Initialize moving average
        
        Args:
            window: Window size
        """
        self.window = window
        
    def calculate(self, series: pd.Series) -> pd.Series:
        """Calculate moving average
        
        Args:
            series: Price series
            
        Returns:
            Moving average series
        """
        return series.rolling(window=self.window).mean()

class RSI:
    """Relative Strength Index"""
    
    def __init__(self, period: int = 14):
        """Initialize RSI
        
        Args:
            period: RSI period
        """
        self.period = period
        
    def calculate(self, series: pd.Series) -> pd.Series:
        """Calculate RSI
        
        Args:
            series: Price series
            
        Returns:
            RSI series
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class MACD:
    """Moving Average Convergence Divergence"""
    
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """Initialize MACD
        
        Args:
            fast: Fast period
            slow: Slow period
            signal: Signal period
        """
        self.fast = fast
        self.slow = slow
        self.signal = signal
        
    def calculate(self, series: pd.Series) -> Dict[str, pd.Series]:
        """Calculate MACD
        
        Args:
            series: Price series
            
        Returns:
            Dictionary with MACD line and signal
        """
        exp1 = series.ewm(span=self.fast).mean()
        exp2 = series.ewm(span=self.slow).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.signal).mean()
        
        return {
            'line': macd,
            'signal': signal,
            'histogram': macd - signal
        }

class BollingerBands:
    """Bollinger Bands indicator"""
    
    def __init__(self, window: int = 20, std_dev: float = 2.0):
        """Initialize Bollinger Bands
        
        Args:
            window: Window size
            std_dev: Number of standard deviations
        """
        self.window = window
        self.std_dev = std_dev
        
    def calculate(self, series: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands
        
        Args:
            series: Price series
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        middle = series.rolling(window=self.window).mean()
        std = series.rolling(window=self.window).std()
        
        return {
            'upper': middle + (std * self.std_dev),
            'middle': middle,
            'lower': middle - (std * self.std_dev)
        }

class IndicatorEngine:
    """Engine for calculating multiple indicators"""
    
    def __init__(self):
        """Initialize indicator engine"""
        self.indicators = {}
        
    def add_indicator(self, name: str, **params):
        """Add indicator to engine
        
        Args:
            name: Indicator name
            **params: Indicator parameters
        """
        if name == "sma":
            self.indicators[f"sma_{params['window']}"] = MovingAverage(**params)
        elif name == "rsi":
            self.indicators[f"rsi_{params['period']}"] = RSI(**params)
        elif name == "macd":
            key = f"macd_{params['fast']}_{params['slow']}_{params['signal']}"
            self.indicators[key] = MACD(**params)
        elif name == "bollinger":
            key = f"bb_{params['window']}"
            self.indicators[key] = BollingerBands(**params)
            
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            DataFrame with indicators
        """
        result = df.copy()
        
        for name, indicator in self.indicators.items():
            if isinstance(indicator, MACD):
                values = indicator.calculate(df['close'])
                result[f"{name}_line"] = values['line']
                result[f"{name}_signal"] = values['signal']
                result[f"{name}_histogram"] = values['histogram']
            elif isinstance(indicator, BollingerBands):
                values = indicator.calculate(df['close'])
                result[f"{name}_upper"] = values['upper']
                result[f"{name}_middle"] = values['middle']
                result[f"{name}_lower"] = values['lower']
            else:
                result[name] = indicator.calculate(df['close'])
                
        return result

# Register indicators
IndicatorRegistry.register("sma", MovingAverage)
IndicatorRegistry.register("rsi", RSI)
IndicatorRegistry.register("macd", MACD)
IndicatorRegistry.register("bollinger", BollingerBands)