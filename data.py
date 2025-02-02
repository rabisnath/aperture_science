"""
Data acquisition and preprocessing module.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
from binance.client import Client

from trading_types import Config, ValidationError, DataError

class DataLoader:
    """Loads market data from various sources"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize data loader
        
        Args:
            config: Optional system configuration
        """
        self.config = config or Config()
        self._client = None
        self.interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
        }
        
    def _initialize_client(self):
        """Initialize API client with error handling"""
        if self._client is None:
            try:
                self._client = Client(
                    os.environ.get('BINANCE_API_KEY'),
                    os.environ.get('BINANCE_API_SECRET'),
                    testnet=True  # Use testnet instead of production
                )
            except Exception as e:
                raise DataError(f"Failed to initialize API client: {str(e)}")
        
    def get_market_data(
        self,
        symbols: List[str],
        interval: str = "5m",
        lookback: int = 100
    ) -> Dict[str, pd.DataFrame]:
        """Fetch market data for symbols
        
        Args:
            symbols: List of trading symbols
            interval: Time interval
            lookback: Number of periods to look back
            
        Returns:
            Dictionary of OHLCV DataFrames by symbol
        """
        try:
            if interval not in self.interval_map:
                raise ValidationError(f"Unsupported interval: {interval}")
                
            self._initialize_client()
            data = {}
            
            for symbol in symbols:
                klines = self._client.get_klines(
                    symbol=symbol,
                    interval=self.interval_map[interval],
                    limit=lookback
                )
                
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'buy_base_volume',
                    'buy_quote_volume', 'ignore'
                ])
                
                # Convert types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                    
                df.set_index('timestamp', inplace=True)
                data[symbol] = df
                
            return data
        except Exception as e:
            raise DataError(f"Failed to fetch market data: {str(e)}")
            
    def fetch_historical_data(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical market data
        
        Args:
            symbols: List of trading symbols
            start: Start datetime
            end: End datetime
            interval: Time interval
            
        Returns:
            Dictionary of OHLCV DataFrames by symbol
        """
        try:
            if interval not in self.interval_map:
                raise ValidationError(f"Unsupported interval: {interval}")
                
            self._initialize_client()
            data = {}
            
            for symbol in symbols:
                klines = self._client.get_historical_klines(
                    symbol=symbol,
                    interval=self.interval_map[interval],
                    start_str=str(int(start.timestamp() * 1000)),
                    end_str=str(int(end.timestamp() * 1000))
                )
                
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'buy_base_volume',
                    'buy_quote_volume', 'ignore'
                ])
                
                # Convert types
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                    
                df.set_index('timestamp', inplace=True)
                data[symbol] = df
                
            return data
        except Exception as e:
            raise DataError(f"Failed to fetch historical data: {str(e)}")
            
    def get_high_volume_symbols(self, N: int = 10, base_currency: str = 'USDT') -> List[str]:
        """Get top N symbols by volume
        
        Args:
            N: Number of symbols to return
            base_currency: Base currency to filter by
            
        Returns:
            List of symbol strings
        """
        try:
            self._initialize_client()
            tickers = pd.DataFrame(self._client.get_orderbook_tickers())
            tickers = tickers[tickers['symbol'].str.endswith(base_currency)]
            tickers['total_qty'] = tickers['bidQty'].astype(float) + tickers['askQty'].astype(float)
            return tickers.nlargest(N, 'total_qty')['symbol'].tolist()
        except Exception as e:
            raise DataError(f"Failed to fetch high volume symbols: {str(e)}")

class DataPreprocessor:
    """Preprocesses market data"""
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """Calculate returns from price data
        
        Args:
            df: Price DataFrame
            
        Returns:
            Returns series
        """
        returns = df['close'].pct_change()
        return returns  # Don't drop NaN values here to maintain length
        
    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        sma_period: int = 20,
        rsi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26
    ) -> pd.DataFrame:
        """Add technical indicators to DataFrame
        
        Args:
            df: Price DataFrame
            sma_period: SMA period
            rsi_period: RSI period
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            
        Returns:
            DataFrame with indicators
        """
        result = df.copy()
        
        # Add SMA
        result['sma_20'] = df['close'].rolling(sma_period).mean()
        
        # Add RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        result['rsi'] = 100 - (100 / (1 + rs))
        result['rsi'] = result['rsi'].fillna(50)  # Fill NaN with neutral value
        result['rsi'] = result['rsi'].clip(0, 100)  # Ensure values are between 0-100
        
        # Add MACD
        exp1 = df['close'].ewm(span=macd_fast).mean()
        exp2 = df['close'].ewm(span=macd_slow).mean()
        result['macd'] = exp1 - exp2
        
        return result
        
    def create_price_matrix(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> np.ndarray:
        """Create price matrix from market data
        
        Args:
            market_data: Dictionary of OHLCV DataFrames
            
        Returns:
            Price matrix [time, assets]
        """
        # Align DataFrames on index
        dfs = []
        for symbol in sorted(market_data.keys()):
            dfs.append(market_data[symbol]['close'])
            
        price_df = pd.concat(dfs, axis=1)
        price_df.columns = sorted(market_data.keys())
        
        return price_df.values
        
    def create_return_matrix(
        self,
        price_matrix: np.ndarray
    ) -> np.ndarray:
        """Create return matrix from price matrix
        
        Args:
            price_matrix: Price matrix [time, assets]
            
        Returns:
            Return matrix [time-1, assets]
        """
        return np.diff(np.log(price_matrix), axis=0)

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            DataFrame with indicators added
        """
        df = df.copy()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)  # Fill NaN with neutral value
        df['rsi'] = df['rsi'].clip(0, 100)  # Ensure values are between 0-100
        
        return df