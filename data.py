"""
Data acquisition and preprocessing module.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

from trading_types import Config, ValidationError, DataError, BrokerType

class DataLoader:
    """Loads market data from various sources"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize data loader
        
        Args:
            config: Optional system configuration
        """
        self.config = config or Config()
        self._alpaca_trading = None
        self._alpaca_data = None
        
        # Map common intervals to Alpaca timeframes
        self.interval_map = {
            '1m': TimeFrame.Minute,
            '5m': TimeFrame(5, TimeFrameUnit.Minute),
            '15m': TimeFrame(15, TimeFrameUnit.Minute),
            '1h': TimeFrame.Hour,
            '4h': TimeFrame(4, TimeFrameUnit.Hour),
            '1d': TimeFrame.Day,
        }
        
    def _initialize_alpaca(self):
        """Initialize Alpaca API clients with error handling"""
        if self._alpaca_trading is None or self._alpaca_data is None:
            try:
                api_key = os.environ.get('ALPACA_API_KEY')
                api_secret = os.environ.get('ALPACA_API_SECRET')
                
                if not api_key or not api_secret:
                    raise ValidationError("Alpaca API credentials not found in environment variables")
                
                self._alpaca_trading = TradingClient(api_key, api_secret, paper=True)
                self._alpaca_data = CryptoHistoricalDataClient(api_key, api_secret)
            except Exception as e:
                raise DataError(f"Failed to initialize Alpaca API client: {str(e)}")
    
    def get_historical_data(
        self,
        symbol: str,
        interval: str = '5m',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get historical market data
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
            start_time: Start time for historical data
            end_time: End time for historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        self._initialize_alpaca()
        
        # Convert interval to Alpaca format
        timeframe = self.interval_map.get(interval)
        if not timeframe:
            raise ValidationError(f"Invalid interval: {interval}")
            
        # Set default time range if not provided
        end_time = end_time or datetime.now()
        start_time = start_time or (end_time - timedelta(days=7))
        
        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_time,
                end=end_time
            )
            bars = self._alpaca_data.get_crypto_bars(request)
            
            # Convert to DataFrame and format columns
            df = bars.df
            if len(df) == 0:
                raise DataError(f"No data returned for {symbol}")
                
            # Rename columns to match our standard format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'trade_count': 'Trades',
                'vwap': 'VWAP'
            })
            
            return df
            
        except Exception as e:
            raise DataError(f"Failed to fetch historical data: {str(e)}")
            
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
        self._initialize_alpaca()
        
        # Calculate start time based on lookback
        end_time = datetime.now()
        if interval.endswith('m'):
            minutes = int(interval[:-1])
            start_time = end_time - timedelta(minutes=minutes * lookback)
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            start_time = end_time - timedelta(hours=hours * lookback)
        elif interval.endswith('d'):
            days = int(interval[:-1])
            start_time = end_time - timedelta(days=days * lookback)
        else:
            raise ValidationError(f"Invalid interval format: {interval}")
            
        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=self.interval_map[interval],
                start=start_time,
                end=end_time
            )
            bars = self._alpaca_data.get_crypto_bars(request)
            
            # Process the multi-symbol response
            data = {}
            for symbol in symbols:
                symbol_data = bars.df[bars.df.symbol == symbol].copy()
                if len(symbol_data) == 0:
                    raise DataError(f"No data returned for {symbol}")
                
                # Rename columns to match our standard format
                symbol_data = symbol_data.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'trade_count': 'Trades',
                    'vwap': 'VWAP'
                })
                
                data[symbol] = symbol_data
                
            return data
            
        except Exception as e:
            raise DataError(f"Failed to fetch market data: {str(e)}")
            
    def get_high_volume_symbols(
        self,
        N: int = 10,
        base_currency: str = 'USD'
    ) -> List[str]:
        """Get top N symbols by volume
        
        Args:
            N: Number of symbols to return
            base_currency: Base currency to filter by
            
        Returns:
            List of symbol strings
        """
        self._initialize_alpaca()
        
        try:
            # Get all available crypto assets
            request = GetAssetsRequest(asset_class=AssetClass.CRYPTO)
            assets = self._alpaca_trading.get_all_assets(request)
            
            # Filter for base currency and get trading volume
            symbols = []
            for asset in assets:
                if asset.status == 'active' and asset.tradable and asset.symbol.endswith(f"/{base_currency}"):
                    # Get recent volume data
                    request = CryptoBarsRequest(
                        symbol_or_symbols=asset.symbol,
                        timeframe=TimeFrame.Day,
                        start=datetime.now() - timedelta(days=7),
                        end=datetime.now()
                    )
                    bars = self._alpaca_data.get_crypto_bars(request)
                    
                    if not bars.df.empty:
                        avg_volume = bars.df['volume'].mean()
                        symbols.append({
                            'symbol': asset.symbol,
                            'volume': avg_volume
                        })
            
            # Sort by volume and take top N
            symbols.sort(key=lambda x: x['volume'], reverse=True)
            return [s['symbol'] for s in symbols[:N]]
            
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
        returns = df['Close'].pct_change()
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
        result['sma_20'] = df['Close'].rolling(sma_period).mean()
        
        # Add RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        result['rsi'] = 100 - (100 / (1 + rs))
        result['rsi'] = result['rsi'].fillna(50)  # Fill NaN with neutral value
        result['rsi'] = result['rsi'].clip(0, 100)  # Ensure values are between 0-100
        
        # Add MACD
        exp1 = df['Close'].ewm(span=macd_fast).mean()
        exp2 = df['Close'].ewm(span=macd_slow).mean()
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