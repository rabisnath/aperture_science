"""
Historical data storage and retrieval module.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import json

from trading_types import (
    MarketConditions, Trade, ValidationError
)

class Historian:
    """Manages historical data storage and retrieval"""
    
    def __init__(self, storage_path: Path):
        """Initialize historian
        
        Args:
            storage_path: Path to storage directory
        """
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.market_conditions_path = storage_path / "market_conditions"
        self.trades_path = storage_path / "trades"
        
        self.market_conditions_path.mkdir(exist_ok=True)
        self.trades_path.mkdir(exist_ok=True)
        
    def save_market_conditions(self, conditions: MarketConditions) -> None:
        """Save market conditions
        
        Args:
            conditions: Market conditions to save
        """
        # Convert to dictionary
        data = {
            'timestamp': conditions.timestamp.isoformat(),
            'volatility': conditions.volatility,
            'skewness': conditions.skewness,
            'mean_return': conditions.mean_return,
            'correlation_matrix': conditions.correlation_matrix.tolist(),
            'basic_stats': conditions.basic_stats,
            'regression_stats': conditions.regression_stats
        }
        
        # Save to file
        filename = conditions.timestamp.strftime("%Y%m%d_%H%M%S.json")
        filepath = self.market_conditions_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_market_conditions(
        self,
        start: datetime,
        end: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Load market conditions
        
        Args:
            start: Start datetime
            end: End datetime
            
        Returns:
            Dictionary of market conditions data
        """
        data = {
            'basic_stats': [],
            'regression_stats': [],
            'correlation_matrices': []
        }
        timestamps = []
        
        # Load files in date range
        for filepath in self.market_conditions_path.glob("*.json"):
            timestamp = datetime.strptime(
                filepath.stem,
                "%Y%m%d_%H%M%S"
            )
            
            if start <= timestamp <= end:
                with open(filepath) as f:
                    market_data = json.load(f)
                    
                timestamps.append(timestamp)
                data['basic_stats'].append(market_data['basic_stats'])
                data['regression_stats'].append(market_data['regression_stats'])
                data['correlation_matrices'].append(
                    np.array(market_data['correlation_matrix'])
                )
                
        # Convert to DataFrames
        if timestamps:
            data['basic_stats'] = pd.DataFrame(
                data['basic_stats'],
                index=timestamps
            )
            data['regression_stats'] = pd.DataFrame(
                data['regression_stats'],
                index=timestamps
            )
            
        return data
        
    def save_trade(self, trade: Trade) -> None:
        """Save trade record
        
        Args:
            trade: Trade to save
        """
        # Convert to dictionary
        data = {
            'symbol': trade.symbol,
            'strategy_id': trade.strategy_id,
            'direction': trade.direction,
            'size': trade.size,
            'entry_time': trade.entry_time.isoformat(),
            'entry_price': trade.entry_price,
            'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
            'exit_price': trade.exit_price,
            'pnl': trade.pnl,
            'status': trade.status
        }
        
        # Save to file
        filename = f"{trade.entry_time.strftime('%Y%m%d_%H%M%S')}_{trade.symbol}.json"
        filepath = self.trades_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_trades(
        self,
        start: datetime,
        end: datetime,
        symbol: Optional[str] = None,
        strategy_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Load trade records
        
        Args:
            start: Start datetime
            end: End datetime
            symbol: Optional symbol filter
            strategy_id: Optional strategy filter
            
        Returns:
            DataFrame of trades
        """
        trades = []
        
        # Load files in date range
        for filepath in self.trades_path.glob("*.json"):
            timestamp = datetime.strptime(
                filepath.stem.split('_')[0],
                "%Y%m%d"
            )
            
            if start <= timestamp <= end:
                with open(filepath) as f:
                    trade_data = json.load(f)
                    
                    # Apply filters
                    if symbol and trade_data['symbol'] != symbol:
                        continue
                    if strategy_id and trade_data['strategy_id'] != strategy_id:
                        continue
                        
                    trades.append(trade_data)
                    
        return pd.DataFrame(trades)
        
    def clean_old_data(self, retention_days: int = 30) -> None:
        """Clean up old data files
        
        Args:
            retention_days: Days to retain data
        """
        cutoff = datetime.now() - timedelta(days=retention_days)
        
        # Clean market conditions
        for filepath in self.market_conditions_path.glob("*.json"):
            timestamp = datetime.strptime(
                filepath.stem,
                "%Y%m%d_%H%M%S"
            )
            if timestamp < cutoff:
                filepath.unlink()
                
        # Clean trades
        for filepath in self.trades_path.glob("*.json"):
            timestamp = datetime.strptime(
                filepath.stem.split('_')[0],
                "%Y%m%d"
            )
            if timestamp < cutoff:
                filepath.unlink()