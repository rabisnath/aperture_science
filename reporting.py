"""
Trading system reporting module.
"""

from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

from trading_types import (
    Config, Trade, MarketConditions, PortfolioState,
    ValidationError
)

class ReportGenerator:
    """Generates trading system reports"""
    
    def __init__(self, config: Config):
        """Initialize report generator
        
        Args:
            config: System configuration
        """
        self.config = config
        
    def generate_portfolio_report(
        self,
        portfolio_state: PortfolioState,
        trades: List[Trade],
        market_conditions: MarketConditions
    ) -> Dict:
        """Generate portfolio report
        
        Args:
            portfolio_state: Current portfolio state
            trades: Recent trades
            market_conditions: Current market conditions
            
        Returns:
            Report dictionary
        """
        report = {
            'portfolio_summary': self._generate_portfolio_summary(portfolio_state),
            'strategy_performance': self._analyze_strategy_performance(trades),
            'market_conditions': self._summarize_market_conditions(market_conditions),
            'risk_metrics': self._calculate_risk_metrics(trades, portfolio_state)
        }
        
        return report
        
    def _generate_portfolio_summary(
        self,
        portfolio_state: PortfolioState
    ) -> Dict:
        """Generate portfolio summary
        
        Args:
            portfolio_state: Current portfolio state
            
        Returns:
            Portfolio summary
        """
        total_value = portfolio_state.capital
        for symbol, position in portfolio_state.positions.items():
            total_value += position['quantity'] * position['price']
            
        return {
            'total_value': total_value,
            'cash': portfolio_state.capital,
            'positions': portfolio_state.positions,
            'allocations': portfolio_state.strategy_allocations
        }
        
    def _analyze_strategy_performance(self, trades: List[Trade]) -> Dict:
        """Analyze strategy performance
        
        Args:
            trades: List of trades
            
        Returns:
            Strategy performance metrics
        """
        if not trades:
            return {}
            
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'strategy_id': t.strategy_id,
                'pnl': t.pnl if t.pnl is not None else 0,
                'duration': (t.exit_time - t.entry_time).total_seconds() / 3600
                if t.exit_time else 0
            }
            for t in trades
        ])
        
        # Calculate metrics by strategy
        metrics = {}
        for strategy in df['strategy_id'].unique():
            strategy_trades = df[df['strategy_id'] == strategy]
            
            metrics[strategy] = {
                'total_pnl': strategy_trades['pnl'].sum(),
                'win_rate': (strategy_trades['pnl'] > 0).mean(),
                'avg_trade_duration': strategy_trades['duration'].mean(),
                'n_trades': len(strategy_trades)
            }
            
        return metrics
        
    def _summarize_market_conditions(
        self,
        market_conditions: MarketConditions
    ) -> Dict:
        """Summarize market conditions
        
        Args:
            market_conditions: Market conditions
            
        Returns:
            Market summary
        """
        return {
            'volatility': market_conditions.volatility,
            'skewness': market_conditions.skewness,
            'mean_return': market_conditions.mean_return,
            'correlation': {
                'mean': np.mean(market_conditions.correlation_matrix),
                'std': np.std(market_conditions.correlation_matrix)
            }
        }
        
    def _calculate_risk_metrics(
        self,
        trades: List[Trade],
        portfolio_state: PortfolioState
    ) -> Dict:
        """Calculate risk metrics
        
        Args:
            trades: List of trades
            portfolio_state: Current portfolio state
            
        Returns:
            Risk metrics
        """
        if not trades:
            return {}
            
        # Calculate returns
        pnls = [t.pnl for t in trades if t.pnl is not None]
        if not pnls:
            return {}
            
        returns = pd.Series(pnls) / portfolio_state.capital
        
        return {
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (
                returns.mean() * np.sqrt(252) / (returns.std() * np.sqrt(252))
                if returns.std() > 0 else 0
            ),
            'max_drawdown': (
                (returns.cumsum() - returns.cumsum().expanding().max()).min()
                if len(returns) > 0 else 0
            ),
            'var_95': np.percentile(returns, 5)
        }