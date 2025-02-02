import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from trading_types import (
    Config, Trade, MarketConditions, PortfolioState
)

class TradingVisualizer:
    """Trading visualization tools"""
    
    def __init__(self, config: Config):
        """Initialize trading visualizer
        
        Args:
            config: System configuration
        """
        self.config = config
        plt.style.use('seaborn')
        
    def plot_portfolio_performance(
        self,
        portfolio_state: PortfolioState,
        trades: List[Trade],
        figsize: tuple = (12, 8)
    ) -> plt.Figure:
        """Plot portfolio performance
        
        Args:
            portfolio_state: Current portfolio state
            trades: List of trades
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot portfolio value
        portfolio_values = self._calculate_portfolio_values(portfolio_state, trades)
        ax1.plot(portfolio_values.index, portfolio_values.values)
        ax1.set_title('Portfolio Value')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        
        # Plot trade PnL
        if trades:
            trade_df = pd.DataFrame([
                {
                    'entry_time': t.entry_time,
                    'pnl': t.pnl if t.pnl is not None else 0,
                    'strategy': t.strategy_id
                }
                for t in trades
            ])
            
            for strategy in trade_df['strategy'].unique():
                strategy_trades = trade_df[trade_df['strategy'] == strategy]
                ax2.scatter(
                    strategy_trades['entry_time'],
                    strategy_trades['pnl'],
                    label=strategy,
                    alpha=0.6
                )
                
            ax2.set_title('Trade PnL')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('PnL')
            ax2.legend()
            
        plt.tight_layout()
        return fig
        
    def plot_market_conditions(
        self,
        market_conditions: MarketConditions,
        figsize: tuple = (12, 8)
    ) -> plt.Figure:
        """Plot market conditions
        
        Args:
            market_conditions: Market conditions
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot correlation matrix
        sns.heatmap(
            market_conditions.correlation_matrix,
            ax=ax1,
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt='.2f'
        )
        ax1.set_title('Correlation Matrix')
        
        # Plot market statistics
        stats = pd.Series({
            'Volatility': market_conditions.volatility,
            'Skewness': market_conditions.skewness,
            'Mean Return': market_conditions.mean_return
        })
        
        stats.plot(kind='bar', ax=ax2)
        ax2.set_title('Market Statistics')
        ax2.set_ylabel('Value')
        
        plt.tight_layout()
        return fig
        
    def _calculate_portfolio_values(
        self,
        portfolio_state: PortfolioState,
        trades: List[Trade]
    ) -> pd.Series:
        """Calculate historical portfolio values
        
        Args:
            portfolio_state: Current portfolio state
            trades: List of trades
            
        Returns:
            Series of portfolio values
        """
        if not trades:
            return pd.Series([portfolio_state.capital])
            
        # Create timeline
        timeline = pd.date_range(
            min(t.entry_time for t in trades),
            max(t.exit_time if t.exit_time else t.entry_time for t in trades),
            freq='D'
        )
        
        # Calculate cumulative PnL
        pnl_df = pd.DataFrame([
            {
                'date': t.exit_time if t.exit_time else t.entry_time,
                'pnl': t.pnl if t.pnl is not None else 0
            }
            for t in trades
        ])
        
        daily_pnl = pnl_df.groupby(
            pd.Grouper(key='date', freq='D')
        )['pnl'].sum()
        
        # Calculate portfolio value
        initial_value = portfolio_state.capital - daily_pnl.sum()
        portfolio_values = pd.Series(
            initial_value,
            index=timeline
        )
        
        portfolio_values += daily_pnl.cumsum()
        
        return portfolio_values

class MarketVisualizer:
    """Market data visualization tools"""
    
    def __init__(self, config: Config):
        """Initialize market visualizer
        
        Args:
            config: System configuration
        """
        self.config = config
        
    def plot_market_data(
        self,
        market_data: Dict[str, pd.DataFrame],
        figsize: tuple = (12, 8)
    ) -> plt.Figure:
        """Plot market data
        
        Args:
            market_data: Dictionary of OHLCV DataFrames
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        n_symbols = len(market_data)
        fig, axes = plt.subplots(n_symbols, 1, figsize=figsize)
        
        if n_symbols == 1:
            axes = [axes]
            
        for ax, (symbol, df) in zip(axes, market_data.items()):
            # Plot OHLC
            ax.plot(df.index, df['close'], label='Close')
            ax.fill_between(
                df.index,
                df['low'],
                df['high'],
                alpha=0.3
            )
            
            # Add volume
            volume_ax = ax.twinx()
            volume_ax.bar(
                df.index,
                df['volume'],
                alpha=0.3,
                color='gray'
            )
            
            ax.set_title(symbol)
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            volume_ax.set_ylabel('Volume')
            
        plt.tight_layout()
        return fig
        
    def create_interactive_chart(
        self,
        market_data: Dict[str, pd.DataFrame],
        trades: Optional[List[Trade]] = None
    ) -> go.Figure:
        """Create interactive chart
        
        Args:
            market_data: Dictionary of OHLCV DataFrames
            trades: Optional list of trades to plot
            
        Returns:
            Plotly figure
        """
        n_symbols = len(market_data)
        fig = make_subplots(
            rows=n_symbols,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        for i, (symbol, df) in enumerate(market_data.items(), 1):
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=symbol
                ),
                row=i,
                col=1
            )
            
            # Add volume bars
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume',
                    opacity=0.3
                ),
                row=i,
                col=1
            )
            
            # Add trades if provided
            if trades:
                symbol_trades = [t for t in trades if t.symbol == symbol]
                if symbol_trades:
                    entries = pd.DataFrame([
                        {
                            'time': t.entry_time,
                            'price': t.entry_price,
                            'type': 'entry',
                            'direction': t.direction
                        }
                        for t in symbol_trades
                    ])
                    
                    exits = pd.DataFrame([
                        {
                            'time': t.exit_time,
                            'price': t.exit_price,
                            'type': 'exit',
                            'direction': t.direction
                        }
                        for t in symbol_trades
                        if t.exit_time is not None
                    ])
                    
                    # Plot entries
                    fig.add_trace(
                        go.Scatter(
                            x=entries['time'],
                            y=entries['price'],
                            mode='markers',
                            name='Entries',
                            marker=dict(
                                symbol='triangle-up',
                                size=10,
                                color='green'
                            )
                        ),
                        row=i,
                        col=1
                    )
                    
                    # Plot exits
                    if not exits.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=exits['time'],
                                y=exits['price'],
                                mode='markers',
                                name='Exits',
                                marker=dict(
                                    symbol='triangle-down',
                                    size=10,
                                    color='red'
                                )
                            ),
                            row=i,
                            col=1
                        )
                        
        fig.update_layout(
            height=300 * n_symbols,
            title='Market Data',
            showlegend=True
        )
        
        return fig

class StrategyVisualizer:
    """Professional visualization tools for strategy and portfolio analysis"""
    
    def __init__(self, style: str = "dark"):
        self.style = style
        self.colors = self._get_color_scheme()
    
    def plot_strategy_performance(self,
                                strategy_results: dict,
                                benchmark: Optional[pd.Series] = None) -> go.Figure:
        """Create comprehensive strategy performance visualization
        
        Args:
            strategy_results: Dictionary of strategy metrics
            benchmark: Optional benchmark returns
        """
        try:
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=(
                    "Cumulative Returns",
                    "Drawdown",
                    "Rolling Sharpe Ratio",
                    "Trade Analysis"
                ),
                vertical_spacing=0.05
            )
            
            # Plot cumulative returns
            equity_curve = strategy_results['equity_curve']
            fig.add_trace(
                go.Scatter(x=equity_curve.index, y=equity_curve.values,
                          name="Strategy"),
                row=1, col=1
            )
            
            if benchmark is not None:
                fig.add_trace(
                    go.Scatter(x=benchmark.index, y=benchmark.values,
                             name="Benchmark"),
                    row=1, col=1
                )
            
            # Plot drawdown
            drawdown = strategy_results['drawdown']
            fig.add_trace(
                go.Scatter(x=drawdown.index, y=drawdown.values,
                          name="Drawdown", fill='tozeroy'),
                row=2, col=1
            )
            
            # Plot rolling Sharpe
            sharpe = strategy_results['rolling_sharpe']
            fig.add_trace(
                go.Scatter(x=sharpe.index, y=sharpe.values,
                          name="Rolling Sharpe"),
                row=3, col=1
            )
            
            # Plot trade markers
            trades = strategy_results['trades']
            for trade in trades:
                color = self.colors['profit'] if trade['pnl'] > 0 else self.colors['loss']
                fig.add_trace(
                    go.Scatter(
                        x=[trade['entry_time'], trade['exit_time']],
                        y=[trade['entry_price'], trade['exit_price']],
                        mode='markers+lines',
                        line=dict(color=color),
                        showlegend=False
                    ),
                    row=4, col=1
                )
            
            fig.update_layout(
                height=1200,
                template="plotly_dark" if self.style == "dark" else "plotly_white"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create strategy performance plot: {str(e)}")
            raise

    def plot_risk_analysis(self,
                          portfolio: dict,
                          scenarios: Optional[List[dict]] = None) -> go.Figure:
        """Create risk analysis dashboard
        
        Args:
            portfolio: Portfolio metrics and positions
            scenarios: Optional scenario analysis results
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Value at Risk",
                    "Position Sizes",
                    "Risk Attribution",
                    "Scenario Analysis"
                )
            )
            
            # Plot VaR distribution
            returns = portfolio['returns']
            fig.add_trace(
                go.Histogram(x=returns, name="Return Distribution",
                            nbinsx=50),
                row=1, col=1
            )
            
            # Plot position sizes
            positions = portfolio['positions']
            fig.add_trace(
                go.Bar(x=list(positions.keys()), 
                      y=list(positions.values()),
                      name="Positions"),
                row=1, col=2
            )
            
            # Plot risk attribution
            risk_contrib = portfolio['risk_contribution']
            fig.add_trace(
                go.Pie(labels=list(risk_contrib.keys()),
                      values=list(risk_contrib.values()),
                      name="Risk Attribution"),
                row=2, col=1
            )
            
            # Plot scenario analysis if available
            if scenarios:
                scenario_results = pd.DataFrame(scenarios)
                fig.add_trace(
                    go.Bar(x=scenario_results['name'],
                          y=scenario_results['impact'],
                          name="Scenario Impact"),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=800,
                template="plotly_dark" if self.style == "dark" else "plotly_white"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to create risk analysis plot: {str(e)}")
            raise

    def _get_color_scheme(self) -> Dict[str, str]:
        """Get color scheme based on style"""
        if self.style == "dark":
            return {
                'profit': '#00ff1c',
                'loss': '#ff355e',
                'neutral': '#808080',
                'primary': '#1f77b4',
                'secondary': '#ff7f0e'
            }
        return {
            'profit': '#008000',
            'loss': '#ff0000',
            'neutral': '#808080',
            'primary': '#1f77b4',
            'secondary': '#ff7f0e'
        }

# Utility functions
def create_subplots(rows: int,
                   cols: int,
                   height_ratios: Optional[List[float]] = None) -> Tuple[go.Figure, List]:
    """Create figure with subplots"""
    fig = make_subplots(
        rows=rows,
        cols=cols,
        row_heights=height_ratios,
        vertical_spacing=0.03
    )
    return fig

def save_interactive_html(fig: go.Figure,
                         filename: str,
                         include_plotlyjs: bool = True) -> None:
    """Save interactive plot as standalone HTML"""
    try:
        fig.write_html(
            filename,
            include_plotlyjs=include_plotlyjs,
            full_html=True
        )
    except Exception as e:
        logger.error(f"Failed to save plot: {str(e)}")
        raise