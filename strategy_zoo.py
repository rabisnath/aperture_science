"""
Strategy Zoo module for generating diverse strategy variants.
Provides tools for creating and managing multiple strategy configurations.
"""

from typing import List, Dict, Optional
from itertools import product
import numpy as np
from dataclasses import dataclass

from strategy import StrategyConfig
from modeling import PCAModel
from trading_types import Config

@dataclass
class StrategyRange:
    """Range configuration for strategy parameter generation"""
    min_value: float
    max_value: float
    step: float
    
    def generate_values(self) -> List[float]:
        """Generate list of values in the range"""
        return list(np.arange(self.min_value, self.max_value + self.step, self.step))

class StrategyZoo:
    """Factory for creating diverse strategy variants"""
    
    @staticmethod
    def create_mean_reversion_variants(
        symbols: List[str],
        base_name: str = "mean_rev",
        entry_range: Optional[StrategyRange] = None,
        exit_range: Optional[StrategyRange] = None,
        lookback_range: Optional[StrategyRange] = None
    ) -> List[StrategyConfig]:
        """Create mean reversion strategy variants
        
        Args:
            symbols: Trading symbols
            base_name: Base name for strategies
            entry_range: Range for entry threshold
            exit_range: Range for exit threshold
            lookback_range: Range for lookback period
            
        Returns:
            List of strategy configurations
        """
        # Default ranges if not provided
        entry_range = entry_range or StrategyRange(1.5, 3.0, 0.5)
        exit_range = exit_range or StrategyRange(0.25, 1.0, 0.25)
        lookback_range = lookback_range or StrategyRange(10, 30, 10)
        
        configs = []
        
        # Generate all combinations
        for entry, exit, lookback in product(
            entry_range.generate_values(),
            exit_range.generate_values(),
            lookback_range.generate_values()
        ):
            # Skip invalid combinations
            if exit >= entry:
                continue
                
            name = f"{base_name}_e{entry}_x{exit}_l{int(lookback)}"
            configs.append(StrategyConfig(
                name=name,
                type="mean_reversion",
                parameters={
                    'entry_threshold': float(entry),
                    'exit_threshold': float(exit),
                    'lookback_period': int(lookback)
                },
                symbols=symbols
            ))
            
        return configs
    
    @staticmethod
    def create_momentum_variants(
        symbols: List[str],
        base_name: str = "momentum",
        fast_range: Optional[StrategyRange] = None,
        slow_range: Optional[StrategyRange] = None,
        signal_range: Optional[StrategyRange] = None
    ) -> List[StrategyConfig]:
        """Create momentum strategy variants
        
        Args:
            symbols: Trading symbols
            base_name: Base name for strategies
            fast_range: Range for fast period
            slow_range: Range for slow period
            signal_range: Range for signal smoothing
            
        Returns:
            List of strategy configurations
        """
        # Default ranges if not provided
        fast_range = fast_range or StrategyRange(5, 20, 5)
        slow_range = slow_range or StrategyRange(20, 60, 20)
        signal_range = signal_range or StrategyRange(5, 15, 5)
        
        configs = []
        
        # Generate all combinations
        for fast, slow, signal in product(
            fast_range.generate_values(),
            slow_range.generate_values(),
            signal_range.generate_values()
        ):
            # Skip invalid combinations
            if fast >= slow:
                continue
                
            name = f"{base_name}_f{int(fast)}_s{int(slow)}_sig{int(signal)}"
            configs.append(StrategyConfig(
                name=name,
                type="momentum",
                parameters={
                    'fast_period': int(fast),
                    'slow_period': int(slow),
                    'signal_period': int(signal)
                },
                symbols=symbols
            ))
            
        return configs
    
    @staticmethod
    def create_pca_variants(
        symbols: List[str],
        base_name: str = "pca",
        component_range: Optional[StrategyRange] = None,
        threshold_range: Optional[StrategyRange] = None,
        window_range: Optional[StrategyRange] = None
    ) -> List[StrategyConfig]:
        """Create PCA strategy variants
        
        Args:
            symbols: Trading symbols
            base_name: Base name for strategies
            component_range: Range for number of components
            threshold_range: Range for signal threshold
            window_range: Range for rolling window
            
        Returns:
            List of strategy configurations
        """
        # Default ranges if not provided
        component_range = component_range or StrategyRange(2, 4, 1)
        threshold_range = threshold_range or StrategyRange(1.0, 2.0, 0.5)
        window_range = window_range or StrategyRange(50, 150, 50)
        
        configs = []
        config = Config()  # Create default config for PCA models
        
        # Generate all combinations
        for n_components, threshold, window in product(
            component_range.generate_values(),
            threshold_range.generate_values(),
            window_range.generate_values()
        ):
            # Skip invalid combinations
            if int(n_components) > len(symbols):
                continue
                
            name = f"{base_name}_c{int(n_components)}_t{threshold}_w{int(window)}"
            model = PCAModel(config=config, n_components=int(n_components))
            
            configs.append(StrategyConfig(
                name=name,
                type="pca",
                parameters={
                    'model': model,
                    'entry_threshold': float(threshold),
                    'window_size': int(window)
                },
                symbols=symbols
            ))
            
        return configs
    
    @staticmethod
    def create_hybrid_variants(
        symbols: List[str],
        base_name: str = "hybrid"
    ) -> List[StrategyConfig]:
        """Create hybrid strategy variants combining multiple signals
        
        Args:
            symbols: Trading symbols
            base_name: Base name for strategies
            
        Returns:
            List of strategy configurations
        """
        configs = []
        
        # Example hybrid combinations
        combinations = [
            {
                'mean_rev_weight': 0.7,
                'momentum_weight': 0.3,
                'entry_threshold': 1.5
            },
            {
                'mean_rev_weight': 0.5,
                'momentum_weight': 0.5,
                'entry_threshold': 2.0
            },
            {
                'mean_rev_weight': 0.3,
                'momentum_weight': 0.7,
                'entry_threshold': 2.5
            }
        ]
        
        for i, params in enumerate(combinations):
            name = f"{base_name}_mr{params['mean_rev_weight']}_mo{params['momentum_weight']}"
            configs.append(StrategyConfig(
                name=name,
                type="hybrid",
                parameters=params,
                symbols=symbols
            ))
            
        return configs
    
    @staticmethod
    def create_all_variants(
        symbols: List[str],
        max_strategies: int = 50
    ) -> List[StrategyConfig]:
        """Create all strategy variants
        
        Args:
            symbols: Trading symbols
            max_strategies: Maximum number of strategies to create
            
        Returns:
            List of strategy configurations
        """
        all_configs = []
        
        # Generate all variants with equal allocation
        max_per_type = max_strategies // 4  # Divide equally among strategy types
        
        # Generate mean reversion variants
        mean_rev_configs = StrategyZoo.create_mean_reversion_variants(symbols)
        all_configs.extend(mean_rev_configs[:max_per_type])
        
        # Generate momentum variants
        momentum_configs = StrategyZoo.create_momentum_variants(symbols)
        all_configs.extend(momentum_configs[:max_per_type])
        
        # Generate PCA variants
        pca_configs = StrategyZoo.create_pca_variants(symbols)
        all_configs.extend(pca_configs[:max_per_type])
        
        # Generate hybrid variants
        hybrid_configs = StrategyZoo.create_hybrid_variants(symbols)
        all_configs.extend(hybrid_configs[:max_per_type])
        
        # Sort by strategy type and name
        all_configs.sort(key=lambda x: (x.type, x.name))
        
        # Limit to max_strategies while maintaining diversity
        return all_configs[:max_strategies] 