# src/rl/environment.py
"""
DEPRECATED: Legacy environment module
For backward compatibility only. Use src.rl.environments.single_agent instead.
"""

import warnings
from .environments.single_agent import StockTradingEnv

warnings.warn(
    "src.rl.environment is deprecated. Use src.rl.environments.single_agent instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backward compatibility
__all__ = ['StockTradingEnv']