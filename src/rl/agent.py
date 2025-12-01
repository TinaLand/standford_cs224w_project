# src/rl/agent.py
"""
DEPRECATED: Legacy agent module
For backward compatibility only. Use src.rl.agents.single_agent instead.
"""

import warnings
from .agents.single_agent import StockTradingAgent

warnings.warn(
    "src.rl.agent is deprecated. Use src.rl.agents.single_agent instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backward compatibility
__all__ = ['StockTradingAgent']