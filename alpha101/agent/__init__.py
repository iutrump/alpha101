"""
Alpha 因子自动挖掘 Agent 模块
支持 LLM 驱动的因子生成、回测和迭代优化
"""

from .config import AgentConfig, get_config
from .llm_interface import LLMInterface, Factor
from .agent import AlphaAgent

__all__ = ['AgentConfig', 'get_config', 'LLMInterface', 'Factor', 'AlphaAgent']
__version__ = '0.1.0'
