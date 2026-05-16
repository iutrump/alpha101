"""
Agent 配置管理模块
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

class AgentConfig:
    """Agent 配置类"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        初始化 Agent 配置
        
        Args:
            env_file: .env 文件路径，如不指定则自动查找
        """
        if env_file is None:
            # 自动查找 .env 文件
            agent_dir = Path(__file__).parent
            env_file = agent_dir / ".env"
        
        # 加载 .env 文件
        if Path(env_file).exists():
            load_dotenv(env_file)
        else:
            print(f"Warning: {env_file} not found, using environment variables or defaults")
        
        # OpenAI API 配置
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4")
        
        # Agent 配置
        self.output_dir = Path(os.getenv("OUTPUT_DIR", "agent_results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_root = os.getenv("DATA_ROOT", "../futures_ml/data")
        self.lookback_days = int(os.getenv("LOOKBACK_DAYS", "252"))
        self.min_train_bars = int(os.getenv("MIN_TRAIN_BARS", "360"))
        self.train_bars = int(os.getenv("TRAIN_BARS", "365"))
        self.retrain_every_bars = int(os.getenv("RETRAIN_EVERY_BARS", str(24 * 6)))
        
        # Backtest 参数
        self.test_start_date = os.getenv("TEST_START_DATE", "2024-01-01")
        self.test_end_date = os.getenv("TEST_END_DATE", "2025-12-31")
        
        # 搜索参数
        self.population_size = int(os.getenv("POPULATION_SIZE", "30"))
        self.generations = int(os.getenv("GENERATIONS", "5"))
        self.max_factor_complexity = int(os.getenv("MAX_FACTOR_COMPLEXITY", "6"))
    
    def validate(self) -> bool:
        """
        验证配置
        
        Returns:
            配置是否有效
        """
        if not self.openai_api_key:
            print("Error: OPENAI_API_KEY not set in .env file")
            return False
        
        if not self.output_dir.exists():
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Error: Cannot create output directory: {e}")
                return False
        
        return True
    
    def __repr__(self) -> str:
        return (
            f"AgentConfig(\n"
            f"  openai_model={self.openai_model},\n"
            f"  output_dir={self.output_dir},\n"
            f"  train_bars={self.train_bars},\n"
            f"  population_size={self.population_size},\n"
            f"  generations={self.generations}\n"
            f")"
        )


# 全局配置实例
_config = None

def get_config(env_file: Optional[str] = None) -> AgentConfig:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = AgentConfig(env_file)
    return _config
