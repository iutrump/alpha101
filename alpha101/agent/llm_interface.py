"""
LLM 与 OpenAI API 的交互接口
"""
import json
import time
import re
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from pathlib import Path

try:
    from openai import OpenAI, RateLimitError, APIError
except ImportError:
    print("Warning: openai library not installed. Install with: pip install openai")
    OpenAI = None

from .config import get_config


@dataclass
class Factor:
    """因子数据类"""
    name: str
    expression: str
    direction: str  # "多空", "反转", "动量" 等
    sharpe_ratio: Optional[float] = None
    returns: Optional[float] = None
    ic_mean: Optional[float] = None
    fitness: Optional[float] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'expression': self.expression,
            'direction': self.direction,
            'sharpe_ratio': self.sharpe_ratio,
            'returns': self.returns,
            'ic_mean': self.ic_mean,
            'fitness': self.fitness,
            'notes': self.notes
        }


class LLMInterface:
    """LLM 交互接口"""
    
    def __init__(self, config=None):
        """
        初始化 LLM 接口
        
        Args:
            config: AgentConfig 实例
        """
        if config is None:
            config = get_config()
        
        self.config = config
        
        if OpenAI is None:
            raise ImportError("openai library is required. Install with: pip install openai")
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_api_base
        )
        
        self.model = self.config.openai_model
        self.conversation_history = []
        self.max_retries = 3
        self.retry_delay = 1
    
    def _make_request(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """
        向 OpenAI API 发送请求，带重试机制
        
        Args:
            messages: 消息列表
            temperature: 温度参数（0-2）
            max_tokens: 最大生成令牌数
            
        Returns:
            LLM 的回复文本
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            except RateLimitError:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise
            
            except APIError as e:
                if attempt < self.max_retries - 1:
                    print(f"API error: {e}. Retrying...")
                    time.sleep(self.retry_delay)
                else:
                    raise
        
        raise RuntimeError("Failed to get response from OpenAI API")
    
    def select_direction(self) -> str:
        """
        让 LLM 随机选择考察方向
        
        Returns:
            选择的方向名称（中文）
        """
        directions = ["量价背离", "动量", "反转", "均值回归", "波动率", "成交量异常"]
        
        prompt = f"""你是一个专业的量化交易因子挖掘专家。

请从以下方向中随机选择一个进行因子挖掘：
{json.dumps(directions, ensure_ascii=False, indent=2)}

你的回复格式应为：
选择的方向: [方向名称]
原因: [简要说明为什么选择这个方向]

请仅返回上述格式的信息，不要额外的解释。"""

        response = self._make_request([{"role": "user", "content": prompt}], temperature=0.9, max_tokens=500)
        
        # 提取方向名称
        for direction in directions:
            if direction in response:
                return direction
        
        # 如果没有找到，使用默认值
        return "动量"
    
    def generate_factor_ideas(self, direction: str, context: Dict[str, Any]) -> List[Dict]:
        """
        基于方向生成因子创意
        
        Args:
            direction: 挖掘方向
            context: 包含市场信息和可用算子的上下文
            
        Returns:
            因子创意列表
        """
        operators_str = json.dumps(context['operators'], ensure_ascii=False, indent=2)
        
        prompt = f"""你是一个专业的量化交易因子挖掘专家。

## 市场信息
- 资产类别: {context['asset_class']}
- 持仓方向: {context['direction_type']}
- 时间频率: {context['frequency']}
- 当前日期: {context['current_date']}

## 可用的计算算子
{operators_str}

## 任务
基于"{direction}"方向，请生成3个高质量的因子创意。

对于每个创意，请提供：
1. 因子名称
2. 核心假设或逻辑
3. 使用的主要算子
4. 预期表现方向

请按照以下 JSON 格式返回结果（仅返回 JSON，不包含其他文本）：
[
  {{
    "name": "因子名称",
    "hypothesis": "核心假设",
    "operators": ["算子1", "算子2"],
    "expected_direction": "正/负"
  }}
]

请确保返回的是有效的 JSON 格式。"""

        response = self._make_request(
            [{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=1500
        )
        
        # 提取 JSON
        try:
            # 查找 JSON 块
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                ideas = json.loads(json_match.group())
                return ideas
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {response}")
        
        return []
    
    def construct_factor_expression(self, idea: Dict, examples: List[str]) -> str:
        """
        基于创意构造因子表达式
        
        Args:
            idea: 因子创意字典
            examples: 表达式示例列表
            
        Returns:
            因子表达式
        """
        examples_str = "\n".join([f"- {ex}" for ex in examples])
        
        prompt = f"""你是一个专业的量化交易因子挖掘专家。

## 因子创意
名称: {idea['name']}
假设: {idea['hypothesis']}
预期方向: {idea['expected_direction']}

## 表达式示例
{examples_str}

## 任务
请根据上述创意和示例，构造一个可执行的因子表达式。

要求：
1. 使用提供的示例中的算子和模式
2. 表达式应该简洁高效，避免过度复杂
3. 使用英文小写
4. 最后一行应为最终的因子表达式

请仅返回因子表达式，不需要其他说明。表达式应该是可以直接执行的代码形式，不要用```包裹"""

        response = self._make_request(
            [{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=500
        )
        
        # 提取最后一行作为表达式
        lines = [line.strip() for line in response.replace('```','').strip().split('\n') if line.strip()]
        if lines:
            return lines[-1]
        return ""
    
    def reflect_on_results(
        self,
        factor: Factor,
        previous_results: List[Factor],
        improvement_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        基于回测结果进行反思，提出改进建议
        
        Args:
            factor: 当前评估的因子
            previous_results: 之前的因子结果列表
            improvement_history: 改进历史（之前迭代的结果）
            
        Returns:
            包含反思和建议的字典
        """
        # 格式化之前的结果
        previous_str = ""
        if previous_results:
            for pf in previous_results[-5:]:  # 只看前5个
                previous_str += f"\n- {pf.name}: Sharpe={pf.sharpe_ratio:.4f}, Returns={pf.returns}"
        
        # 格式化改进历史
        history_str = ""
        if improvement_history:
            for idx, hist in enumerate(improvement_history[-3:], 1):  # 只看最近3次迭代
                improved = hist.get('improved_factor', {})
                history_str += f"\n迭代 {idx}:"
                history_str += f"\n  - Sharpe: {improved.get('sharpe_ratio', 0):.4f}"
                history_str += f"\n  - Returns: {improved.get('returns', 0)}"
                history_str += f"\n  - IC 均值: {improved.get('ic_mean', 0):.4f}"
                history_str += f"\n  - 改进建议: {', '.join(hist.get('reflection', {}).get('improvements', [])[:2])}  "
        
        prompt = f"""你是一个专业的量化交易因子挖掘专家。

## 当前因子评估结果
名称: {factor.name}
表达式: {factor.expression}
Sharpe 比率: {factor.sharpe_ratio:.4f}
年化收益: {factor.returns}
IC 均值: {factor.ic_mean:.4f}
综合评分: {factor.fitness:.2f}

## 之前的结果
{previous_str if previous_str else "无"}

## 改进迭代历史
{history_str if history_str else "这是首次迭代"}"

## 任务
请分析这个因子的表现，并提出改进建议。

请回复以下内容：
1. 性能评估（好/一般/差）
2. 主要优势
3. 主要劣势
4. 改进方向（3个建议）
5. 下一步策略

请按照以下 JSON 格式返回（仅返回 JSON）：
{{
  "rating": "评级",
  "strengths": ["优势1", "优势2"],
  "weaknesses": ["劣势1", "劣势2"],
  "improvements": ["建议1", "建议2", "建议3"],
  "next_steps": "下一步策略"
}}"""

        response = self._make_request(
            [{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=1000
        )
        
        # 提取 JSON
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                reflection = json.loads(json_match.group())
                return reflection
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {response}")
        
        return {
            "rating": "无法解析",
            "strengths": [],
            "weaknesses": [],
            "improvements": [],
            "next_steps": ""
        }
    
    def iterate_factor(self, factor: Factor, reflection: Dict, examples: List[str]) -> str:
        """
        基于反思结果迭代改进因子表达式
        
        Args:
            factor: 原始因子
            reflection: 反思结果字典
            examples: 表达式示例列表
            
        Returns:
            改进后的因子表达式
        """
        improvements_str = "\n".join([f"- {imp}" for imp in reflection.get('improvements', [])])
        examples_str = "\n".join([f"- {ex}" for ex in examples])
        
        prompt = f"""你是一个专业的量化交易因子挖掘专家。

## 原始因子
名称: {factor.name}
表达式: {factor.expression}
Sharpe: {factor.sharpe_ratio:.4f}

## 改进建议
{improvements_str}

## 表达式示例
{examples_str}

## 任务
基于改进建议，生成一个改进版本的因子表达式。

要求：
1. 核心思想保持一致，但设计更优化
2. 避免过度复杂
3. 仅返回最终的改进表达式，不需要其他说明，表达式应该是可以直接执行的代码形式。不要用```包裹"""

        response = self._make_request(
            [{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=500
        )
        
        # 提取最后一行作为表达式
        lines = [line.strip() for line in response.replace('```','').strip().split('\n') if line.strip()]
        if lines:
            return lines[-1]
        return ""
    
    def generate_factor_candidates(
        self,
        factor: Factor,
        reflection: Dict,
        examples: List[str],
        num_candidates: int = 3
    ) -> List[str]:
        """
        基于反思结果生成多个改进候选表达式
        
        Args:
            factor: 原始因子
            reflection: 反思结果字典
            examples: 表达式示例列表
            num_candidates: 生成候选数量
            
        Returns:
            改进后的因子表达式列表
        """
        improvements_str = "\n".join([f"- {imp}" for imp in reflection.get('improvements', [])])
        examples_str = "\n".join([f"- {ex}" for ex in examples])
        
        prompt = f"""你是一个专业的量化交易因子挖掘专家。

## 原始因子
名称: {factor.name}
表达式: {factor.expression}
Sharpe: {factor.sharpe_ratio:.4f}

## 改进建议
{improvements_str}

## 表达式示例
{examples_str}

## 任务
基于改进建议，生成 {num_candidates} 个不同的改进候选表达式。

要求：
1. 每个候选需要新颖，体现不同的改进思路
2. 避免过度复杂
3. 用可执行的代码形式，不要用```包裹
4. 按行返回，每行一个表达式

仅返回 {num_candidates} 行表达式，不需要其他说明。"""

        response = self._make_request(
            [{"role": "user", "content": prompt}],
            temperature=0.8,  # 提高温度以获得更多样性
            max_tokens=800
        )
        
        # 提取所有行作为候选表达式
        lines = [
            line.strip()
            for line in response.replace('```', '').strip().split('\n')
            if line.strip() and not line.strip().startswith('#')
        ]
        
        return lines[:num_candidates]
