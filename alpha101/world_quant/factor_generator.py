"""
因子生成器模块
支持多种因子生成策略：随机生成、模板变异、参数化搜索
"""
import ast
import copy
import random
import numpy as np
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
import itertools


@dataclass(frozen=True)
class AstNode:
    pass


@dataclass(frozen=True)
class NumberNode(AstNode):
    value: float


@dataclass(frozen=True)
class FieldNode(AstNode):
    name: str


@dataclass(frozen=True)
class UnaryOpNode(AstNode):
    op: str
    operand: AstNode


@dataclass(frozen=True)
class BinaryOpNode(AstNode):
    op: str
    left: AstNode
    right: AstNode


@dataclass(frozen=True)
class FunctionNode(AstNode):
    name: str
    args: Tuple[AstNode, ...]


class FactorGenerator:
    """因子表达式生成器"""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 基础数据字段
        self.data_fields = ['open', 'high', 'low', 'close', 'volume', 'returns', 'vwap', 'cap', 'market_return', 'funding']
        
        # 时序算子及其参数范围（单变量）
        self.ts_operators = {
            'ts_mean': [3, 12, 20, 30, 60],
            'ts_rank': [3, 5, 7, 10, 14, 20, 30],
            'ts_min': [3, 5, 7, 10, 14, 20, 30],
            'ts_max': [3, 5, 7, 10, 14, 20, 30],
            'ts_std_dev': [7, 14, 21, 28, 54],
            'ts_arg_max': [7, 14, 21, 28],
            'ts_arg_min': [7, 14, 21, 28],
            'ts_sum': [5, 10, 20, 30, 60],
            'ts_product': [5, 10, 20],
            'ts_skewness': [10, 20, 30, 60],
            'ts_kurtosis': [10, 20, 30, 60],
            'ts_decay_linear': [7, 14, 21, 28],
            'ts_drawdown': [7, 14, 21, 28],
            'ts_pos': [7, 14, 21, 28],
            'ts_zscore': [7, 14, 21, 28],
            'ts_ema': [6, 12, 24, 48],
            'ts_slope': [6, 12, 24, 48],
        }
        
        # 双变量时序算子（需要两个输入序列）
        self.ts_dual_operators = {
            'ts_corr': [5, 10, 20, 30, 60],
            'ts_covariance': [5, 10, 20, 30, 60],
            'ts_alpha': [7, 14, 21, 28],
            'ts_r2': [7, 14, 21, 28],
            'ts_beta': [7, 14, 21, 28],
            'ts_resid': [6, 12, 24, 48],
        }
        
        # 滞后算子参数范围
        self.delay_params = [1, 3, 5, 7, 10, 14, 20]
        self.delta_params = [1, 3, 5, 7, 10]
        
        # 截面算子（无参数）
        self.cross_operators = ['rank', 'scale', 'zscore', 'winsorize']
        
        # 数学运算符
        self.binary_ops = ['+', '-', '*', '/']
        self.unary_ops = ['log', '-', 'abs', 'sqrt', 'sign']
        
        # 复杂度控制
        self.max_depth = 4
        self.max_operators = 8
        
        # 语义规则：禁止的嵌套模式
        self.forbidden_patterns = [
            (r'ts_mean\(ts_mean', 'nested ts_mean'),  # 避免重复平均
            (r'ts_rank\(ts_rank', 'nested ts_rank'),  # 避免重复排名
            (r'rank\(rank', 'nested rank'),
            (r'scale\(scale', 'nested scale'),
            (r'abs\(abs', 'nested abs'),
            (r'ts_std_dev\(ts_std_dev', 'nested ts_std_dev'),
            (r'ts_arg_max\([^)]*\)[^)]*ts_mean', 'ts_arg_max with ts_mean'),  # argmax后接ts_mean无意义
            (r'ts_arg_min\([^)]*\)[^)]*ts_mean', 'ts_arg_min with ts_mean'),
            (r'ts_corr\(ts_corr', 'nested ts_corr'),  # 避免嵌套相关性
            (r'ts_covariance\(ts_covariance', 'nested ts_covariance'),
        ]
    
    def calculate_complexity(self, expr: str) -> Dict[str, int]:
        """
        ????????????????????
        
        Args:
            expr: ????????????
            
        Returns:
            ??????????????????
        """
        try:
            root = self._parse_expr_to_ast(expr)
            stats = self._calculate_ast_stats(root)
            ts_op_count = stats['ts_operators']
            ts_dual_op_count = stats['ts_dual_operators']
            cross_op_count = stats['cross_operators']
            delay_delta_count = stats['delay_delta_operators']
            unary_op_count = stats['unary_operators']
            binary_op_count = stats['binary_operators']
            max_depth = stats['max_depth']
        except Exception:
            ts_op_count = sum(expr.count(op) for op in self.ts_operators.keys())
            ts_dual_op_count = sum(expr.count(op) for op in self.ts_dual_operators.keys())
            cross_op_count = sum(expr.count(op) for op in self.cross_operators)
            delay_delta_count = expr.count('ts_delay') + expr.count('ts_delta')
            unary_op_count = expr.count('log') + expr.count('abs')
            binary_op_count = sum(expr.count(f' {op} ') for op in self.binary_ops)
            max_depth = 0
            current_depth = 0
            for char in expr:
                if char == '(':
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif char == ')':
                    current_depth -= 1

        operator_count = (
            ts_op_count
            + ts_dual_op_count
            + cross_op_count
            + delay_delta_count
            + unary_op_count
            + binary_op_count
        )
        expr_length = len(expr)

        return {
            'total_operators': operator_count,
            'max_depth': max_depth,
            'expression_length': expr_length,
            'ts_operators': ts_op_count,
            'ts_dual_operators': ts_dual_op_count,
            'cross_operators': cross_op_count,
            'binary_operators': binary_op_count,
            'complexity_score': operator_count * 1.0 + max_depth * 0.5 + expr_length * 0.01
        }

    def is_semantically_valid(self, expr: str) -> Tuple[bool, str]:
        """
        检查因子表达式是否具有语义有效性
        
        Args:
            expr: 因子表达式字符串
            
        Returns:
            (是否有效, 失败原因)
        """
        # 检查禁止的嵌套模式
        for pattern, reason in self.forbidden_patterns:
            if re.search(pattern, expr):
                return False, f"Forbidden pattern: {reason}"
        
        # 检查是否过于简单（只有一个字段）
        if expr in self.data_fields:
            return False, "Too simple: single field"
        
        # 检查是否包含有效算子
        has_operator = False
        all_operators = (list(self.ts_operators.keys()) + 
                        list(self.ts_dual_operators.keys()) + 
                        self.cross_operators + 
                        ['ts_delay', 'ts_delta'])
        for op in all_operators:
            if op in expr:
                has_operator = True
                break
        
        if not has_operator:
            return False, "No valid operator found"
        
        return True, "Valid"
    
    def filter_by_complexity(self, expr: str, max_complexity: float = 10.0) -> bool:
        """
        根据复杂度过滤因子
        
        Args:
            expr: 因子表达式
            max_complexity: 最大允许的复杂度分数
            
        Returns:
            是否通过过滤
        """
        complexity = self.calculate_complexity(expr)
        return complexity['complexity_score'] <= max_complexity
    
    def generate_random_factor(self, depth: int = 0, used_ops: int = 0) -> str:
        """
        递归生成随机因子表达式
        
        Args:
            depth: 当前递归深度
            used_ops: 已使用的算子数量
            
        Returns:
            因子表达式字符串
        """
        ast_root = self._generate_random_ast(depth, used_ops)
        ast_root = self._sanitize_ast(ast_root)
        return self._ast_to_string(ast_root)
    
    def _ast_to_string(self, node: AstNode) -> str:
        if isinstance(node, NumberNode):
            if float(node.value).is_integer():
                return str(int(node.value))
            return str(float(node.value))
        if isinstance(node, FieldNode):
            return node.name
        if isinstance(node, UnaryOpNode):
            inner = self._ast_to_string(node.operand)
            if node.op == '-':
                return f"-({inner})" if isinstance(node.operand, BinaryOpNode) else f"-{inner}"
            return f"{node.op}({inner})"
        if isinstance(node, BinaryOpNode):
            left = self._ast_to_string(node.left)
            right = self._ast_to_string(node.right)
            return f"({left} {node.op} {right})"
        if isinstance(node, FunctionNode):
            args = ", ".join(self._ast_to_string(arg) for arg in node.args)
            return f"{node.name}({args})"
        raise ValueError(f"Unsupported AST node: {type(node)}")

    def _generate_random_ast(self, depth: int = 0, used_ops: int = 0) -> AstNode:
        # 终止条件：达到最大深度或算子数量
        if depth >= self.max_depth or used_ops >= self.max_operators or (random.random() < 0.3 and depth > 0):
            return self._generate_simple_ast()

        strategies = [
            self._generate_ts_operator_ast,
            self._generate_ts_dual_operator_ast,
            self._generate_cross_operator_ast,
            self._generate_delay_delta_ast,
            self._generate_binary_ast,
            self._generate_unary_ast,
            self._generate_simple_ast,
        ]

        weights = [0.30, 0.17, 0.17, 0.15, 0.15, 0.03, 0.03]
        strategy = random.choices(strategies, weights=weights)[0]
        return strategy(depth, used_ops)

    def _generate_simple_ast(self, depth: int = 0, used_ops: int = 0) -> AstNode:
        if random.random() < 0.9:
            return FieldNode(random.choice(self.data_fields))
        return NumberNode(float(random.choice([0.1, 0.5, 1, 2, -1])))

    def _generate_ts_operator_ast(self, depth: int, used_ops: int) -> AstNode:
        op_name = random.choice(list(self.ts_operators.keys()))
        param = random.choice(self.ts_operators[op_name])
        inner = self._generate_random_ast(depth + 1, used_ops + 1)
        inner = self._ensure_series_node(inner)
        return FunctionNode(op_name, (inner, NumberNode(float(param))))

    def _generate_ts_dual_operator_ast(self, depth: int, used_ops: int) -> AstNode:
        op_name = random.choice(list(self.ts_dual_operators.keys()))
        param = random.choice(self.ts_dual_operators[op_name])
        field1 = random.choice(self.data_fields)
        field2 = random.choice([f for f in self.data_fields if f != field1])
        return FunctionNode(op_name, (FieldNode(field1), FieldNode(field2), NumberNode(float(param))))

    def _generate_cross_operator_ast(self, depth: int, used_ops: int) -> AstNode:
        op_name = random.choice(self.cross_operators)
        inner = self._generate_random_ast(depth + 1, used_ops + 1)
        inner = self._ensure_series_node(inner)
        return FunctionNode(op_name, (inner,))

    def _generate_delay_delta_ast(self, depth: int, used_ops: int) -> AstNode:
        if random.random() < 0.5:
            param = random.choice(self.delay_params)
            inner = self._generate_random_ast(depth + 1, used_ops + 1)
            inner = self._ensure_series_node(inner)
            return FunctionNode('ts_delay', (inner, NumberNode(float(param))))
        param = random.choice(self.delta_params)
        inner = self._generate_random_ast(depth + 1, used_ops + 1)
        inner = self._ensure_series_node(inner)
        return FunctionNode('ts_delta', (inner, NumberNode(float(param))))

    def _generate_binary_ast(self, depth: int, used_ops: int) -> AstNode:
        op = random.choice(self.binary_ops)
        left = self._generate_random_ast(depth + 1, used_ops + 1)
        right = self._generate_random_ast(depth + 1, used_ops + 1)
        return BinaryOpNode(op, left, right)

    def _generate_unary_ast(self, depth: int, used_ops: int) -> AstNode:
        op = random.choice(self.unary_ops)
        inner = self._generate_random_ast(depth + 1, used_ops + 1)
        if op == '-':
            return UnaryOpNode('-', inner)
        inner = self._ensure_series_node(inner)
        return FunctionNode(op, (inner,))

    def _parse_expr_to_ast(self, expr: str) -> AstNode:
        py_ast = ast.parse(expr, mode='eval')
        return self._from_py_ast(py_ast.body)

    def _from_py_ast(self, node: ast.AST) -> AstNode:
        if isinstance(node, ast.Name):
            return FieldNode(node.id)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return NumberNode(float(node.value))
            raise ValueError(f"Unsupported constant: {node.value}")
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return UnaryOpNode('-', self._from_py_ast(node.operand))
            if isinstance(node.op, ast.UAdd):
                return self._from_py_ast(node.operand)
            raise ValueError(f"Unsupported unary op: {node.op}")
        if isinstance(node, ast.BinOp):
            op_map = {
                ast.Add: '+',
                ast.Sub: '-',
                ast.Mult: '*',
                ast.Div: '/',
            }
            for op_type, op_str in op_map.items():
                if isinstance(node.op, op_type):
                    return BinaryOpNode(op_str, self._from_py_ast(node.left), self._from_py_ast(node.right))
            raise ValueError(f"Unsupported binary op: {node.op}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            else:
                raise ValueError("Unsupported call target")
            args = tuple(self._from_py_ast(arg) for arg in node.args)
            return FunctionNode(name, args)
        raise ValueError(f"Unsupported AST node: {type(node)}")

    def _collect_nodes(self, node: AstNode) -> List[AstNode]:
        nodes = [node]
        if isinstance(node, UnaryOpNode):
            nodes.extend(self._collect_nodes(node.operand))
        elif isinstance(node, BinaryOpNode):
            nodes.extend(self._collect_nodes(node.left))
            nodes.extend(self._collect_nodes(node.right))
        elif isinstance(node, FunctionNode):
            for arg in node.args:
                nodes.extend(self._collect_nodes(arg))
        return nodes

    def _replace_subtree(self, node: AstNode, target: AstNode, replacement: AstNode) -> AstNode:
        if node is target:
            return replacement
        if isinstance(node, UnaryOpNode):
            return UnaryOpNode(node.op, self._replace_subtree(node.operand, target, replacement))
        if isinstance(node, BinaryOpNode):
            return BinaryOpNode(
                node.op,
                self._replace_subtree(node.left, target, replacement),
                self._replace_subtree(node.right, target, replacement),
            )
        if isinstance(node, FunctionNode):
            new_args = tuple(self._replace_subtree(arg, target, replacement) for arg in node.args)
            return FunctionNode(node.name, new_args)
        return node

    def _infer_node_type(self, node: AstNode) -> str:
        """
        Infer node output type for typed crossover.
        Returns one of: "series", "scalar".
        """
        if isinstance(node, NumberNode):
            return "scalar"
        if isinstance(node, FieldNode):
            return "series"
        if isinstance(node, UnaryOpNode):
            return self._infer_node_type(node.operand)
        if isinstance(node, BinaryOpNode):
            left_type = self._infer_node_type(node.left)
            right_type = self._infer_node_type(node.right)
            if left_type == "series" or right_type == "series":
                return "series"
            return "scalar"
        if isinstance(node, FunctionNode):
            # All supported factor functions output a timeseries.
            return "series"
        return "series"

    def _collect_nodes_by_type(self, node: AstNode) -> Dict[str, List[AstNode]]:
        nodes = self._collect_nodes(node)
        grouped: Dict[str, List[AstNode]] = {"series": [], "scalar": []}
        for n in nodes:
            grouped[self._infer_node_type(n)].append(n)
        return grouped

    def _calculate_ast_stats(self, node: AstNode, depth: int = 1) -> Dict[str, int]:
        stats = {
            'ts_operators': 0,
            'ts_dual_operators': 0,
            'cross_operators': 0,
            'delay_delta_operators': 0,
            'unary_operators': 0,
            'binary_operators': 0,
            'max_depth': depth,
        }

        if isinstance(node, UnaryOpNode):
            stats['unary_operators'] += 1
            child = self._calculate_ast_stats(node.operand, depth + 1)
            for k, v in child.items():
                if k == 'max_depth':
                    stats[k] = max(stats[k], v)
                else:
                    stats[k] += v
            return stats

        if isinstance(node, BinaryOpNode):
            stats['binary_operators'] += 1
            left = self._calculate_ast_stats(node.left, depth + 1)
            right = self._calculate_ast_stats(node.right, depth + 1)
            for child in (left, right):
                for k, v in child.items():
                    if k == 'max_depth':
                        stats[k] = max(stats[k], v)
                    else:
                        stats[k] += v
            return stats

        if isinstance(node, FunctionNode):
            if node.name in self.ts_operators:
                stats['ts_operators'] += 1
            elif node.name in self.ts_dual_operators:
                stats['ts_dual_operators'] += 1
            elif node.name in self.cross_operators:
                stats['cross_operators'] += 1
            elif node.name in ('ts_delay', 'ts_delta'):
                stats['delay_delta_operators'] += 1
            elif node.name in self.unary_ops:
                stats['unary_operators'] += 1

            for arg in node.args:
                child = self._calculate_ast_stats(arg, depth + 1)
                for k, v in child.items():
                    if k == 'max_depth':
                        stats[k] = max(stats[k], v)
                    else:
                        stats[k] += v
            return stats

        return stats

    def _ensure_series_node(self, node: AstNode) -> AstNode:
        if self._is_constant_expr(node):
            return FieldNode(random.choice(self.data_fields))
        return node

    def _is_constant_expr(self, node: AstNode) -> bool:
        if isinstance(node, NumberNode):
            return True
        if isinstance(node, FieldNode):
            return False
        if isinstance(node, UnaryOpNode):
            return self._is_constant_expr(node.operand)
        if isinstance(node, BinaryOpNode):
            return self._is_constant_expr(node.left) and self._is_constant_expr(node.right)
        if isinstance(node, FunctionNode):
            return all(self._is_constant_expr(arg) for arg in node.args)
        return False

    def _sanitize_positive_window(self, node: AstNode, fallback: int = 1) -> NumberNode:
        """
        Ensure a window/lag argument is a positive integer NumberNode.
        Non-numeric expressions or non-positive values are replaced by fallback.
        """
        if isinstance(node, NumberNode):
            try:
                val = int(round(float(node.value)))
            except Exception:
                val = fallback
            if val <= 0:
                val = fallback
            return NumberNode(float(val))
        return NumberNode(float(max(1, int(fallback))))

    def _sanitize_ast(self, node: AstNode) -> AstNode:
        if isinstance(node, UnaryOpNode):
            operand = self._sanitize_ast(node.operand)
            return UnaryOpNode(node.op, operand)
        if isinstance(node, BinaryOpNode):
            left = self._sanitize_ast(node.left)
            right = self._sanitize_ast(node.right)
            return BinaryOpNode(node.op, left, right)
        if isinstance(node, FunctionNode):
            args = tuple(self._sanitize_ast(arg) for arg in node.args)
            # Collapse redundant abs nesting: abs(abs(x)) -> abs(x)
            if node.name == 'abs' and args and isinstance(args[0], FunctionNode) and args[0].name == 'abs':
                return args[0]
            if node.name in self.ts_operators or node.name in self.cross_operators:
                if args:
                    args = (self._ensure_series_node(args[0]),) + args[1:]
            if node.name in self.ts_operators and len(args) >= 2:
                # ts_xxx(series, window): window must be positive.
                args = (args[0], self._sanitize_positive_window(args[1], fallback=1)) + args[2:]
            if node.name in ('ts_delay', 'ts_delta'):
                if args:
                    args = (self._ensure_series_node(args[0]),) + args[1:]
                if len(args) >= 2:
                    # ts_delay/ts_delta(series, lag): lag must be positive.
                    args = (args[0], self._sanitize_positive_window(args[1], fallback=1)) + args[2:]
            if node.name in self.ts_dual_operators:
                if len(args) >= 2:
                    args = (self._ensure_series_node(args[0]), self._ensure_series_node(args[1])) + args[2:]
                if len(args) >= 3:
                    # ts_corr/ts_cov/...(..., ..., window): window must be positive.
                    args = (args[0], args[1], self._sanitize_positive_window(args[2], fallback=1)) + args[3:]
            if node.name in self.unary_ops and node.name != '-':
                if args:
                    args = (self._ensure_series_node(args[0]),) + args[1:]
            return FunctionNode(node.name, args)
        return node
    
    def generate_template_based_factors(self, templates: List[str]) -> List[Tuple[str, str]]:
        """
        基于模板生成因子变体
        
        Args:
            templates: 因子模板列表，使用占位符如 {{field}}, {{window}}
            
        Returns:
            (因子名, 因子表达式) 列表
        """
        factors = []
        
        for template_idx, template in enumerate(templates):
            # 替换字段占位符
            for field in self.data_fields:
                expr = template.replace('{{field}}', field)
                
                # 替换窗口占位符
                for window in [3, 5, 7, 10, 14, 20, 30]:
                    final_expr = expr.replace('{{window}}', str(window))
                    
                    # 替换延迟占位符
                    for ts_delay in [1, 2, 3, 5]:
                        final_final_expr = final_expr.replace('{{ts_delay}}', str(ts_delay))
                        
                        # 如果还有占位符，跳过
                        if '{{' in final_final_expr:
                            continue
                        
                        factor_name = f"template_{template_idx}_f{field}_w{window}_d{ts_delay}"
                        factors.append((factor_name, final_final_expr))
                        
                        # 限制每个模板的变体数量
                        if len([f for f in factors if f[0].startswith(f"template_{template_idx}")]) >= 50:
                            break
                    
                    if len([f for f in factors if f[0].startswith(f"template_{template_idx}")]) >= 50:
                        break
                
                if len([f for f in factors if f[0].startswith(f"template_{template_idx}")]) >= 50:
                    break
        
        return factors
    
    def generate_grid_search_factors(self) -> List[Tuple[str, str]]:
        """
        网格搜索：系统地组合算子和参数
        
        Returns:
            (因子名, 因子表达式) 列表
        """
        factors = []
        idx = 0
        
        # 1. 单一时序算子 + 单一字段
        for field in self.data_fields:
            for op_name, params in self.ts_operators.items():
                for param in params[:5]:  # 限制参数数量
                    expr = f"{op_name}({field}, {param})"
                    factors.append((f"grid_single_{idx}", expr))
                    idx += 1
        
        # 2. 时序算子组合
        for field in ['close', 'returns']:
            for op1, params1 in list(self.ts_operators.items())[:3]:
                for op2, params2 in list(self.ts_operators.items())[:3]:
                    p1 = random.choice(params1)
                    p2 = random.choice(params2)
                    
                    # 不同参数的组合
                    if p1 != p2:
                        expr = f"{op1}({field}, {p1}) - {op2}({field}, {p2})"
                        factors.append((f"grid_combo_{idx}", expr))
                        idx += 1
        
        # 3. 延迟对比
        for field in ['close', 'high', 'low', 'open']:
            for delay_param in [1, 2, 3, 5]:
                expr = f"{field} - ts_delay({field}, {delay_param})"
                factors.append((f"grid_delay_{idx}", expr))
                idx += 1
        
        # 4. 排名相关
        for field in ['close', 'volume', 'returns']:
            for window in [5, 10, 20]:
                expr = f"rank(ts_mean({field}, {window}))"
                factors.append((f"grid_rank_{idx}", expr))
                idx += 1
        
        return factors
    
    def mutate_expression(self, expr: str) -> str:
        """
        对现有因子表达式进行变异
        
        变异策略：
        1. 修改参数值
        2. 替换算子
        3. 添加/删除一层运算
        """
        try:
            root = self._parse_expr_to_ast(expr)
        except Exception:
            root = self._generate_random_ast()

        nodes = self._collect_nodes(root)
        if not nodes:
            return expr

        def modify_param(node: AstNode) -> Optional[AstNode]:
            candidates = [n for n in nodes if isinstance(n, FunctionNode)]
            if not candidates:
                return None
            target_node = random.choice(candidates)
            if target_node.name in self.ts_operators and len(target_node.args) >= 2:
                params = self.ts_operators[target_node.name]
                current = target_node.args[1]
                if isinstance(current, NumberNode) and len(params) > 1:
                    new_param = random.choice([p for p in params if p != int(current.value)])
                    new_args = (target_node.args[0], NumberNode(float(new_param)))
                    return self._replace_subtree(root, target_node, FunctionNode(target_node.name, new_args))
            if target_node.name in self.ts_dual_operators and len(target_node.args) >= 3:
                params = self.ts_dual_operators[target_node.name]
                current = target_node.args[2]
                if isinstance(current, NumberNode) and len(params) > 1:
                    new_param = random.choice([p for p in params if p != int(current.value)])
                    new_args = (target_node.args[0], target_node.args[1], NumberNode(float(new_param)))
                    return self._replace_subtree(root, target_node, FunctionNode(target_node.name, new_args))
            if target_node.name in ('ts_delay', 'ts_delta') and len(target_node.args) >= 2:
                params = self.delay_params if target_node.name == 'ts_delay' else self.delta_params
                current = target_node.args[1]
                if isinstance(current, NumberNode) and len(params) > 1:
                    new_param = random.choice([p for p in params if p != int(current.value)])
                    new_args = (target_node.args[0], NumberNode(float(new_param)))
                    return self._replace_subtree(root, target_node, FunctionNode(target_node.name, new_args))
            return None

        def replace_operator(node: AstNode) -> Optional[AstNode]:
            func_nodes = [n for n in nodes if isinstance(n, FunctionNode)]
            bin_nodes = [n for n in nodes if isinstance(n, BinaryOpNode)]
            unary_nodes = [n for n in nodes if isinstance(n, UnaryOpNode)]
            pool = func_nodes + bin_nodes + unary_nodes
            if not pool:
                return None
            target_node = random.choice(pool)
            if isinstance(target_node, FunctionNode):
                if target_node.name in self.ts_operators:
                    new_name = random.choice([n for n in self.ts_operators.keys() if n != target_node.name])
                    return self._replace_subtree(root, target_node, FunctionNode(new_name, target_node.args))
                if target_node.name in self.ts_dual_operators:
                    new_name = random.choice([n for n in self.ts_dual_operators.keys() if n != target_node.name])
                    return self._replace_subtree(root, target_node, FunctionNode(new_name, target_node.args))
                if target_node.name in self.cross_operators:
                    new_name = random.choice([n for n in self.cross_operators if n != target_node.name])
                    return self._replace_subtree(root, target_node, FunctionNode(new_name, target_node.args))
                if target_node.name in ('ts_delay', 'ts_delta'):
                    new_name = 'ts_delta' if target_node.name == 'ts_delay' else 'ts_delay'
                    return self._replace_subtree(root, target_node, FunctionNode(new_name, target_node.args))
                if target_node.name in self.unary_ops and target_node.name != '-':
                    candidates = [n for n in self.unary_ops if n != '-' and n != target_node.name]
                    if candidates:
                        new_name = random.choice(candidates)
                        return self._replace_subtree(root, target_node, FunctionNode(new_name, target_node.args))
            if isinstance(target_node, UnaryOpNode) and target_node.op == '-':
                new_name = random.choice([n for n in self.unary_ops if n != '-'])
                new_node = FunctionNode(new_name, (self._ensure_series_node(target_node.operand),))
                return self._replace_subtree(root, target_node, new_node)
            if isinstance(target_node, BinaryOpNode):
                new_op = random.choice([op for op in self.binary_ops if op != target_node.op])
                new_node = BinaryOpNode(new_op, target_node.left, target_node.right)
                return self._replace_subtree(root, target_node, new_node)
            return None

        def add_or_remove_layer(node: AstNode) -> Optional[AstNode]:
            target_node = random.choice(nodes)
            if random.random() < 0.5:
                # 添加一层运算
                choice = random.random()
                if choice < 0.4:
                    op = random.choice(self.unary_ops)
                    if op == '-':
                        new_node = UnaryOpNode('-', target_node)
                    else:
                        new_node = FunctionNode(op, (self._ensure_series_node(target_node),))
                elif choice < 0.7:
                    op_name = random.choice(list(self.ts_operators.keys()))
                    param = random.choice(self.ts_operators[op_name])
                    new_node = FunctionNode(op_name, (self._ensure_series_node(target_node), NumberNode(float(param))))
                else:
                    op_name = random.choice(self.cross_operators)
                    new_node = FunctionNode(op_name, (self._ensure_series_node(target_node),))
                return self._replace_subtree(root, target_node, new_node)

            # 删除一层运算
            if isinstance(target_node, FunctionNode) and target_node.args:
                return self._replace_subtree(root, target_node, target_node.args[0])
            if isinstance(target_node, UnaryOpNode):
                return self._replace_subtree(root, target_node, target_node.operand)
            if isinstance(target_node, BinaryOpNode):
                return self._replace_subtree(root, target_node, random.choice([target_node.left, target_node.right]))
            return None

        r = random.random()
        mutated = None
        if r < 0.30:
            mutated = modify_param(root)
        elif r < 0.60:
            mutated = replace_operator(root)
        else:
            mutated = add_or_remove_layer(root)

        if mutated is None:
            target = random.choice(nodes)
            new_subtree = self._generate_random_ast()
            mutated = self._replace_subtree(root, target, new_subtree)

        mutated = self._sanitize_ast(mutated)
        return self._ast_to_string(mutated)
    
    def crossover_expressions(self, expr1: str, expr2: str) -> str:
        """
        ??????????????????
        
        ??????* / ??ts_dual_operators ??????????????? + ??-??
        """
        def _fallback_combine() -> str:
            if random.random() < 0.5:
                op = random.choice(['*', '/', '+', '-'])
                return f"({expr1}) {op} ({expr2})"
            dual_op = random.choice(list(self.ts_dual_operators.keys()))
            param = random.choice(self.ts_dual_operators[dual_op])
            return f"{dual_op}({expr1}, {expr2}, {param})"

        try:
            root1 = self._parse_expr_to_ast(expr1)
            root2 = self._parse_expr_to_ast(expr2)
        except Exception:
            return _fallback_combine()

        groups1 = self._collect_nodes_by_type(root1)
        groups2 = self._collect_nodes_by_type(root2)
        candidate_types = [t for t in ("series", "scalar") if groups1[t] and groups2[t]]
        if not candidate_types:
            return _fallback_combine()

        # Typed subtree crossover with retries to avoid trivial children.
        for _ in range(8):
            picked_type = random.choice(candidate_types)
            subtree1 = random.choice(groups1[picked_type])
            subtree2 = random.choice(groups2[picked_type])
            swapped = self._replace_subtree(root1, subtree1, copy.deepcopy(subtree2))
            swapped = self._sanitize_ast(swapped)
            child = self._ast_to_string(swapped)
            is_valid, _ = self.is_semantically_valid(child)
            if is_valid:
                return child

        child = _fallback_combine()
        try:
            child_root = self._sanitize_ast(self._parse_expr_to_ast(child))
            child = self._ast_to_string(child_root)
        except Exception:
            pass
        return child


    def _simplify_ast(self, node: AstNode) -> AstNode:
        if isinstance(node, NumberNode):
            return node
        if isinstance(node, FieldNode):
            return node
        if isinstance(node, UnaryOpNode):
            operand = self._simplify_ast(node.operand)
            if node.op == '-' and isinstance(operand, UnaryOpNode) and operand.op == '-':
                return self._simplify_ast(operand.operand)
            if node.op == '-' and isinstance(operand, NumberNode):
                return NumberNode(-operand.value)
            return UnaryOpNode(node.op, operand)
        if isinstance(node, BinaryOpNode):
            left = self._simplify_ast(node.left)
            right = self._simplify_ast(node.right)

            if node.op == '+':
                if isinstance(left, NumberNode) and left.value == 0:
                    return right
                if isinstance(right, NumberNode) and right.value == 0:
                    return left
            elif node.op == '-':
                if isinstance(right, NumberNode) and right.value == 0:
                    return left
                if isinstance(left, NumberNode) and left.value == 0:
                    return UnaryOpNode('-', right)
            elif node.op == '*':
                if isinstance(left, NumberNode) and left.value == 0:
                    return NumberNode(0.0)
                if isinstance(right, NumberNode) and right.value == 0:
                    return NumberNode(0.0)
                if isinstance(left, NumberNode) and left.value == 1:
                    return right
                if isinstance(right, NumberNode) and right.value == 1:
                    return left
            elif node.op == '/':
                if isinstance(left, NumberNode) and left.value == 0:
                    return NumberNode(0.0)
                if isinstance(right, NumberNode) and right.value == 1:
                    return left

            return BinaryOpNode(node.op, left, right)
        if isinstance(node, FunctionNode):
            args = tuple(self._simplify_ast(arg) for arg in node.args)
            return FunctionNode(node.name, args)
        return node


class FactorLibrary:
    """因子模板库"""
    
    # 经典 Alpha101 模式
    ALPHA101_PATTERNS = [
        # 动量类
        "rank(ts_delta({{field}}, {{ts_delay}}))",
        "ts_rank({{field}}, {{window}})",
        "{{field}} - ts_delay({{field}}, {{ts_delay}})",
        "{{field}} / ts_delay({{field}}, {{ts_delay}}) - 1",
        
        # 反转类
        "-1 * rank(ts_rank({{field}}, {{window}}))",
        "-1 * ts_delta({{field}}, {{ts_delay}})",
        
        # 波动率类
        "ts_std_dev({{field}}, {{window}})",
        "rank(ts_std_dev({{field}}, {{window}}))",
        "(ts_max({{field}}, {{window}}) - ts_min({{field}}, {{window}})) / ts_mean({{field}}, {{window}})",
        
        # 均值回归
        "({{field}} - ts_mean({{field}}, {{window}})) / ts_std_dev({{field}}, {{window}})",
        "rank({{field}} - ts_mean({{field}}, {{window}}))",
        
        # 组合类
        "rank(ts_delta(ts_mean({{field}}, {{window}}), {{ts_delay}}))",
        "ts_mean(rank({{field}}), {{window}})",
    ]
    
    # WorldQuant 101 类似模式
    COMPLEX_PATTERNS = [
        "(rank(open - ts_delay(high, 1)) * rank(open - ts_delay(close, 1))) * rank(open - ts_delay(low, 1))",
        "rank(ts_delta(close, 1)) * rank((-1 * ts_delta(volume, 1)))",
        "(-1 * rank(ts_rank(close, 5))) * rank(volume - ts_delay(volume, 5))",
        "rank(volume / ts_mean(volume, 20)) * rank((-1 * ts_delta(close, 7)))",
    ]
if __name__ == "__main__":
    generator = FactorGenerator(seed=42)
    
    # 生成随机因子
    random_factors = [generator.generate_random_factor() for _ in range(1000)]
    print("随机生成的因子表达式：")
    for factor in random_factors:
        print(factor)
    
    # 基于模板生成因子
    template_factors = generator.generate_template_based_factors(FactorLibrary.ALPHA101_PATTERNS)
    print("\n基于模板生成的因子表达式：")
    for name, expr in template_factors[:1000]:  # 只展示前10个
        print(f"{name}: {expr}")
    
    # 网格搜索生成因子
    grid_factors = generator.generate_grid_search_factors()
    print("\n网格搜索生成的因子表达式：")
    for name, expr in grid_factors[:10]:  # 只展示前10个
        print(f"{name}: {expr}")
