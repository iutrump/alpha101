from __future__ import annotations

from typing import Dict

from alpha101.factors.generation.ast_nodes import AstNode, BinaryOpNode, FunctionNode, UnaryOpNode


class ComplexityMixin:
    def calculate_complexity(self, expr: str) -> Dict[str, int]:
        try:
            root = self._parse_expr_to_ast(expr)
            stats = self._calculate_ast_stats(root)
            ts_op_count = stats["ts_operators"]
            ts_dual_op_count = stats["ts_dual_operators"]
            cross_op_count = stats["cross_operators"]
            delay_delta_count = stats["delay_delta_operators"]
            unary_op_count = stats["unary_operators"]
            binary_op_count = stats["binary_operators"]
            max_depth = stats["max_depth"]
        except Exception:
            ts_op_count = sum(expr.count(op) for op in self.ts_operators.keys())
            ts_dual_op_count = sum(expr.count(op) for op in self.ts_dual_operators.keys())
            cross_op_count = sum(expr.count(op) for op in self.cross_operators)
            delay_delta_count = expr.count("ts_delay") + expr.count("ts_delta")
            unary_op_count = expr.count("log") + expr.count("abs")
            binary_op_count = sum(expr.count(f" {op} ") for op in self.binary_ops)
            max_depth = 0
            current_depth = 0
            for char in expr:
                if char == "(":
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif char == ")":
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
            "total_operators": operator_count,
            "max_depth": max_depth,
            "expression_length": expr_length,
            "ts_operators": ts_op_count,
            "ts_dual_operators": ts_dual_op_count,
            "cross_operators": cross_op_count,
            "binary_operators": binary_op_count,
            "complexity_score": operator_count * 1.0 + max_depth * 0.5 + expr_length * 0.01,
        }

    def filter_by_complexity(self, expr: str, max_complexity: float = 10.0) -> bool:
        complexity = self.calculate_complexity(expr)
        return complexity["complexity_score"] <= max_complexity

    def _calculate_ast_stats(self, node: AstNode, depth: int = 1) -> Dict[str, int]:
        stats = {
            "ts_operators": 0,
            "ts_dual_operators": 0,
            "cross_operators": 0,
            "delay_delta_operators": 0,
            "unary_operators": 0,
            "binary_operators": 0,
            "max_depth": depth,
        }

        if isinstance(node, UnaryOpNode):
            stats["unary_operators"] += 1
            child = self._calculate_ast_stats(node.operand, depth + 1)
            return self._merge_stats(stats, child)

        if isinstance(node, BinaryOpNode):
            stats["binary_operators"] += 1
            for child in (
                self._calculate_ast_stats(node.left, depth + 1),
                self._calculate_ast_stats(node.right, depth + 1),
            ):
                stats = self._merge_stats(stats, child)
            return stats

        if isinstance(node, FunctionNode):
            if node.name in self.ts_operators:
                stats["ts_operators"] += 1
            elif node.name in self.ts_dual_operators:
                stats["ts_dual_operators"] += 1
            elif node.name in self.cross_operators:
                stats["cross_operators"] += 1
            elif node.name in ("ts_delay", "ts_delta"):
                stats["delay_delta_operators"] += 1
            elif node.name in self.unary_ops:
                stats["unary_operators"] += 1

            for arg in node.args:
                stats = self._merge_stats(stats, self._calculate_ast_stats(arg, depth + 1))
            return stats

        return stats

    @staticmethod
    def _merge_stats(base: Dict[str, int], child: Dict[str, int]) -> Dict[str, int]:
        for key, value in child.items():
            if key == "max_depth":
                base[key] = max(base[key], value)
            else:
                base[key] += value
        return base
