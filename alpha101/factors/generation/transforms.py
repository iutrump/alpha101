from __future__ import annotations

import ast
import random
from typing import Dict, List

from alpha101.factors.generation.ast_nodes import (
    AstNode,
    BinaryOpNode,
    FieldNode,
    FunctionNode,
    NumberNode,
    UnaryOpNode,
)


class AstTransformMixin:
    def _ast_to_string(self, node: AstNode) -> str:
        if isinstance(node, NumberNode):
            if float(node.value).is_integer():
                return str(int(node.value))
            return str(float(node.value))
        if isinstance(node, FieldNode):
            return node.name
        if isinstance(node, UnaryOpNode):
            inner = self._ast_to_string(node.operand)
            if node.op == "-":
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

    def _parse_expr_to_ast(self, expr: str) -> AstNode:
        py_ast = ast.parse(expr, mode="eval")
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
                return UnaryOpNode("-", self._from_py_ast(node.operand))
            if isinstance(node.op, ast.UAdd):
                return self._from_py_ast(node.operand)
            raise ValueError(f"Unsupported unary op: {node.op}")
        if isinstance(node, ast.BinOp):
            op_map = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
            }
            for op_type, op_str in op_map.items():
                if isinstance(node.op, op_type):
                    return BinaryOpNode(op_str, self._from_py_ast(node.left), self._from_py_ast(node.right))
            raise ValueError(f"Unsupported binary op: {node.op}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Unsupported call target")
            return FunctionNode(node.func.id, tuple(self._from_py_ast(arg) for arg in node.args))
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
        if isinstance(node, NumberNode):
            return "scalar"
        if isinstance(node, FieldNode):
            return "series"
        if isinstance(node, UnaryOpNode):
            return self._infer_node_type(node.operand)
        if isinstance(node, BinaryOpNode):
            left_type = self._infer_node_type(node.left)
            right_type = self._infer_node_type(node.right)
            return "series" if left_type == "series" or right_type == "series" else "scalar"
        if isinstance(node, FunctionNode):
            return "series"
        return "series"

    def _collect_nodes_by_type(self, node: AstNode) -> Dict[str, List[AstNode]]:
        grouped: Dict[str, List[AstNode]] = {"series": [], "scalar": []}
        for n in self._collect_nodes(node):
            grouped[self._infer_node_type(n)].append(n)
        return grouped

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
            return UnaryOpNode(node.op, self._sanitize_ast(node.operand))
        if isinstance(node, BinaryOpNode):
            left = self._sanitize_ast(node.left)
            right = self._sanitize_ast(node.right)
            return BinaryOpNode(node.op, left, right)
        if isinstance(node, FunctionNode):
            args = tuple(self._sanitize_ast(arg) for arg in node.args)
            if node.name == "abs" and args and isinstance(args[0], FunctionNode) and args[0].name == "abs":
                return args[0]
            if node.name in self.ts_operators or node.name in self.cross_operators:
                if args:
                    args = (self._ensure_series_node(args[0]),) + args[1:]
            if node.name in self.ts_operators and len(args) >= 2:
                args = (args[0], self._sanitize_positive_window(args[1], fallback=1)) + args[2:]
            if node.name in ("ts_delay", "ts_delta"):
                if args:
                    args = (self._ensure_series_node(args[0]),) + args[1:]
                if len(args) >= 2:
                    args = (args[0], self._sanitize_positive_window(args[1], fallback=1)) + args[2:]
            if node.name in self.ts_dual_operators:
                if len(args) >= 2:
                    args = (self._ensure_series_node(args[0]), self._ensure_series_node(args[1])) + args[2:]
                if len(args) >= 3:
                    args = (args[0], args[1], self._sanitize_positive_window(args[2], fallback=1)) + args[3:]
            if node.name in self.unary_ops and node.name != "-" and args:
                args = (self._ensure_series_node(args[0]),) + args[1:]
            return FunctionNode(node.name, args)
        return node

    def _simplify_ast(self, node: AstNode) -> AstNode:
        if isinstance(node, NumberNode):
            return node
        if isinstance(node, FieldNode):
            return node
        if isinstance(node, UnaryOpNode):
            operand = self._simplify_ast(node.operand)
            if node.op == "-" and isinstance(operand, UnaryOpNode) and operand.op == "-":
                return self._simplify_ast(operand.operand)
            if node.op == "-" and isinstance(operand, NumberNode):
                return NumberNode(-operand.value)
            return UnaryOpNode(node.op, operand)
        if isinstance(node, BinaryOpNode):
            left = self._simplify_ast(node.left)
            right = self._simplify_ast(node.right)
            if node.op == "+":
                if isinstance(left, NumberNode) and left.value == 0:
                    return right
                if isinstance(right, NumberNode) and right.value == 0:
                    return left
            elif node.op == "-":
                if isinstance(right, NumberNode) and right.value == 0:
                    return left
                if isinstance(left, NumberNode) and left.value == 0:
                    return UnaryOpNode("-", right)
            elif node.op == "*":
                if isinstance(left, NumberNode) and left.value == 0:
                    return NumberNode(0.0)
                if isinstance(right, NumberNode) and right.value == 0:
                    return NumberNode(0.0)
                if isinstance(left, NumberNode) and left.value == 1:
                    return right
                if isinstance(right, NumberNode) and right.value == 1:
                    return left
            elif node.op == "/":
                if isinstance(left, NumberNode) and left.value == 0:
                    return NumberNode(0.0)
                if isinstance(right, NumberNode) and right.value == 1:
                    return left
            return BinaryOpNode(node.op, left, right)
        if isinstance(node, FunctionNode):
            args = tuple(self._simplify_ast(arg) for arg in node.args)
            return FunctionNode(node.name, args)
        return node
