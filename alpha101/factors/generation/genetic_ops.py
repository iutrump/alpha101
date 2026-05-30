from __future__ import annotations

import copy
import random
from typing import Optional

from alpha101.factors.generation.ast_nodes import (
    AstNode,
    BinaryOpNode,
    FunctionNode,
    NumberNode,
    UnaryOpNode,
)


class GeneticOpsMixin:
    def mutate_expression(self, expr: str) -> str:
        try:
            root = self._parse_expr_to_ast(expr)
        except Exception:
            root = self._generate_random_ast()

        nodes = self._collect_nodes(root)
        if not nodes:
            return expr

        mutations = (self._mutate_param, self._mutate_operator, self._mutate_layer)
        r = random.random()
        if r < 0.30:
            mutated = mutations[0](root, nodes)
        elif r < 0.60:
            mutated = mutations[1](root, nodes)
        else:
            mutated = mutations[2](root, nodes)

        if mutated is None:
            target = random.choice(nodes)
            mutated = self._replace_subtree(root, target, self._generate_random_ast())

        mutated = self._sanitize_ast(mutated)
        return self._ast_to_string(mutated)

    def crossover_expressions(self, expr1: str, expr2: str) -> str:
        def _fallback_combine() -> str:
            if random.random() < 0.5:
                op = random.choice(["*", "/", "+", "-"])
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

    def _mutate_param(self, root: AstNode, nodes: list[AstNode]) -> Optional[AstNode]:
        candidates = [n for n in nodes if isinstance(n, FunctionNode)]
        if not candidates:
            return None
        target_node = random.choice(candidates)

        if target_node.name in self.ts_operators and len(target_node.args) >= 2:
            return self._replace_window_param(root, target_node, self.ts_operators[target_node.name], arg_idx=1)
        if target_node.name in self.ts_dual_operators and len(target_node.args) >= 3:
            return self._replace_window_param(root, target_node, self.ts_dual_operators[target_node.name], arg_idx=2)
        if target_node.name in ("ts_delay", "ts_delta") and len(target_node.args) >= 2:
            params = self.delay_params if target_node.name == "ts_delay" else self.delta_params
            return self._replace_window_param(root, target_node, params, arg_idx=1)
        return None

    def _replace_window_param(
        self,
        root: AstNode,
        target_node: FunctionNode,
        params: list[int],
        *,
        arg_idx: int,
    ) -> Optional[AstNode]:
        current = target_node.args[arg_idx]
        if not isinstance(current, NumberNode) or len(params) <= 1:
            return None
        new_param = random.choice([p for p in params if p != int(current.value)])
        new_args = list(target_node.args)
        new_args[arg_idx] = NumberNode(float(new_param))
        return self._replace_subtree(root, target_node, FunctionNode(target_node.name, tuple(new_args)))

    def _mutate_operator(self, root: AstNode, nodes: list[AstNode]) -> Optional[AstNode]:
        func_nodes = [n for n in nodes if isinstance(n, FunctionNode)]
        bin_nodes = [n for n in nodes if isinstance(n, BinaryOpNode)]
        unary_nodes = [n for n in nodes if isinstance(n, UnaryOpNode)]
        pool = func_nodes + bin_nodes + unary_nodes
        if not pool:
            return None

        target_node = random.choice(pool)
        if isinstance(target_node, FunctionNode):
            new_name = self._replacement_function_name(target_node.name)
            if new_name is not None:
                return self._replace_subtree(root, target_node, FunctionNode(new_name, target_node.args))
        if isinstance(target_node, UnaryOpNode) and target_node.op == "-":
            new_name = random.choice([n for n in self.unary_ops if n != "-"])
            return self._replace_subtree(
                root,
                target_node,
                FunctionNode(new_name, (self._ensure_series_node(target_node.operand),)),
            )
        if isinstance(target_node, BinaryOpNode):
            new_op = random.choice([op for op in self.binary_ops if op != target_node.op])
            return self._replace_subtree(root, target_node, BinaryOpNode(new_op, target_node.left, target_node.right))
        return None

    def _replacement_function_name(self, name: str) -> Optional[str]:
        if name in self.ts_operators:
            return random.choice([n for n in self.ts_operators.keys() if n != name])
        if name in self.ts_dual_operators:
            return random.choice([n for n in self.ts_dual_operators.keys() if n != name])
        if name in self.cross_operators:
            return random.choice([n for n in self.cross_operators if n != name])
        if name in ("ts_delay", "ts_delta"):
            return "ts_delta" if name == "ts_delay" else "ts_delay"
        if name in self.unary_ops and name != "-":
            candidates = [n for n in self.unary_ops if n != "-" and n != name]
            return random.choice(candidates) if candidates else None
        return None

    def _mutate_layer(self, root: AstNode, nodes: list[AstNode]) -> Optional[AstNode]:
        target_node = random.choice(nodes)
        if random.random() < 0.5:
            return self._replace_subtree(root, target_node, self._wrap_node(target_node))

        if isinstance(target_node, FunctionNode) and target_node.args:
            return self._replace_subtree(root, target_node, target_node.args[0])
        if isinstance(target_node, UnaryOpNode):
            return self._replace_subtree(root, target_node, target_node.operand)
        if isinstance(target_node, BinaryOpNode):
            return self._replace_subtree(root, target_node, random.choice([target_node.left, target_node.right]))
        return None

    def _wrap_node(self, target_node: AstNode) -> AstNode:
        choice = random.random()
        if choice < 0.4:
            op = random.choice(self.unary_ops)
            if op == "-":
                return UnaryOpNode("-", target_node)
            return FunctionNode(op, (self._ensure_series_node(target_node),))
        if choice < 0.7:
            op_name = random.choice(list(self.ts_operators.keys()))
            param = random.choice(self.ts_operators[op_name])
            return FunctionNode(op_name, (self._ensure_series_node(target_node), NumberNode(float(param))))
        op_name = random.choice(self.cross_operators)
        return FunctionNode(op_name, (self._ensure_series_node(target_node),))
