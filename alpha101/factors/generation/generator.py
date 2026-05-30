from __future__ import annotations

import random
from typing import Optional

import numpy as np

from alpha101.factors.generation.ast_nodes import BinaryOpNode, FieldNode, FunctionNode, NumberNode, UnaryOpNode
from alpha101.factors.generation.complexity import ComplexityMixin
from alpha101.factors.generation.genetic_ops import GeneticOpsMixin
from alpha101.factors.generation.grammar import (
    BINARY_OPS,
    CROSS_OPERATORS,
    DATA_FIELDS,
    DELAY_PARAMS,
    DELTA_PARAMS,
    FORBIDDEN_PATTERNS,
    MAX_DEPTH,
    MAX_OPERATORS,
    TS_DUAL_OPERATORS,
    TS_OPERATORS,
    UNARY_OPS,
)
from alpha101.factors.generation.transforms import AstTransformMixin
from alpha101.factors.generation.validation import ValidationMixin


class FactorGenerator(AstTransformMixin, ComplexityMixin, ValidationMixin, GeneticOpsMixin):
    """Generate, mutate, and crossover Alpha101-style factor expressions."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.data_fields = DATA_FIELDS
        self.ts_operators = TS_OPERATORS
        self.ts_dual_operators = TS_DUAL_OPERATORS
        self.delay_params = DELAY_PARAMS
        self.delta_params = DELTA_PARAMS
        self.cross_operators = CROSS_OPERATORS
        self.binary_ops = BINARY_OPS
        self.unary_ops = UNARY_OPS
        self.max_depth = MAX_DEPTH
        self.max_operators = MAX_OPERATORS
        self.forbidden_patterns = FORBIDDEN_PATTERNS

    def generate_random_factor(self, depth: int = 0, used_ops: int = 0) -> str:
        ast_root = self._generate_random_ast(depth, used_ops)
        ast_root = self._sanitize_ast(ast_root)
        return self._ast_to_string(ast_root)

    def _generate_random_ast(self, depth: int = 0, used_ops: int = 0):
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

    def _generate_simple_ast(self, depth: int = 0, used_ops: int = 0):
        if random.random() < 0.9:
            return FieldNode(random.choice(self.data_fields))
        return NumberNode(float(random.choice([0.1, 0.5, 1, 2, -1])))

    def _generate_ts_operator_ast(self, depth: int, used_ops: int):
        op_name = random.choice(list(self.ts_operators.keys()))
        param = random.choice(self.ts_operators[op_name])
        inner = self._ensure_series_node(self._generate_random_ast(depth + 1, used_ops + 1))
        return FunctionNode(op_name, (inner, NumberNode(float(param))))

    def _generate_ts_dual_operator_ast(self, depth: int, used_ops: int):
        op_name = random.choice(list(self.ts_dual_operators.keys()))
        param = random.choice(self.ts_dual_operators[op_name])
        field1 = random.choice(self.data_fields)
        field2 = random.choice([f for f in self.data_fields if f != field1])
        return FunctionNode(op_name, (FieldNode(field1), FieldNode(field2), NumberNode(float(param))))

    def _generate_cross_operator_ast(self, depth: int, used_ops: int):
        op_name = random.choice(self.cross_operators)
        inner = self._ensure_series_node(self._generate_random_ast(depth + 1, used_ops + 1))
        return FunctionNode(op_name, (inner,))

    def _generate_delay_delta_ast(self, depth: int, used_ops: int):
        if random.random() < 0.5:
            param = random.choice(self.delay_params)
            inner = self._ensure_series_node(self._generate_random_ast(depth + 1, used_ops + 1))
            return FunctionNode("ts_delay", (inner, NumberNode(float(param))))
        param = random.choice(self.delta_params)
        inner = self._ensure_series_node(self._generate_random_ast(depth + 1, used_ops + 1))
        return FunctionNode("ts_delta", (inner, NumberNode(float(param))))

    def _generate_binary_ast(self, depth: int, used_ops: int):
        op = random.choice(self.binary_ops)
        left = self._generate_random_ast(depth + 1, used_ops + 1)
        right = self._generate_random_ast(depth + 1, used_ops + 1)
        return BinaryOpNode(op, left, right)

    def _generate_unary_ast(self, depth: int, used_ops: int):
        op = random.choice(self.unary_ops)
        inner = self._generate_random_ast(depth + 1, used_ops + 1)
        if op == "-":
            return UnaryOpNode("-", inner)
        return FunctionNode(op, (self._ensure_series_node(inner),))
