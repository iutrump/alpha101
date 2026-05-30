from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


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
