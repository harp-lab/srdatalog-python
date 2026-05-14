'''ir_core — dialect-agnostic IR framework.

Stage 1 of the multi-dialect IR refactor. The framework hosts dialects
(data-structure, parallelism, target) without knowing what they are.

Spec: docs/ir_lowering_semantics.md.
Discipline rules: docs/design_principles.md.

Public API:

  Compiler            — registry of dialects.
  Dialect             — what every dialect provides.
  Op, Type            — frozen+slots dataclass bases for IR ops/types.
  Lowering, Rewrite   — pass kinds.
  PassDriver          — runs registered passes.
  VerificationError   — well-formedness violation.
  Strategy combinators — top_down, bottom_up, seq, choice, repeat,
                         try_, all_, one, some, id_, fail.
  assert_never        — exhaustiveness check for match defaults.

Design invariants:

  P1  Open dialect registration. New dialects are purely additive.
  P2  Polymorphic relation references. MIR ops never name a dialect.
  P3  Dialect-local pass libraries. Each dialect ships its own passes.

  D1  Op/Type subclasses are pure data. No methods.
  D2  IR is immutable (frozen dataclasses).
  D3  Dispatch via match statements over closed unions; never via
      virtual methods or `isinstance` chains.
  D4  Rewrites are pure functions. No mutable rewriter state.
'''

from __future__ import annotations

from typing import NoReturn

from srdatalog.ir.core.dialect import Compiler, Dialect
from srdatalog.ir.core.ops import Op, Type
from srdatalog.ir.core.passes import (
  Lowering,
  PassDriver,
  Rewrite,
  RewriteContext,
  RewriteRegistrationConflict,
  UnrenderableOp,
  UnrenderableOpError,
)
from srdatalog.ir.core.strategy import (
  Strategy,
  all_,
  bottom_up,
  choice,
  fail,
  id_,
  one,
  repeat,
  seq,
  some,
  top_down,
  try_,
)
from srdatalog.ir.core.verifier import VerificationError


def assert_never(value: object) -> NoReturn:
  '''Exhaustiveness check for `match` defaults.

  Use as the body of the catch-all branch in dispatchers over closed
  unions. Type checkers (mypy --strict, pyright) infer that the
  argument has type `Never` if all cases are covered, flagging
  unreachable-but-not-actually-handled cases at type-check time.
  '''
  raise AssertionError(f'Unhandled case: {type(value).__name__}({value!r})')


__all__ = [
  'Compiler',
  'Dialect',
  'Lowering',
  'Op',
  'PassDriver',
  'Rewrite',
  'RewriteContext',
  'RewriteRegistrationConflict',
  'Strategy',
  'Type',
  'UnrenderableOp',
  'UnrenderableOpError',
  'VerificationError',
  'all_',
  'assert_never',
  'bottom_up',
  'choice',
  'fail',
  'id_',
  'one',
  'repeat',
  'seq',
  'some',
  'top_down',
  'try_',
]
