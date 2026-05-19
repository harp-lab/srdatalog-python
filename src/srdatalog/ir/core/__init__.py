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
  Pragma + @pragma_handler — typed compile-time pragma objects (F-pragma).
  Pass kinds (LoweringPass, RewritePass, ProgramPass) + program_pass
                       — F1 declarative-pipeline framework.
  LowerCtx + NameGen + ViewLayout
                       — F3 per-pass state for `LoweringPass`
                         dispatchers (5 fields, D10-pinned).
  Scope + EmptyScope   — F3 per-op-family lexical-context base.

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
from srdatalog.ir.core.lower_ctx import LowerCtx, NameGen, ViewLayout
from srdatalog.ir.core.ops import Op, Type
from srdatalog.ir.core.passes import (
  AmbiguousLowering,
  Lowering,
  LoweringMissingError,
  LoweringPass,
  Pass,
  PassDriver,
  PassOrderingError,
  ProgramPass,
  Rewrite,
  RewritePass,
  program_pass,
)
from srdatalog.ir.core.plugin import (
  PluginConflictError,
  PluginCycleError,
  PluginInfo,
  PluginLoadError,
)
from srdatalog.ir.core.pragma import (
  Pragma,
  PragmaConfigError,
  PragmaCtx,
  PragmaOrderingError,
  UnconsumedPragmaError,
  UnregisteredPragmaError,
  get_pragma,
  has_pragma,
  pragma_handler,
)
from srdatalog.ir.core.scope import EmptyScope, Scope
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
from srdatalog.ir.core.verifier import (
  UnrenderableOpError,
  VerificationError,
  verify_renderability,
)


def assert_never(value: object) -> NoReturn:
  '''Exhaustiveness check for `match` defaults.

  Use as the body of the catch-all branch in dispatchers over closed
  unions. Type checkers (mypy --strict, pyright) infer that the
  argument has type `Never` if all cases are covered, flagging
  unreachable-but-not-actually-handled cases at type-check time.
  '''
  raise AssertionError(f'Unhandled case: {type(value).__name__}({value!r})')


__all__ = [
  'AmbiguousLowering',
  'Compiler',
  'Dialect',
  'EmptyScope',
  'LowerCtx',
  'Lowering',
  'LoweringMissingError',
  'LoweringPass',
  'NameGen',
  'Op',
  'Pass',
  'PassDriver',
  'PassOrderingError',
  'PluginConflictError',
  'PluginCycleError',
  'PluginInfo',
  'PluginLoadError',
  'Pragma',
  'PragmaConfigError',
  'PragmaCtx',
  'PragmaOrderingError',
  'ProgramPass',
  'Rewrite',
  'RewritePass',
  'Scope',
  'Strategy',
  'Type',
  'UnconsumedPragmaError',
  'UnregisteredPragmaError',
  'UnrenderableOpError',
  'VerificationError',
  'ViewLayout',
  'all_',
  'assert_never',
  'bottom_up',
  'choice',
  'fail',
  'get_pragma',
  'has_pragma',
  'id_',
  'one',
  'pragma_handler',
  'program_pass',
  'repeat',
  'seq',
  'some',
  'top_down',
  'try_',
  'verify_renderability',
]
