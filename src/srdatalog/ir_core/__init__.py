'''ir_core — dialect-agnostic IR framework.

Stage 1 of the multi-dialect IR refactor. The framework hosts dialects
(data-structure, parallelism, target) without knowing what they are.
See docs/ir_lowering_semantics.md, Part VI (sections 19-22) for the
spec.

Public API:

  Compiler            — registry of dialects.
  Dialect             — what every dialect provides.
  Op, Type            — bases for IR ops and types.
  Lowering, Rewrite   — pass kinds.
  PassDriver          — runs registered passes.
  VerificationError   — well-formedness violation.

Design invariants (see Part I, section 4 of the spec):

  P1  Open dialect registration. New dialects are purely additive.
  P2  Polymorphic relation references. MIR ops never name a dialect.
  P3  Dialect-local pass libraries. Each dialect ships its own passes.

Stage 1 status: skeleton in place. Real implementations land as
dialects come online (Stage 3+).
'''

from __future__ import annotations

from srdatalog.ir_core.dialect import Compiler, Dialect
from srdatalog.ir_core.ops import Op, Type
from srdatalog.ir_core.passes import Lowering, PassDriver, Rewrite
from srdatalog.ir_core.verifier import VerificationError

__all__ = [
  'Compiler',
  'Dialect',
  'Lowering',
  'Op',
  'PassDriver',
  'Rewrite',
  'Type',
  'VerificationError',
]
