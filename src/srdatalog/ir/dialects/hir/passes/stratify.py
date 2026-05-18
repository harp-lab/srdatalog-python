'''D-Stratify: wrap `srdatalog.ir.hir.stratify.stratify` as a `ProgramPass`.

Spec: `docs/phase_d_hir_passes.md` section 3 (per-pass migration
table) + section 3.1 (per-PR template). This module is the Wave 2B
landing for the SCC analysis on the rule dependency graph.

The legacy function (`hir/stratify.py:stratify(rules, decls)`) is the
HIR entry point — it takes `(rules, decls)` and produces a fresh
`HirProgram`, unlike the later HIR transforms which mutate an existing
`HirProgram` in place. To fit the `ProgramPass.apply(prog, compiler)`
shape uniformly, we thread a tiny `HirPlanState` through-state
(rules + decls + hir) instead of either a bare `HirProgram` (no
pre-stratify shape) or the F5.1 `InitialProg` (heavier, also carries
MIR fields that don't belong in an HIR-only pipeline).

`compile_to_hir`'s body is NOT touched in this PR — the wrapper is
addition-only so a future PR can put it into a `DEFAULT_HIR_PIPELINE`
and reduce `compile_to_hir` to `Compiler.run` (spec section 4).
'''

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from srdatalog.ir.core import ProgramPass

if TYPE_CHECKING:
  from srdatalog.dsl import Program, Rule
  from srdatalog.ir.hir.types import HirProgram, RelationDecl


# -----------------------------------------------------------------------------
# Through-state
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HirPlanState:
  '''HIR-pipeline through-state for the Wave 2B passes.

  Threaded by every `ProgramPass` in `dialects/hir/passes/`. Each
  pass either populates `hir` (the entry, `StratifyPass`) or mutates
  it in place (every subsequent transform, including `SplitPass`).

  `program` is optional — callers that already have `(rules, decls)`
  may build the state directly without round-tripping through
  `Program`. `StratifyPass._fn` extracts rules + decls from `program`
  only if the explicit fields are empty.
  '''

  program: Program | None = None
  rules: list[Rule] = field(default_factory=list)
  decls: list[RelationDecl] = field(default_factory=list)
  hir: HirProgram | None = None


# -----------------------------------------------------------------------------
# Pass
# -----------------------------------------------------------------------------


class StratifyPass(ProgramPass):
  '''Wraps `srdatalog.ir.hir.stratify.stratify`. Adds `hir` to state.

  Mirrors the F5.1 `HirPlanningShim` shape — subclass `ProgramPass`,
  declare `consumes` / `produces`, dispatch via a `@staticmethod`
  `_fn`. Future per-pass migrations in Wave 2B follow the same
  template (see `SplitPass` in `split.py`).
  '''

  def __init__(self) -> None:
    super().__init__(
      name='stratify',
      consumes=(),
      produces=('hir',),
      fn=self._fn,
    )

  @staticmethod
  def _fn(state: HirPlanState, _compiler: Any) -> HirPlanState:
    from srdatalog.ir.hir.pass_ import program_to_decls
    from srdatalog.ir.hir.stratify import stratify as _legacy_stratify

    rules = state.rules
    decls = state.decls
    if not rules and not decls and state.program is not None:
      rules = list(state.program.rules)
      decls = program_to_decls(state.program)
    hir = _legacy_stratify(rules, decls)
    return dataclasses.replace(state, rules=rules, decls=decls, hir=hir)


__all__ = ['HirPlanState', 'StratifyPass']
