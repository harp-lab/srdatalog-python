'''D-Plan: wrap `srdatalog.ir.hir.plan.plan_joins` as a `ProgramPass`.

Spec: `docs/phase_d_hir_passes.md` section 3 (per-pass migration
table) + section 3.1 (per-PR template). This module is the Wave 2B
landing for the join-planning pass (Pass 4): produces per-variant
`var_order`, `clause_order`, `access_patterns`, `negation_patterns`,
and propagates pragma fields onto each `HirRuleVariant`.

The spec's table names the legacy entry as `plan.py:plan(prog)`; the
in-tree `hir/plan.py` actually exposes `plan_joins(hir) -> HirProgram`
(plus a `JoinPlannerPass` class with `.run(hir)`). Per the per-PR
template, we wrap the existing imperative function with no behavior
change. `plan_joins` mutates `hir` in place and returns it — the
wrapper preserves that contract (the through-state's `hir` retains
the same object identity after `PlanPass`).

`compile_to_hir`'s body and the legacy `hir/plan.py` are NOT touched
in this PR — wrapping is addition-only so a future PR can drop the
wrapper into a `DEFAULT_HIR_PIPELINE` (spec section 4).
'''

from __future__ import annotations

import dataclasses
from typing import Any

from srdatalog.ir.core import ProgramPass
from srdatalog.ir.dialects.hir.passes.stratify import HirPlanState


class PlanPass(ProgramPass):
  '''Wraps `srdatalog.ir.hir.plan.plan_joins`. Mutates the through-state's
  `hir` in place (legacy contract preserved) and returns the same
  `HirPlanState` instance via `dataclasses.replace`.

  Pre-flight ordering: `consumes=('hir',)` so the framework's
  pre-flight check accepts a pipeline that places `PlanPass` after
  any pass that produces `'hir'` (typically `StratifyPass`, optionally
  with `SplitPass` and the semi-naive variant pass in between).
  '''

  def __init__(self) -> None:
    super().__init__(
      name='plan',
      consumes=('hir',),
      produces=('hir',),
      fn=self._fn,
    )

  @staticmethod
  def _fn(state: HirPlanState, _compiler: Any) -> HirPlanState:
    from srdatalog.ir.hir.plan import plan_joins as _legacy_plan_joins

    assert state.hir is not None, 'PlanPass: hir not set (StratifyPass must run first)'
    new_hir = _legacy_plan_joins(state.hir)
    return dataclasses.replace(state, hir=new_hir)


__all__ = ['PlanPass']
