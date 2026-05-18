'''D-TempIndex: wrap `srdatalog.ir.hir.split.TempIndexRegistrationPass`
as a `ProgramPass`.

Spec: `docs/phase_d_hir_passes.md` section 3 (per-pass migration
table) + section 3.1 (per-PR template). This module is the Wave 2B
landing for temp-relation index registration (Pass 5.5): for each
split variant in each stratum, register the identity index
`[0..arity-1]` on the variant's temp relation in the enclosing
stratum's `required_indices` and `canonical_index` maps. Runs after
the main `IndexSelectionPass` so it only affects temp relations
(which the selection pass doesn't see — they appear only as
synthetic rule heads).

Naming drift relative to the spec table:

  * D-Split (PR #48) wrapped `TempRelSynthesisPass` from
    `hir/split.py` as `SplitPass`. The spec's migration table calls
    that slot "D-TempRel" / `TempRelSynthesisPass wrapper`, so
    PR #48 effectively filled D-TempRel under the name D-Split.
  * The spec's table lists `temp_index.py:register(prog)` as the
    legacy entry for D-TempIndex; in-tree, `TempIndexRegistrationPass`
    actually lives in the SAME `hir/split.py` module as
    `TempRelSynthesisPass` (split.py defines both: Pass 4.5 + Pass
    5.5). This PR fills the remaining D-TempIndex slot by wrapping
    the second class from `split.py`.

The legacy `TempIndexRegistrationPass.run(hir)` mutates `hir` in
place (`stratum.required_indices[...] = ...` /
`stratum.canonical_index[...] = ...`) and returns it —
`TempIndexPass._fn` preserves that contract, replacing only the
through-state's `hir` reference (same object identity).
'''

from __future__ import annotations

import dataclasses
from typing import Any

from srdatalog.ir.core import ProgramPass
from srdatalog.ir.dialects.hir.passes.stratify import HirPlanState


class TempIndexPass(ProgramPass):
  '''Wraps `srdatalog.ir.hir.split.TempIndexRegistrationPass`. Mutates
  the through-state's `hir` in place (legacy contract preserved) and
  returns the same `HirPlanState` instance via `dataclasses.replace`.

  Pre-flight ordering: `consumes=('hir',)` so the framework's
  pre-flight check accepts a pipeline that places `TempIndexPass`
  after any pass that produces `'hir'` (per the spec's pass list,
  this is `IndexSelectionPass`, which itself follows
  `StratifyPass` / `SplitPass` / `SemiNaivePass` / `PlanPass`).
  '''

  def __init__(self) -> None:
    super().__init__(
      name='temp_index',
      consumes=('hir',),
      produces=('hir',),
      fn=self._fn,
    )

  @staticmethod
  def _fn(state: HirPlanState, _compiler: Any) -> HirPlanState:
    from srdatalog.ir.hir.split import TempIndexRegistrationPass as _legacy

    assert state.hir is not None, 'TempIndexPass: hir not set (StratifyPass must run first)'
    new_hir = _legacy().run(state.hir)
    return dataclasses.replace(state, hir=new_hir)


__all__ = ['TempIndexPass']
