'''D-Index: wrap `srdatalog.ir.hir.index.IndexSelectionPass` as a
`ProgramPass`.

Spec: `docs/phase_d_hir_passes.md` section 3 (per-pass migration
table) + section 3.1 (per-PR template). This module is the Wave 2B
landing for index selection (Nim's `selectIndices`, Pass 5): populate
per-stratum `required_indices` / `canonical_index` and program-level
`global_index_map` from per-variant access / negation patterns
(populated upstream by `PlanPass`).

The spec's migration table cites `index.py:select_indices(prog)` as
the legacy entry; the in-tree `hir/index.py` ships both `select_indices`
as the free function and `IndexSelectionPass` as the pipeline wrapper
used by `default_pipeline` in `ir/hir/__init__.py`. Per the per-PR
template (section 3.1) we wrap the existing pipeline-shaped class
without changing behavior — the wrap matches what `compile_to_hir`
already invokes.

The legacy `IndexSelectionPass.run(hir)` mutates `hir` in place
(`stratum.required_indices[...] = ...`,
`hir.global_index_map[...] = ...`) and returns it — `IndexSelectionPass._fn`
preserves that contract, replacing only the through-state's `hir`
reference (same object identity).

Pre-flight ordering: `consumes=('hir',)` so a pipeline that places
`IndexSelectionPass` after any `'hir'` producer validates — in
practice the spec ordering is `[StratifyPass, SplitPass, SemiNaivePass,
PlanPass, IndexSelectionPass]` because index selection reads the
per-variant `access_patterns` / `negation_patterns` populated by
`PlanPass` on top of `SemiNaivePass`-generated variants.
'''

from __future__ import annotations

import dataclasses
from typing import Any

from srdatalog.ir.core import ProgramPass
from srdatalog.ir.dialects.hir.passes.stratify import HirPlanState


class IndexSelectionPass(ProgramPass):
  '''Wraps `srdatalog.ir.hir.index.IndexSelectionPass`. Mutates the
  through-state's `hir` in place (legacy contract preserved) and
  returns the same `HirPlanState` instance via `dataclasses.replace`.
  '''

  def __init__(self) -> None:
    super().__init__(
      name='index_selection',
      consumes=('hir',),
      produces=('hir',),
      fn=self._fn,
    )

  @staticmethod
  def _fn(state: HirPlanState, _compiler: Any) -> HirPlanState:
    from srdatalog.ir.hir.index import IndexSelectionPass as _legacy

    assert state.hir is not None, 'IndexSelectionPass: hir not set (StratifyPass must run first)'
    new_hir = _legacy().run(state.hir)
    return dataclasses.replace(state, hir=new_hir)


__all__ = ['IndexSelectionPass']
