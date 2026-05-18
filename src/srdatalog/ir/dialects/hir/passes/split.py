'''D-Split: wrap `srdatalog.ir.hir.split.TempRelSynthesisPass` as a
`ProgramPass`.

Spec: `docs/phase_d_hir_passes.md` section 3 (per-pass migration
table) + section 3.1 (per-PR template). The spec's table cites
`split.py:split_multihead(prog)` as the legacy entry point; the
in-tree split.py actually ships two `HirTransformPass` classes
(`TempRelSynthesisPass` Pass 4.5, `TempIndexRegistrationPass`
Pass 5.5) — `split_multihead` was never implemented (multi-head
unioning lives inside `_merge_sccs_for_multi_head` in
`stratify.py`). Per the per-PR template (section 3.1), each
Wave 2B PR wraps the existing imperative function from its named
module without changing behavior; the first / principal class in
`split.py` is `TempRelSynthesisPass`, so that is what `SplitPass`
wraps. `TempIndexRegistrationPass` lands in its own Wave 2B PR
(`D-TempIndex`) per the migration table.

The legacy `TempRelSynthesisPass.run(hir)` mutates `hir` in place
(`hir.relation_decls.append(...)`) and returns it — `SplitPass._fn`
preserves that contract, replacing only the through-state's `hir`
reference (same object identity).
'''

from __future__ import annotations

import dataclasses
from typing import Any

from srdatalog.ir.core import ProgramPass
from srdatalog.ir.dialects.hir.passes.stratify import HirPlanState


class SplitPass(ProgramPass):
  '''Wraps `srdatalog.ir.hir.split.TempRelSynthesisPass`. Mutates the
  through-state's `hir` in place (legacy contract preserved) and
  returns the same `HirPlanState` instance via `dataclasses.replace`.
  '''

  def __init__(self) -> None:
    super().__init__(
      name='split',
      consumes=('hir',),
      produces=('hir',),
      fn=self._fn,
    )

  @staticmethod
  def _fn(state: HirPlanState, _compiler: Any) -> HirPlanState:
    from srdatalog.ir.hir.split import TempRelSynthesisPass as _legacy

    assert state.hir is not None, 'SplitPass: hir not set (StratifyPass must run first)'
    new_hir = _legacy().run(state.hir)
    return dataclasses.replace(state, hir=new_hir)


__all__ = ['SplitPass']
