'''D-SemiNaive: wrap `srdatalog.ir.hir.semi_naive.SemiNaiveVariantPass`
as a `ProgramPass`.

Spec: `docs/phase_d_hir_passes.md` section 3 (per-pass migration
table) + section 3.1 (per-PR template). This module is the Wave 2B
landing for semi-naive variant generation (Nim's `generateVariants`,
Pass 3): for each `HirStratum`, populate `base_variants` (one per
rule for non-recursive strata) or `recursive_variants` (one DELTA
variant per SCC-member body clause for recursive strata).

The spec's migration table cites `semi_naive.py:gen_variants(prog)`
as the legacy entry; the in-tree module ships `generate_variants` as
the free function and `SemiNaiveVariantPass` as the pipeline wrapper
used by `default_pipeline` in `ir/hir/__init__.py`. Per the per-PR
template (section 3.1) we wrap the existing pipeline-shaped class
without changing behavior — the wrap matches what `compile_to_hir`
already invokes.

The legacy `SemiNaiveVariantPass.run(hir)` mutates `hir` in place
(`stratum.base_variants.append(...)` / `recursive_variants.append`)
and returns it — `SemiNaivePass._fn` preserves that contract,
replacing only the through-state's `hir` reference (same object
identity).
'''

from __future__ import annotations

import dataclasses
from typing import Any

from srdatalog.ir.core import ProgramPass
from srdatalog.ir.dialects.hir.passes.stratify import HirPlanState


class SemiNaivePass(ProgramPass):
  '''Wraps `srdatalog.ir.hir.semi_naive.SemiNaiveVariantPass`. Mutates
  the through-state's `hir` in place (legacy contract preserved) and
  returns the same `HirPlanState` instance via `dataclasses.replace`.
  '''

  def __init__(self) -> None:
    super().__init__(
      name='semi_naive',
      consumes=('hir',),
      produces=('hir',),
      fn=self._fn,
    )

  @staticmethod
  def _fn(state: HirPlanState, _compiler: Any) -> HirPlanState:
    from srdatalog.ir.hir.semi_naive import SemiNaiveVariantPass as _legacy

    assert state.hir is not None, 'SemiNaivePass: hir not set (StratifyPass must run first)'
    new_hir = _legacy().run(state.hir)
    return dataclasses.replace(state, hir=new_hir)


__all__ = ['SemiNaivePass']
