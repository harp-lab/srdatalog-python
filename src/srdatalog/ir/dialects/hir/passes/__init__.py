'''Per-HIR-pass `ProgramPass` wrappers (Phase D, Wave 2B).

Re-exports each migrated pass class so callers can build pipelines
without knowing the per-pass module layout:

    from srdatalog.ir.dialects.hir.passes import StratifyPass, SplitPass

Spec: `docs/phase_d_hir_passes.md` section 3 (per-pass migration
table) + section 3.1 (per-PR template).
'''

from __future__ import annotations

from srdatalog.ir.dialects.hir.passes.split import SplitPass
from srdatalog.ir.dialects.hir.passes.stratify import HirPlanState, StratifyPass

__all__ = [
  'HirPlanState',
  'SplitPass',
  'StratifyPass',
]
