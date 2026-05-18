'''HIR pass subpackage (Phase D).

Per `docs/phase_d_hir_passes.md` section 1: each HIR planning pass
lands here as a `ProgramPass` instance in its own module under
`passes/`. Wave 2B (D-Stratify + D-Split + ...) wraps the legacy
imperative functions in `srdatalog.ir.hir` without changing
`compile_to_hir` — the wrappers exist so future PRs can put them into
`DEFAULT_HIR_PIPELINE` (spec section 4).

This subpackage carries no `Dialect` registration today; HIR types
stay as planning records, not Op subclasses (per the Stage 3B
"wrong abstraction" decision restated in section 1 of the spec).
'''

from __future__ import annotations
