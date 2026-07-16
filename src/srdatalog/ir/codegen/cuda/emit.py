'''codegen.cuda.emit — backwards-compatible re-export shim.

Per docs/stage3a_execution_plan.md §7 task S3A.3, the CUDA renderers
moved from a 41-case match in this file into per-dialect modules under
`codegen/cuda/render/`. Each dialect's renderer self-registers via
`@register_render` decorators in its own file, so adding a new IIR
dialect no longer requires editing this file (P1 fix).

This module is now a thin shim that re-exports the same public API
(`EmitCtx`, `emit`, `emit_expr`) from the new location. Existing
callers (`codegen/cuda/api.py`, `codegen/cuda/runner.py`,
`codegen/cuda/complete_runner.py`) keep working without changes.

When the module-global registry moves to a per-`Codegen` instance in
S3A.8, this shim disappears entirely; until then it preserves the
import surface external code expects.
'''

from __future__ import annotations

from srdatalog.ir.codegen.cuda.render import EmitCtx, emit, emit_expr

__all__ = ['EmitCtx', 'emit', 'emit_expr']
