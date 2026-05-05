'''target.cuda — IIR -> CUDA C++ emission.

M1: emits the inner kernel body for [Scan, InsertInto] pipelines.
The functor envelope, view declarations, and runner remain produced
by the legacy emitter for now — M1's surface is what `jit_pipeline()`
emits, not what `gen_jit_file_content_from_execute_pipeline()` emits.

See docs/stage2_emitter_audit.md §10 for the full per-rule emission
shape this module reproduces.
'''

from __future__ import annotations

from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
from srdatalog.ir.core import Dialect

DIALECT = Dialect(name='target.cuda')


def register_default_plugins() -> None:
  '''Register the default CUDA index plugins. Idempotent.

  Per S3A.8 / docs/stage3a_execution_plan.md §7 (kill A7), each compile
  entry point in `codegen.cuda.api` calls this at the start of compile
  time. No module-import-time side effects from dialect packages — the
  codegen is the registry's owner and explicitly opts into each plugin
  here. Adding a new index dialect = adding a `register_X_cuda_plugin()`
  function in that dialect and calling it from this function.
  '''
  from srdatalog.ir.dialects.relation.d2l.cuda import register_d2l_cuda_plugin

  register_d2l_cuda_plugin()


__all__ = ['DIALECT', 'EmitCtx', 'emit', 'register_default_plugins']
