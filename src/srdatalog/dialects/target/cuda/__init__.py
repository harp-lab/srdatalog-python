'''target.cuda — IIR -> CUDA C++ emission.

M1: emits the inner kernel body for [Scan, InsertInto] pipelines.
The functor envelope, view declarations, and runner remain produced
by the legacy emitter for now — M1's surface is what `jit_pipeline()`
emits, not what `gen_jit_file_content_from_execute_pipeline()` emits.

See docs/stage2_emitter_audit.md §10 for the full per-rule emission
shape this module reproduces.
'''

from __future__ import annotations

from srdatalog.dialects.target.cuda.emit import EmitCtx, emit
from srdatalog.ir_core import Dialect

DIALECT = Dialect(name='target.cuda')


__all__ = ['DIALECT', 'EmitCtx', 'emit']
