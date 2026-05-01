'''Public compile entry point for the dialect-based codegen.

`compile_pipeline(ep, target='cuda')` is the public surface that
Stage 2+ migrates the JIT codegen behind. For M0 (test harness setup,
no dialect lowering yet), it is a back-compat shim around
`gen_jit_file_content_from_execute_pipeline` — which is what every
existing test path ultimately calls into. As Stage 2 milestones land,
the body progressively routes through the dialect framework
(Compiler + registered dialects) instead of calling the legacy
emitter directly.

The contract:

  - Same input as the legacy path: an `m.ExecutePipeline` MIR node.
  - Same output: a C++ source string.
  - Byte-equivalent (after _cpp_norm normalization) to the legacy
    path on every existing test fixture, at every milestone.

Once Stage 2 is complete, the legacy emitter (`codegen/jit/`) becomes
internal implementation of specific lowering rules; callers use
`compile_pipeline` exclusively.

See:
  - docs/stage2_emitter_audit.md — the per-milestone migration plan.
  - docs/ir_lowering_semantics.md — the formal lowering rules.
  - docs/design_principles.md — discipline rules for the rewrite.
'''

from __future__ import annotations

from typing import Literal

import srdatalog.mir.types as m

Target = Literal['cuda']


def compile_pipeline(ep: m.ExecutePipeline, *, target: Target = 'cuda') -> str:
  '''Compile an MIR ExecutePipeline to target C++ source.

  M0 status: delegates to the legacy emitter. Stage 2 milestones M1+
  progressively route through the dialect framework while preserving
  byte-equivalence with this baseline.

  Args:
    ep: The MIR ExecutePipeline to compile.
    target: Codegen backend. Currently 'cuda' only; CPU backends
      arrive in Stage 3.

  Returns:
    C++ source string for the kernel + runner envelope.

  Raises:
    ValueError: if `target` is not supported.
  '''
  if target != 'cuda':
    raise ValueError(f'compile_pipeline: unsupported target {target!r}')
  # Delayed import: the legacy emitter pulls in heavy modules; keep
  # `compile_pipeline` cheap to import for tests that don't actually
  # invoke it.
  from srdatalog.codegen.jit.file import gen_jit_file_content_from_execute_pipeline

  return gen_jit_file_content_from_execute_pipeline(ep)


__all__ = ['Target', 'compile_pipeline']
