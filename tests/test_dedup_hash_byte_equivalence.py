'''Structural tests for the dialect's dedup-hash WriteOutput emit (N6).

There are no `jit_batch.*.cpp` goldens that exercise `dedup_hash=True`
in the project's fixture set, so we synthesize a handful of
representative pipelines and assert that `compile_pipeline` produces
the expected dedup-hash structural shapes:

  - `try_insert(thread_id, v0, ...)` gate around the write.
  - `if (_p) {` guard for the actual emit.
  - Materialize: `atomicAdd(atomic_write_pos, 1u)` + `out_data_0[...]`.
  - Count: `output.emit_direct()` (no atomicAdd).
  - Without `dedup_hash`: standard `output.emit_direct(args)` and no
    dedup machinery.

Earlier this file compared dialect output byte-for-byte against the
legacy `jit_kernel_full` emit. That oracle was retired alongside the
legacy `ir/dialects/target/cuda/` chain — once the runner-level byte-equivalence
suite (`tests/test_runner_byte_equivalence.py`,
`tests/test_byte_equivalence_jit.py`) established equivalence for all
in-tree fixtures, the synthetic dedup pipelines need only structural
sanity checks.
'''

from __future__ import annotations

import srdatalog.ir.mir.types as m
from srdatalog.compile import compile_pipeline
from srdatalog.ir.hir.types import Version


def _scan_insert_dedup(arity: int, rule_name: str = 'Dupy') -> m.ExecutePipeline:
  vars_ = [f'v{i}' for i in range(arity)]
  cols = list(range(arity))
  scan = m.Scan(
    vars=vars_, rel_name='Src', version=Version.FULL, index=cols, handle_start=0,
  )
  insert = m.InsertInto(
    rel_name='Dst', version=Version.NEW, vars=vars_, index=cols,
  )
  return m.ExecutePipeline(
    pipeline=[scan, insert],
    source_specs=[scan],
    dest_specs=[insert],
    rule_name=rule_name,
    dedup_hash=True,
  )


def test_dedup_hash_emits_try_insert_gate():
  '''Structural sanity check: the dialect emits the dedup_hash
  three-tier shape — try_insert + `if (_p)` + atomic-add write
  block, with one column-write per arity.'''
  ep = _scan_insert_dedup(2)
  out = compile_pipeline(ep)
  assert 'dedup_table.try_insert(thread_id, v0, v1)' in out
  assert 'if (_p) {' in out
  assert 'atomicAdd(atomic_write_pos, 1u)' in out
  assert 'out_data_0[(pos + out_base_0) + 0 * out_stride_0] = v0' in out
  assert 'out_data_0[(pos + out_base_0) + 1 * out_stride_0] = v1' in out


def test_dedup_hash_high_arity_writes_each_column():
  '''Each output column gets its own `out_data_0[...] = vN` line.'''
  ep = _scan_insert_dedup(4)
  out = compile_pipeline(ep)
  for col in range(4):
    assert (
      f'out_data_0[(pos + out_base_0) + {col} * out_stride_0] = v{col}'
      in out
    )


def test_dedup_hash_off_does_not_emit_dedup_machinery():
  '''Sanity: with `dedup_hash=False`, the standard emit_direct path
  fires and no dedup_table / atomic_write_pos symbols leak in.'''
  scan = m.Scan(
    vars=['x'], rel_name='Src', version=Version.FULL, index=[0], handle_start=0,
  )
  insert = m.InsertInto(rel_name='Dst', version=Version.NEW, vars=['x'], index=[0])
  ep = m.ExecutePipeline(
    pipeline=[scan, insert],
    source_specs=[scan],
    dest_specs=[insert],
    rule_name='NoDupe',
    dedup_hash=False,
  )
  out = compile_pipeline(ep)
  assert 'dedup_table' not in out
  assert 'atomic_write_pos' not in out
  assert 'output.emit_direct(x)' in out
