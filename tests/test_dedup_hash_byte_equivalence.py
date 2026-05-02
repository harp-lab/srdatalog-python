'''Byte-equivalence gate for the dialect's dedup_hash WriteOutput emit
(N6).

There are no `jit_batch.*.cpp` goldens that exercise `dedup_hash=True`,
so we synthesize a handful of representative pipelines and compare the
dialect's `compile_pipeline` output (kernel body) against the legacy
`jit_kernel_full` emit. Any divergence in the dedup-hash try_insert
gate or the materialize-phase atomic-add write surfaces here.

Once the legacy `jit_kernel_full` is retired, this file can be flipped
to compare against checked-in goldens — the pipelines below are stable
enough to capture as fixtures.
'''

from __future__ import annotations

import re

import pytest

import srdatalog.mir.types as m
from srdatalog.codegen.jit.kernel_functor import jit_kernel_full
from srdatalog.compile import compile_pipeline
from srdatalog.hir.types import Version


def _norm(src: str) -> str:
  '''Same shape as `tests/integration_helpers._cpp_norm` — strip line
  comments, collapse whitespace, drop spaces adjacent to punctuation.'''
  src = re.sub(r'//[^\n]*', '', src)
  src = re.sub(r'\s+', ' ', src).strip()
  src = re.sub(r'\s*([(),;{}])\s*', r'\1', src)
  return src


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


def _kernel_body_only(s: str) -> str:
  '''Extract the operator()-body region we care about. The dialect's
  `compile_pipeline` adds a JIT_FILE_PRELUDE wrapper that
  `jit_kernel_full` doesn't, so trim everything up to the first
  `__device__ void operator()(` and everything after the matching
  `}\n};` for a fair comparison.'''
  start = s.index('__device__ void operator()(')
  return s[start:]


@pytest.mark.parametrize('arity', [1, 2, 3, 4])
def test_dedup_hash_scan_insert_byte_equivalence(arity: int):
  ep = _scan_insert_dedup(arity)
  dialect_out = _kernel_body_only(compile_pipeline(ep))
  legacy_out = _kernel_body_only(jit_kernel_full(ep))
  assert _norm(dialect_out) == _norm(legacy_out), (
    f'dedup_hash arity={arity}: dialect output diverges from legacy '
    f'jit_kernel_full.\nDIALECT:\n{dialect_out}\n\nLEGACY:\n{legacy_out}'
  )


def test_dedup_hash_emits_try_insert_gate():
  '''Sanity check on the structural shape: try_insert + if(_p) +
  atomicAdd write block.'''
  ep = _scan_insert_dedup(2)
  out = compile_pipeline(ep)
  assert 'dedup_table.try_insert(thread_id, v0, v1)' in out
  assert 'if (_p) {' in out
  assert 'atomicAdd(atomic_write_pos, 1u)' in out
  assert 'out_data_0[(pos + out_base_0) + 0 * out_stride_0] = v0' in out
  assert 'out_data_0[(pos + out_base_0) + 1 * out_stride_0] = v1' in out


def test_dedup_hash_off_does_not_emit_dedup_machinery():
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
