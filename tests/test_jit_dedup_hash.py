'''Tests for dedup-hash codegen path.

When `ExecutePipeline.dedup_hash=True`, the kernel functor emit:
  1. Sets `ctx.dedup_hash_enabled=True` + populates `ctx.dedup_hash_vars`
  2. Inserts the `struct DedupTable` definition inside the kernel struct
  3. Extends `operator()` signature with a `DedupTable dedup_table` param
  4. The InsertInto emit (already ported) wraps `emit_direct` in a
     `dedup_table.try_insert(thread_id, ...)` gate, then the materialize
     path uses atomicAdd for the write position.
'''

import sys

import srdatalog.mir.types as m
from srdatalog.codegen.jit.kernel_functor import (
  gen_dedup_table_struct,
  jit_functor_start,
  jit_kernel_full,
)
from srdatalog.hir.types import Version

# -----------------------------------------------------------------------------
# gen_dedup_table_struct
# -----------------------------------------------------------------------------


def test_dedup_struct_arity_2():
  out = gen_dedup_table_struct(2)
  # struct header
  assert "struct DedupTable {" in out
  assert "unsigned long long* hash_slots;" in out
  assert "uint32_t* tid_slots;" in out
  assert "uint32_t capacity;" in out
  # compute_hash with 2 column params
  assert "__device__ __forceinline__ unsigned long long compute_hash(" in out
  assert "uint32_t v0,\n" in out
  assert "uint32_t v1)\n" in out
  # FNV-1a constants
  assert "14695981039346656037ULL;" in out
  assert "1099511628211ULL;" in out
  # try_insert + check_winner, each with thread_id + 2 v-params
  assert "__device__ __forceinline__ bool try_insert(" in out
  assert "__device__ __forceinline__ bool check_winner(" in out
  # atomicCAS linear-probing up to 128 slots
  assert "atomicCAS(&hash_slots[s], 0ULL, h);" in out
  assert "for (uint32_t p = 0; p < 128; p++) {" in out


def test_dedup_struct_arity_3_adds_third_var():
  out = gen_dedup_table_struct(3)
  assert "uint32_t v0,\n" in out
  assert "uint32_t v1,\n" in out
  assert "uint32_t v2)\n" in out
  # FNV-1a mix step per column
  assert out.count("h ^= (uint64_t)v") == 3


# -----------------------------------------------------------------------------
# jit_functor_start signature variations
# -----------------------------------------------------------------------------


def test_functor_start_without_dedup_no_dedup_param():
  out = jit_functor_start("R", dedup_hash=False)
  assert "DedupTable dedup_table" not in out
  assert "Output& output" in out


def test_functor_start_with_dedup_adds_param_before_output():
  out = jit_functor_start("R", dedup_hash=True)
  # Param list order: ... num_warps, DedupTable dedup_table, Output& output
  ddt_idx = out.index("DedupTable dedup_table,")
  out_idx = out.index("Output& output")
  num_warps_idx = out.index("uint32_t num_warps,")
  assert num_warps_idx < ddt_idx < out_idx


# -----------------------------------------------------------------------------
# jit_full_kernel integration
# -----------------------------------------------------------------------------


def _simple_dedup_ep(rule_name="Dupy") -> m.ExecutePipeline:
  '''Synthetic pipeline: scan + insert with dedup_hash enabled.'''
  scan = m.Scan(
    vars=["x", "y"],
    rel_name="Src",
    version=Version.FULL,
    index=[0, 1],
    handle_start=0,
  )
  insert = m.InsertInto(
    rel_name="Dst",
    version=Version.NEW,
    vars=["x", "y"],
    index=[0, 1],
  )
  return m.ExecutePipeline(
    pipeline=[scan, insert],
    source_specs=[scan],
    dest_specs=[insert],
    rule_name=rule_name,
    dedup_hash=True,
  )


def test_full_kernel_emits_dedup_table_struct_when_flag_set():
  ep = _simple_dedup_ep()
  out = jit_kernel_full(ep)
  # DedupTable struct present
  assert "struct DedupTable {" in out
  # Positioned between the constexpr + operator() — i.e., inside the
  # kernel struct but before operator().
  kernel_open = out.index("struct Kernel_Dupy {")
  op_start = out.index("__device__ void operator()(")
  ddt_pos = out.index("struct DedupTable {")
  assert kernel_open < ddt_pos < op_start
  # Functor signature now carries the dedup param
  assert "DedupTable dedup_table," in out


def test_full_kernel_no_dedup_does_not_emit_struct():
  scan = m.Scan(
    vars=["x"],
    rel_name="Src",
    version=Version.FULL,
    index=[0],
    handle_start=0,
  )
  insert = m.InsertInto(
    rel_name="Dst",
    version=Version.NEW,
    vars=["x"],
    index=[0],
  )
  ep = m.ExecutePipeline(
    pipeline=[scan, insert],
    source_specs=[scan],
    dest_specs=[insert],
    rule_name="NoDupe",
    dedup_hash=False,
  )
  out = jit_kernel_full(ep)
  assert "struct DedupTable" not in out
  assert "DedupTable dedup_table" not in out


def test_full_kernel_insert_into_wraps_emit_in_try_insert():
  '''With dedup_hash on, the InsertInto emit wraps `emit_direct` in a
  `dedup_table.try_insert(thread_id, ...)` gate — verified end-to-end
  through jit_kernel_full.'''
  ep = _simple_dedup_ep()
  out = jit_kernel_full(ep)
  assert "dedup_table.try_insert(thread_id, x, y)" in out
  # Atomic write path emitted (materialize phase), not plain emit_direct
  assert "atomicAdd(atomic_write_pos, 1u)" in out
  # dedup materialize uses out_data_0 / out_stride_0 layout
  assert "out_data_0[(pos + out_base_0) +" in out


def test_full_kernel_dedup_vars_captured_in_context():
  '''jit_kernel_full should populate ctx.dedup_hash_vars from the
  first InsertInto's vars so emit paths can reference them — sanity
  check by inspecting the emitted try_insert args.'''
  ep = _simple_dedup_ep()
  out = jit_kernel_full(ep)
  # try_insert args list the sanitized InsertInto vars in order
  assert "try_insert(thread_id, x, y)" in out


if __name__ == "__main__":
  import inspect

  this = sys.modules[__name__]
  passed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if name.startswith("test_"):
      fn()
      passed += 1
  print(f"{passed} passed")
