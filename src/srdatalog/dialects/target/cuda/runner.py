'''target.cuda — per-rule runner emission.

`emit_runner_full(ep, db, rel_index_types)` is the canonical entry
point that `compile.compile_runner` calls into. It produces the
per-rule `JitRunner_<rule>` struct plus all kernel definitions and
out-of-line phase methods — the content of `jit_runner.<rule>.cpp`.

Today the implementation delegates to the legacy
`codegen.jit.complete_runner.gen_complete_runner` for the runner
scaffolding (phase methods, type aliases, execute() dispatcher,
LaunchParams struct, BG variants, fused kernel) and routes kernel
*bodies* through `compile_kernel_body` when `_dialect_safe_kernel`
holds. Subsequent milestones port the remaining pieces:

  - N2  Fused composer (count + materialize back-to-back operator())
  - N4  par.data.block_group dialect (BG warp-cumulative dispatch)
  - N5  relation.d2l dialect (multi-view plugin dispatch + setup)
  - N6  Dedup-hash WriteOutput variant
  - N7  Tiled-Cartesian ballot-reuse on relation.sorted_array
  - N8  par.data.atomic_ws dialect (WCOJ task queue)

Each milestone collapses one slice of the delegation into native
dialect emission, validated by `tests/test_runner_byte_equivalence.py`.

The emission output of this module is byte-equivalent (modulo
`_cpp_norm`) to the upstream Nim `jit_runner.<rule>.cpp` goldens
on every fixture that the legacy emitter handled.
'''

from __future__ import annotations

import srdatalog.mir.types as m

# -----------------------------------------------------------------------------
# Pure-template phase-method emitters
#
# Each takes the runner type prefix + a few configuration knobs and returns
# the out-of-line C++ definition body. No IR walks, no plugin dispatch, no
# shared state — these are the pieces of the runner that are genuinely
# fixed-shape and were the natural first to lift.
# -----------------------------------------------------------------------------


def emit_launch_count(runner_prefix: str, *, is_block_group: bool = False) -> str:
  '''`<runner_prefix>::launch_count` — fires kernel_count (and the BG
  variant when `is_block_group=True`) on the given stream after the
  zero-key fast path.'''
  code = f'void {runner_prefix}::launch_count(LaunchParams& p, GPU_STREAM_T stream) {{\n'
  code += '  if (p.num_threads == 0) return;\n'
  code += '  if (p.num_unique_root_keys == 0) {\n'
  code += '    cudaMemsetAsync(p.thread_counts_ptr, 0, p.num_threads * sizeof(uint32_t), stream);\n'
  code += '    return;\n'
  code += '  }\n'
  if is_block_group:
    code += '  if (p.bg_total_work > 0) {\n'
    code += (
      '    kernel_bg_count<<<p.num_blocks, kBlockSize, 0, stream>>>'
      '(p.d_views.data(), p.root_unique_values_ptr, '
      'p.head_root_unique_values_ptr, p.num_unique_root_keys, '
      'p.num_full_unique_root_keys, p.num_root_keys, '
      'p.bg_cumulative_work_ptr, p.bg_total_work, p.thread_counts_ptr);\n'
    )
    code += '  } else {\n'
    code += (
      '    kernel_count<<<p.num_blocks, kBlockSize, 0, stream>>>'
      '(p.d_views.data(), p.root_unique_values_ptr, '
      'p.num_unique_root_keys, p.num_root_keys, p.thread_counts_ptr);\n'
    )
    code += '  }\n'
  else:
    code += (
      '  kernel_count<<<p.num_blocks, kBlockSize, 0, stream>>>'
      '(p.d_views.data(), p.root_unique_values_ptr, '
      'p.num_unique_root_keys, p.num_root_keys, p.thread_counts_ptr);\n'
    )
  code += '}\n\n'
  return code


def emit_scan_and_resize(
  ep: m.ExecutePipeline,
  runner_prefix: str,
) -> str:
  '''`<runner_prefix>::scan_and_resize` — exclusive prefix-scan over
  thread_counts, read total, resize each dest relation in place.'''
  code = '// Phase 3: Prefix scan + readback total + resize destinations\n'
  code += (
    f'uint32_t {runner_prefix}::scan_and_resize(DB& db, LaunchParams& p, GPU_STREAM_T stream) {{\n'
  )
  code += (
    '  thrust::exclusive_scan(rmm::exec_policy(stream), '
    'p.thread_counts_ptr, p.thread_counts_ptr + p.num_threads + 1, '
    'p.thread_counts_ptr, 0, thrust::plus<uint32_t>());\n'
  )
  code += '  uint32_t total_count = 0;\n'
  code += (
    '  GPU_MEMCPY_ASYNC(&total_count, p.thread_counts_ptr + p.num_threads, '
    'sizeof(uint32_t), GPU_DEVICE_TO_HOST, stream);\n'
  )
  code += '  GPU_STREAM_SYNCHRONIZE(stream);\n'
  code += '  if (total_count == 0) return 0;\n\n'
  for i, dest in enumerate(ep.dest_specs):
    code += f'  auto& dest_rel_{i} = get_relation_by_schema<{dest.rel_name}, NEW_VER>(db);\n'
    code += f'  p.old_size_{i} = static_cast<uint32_t>(dest_rel_{i}.size());\n'
    code += f'  dest_rel_{i}.resize_interned_columns(p.old_size_{i} + total_count, stream);\n'
  code += '  return total_count;\n'
  code += '}\n\n'
  return code


def emit_scan_only(runner_prefix: str) -> str:
  '''`<runner_prefix>::scan_only` — async prefix-scan, no host sync.'''
  code = '// Phase 3a: Prefix scan only (async, no sync)\n'
  code += f'void {runner_prefix}::scan_only(LaunchParams& p, GPU_STREAM_T stream) {{\n'
  code += '  if (p.num_threads == 0) return;\n'
  code += (
    '  thrust::exclusive_scan(rmm::exec_policy(stream), '
    'p.thread_counts_ptr, p.thread_counts_ptr + p.num_threads + 1, '
    'p.thread_counts_ptr, 0, thrust::plus<uint32_t>());\n'
  )
  code += '}\n\n'
  return code


def emit_read_total(runner_prefix: str) -> str:
  '''`<runner_prefix>::read_total` — read the post-scan total count
  (call after device sync).'''
  code = '// Phase 3b: Read total count (call after device sync)\n'
  code += f'uint32_t {runner_prefix}::read_total(LaunchParams& p) {{\n'
  code += '  if (p.num_threads == 0) return 0;\n'
  code += '  uint32_t total_count = 0;\n'
  code += (
    '  GPU_MEMCPY(&total_count, p.thread_counts_ptr + p.num_threads, '
    'sizeof(uint32_t), GPU_DEVICE_TO_HOST);\n'
  )
  code += '  return total_count;\n'
  code += '}\n\n'
  return code


# -----------------------------------------------------------------------------
# Top-level entry — full / decl
# -----------------------------------------------------------------------------


def emit_runner_full(
  ep: m.ExecutePipeline,
  db_type_name: str,
  rel_index_types: dict[str, str] | None = None,
) -> str:
  '''Emit the full per-rule runner — struct + kernel defs + out-of-line
  phase methods + execute(). Goes into the per-rule `jit_batch_N.cpp`
  file at production-build time.'''
  from srdatalog.codegen.jit.complete_runner import gen_complete_runner

  _decl, full = gen_complete_runner(
    ep, db_type_name, rel_index_types=rel_index_types,
  )
  return full


def emit_runner_decl(
  ep: m.ExecutePipeline,
  db_type_name: str,
  rel_index_types: dict[str, str] | None = None,
) -> str:
  '''Emit the forward-declaration variant — type aliases + LaunchParams
  + method declarations only. Goes into the main compile unit so the
  orchestrator can call `JitRunner_<rule>::execute()`.'''
  from srdatalog.codegen.jit.complete_runner import gen_complete_runner

  decl, _full = gen_complete_runner(
    ep, db_type_name, rel_index_types=rel_index_types,
  )
  return decl


__all__ = [
  'emit_launch_count',
  'emit_read_total',
  'emit_runner_decl',
  'emit_runner_full',
  'emit_scan_and_resize',
  'emit_scan_only',
]
