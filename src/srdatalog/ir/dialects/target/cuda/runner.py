'''target.cuda — per-rule runner emission.

`emit_runner_full(ep, db, rel_index_types)` is the canonical entry
point that `compile.compile_runner` calls into. It produces the
per-rule `JitRunner_<rule>` struct plus all kernel definitions and
out-of-line phase methods — the content of `jit_runner.<rule>.cpp`.

Today the implementation delegates to the legacy
`ir.dialects.target.cuda.complete_runner.gen_complete_runner` for the runner
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

import srdatalog.ir.mir.types as m

# -----------------------------------------------------------------------------
# Pure-template phase-method emitters
#
# Each takes the runner type prefix + a few configuration knobs and returns
# the out-of-line C++ definition body. No IR walks, no plugin dispatch, no
# shared state — these are the pieces of the runner that are genuinely
# fixed-shape and were the natural first to lift.
# -----------------------------------------------------------------------------


def emit_launch_count(
  runner_prefix: str,
  *,
  is_block_group: bool = False,
  is_dedup_hash: bool = False,
) -> str:
  '''`<runner_prefix>::launch_count` — fires kernel_count (and the BG
  variant when `is_block_group=True`) on the given stream after the
  zero-key fast path. When `is_dedup_hash=True`, passes `p.dedup_table`
  to the kernel.'''
  dedup_arg = 'p.dedup_table, ' if is_dedup_hash else ''
  code = f'void {runner_prefix}::launch_count(LaunchParams& p, GPU_STREAM_T stream) {{\n'
  code += '  if (p.num_threads == 0) return;\n'
  if is_dedup_hash:
    # Single-line zero-key fast path on the dedup-hash path (matches Nim).
    code += (
      '  if (p.num_unique_root_keys == 0) { '
      'cudaMemsetAsync(p.thread_counts_ptr, 0, p.num_threads * sizeof(uint32_t), stream); return; }\n'
    )
  else:
    code += '  if (p.num_unique_root_keys == 0) {\n'
    code += (
      '    cudaMemsetAsync(p.thread_counts_ptr, 0, p.num_threads * sizeof(uint32_t), stream);\n'
    )
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
      f'p.num_unique_root_keys, p.num_root_keys, {dedup_arg}p.thread_counts_ptr);\n'
    )
    code += '  }\n'
  else:
    code += (
      '  kernel_count<<<p.num_blocks, kBlockSize, 0, stream>>>'
      '(p.d_views.data(), p.root_unique_values_ptr, '
      f'p.num_unique_root_keys, p.num_root_keys, {dedup_arg}p.thread_counts_ptr);\n'
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


def emit_launch_materialize(
  ep: m.ExecutePipeline,
  runner_prefix: str,
) -> str:
  '''`<runner_prefix>::launch_materialize` — fires the materialize kernel
  (and BG variant when ep.block_group). Pure template; ProvPtrType is
  always nullptr today (no provenance materialization yet).'''
  code = '// Phase 4: Launch materialize kernel on given stream (no sync)\n'
  code += (
    f'void {runner_prefix}::launch_materialize'
    '(DB& db, LaunchParams& p, uint32_t total_count, GPU_STREAM_T stream) {\n'
  )
  code += '  using ProvPtrType = semiring_value_t<SR>*;\n'
  code += '  ProvPtrType prov_ptr = nullptr;\n\n'
  for i, dest in enumerate(ep.dest_specs):
    code += f'  auto& dest_rel_{i} = get_relation_by_schema<{dest.rel_name}, NEW_VER>(db);\n'
    code += f'  uint32_t old_size_{i} = p.old_size_{i};\n'

  def _baseline_launch() -> str:
    out = '  kernel_materialize<<<p.num_blocks, kBlockSize, 0, stream>>>(\n'
    out += (
      '      p.d_views.data(), p.root_unique_values_ptr, p.num_unique_root_keys, p.num_root_keys,\n'
    )
    if ep.dedup_hash:
      # Dedup uses atomic_write_pos + dedup_table instead of thread_offsets.
      out += '      p.atomic_write_pos_ptr,\n'
      out += '      p.dedup_table,\n'
      out += (
        '      dest_rel_0.template interned_column<0>(), '
        'prov_ptr, dest_rel_0.interned_stride(), old_size_0'
      )
      for i in range(1, len(ep.dest_specs)):
        out += (
          f', dest_rel_{i}.template interned_column<0>(), '
          f'prov_ptr, dest_rel_{i}.interned_stride(), old_size_{i}'
        )
      out += ');\n'
      return out
    out += '      p.thread_counts_ptr'
    for i in range(len(ep.dest_specs)):
      out += (
        f', dest_rel_{i}.template interned_column<0>(), '
        f'prov_ptr, dest_rel_{i}.interned_stride(), old_size_{i}'
      )
    out += ');\n'
    return out

  if ep.block_group:
    code += '\n  if (p.bg_total_work > 0) {\n'
    code += '    kernel_bg_materialize<<<p.bg_num_blocks, kBlockSize, 0, stream>>>(\n'
    code += (
      '        p.d_views.data(), p.root_unique_values_ptr, '
      'p.head_root_unique_values_ptr, p.num_unique_root_keys, '
      'p.num_full_unique_root_keys, p.num_root_keys, '
      'p.bg_cumulative_work_ptr, p.bg_total_work, p.thread_counts_ptr'
    )
    for i in range(len(ep.dest_specs)):
      code += (
        f', dest_rel_{i}.template interned_column<0>(), prov_ptr, '
        f'dest_rel_{i}.interned_stride(), old_size_{i}'
      )
    code += ');\n'
    code += '  } else {\n  '
    code += _baseline_launch().replace('\n', '\n  ').rstrip('  ')
    code += '}\n'
  else:
    code += _baseline_launch()
  code += '}\n\n'
  return code


def emit_launch_fused(
  ep: m.ExecutePipeline,
  runner_prefix: str,
) -> str:
  '''`<runner_prefix>::launch_fused` — fires kernel_fused (or
  kernel_bg_fused with stream-ordered histogram) into the given stream.'''
  code = '// launch_fused: launch fused kernel on given stream (no sync)\n'
  code += f'void {runner_prefix}::launch_fused(DB& db, LaunchParams& p, GPU_STREAM_T stream) {{\n'
  code += '  if (p.num_unique_root_keys == 0) return;\n\n'
  for i, dest in enumerate(ep.dest_specs):
    code += f'  auto& dest_rel_{i} = get_relation_by_schema<{dest.rel_name}, NEW_VER>(db);\n'

  if ep.block_group:
    code += '\n  // BG fused: compute histogram for block assignment (stream-ordered)\n'
    code += '  int num_sms = 0;\n'
    code += '  GPU_DEVICE_GET_ATTRIBUTE(&num_sms, GPU_DEV_ATTR_MULTIPROCESSOR_COUNT, 0);\n'
    code += '  static SRDatalog::GPU::DeviceArray<uint64_t> bg_work_per_key;\n'
    code += '  bg_work_per_key.resize(p.num_unique_root_keys);\n'
    code += '  {\n'
    code += (
      '    uint32_t hist_blocks = std::max((uint32_t)'
      '((p.num_unique_root_keys + (kBlockSize/kGroupSize) - 1) / '
      '(kBlockSize/kGroupSize)), (uint32_t)(num_sms * 4));\n'
    )
    code += (
      '    kernel_bg_histogram<<<hist_blocks, kBlockSize, 0, stream>>>'
      '(p.d_views.data(), p.root_unique_values_ptr, '
      'p.head_root_unique_values_ptr, p.num_unique_root_keys, '
      'p.num_full_unique_root_keys, p.num_root_keys, '
      'bg_work_per_key.data());\n'
    )
    code += '  }\n'
    code += '  static SRDatalog::GPU::DeviceArray<uint64_t> bg_cumulative_work;\n'
    code += '  bg_cumulative_work.resize(p.num_unique_root_keys);\n'
    code += (
      '  thrust::inclusive_scan(rmm::exec_policy(stream), '
      'bg_work_per_key.data(), bg_work_per_key.data() + '
      'p.num_unique_root_keys, bg_cumulative_work.data());\n'
    )
    code += '  uint64_t bg_total_work = 0;\n'
    code += '  cudaStreamSynchronize(stream);\n'
    code += (
      '  GPU_MEMCPY(&bg_total_work, bg_cumulative_work.data() + '
      'p.num_unique_root_keys - 1, sizeof(uint64_t), GPU_DEVICE_TO_HOST);\n'
    )
    code += '  if (bg_total_work == 0) return;\n'
    code += '  uint32_t bg_num_blocks = num_sms * 8;\n\n'

    code += '  kernel_bg_fused<<<bg_num_blocks, kBlockSize, 0, stream>>>(\n'
    code += (
      '      p.d_views.data(), p.root_unique_values_ptr, '
      'p.head_root_unique_values_ptr, p.num_unique_root_keys, '
      'p.num_full_unique_root_keys, p.num_root_keys, '
      'bg_cumulative_work.data(), bg_total_work'
    )
    for i in range(len(ep.dest_specs)):
      code += (
        f', dest_rel_{i}.template interned_column<0>(), '
        f'dest_rel_{i}.interned_stride(), p.old_size_{i}, '
        f'p.fused_wp_ptr_{i}'
      )
    code += ', p.fused_capacity, p.fused_of_ptr);\n'
  else:
    code += '  kernel_fused<<<p.num_blocks, kBlockSize, 0, stream>>>(\n'
    code += (
      '      p.d_views.data(), p.root_unique_values_ptr, p.num_unique_root_keys, p.num_root_keys,\n'
    )
    parts = []
    for i in range(len(ep.dest_specs)):
      parts.append(
        f'dest_rel_{i}.template interned_column<0>(), '
        f'dest_rel_{i}.interned_stride(), p.old_size_{i}, p.fused_wp_ptr_{i}'
      )
    code += '      ' + ', '.join(parts) + ',\n'
    code += '      p.fused_capacity, p.fused_of_ptr);\n'
  code += '}\n\n'
  return code


def emit_read_fused_result(
  ep: m.ExecutePipeline,
  runner_prefix: str,
) -> str:
  '''`<runner_prefix>::read_fused_result` — readback fused write counts
  + overflow flag (call after device sync).'''
  code = '// read_fused_result: readback fused write counts (call after device sync)\n'
  code += f'void {runner_prefix}::read_fused_result(LaunchParams& p) {{\n'
  for i in range(len(ep.dest_specs)):
    code += (
      f'  GPU_MEMCPY(&p.fused_h_wp_{i}, p.fused_wp_ptr_{i}, '
      'sizeof(uint32_t), GPU_DEVICE_TO_HOST);\n'
    )
  code += '  uint32_t h_of = 0;\n'
  code += '  GPU_MEMCPY(&h_of, p.fused_of_ptr, sizeof(uint32_t), GPU_DEVICE_TO_HOST);\n'
  code += '  p.fused_overflow = (h_of != 0);\n'
  code += '}\n\n'
  return code


# -----------------------------------------------------------------------------
# Pre-operator() per-struct templates (declared inside `struct JitRunner_X`)
# -----------------------------------------------------------------------------


def emit_struct_type_aliases(
  rule_name: str,
  db_type_name: str,
  first_schema: str,
  first_version: str,
  dest_specs: list[m.InsertInto],
  dest_arities: list[int],
  total_view_count: int,
) -> str:
  '''Type alias block shared between `full` and `decl`. Does NOT include
  `struct JitRunner_X {` or the closing brace.
  '''
  code = '  // Type aliases - all concrete, resolved at Nim JIT time\n'
  code += f'  using DB = {db_type_name};\n'
  code += f'  using FirstSchema = {first_schema};\n'
  if dest_specs:
    code += f'  using DestSchema = {dest_specs[0].rel_name};\n'
  else:
    code += '  using DestSchema = void;\n'
  code += '  using SR = NoProvenance;\n'
  code += '  using ValueType = typename FirstSchema::intern_value_type;\n'
  code += (
    '  using RelType = std::decay_t<decltype(get_relation_by_schema'
    f'<FirstSchema, {first_version}>(std::declval<DB&>()))>;\n'
  )
  code += '  using IndexType = typename RelType::IndexTypeInst;\n'
  code += '  using ViewType = typename IndexType::NodeView;\n'
  code += '  static constexpr auto Layout = SRDatalog::GPU::StorageLayout::SoA;\n'
  code += '  static constexpr int kBlockSize = 256;\n'
  code += '  static constexpr int kGroupSize = 32;\n'
  for i, arity in enumerate(dest_arities):
    code += f'  static constexpr std::size_t OutputArity_{i} = {arity};\n'
    if i == 0:
      code += '  static constexpr std::size_t OutputArity = OutputArity_0;  // Legacy alias\n'
  code += f'  static constexpr std::size_t NumSources = {total_view_count};\n\n'
  return code


def emit_launch_params_struct(
  num_dests: int,
  is_fused_eligible: bool,
  is_block_group: bool = False,
  is_dedup_hash: bool = False,
  for_decl: bool = False,
) -> str:
  '''LaunchParams block — shared between `full` and `decl` emission.
  When `for_decl` is True the BG-block comment uses the decl variant
  ("must match JIT batch definition exactly!") to mirror Nim exactly.
  '''
  code = '  // State carried between decomposed phases\n'
  code += '  struct LaunchParams {\n'
  code += '    std::vector<ViewType> views_vec;\n'
  code += '    SRDatalog::GPU::DeviceArray<ViewType> d_views{0};\n'
  code += '    SRDatalog::GPU::DeviceArray<uint32_t> thread_counts{0};\n'
  code += '    uint32_t* thread_counts_ptr = nullptr;\n'
  code += '    uint32_t num_unique_root_keys = 0;\n'
  code += '    uint32_t num_root_keys = 0;\n'
  code += '    const ValueType* root_unique_values_ptr = nullptr;\n'
  code += '    const ValueType* head_root_unique_values_ptr = nullptr;\n'
  code += '    uint32_t num_full_unique_root_keys = 0;\n'
  code += '    uint32_t num_head_unique_root_keys = 0;\n'
  code += '    uint32_t num_blocks = 0;\n'
  code += '    uint32_t num_threads = 0;\n'
  if is_block_group:
    if for_decl:
      code += '    // Block-group state (must match JIT batch definition exactly!)\n'
    else:
      code += (
        '    // Block-group state (raw ptr to avoid DeviceArray default ctor using wrong stream)\n'
      )
    code += '    uint64_t* bg_cumulative_work_ptr = nullptr;\n'
    code += '    uint64_t bg_total_work = 0;\n'
    code += '    uint32_t bg_num_blocks = 0;\n'
    code += '    uint32_t bg_num_threads = 0;\n'
  for i in range(num_dests):
    code += f'    uint32_t old_size_{i} = 0;\n'
  if is_fused_eligible:
    code += '    // Fused execution state (set by orchestrator)\n'
    for j in range(num_dests):
      code += f'    uint32_t* fused_wp_ptr_{j} = nullptr;\n'
    code += '    uint32_t* fused_of_ptr = nullptr;\n'
    code += '    uint32_t fused_capacity = 0;\n'
    for j in range(num_dests):
      code += f'    uint32_t fused_h_wp_{j} = 0;\n'
    code += '    bool fused_overflow = false;\n'
  if is_dedup_hash:
    code += '    // Dedup hash table for in-kernel duplicate elimination\n'
    code += '    SRDatalog::GPU::DeviceArray<unsigned long long> dedup_hash_arr{0};\n'
    code += '    SRDatalog::GPU::DeviceArray<uint32_t> dedup_tid_arr{0};\n'
    code += '    DedupTable dedup_table{};\n'
    code += '    SRDatalog::GPU::DeviceArray<uint32_t> atomic_write_pos{0};\n'
    code += '    uint32_t* atomic_write_pos_ptr = nullptr;\n'
  code += '  };\n\n'
  return code


def emit_method_forward_decls(
  is_count: bool,
  is_fused_eligible: bool,
) -> str:
  '''Phase-method forward declarations inside `struct JitRunner_X`.'''
  code = '  // Phase-decomposed methods for stream-parallel execution\n'
  code += '  static LaunchParams setup(DB& db, uint32_t iteration, GPU_STREAM_T stream = 0);\n'
  code += '  static void launch_count(LaunchParams& p, GPU_STREAM_T stream = 0);\n'
  code += '  static uint32_t scan_and_resize(DB& db, LaunchParams& p, GPU_STREAM_T stream = 0);\n'
  code += '  static void scan_only(LaunchParams& p, GPU_STREAM_T stream = 0);\n'
  code += '  static uint32_t read_total(LaunchParams& p);\n'
  if not is_count:
    code += (
      '  static void launch_materialize(DB& db, LaunchParams& p, '
      'uint32_t total_count, GPU_STREAM_T stream = 0);\n\n'
    )

  code += '  // Non-template execute - calls kernels directly\n'
  code += '  static void execute(DB& db, uint32_t iteration);\n'
  if is_fused_eligible:
    code += '  static void execute_fused(DB& db, uint32_t iteration);\n'
    code += '  static void launch_fused(DB& db, LaunchParams& p, GPU_STREAM_T stream = 0);\n'
    code += '  static void read_fused_result(LaunchParams& p);\n'
    code += '  static inline uint32_t prev_fused_total_ = 4096;\n'
  return code


def emit_grid_config_code(prefix: str, root_is_scan: bool) -> str:
  '''Grid configuration template — populates `<prefix>num_threads` /
  `num_blocks` based on whether the rule is a binary join (row-based)
  or WCOJ (unique-key-based).'''
  code = '  int num_sms = 0;\n'
  code += '  GPU_DEVICE_GET_ATTRIBUTE(&num_sms, GPU_DEV_ATTR_MULTIPROCESSOR_COUNT, 0);\n'
  if root_is_scan:
    code += '  // Binary join: row-based iteration over ALL rows\n'
    code += (
      f'  {prefix}num_threads = (({prefix}num_root_keys + kGroupSize - 1) '
      '/ kGroupSize) * kGroupSize;\n'
    )
  else:
    code += '  // WCOJ: unique-key-based iteration\n'
    code += (
      f'  {prefix}num_threads = (({prefix}num_unique_root_keys + '
      'kGroupSize - 1) / kGroupSize) * kGroupSize;\n'
    )
  code += (
    f'  {prefix}num_threads = std::max({prefix}num_threads, static_cast<uint32_t>(kBlockSize));\n'
  )
  code += f'  {prefix}num_blocks = ({prefix}num_threads + kBlockSize - 1) / kBlockSize;\n'
  code += (
    f'  {prefix}num_blocks = std::max({prefix}num_blocks, static_cast<uint32_t>(num_sms) * 8);\n'
  )
  code += f'  {prefix}num_threads = {prefix}num_blocks * kBlockSize;\n\n'
  return code


# -----------------------------------------------------------------------------
# execute() and execute_fused() — top-level dispatchers
# -----------------------------------------------------------------------------


def emit_execute(
  rule_name: str,
  runner_prefix: str,
  is_count: bool,
  *,
  is_block_group: bool = False,
  is_dedup_hash: bool = False,
  dest_specs: list[m.InsertInto] | None = None,
) -> str:
  '''`<runner_prefix>::execute` — top-level dispatcher. For BG materialize
  rules, fans out into a 5-step pipeline (histogram → prefix sum → BG
  count → scan + resize → BG materialize) with adaptive fallback to
  baseline below the size threshold.'''
  code = '// Execute definition - calls decomposed phases sequentially\n'
  code += f'void {runner_prefix}::execute(DB& db, uint32_t iteration) {{\n'
  code += f'  nvtxRangePushA("{rule_name}");\n\n'

  if is_block_group and not is_count:
    code += '  auto p = setup(db, iteration);\n'
    code += '  if (p.num_unique_root_keys == 0) { nvtxRangePop(); return; }\n\n'
    code += '  // Adaptive: fall back to baseline for small deltas (histogram overhead > benefit)\n'
    code += '  constexpr uint32_t kBGAdaptiveThreshold = 256;\n'
    code += '  if (p.num_root_keys < kBGAdaptiveThreshold || p.num_unique_root_keys < 32) {\n'
    code += '    launch_count(p, 0);\n'
    code += '    uint32_t total_count = scan_and_resize(db, p, 0);\n'
    code += '    if (total_count == 0) { nvtxRangePop(); return; }\n'
    code += '    launch_materialize(db, p, total_count, 0);\n'
    code += '    nvtxRangePop();\n'
    code += '    return;\n'
    code += '  }\n\n'
    code += '  int num_sms = 0;\n'
    code += '  GPU_DEVICE_GET_ATTRIBUTE(&num_sms, GPU_DEV_ATTR_MULTIPROCESSOR_COUNT, 0);\n\n'
    code += '  // BG Step 1: Per-key work histogram (grid-stride, warp-reduced)\n'
    code += '  SRDatalog::GPU::DeviceArray<uint64_t> bg_work_per_key(p.num_unique_root_keys);\n'
    code += '  {\n'
    code += (
      '    uint32_t hist_blocks = std::max((uint32_t)'
      '((p.num_unique_root_keys + (kBlockSize/kGroupSize) - 1) / '
      '(kBlockSize/kGroupSize)), (uint32_t)(num_sms * 4));\n'
    )
    code += (
      '    kernel_bg_histogram<<<hist_blocks, kBlockSize>>>'
      '(p.d_views.data(), p.root_unique_values_ptr, '
      'p.head_root_unique_values_ptr, p.num_unique_root_keys, '
      'p.num_full_unique_root_keys, p.num_root_keys, '
      'bg_work_per_key.data());\n'
    )
    code += '    cudaDeviceSynchronize();\n'
    code += '  }\n\n'
    code += '  // BG Step 2: Inclusive prefix sum on per-key work\n'
    code += '  SRDatalog::GPU::DeviceArray<uint64_t> bg_cumulative_work(p.num_unique_root_keys);\n'
    code += (
      '  thrust::inclusive_scan(thrust::device, bg_work_per_key.data(), '
      'bg_work_per_key.data() + p.num_unique_root_keys, '
      'bg_cumulative_work.data());\n'
    )
    code += '  cudaDeviceSynchronize();\n'
    code += (
      '  GPU_MEMCPY(&p.bg_total_work, bg_cumulative_work.data() + '
      'p.num_unique_root_keys - 1, sizeof(uint64_t), GPU_DEVICE_TO_HOST);\n'
    )
    code += '  if (p.bg_total_work == 0) { nvtxRangePop(); return; }\n\n'
    code += '  // BG Step 3: Block-group count kernel\n'
    code += '  p.bg_num_blocks = num_sms * 8;\n'
    code += '  p.bg_num_threads = p.bg_num_blocks * kBlockSize;\n'
    code += '  p.thread_counts = SRDatalog::GPU::DeviceArray<uint32_t>(p.bg_num_threads + 1);\n'
    code += '  cudaMemset(p.thread_counts.data(), 0, (p.bg_num_threads + 1) * sizeof(uint32_t));\n'
    code += '  p.thread_counts_ptr = p.thread_counts.data();\n'
    code += (
      '  kernel_bg_count<<<p.bg_num_blocks, kBlockSize>>>'
      '(p.d_views.data(), p.root_unique_values_ptr, '
      'p.head_root_unique_values_ptr, p.num_unique_root_keys, '
      'p.num_full_unique_root_keys, p.num_root_keys, '
      'bg_cumulative_work.data(), p.bg_total_work, '
      'p.thread_counts_ptr);\n'
    )
    code += '  cudaDeviceSynchronize();\n\n'
    code += '  // BG Step 4: Prefix sum + resize\n'
    code += (
      '  thrust::exclusive_scan(thrust::device, p.thread_counts_ptr, '
      'p.thread_counts_ptr + p.bg_num_threads + 1, p.thread_counts_ptr, '
      '0, thrust::plus<uint32_t>());\n'
    )
    code += '  uint32_t total_count = 0;\n'
    code += (
      '  GPU_MEMCPY(&total_count, p.thread_counts_ptr + p.bg_num_threads, '
      'sizeof(uint32_t), GPU_DEVICE_TO_HOST);\n'
    )
    code += '  if (total_count == 0) { nvtxRangePop(); return; }\n\n'
    assert dest_specs is not None
    for i, dest in enumerate(dest_specs):
      code += f'  auto& bg_dest_{i} = get_relation_by_schema<{dest.rel_name}, NEW_VER>(db);\n'
      code += f'  uint32_t bg_old_size_{i} = static_cast<uint32_t>(bg_dest_{i}.size());\n'
      code += f'  bg_dest_{i}.resize_interned_columns(bg_old_size_{i} + total_count);\n'
    code += '\n  // BG Step 5: Block-group materialize kernel\n'
    code += (
      '  kernel_bg_materialize<<<p.bg_num_blocks, kBlockSize>>>'
      '(p.d_views.data(), p.root_unique_values_ptr, '
      'p.head_root_unique_values_ptr, p.num_unique_root_keys, '
      'p.num_full_unique_root_keys, p.num_root_keys, '
      'bg_cumulative_work.data(), p.bg_total_work, '
      'p.thread_counts_ptr'
    )
    for i in range(len(dest_specs)):
      code += (
        f', bg_dest_{i}.template interned_column<0>(), '
        f'static_cast<semiring_value_t<SR>*>(nullptr), '
        f'bg_dest_{i}.interned_stride(), bg_old_size_{i}'
      )
    code += ');\n'
    code += '  cudaDeviceSynchronize();\n'
    code += '  nvtxRangePop();\n'
    code += '}\n\n'
    return code

  # Baseline (non-BG) execute.
  code += '  auto p = setup(db, iteration);\n'
  code += '  launch_count(p, 0);\n'
  if is_count:
    code += '  scan_only(p, 0);\n'
    code += '  GPU_STREAM_SYNCHRONIZE(0);\n'
    code += '  uint32_t total_count = read_total(p);\n'
    code += '  if (total_count == 0) { nvtxRangePop(); return; }\n\n'
  elif is_dedup_hash:
    code += '  uint32_t total_count = scan_and_resize(db, p, 0);\n'
    code += '  if (total_count == 0) { nvtxRangePop(); return; }\n'
    code += '  // Clear for materialize: fresh table ensures identical dedup decisions\n'
    code += (
      '  cudaMemsetAsync(p.dedup_hash_arr.data(), 0, '
      'p.dedup_table.capacity * sizeof(unsigned long long), 0);\n'
    )
    code += '  cudaMemsetAsync(p.atomic_write_pos_ptr, 0, sizeof(uint32_t), 0);\n'
    code += '  launch_materialize(db, p, total_count, 0);\n'
  else:
    code += '  uint32_t total_count = scan_and_resize(db, p, 0);\n'
    code += '  if (total_count == 0) { nvtxRangePop(); return; }\n\n'
    code += '  launch_materialize(db, p, total_count, 0);\n'
  code += '  nvtxRangePop();\n'
  code += '}\n\n'
  return code


def emit_execute_fused(
  ep: m.ExecutePipeline,
  runner_prefix: str,
) -> str:
  '''`<runner_prefix>::execute_fused` — single-pass fused dispatcher with
  speculative output buffer + automatic capacity growth on overflow.'''
  code = '// Tail-mode fused execution: single kernel, no count/scan phase\n'
  code += f'void {runner_prefix}::execute_fused(DB& db, uint32_t iteration) {{\n'
  code += '  auto p = setup(db, iteration);\n'
  code += '  if (p.num_unique_root_keys == 0) return;\n\n'
  for i, dest in enumerate(ep.dest_specs):
    code += f'  auto& dest_rel_{i} = get_relation_by_schema<{dest.rel_name}, NEW_VER>(db);\n'
    code += f'  uint32_t old_size_{i} = static_cast<uint32_t>(dest_rel_{i}.size());\n'
  code += '  uint32_t capacity = prev_fused_total_;\n'
  for i in range(len(ep.dest_specs)):
    code += f'  dest_rel_{i}.resize_interned_columns(old_size_{i} + capacity);\n'
  code += '\n'
  for i in range(len(ep.dest_specs)):
    code += f'  static SRDatalog::GPU::DeviceArray<uint32_t> s_wp_{i}(1);\n'
    code += f'  cudaMemsetAsync(s_wp_{i}.data(), 0, sizeof(uint32_t), 0);\n'
  code += '  static SRDatalog::GPU::DeviceArray<uint32_t> s_of(1);\n'
  code += '  cudaMemsetAsync(s_of.data(), 0, sizeof(uint32_t), 0);\n'

  if ep.block_group:
    code += '\n  // BG fused: compute histogram for block assignment\n'
    code += '  int num_sms = 0;\n'
    code += '  GPU_DEVICE_GET_ATTRIBUTE(&num_sms, GPU_DEV_ATTR_MULTIPROCESSOR_COUNT, 0);\n'
    code += '  SRDatalog::GPU::DeviceArray<uint64_t> bg_work_per_key(p.num_unique_root_keys);\n'
    code += '  {\n'
    code += (
      '    uint32_t hist_blocks = std::max((uint32_t)'
      '((p.num_unique_root_keys + (kBlockSize/kGroupSize) - 1) / '
      '(kBlockSize/kGroupSize)), (uint32_t)(num_sms * 4));\n'
    )
    code += (
      '    kernel_bg_histogram<<<hist_blocks, kBlockSize>>>'
      '(p.d_views.data(), p.root_unique_values_ptr, '
      'p.head_root_unique_values_ptr, p.num_unique_root_keys, '
      'p.num_full_unique_root_keys, p.num_root_keys, '
      'bg_work_per_key.data());\n'
    )
    code += '    cudaDeviceSynchronize();\n'
    code += '  }\n'
    code += '  SRDatalog::GPU::DeviceArray<uint64_t> bg_cumulative_work(p.num_unique_root_keys);\n'
    code += (
      '  thrust::inclusive_scan(thrust::device, bg_work_per_key.data(), '
      'bg_work_per_key.data() + p.num_unique_root_keys, '
      'bg_cumulative_work.data());\n'
    )
    code += '  uint64_t bg_total_work = 0;\n'
    code += (
      '  GPU_MEMCPY(&bg_total_work, bg_cumulative_work.data() + '
      'p.num_unique_root_keys - 1, sizeof(uint64_t), GPU_DEVICE_TO_HOST);\n'
    )
    code += '  if (bg_total_work == 0) {\n'
    for i in range(len(ep.dest_specs)):
      code += f'    dest_rel_{i}.resize_interned_columns(old_size_{i});\n'
    code += '    return;\n'
    code += '  }\n'
    code += '  uint32_t bg_num_blocks = num_sms * 8;\n\n'

    code += '  kernel_bg_fused<<<bg_num_blocks, kBlockSize>>>(\n'
    code += (
      '      p.d_views.data(), p.root_unique_values_ptr, '
      'p.head_root_unique_values_ptr, p.num_unique_root_keys, '
      'p.num_full_unique_root_keys, p.num_root_keys, '
      'bg_cumulative_work.data(), bg_total_work'
    )
    for i in range(len(ep.dest_specs)):
      code += (
        f', dest_rel_{i}.template interned_column<0>(), '
        f'dest_rel_{i}.interned_stride(), old_size_{i}, s_wp_{i}.data()'
      )
    code += ', capacity, s_of.data());\n'
  else:
    code += '  kernel_fused<<<p.num_blocks, kBlockSize>>>(\n'
    code += (
      '      p.d_views.data(), p.root_unique_values_ptr, p.num_unique_root_keys, p.num_root_keys,\n'
    )
    parts = []
    for i in range(len(ep.dest_specs)):
      parts.append(
        f'dest_rel_{i}.template interned_column<0>(), '
        f'dest_rel_{i}.interned_stride(), old_size_{i}, s_wp_{i}.data()'
      )
    code += '      ' + ', '.join(parts) + ',\n'
    code += '      capacity, s_of.data());\n'

  code += '  GPU_DEVICE_SYNCHRONIZE();\n'
  code += '  uint32_t h_of = 0;\n'
  for i in range(len(ep.dest_specs)):
    code += f'  uint32_t h_wp_{i} = 0;\n'
    code += f'  GPU_MEMCPY(&h_wp_{i}, s_wp_{i}.data(), sizeof(uint32_t), GPU_DEVICE_TO_HOST);\n'
  code += '  GPU_MEMCPY(&h_of, s_of.data(), sizeof(uint32_t), GPU_DEVICE_TO_HOST);\n'
  code += '  if (!h_of) {\n'
  for i in range(len(ep.dest_specs)):
    code += f'    dest_rel_{i}.resize_interned_columns(old_size_{i} + h_wp_{i});\n'
  code += '    uint32_t max_wp = 0;\n'
  for i in range(len(ep.dest_specs)):
    code += f'    max_wp = std::max(max_wp, h_wp_{i});\n'
  code += '    prev_fused_total_ = std::max(max_wp, 1u) * 4;\n'
  code += '  } else {\n'
  for i in range(len(ep.dest_specs)):
    code += f'    dest_rel_{i}.resize_interned_columns(old_size_{i});\n'
  code += '    uint32_t max_wp = 0;\n'
  for i in range(len(ep.dest_specs)):
    code += f'    max_wp = std::max(max_wp, h_wp_{i});\n'
  code += '    prev_fused_total_ = max_wp * 4;\n'
  code += '    execute(db, iteration);\n'
  code += '  }\n'
  code += '}\n'
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
  from srdatalog.ir.dialects.target.cuda.complete_runner import gen_complete_runner

  _decl, full = gen_complete_runner(
    ep,
    db_type_name,
    rel_index_types=rel_index_types,
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
  from srdatalog.ir.dialects.target.cuda.complete_runner import gen_complete_runner

  decl, _full = gen_complete_runner(
    ep,
    db_type_name,
    rel_index_types=rel_index_types,
  )
  return decl


__all__ = [
  'emit_execute',
  'emit_execute_fused',
  'emit_grid_config_code',
  'emit_launch_count',
  'emit_launch_fused',
  'emit_launch_materialize',
  'emit_launch_params_struct',
  'emit_method_forward_decls',
  'emit_read_fused_result',
  'emit_read_total',
  'emit_runner_decl',
  'emit_runner_full',
  'emit_scan_and_resize',
  'emit_scan_only',
  'emit_struct_type_aliases',
]
