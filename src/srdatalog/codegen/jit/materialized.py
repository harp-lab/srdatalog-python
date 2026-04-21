'''Materialized binary-join code generation.

Port of src/srdatalog/codegen/target_jit/jit_materialized.nim.

Materialized joins take a different shape from fused joins (kernel
functors with nested loops). They emit host-side C++ that:
  1. Materializes row-id pairs in buffers between join stages
  2. Uses binary-search probes against already-indexed relations
  3. Applies merge-path load balancing for unbalanced outputs

The generated C++ uses Thrust primitives (thrust::lower_bound /
upper_bound, exclusive_scan, gather) plus three CUDA kernels declared
in `gen_materialized_join_helpers()`.

Public API:

  is_materialized_pipeline(ops) -> bool
    True iff any op is a ProbeJoin. `jit_complete_runner` uses this
    as an early-dispatch check; when True it emits a materialized
    runner instead of a kernel functor.

  gen_materialized_runner(node, db_type_name) -> str
    The main entry. Given an ExecutePipeline MIR node, emits a host-
    side Thrust executor:
      struct JitRunner_<rule> {
        using DB = <dbTypeName>;
        using FirstSchema = ...;
        ...
        static void execute(DB& db, uint32_t iteration = 0) { ... }
      };

  gen_materialized_join_helpers() -> str
    Returns the three __global__ CUDA kernels
    (probe_count_matches_kernel, probe_materialize_pairs_kernel,
    gather_column_kernel) that the runner's execute() body calls.

  gen_materialized_join_kernel(ops, rule_name, ctx) -> str
    Legacy in-kernel variant. Not used by Nim's live code and not
    exercised by our fixtures — ported for completeness.
'''
from __future__ import annotations
from typing import Optional

import srdatalog.mir.types as m
from srdatalog.hir.types import Version
from srdatalog.codegen.jit.context import CodeGenContext, ind


# -----------------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------------

def is_materialized_pipeline(ops: list[m.MirNode]) -> bool:
  '''True iff the pipeline contains a ProbeJoin — which switches
  dispatch to the materialized runner.'''
  return any(isinstance(op, m.ProbeJoin) for op in ops)


# -----------------------------------------------------------------------------
# Helper kernels (fixed string, matches Nim)
# -----------------------------------------------------------------------------

def gen_materialized_join_helpers() -> str:
  '''Three __global__ CUDA helper kernels + comments. Embedded verbatim
  into the generated batch file when a materialized pipeline is present.
  '''
  return """
// ==========================================================================
// Materialized Join Helpers (Thrust-based)
// ==========================================================================

// Count matches for each input key (for load balancing)
template<typename KeyT, typename ViewT>
__global__ void probe_count_matches_kernel(
    const uint32_t* input_rowids,
    const KeyT* input_keys,
    ViewT view,
    uint32_t* counts,
    size_t n
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    KeyT key = input_keys[tid];
    auto range = view.get_range(key);
    counts[tid] = range.second - range.first;
}

// Materialize (left_rowid, right_rowid) pairs with merge-path partitioning
template<typename KeyT, typename ViewT>
__global__ void probe_materialize_pairs_kernel(
    const uint32_t* input_rowids,
    const KeyT* input_keys,
    ViewT view,
    const uint32_t* offsets,
    uint32_t* out_left,
    uint32_t* out_right,
    size_t n
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    uint32_t left_rowid = input_rowids[tid];
    KeyT key = input_keys[tid];
    auto range = view.get_range(key);

    uint32_t out_offset = offsets[tid];
    for (uint32_t i = range.first; i < range.second; ++i) {
        out_left[out_offset] = left_rowid;
        out_right[out_offset] = i;  // Right side row ID
        ++out_offset;
    }
}

// Gather column values using row IDs
template<typename Schema, int Col, typename T>
__global__ void gather_column_kernel(
    const uint32_t* rowids,
    const T* column_data,
    T* output,
    size_t n
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    output[tid] = column_data[rowids[tid]];
}

"""


# -----------------------------------------------------------------------------
# Version normalization
# -----------------------------------------------------------------------------

def _version_cpp(ver: "str | Version") -> str:
  '''Map whatever the MIR carries (string or Version) to the C++
  `*_VER` constant. Matches Nim's `case` on the stringly-typed version.
  '''
  if isinstance(ver, Version):
    return ver.code
  v = str(ver)
  if v in ("DELTA", "DeltaVer", "DELTA_VER"):
    return "DELTA_VER"
  if v in ("FULL", "FullVer", "FULL_VER"):
    return "FULL_VER"
  if v in ("NEW", "NewVer", "NEW_VER"):
    return "NEW_VER"
  return "FULL_VER"


# -----------------------------------------------------------------------------
# Legacy in-kernel variant (gen_materialized_join_kernel)
# -----------------------------------------------------------------------------

def gen_materialized_join_kernel(
  ops: list[m.MirNode], rule_name: str, ctx: CodeGenContext,
) -> str:
  '''Emit in-kernel materialized-join code. Unused by Nim's live
  codegen (superseded by the host-side runner below) but ported for
  completeness.
  '''
  i = ind(ctx)
  code = ""

  scan_op: Optional[m.Scan] = None
  probe_join_ops: list[m.ProbeJoin] = []
  gather_ops: list[m.GatherColumn] = []
  insert_ops: list[m.InsertInto] = []

  for op in ops:
    if isinstance(op, m.Scan):
      scan_op = op
    elif isinstance(op, m.ProbeJoin):
      probe_join_ops.append(op)
    elif isinstance(op, m.GatherColumn):
      gather_ops.append(op)
    elif isinstance(op, m.InsertInto):
      insert_ops.append(op)

  if scan_op is None:
    return code + i + "// ERROR: Materialized join requires moScan as first op\n"

  # Phase 1: Initial scan
  code += i + f"// ===== Materialized Join: {rule_name} =====\n"
  code += i + "// Phase 1: Scan first relation\n"

  scan_rel = scan_op.rel_name
  scan_idx_join_under = "_".join(str(c) for c in scan_op.index)
  scan_view_var = f"view_{scan_rel}_{scan_idx_join_under}"
  scan_idx_join_comma = ", ".join(str(c) for c in scan_op.index)

  code += (
    i + f"auto {scan_view_var} = db.get_view<{scan_rel}, "
    f"{scan_idx_join_comma}>();\n"
  )
  code += i + f"auto {scan_rel}_size = {scan_view_var}.size();\n"
  code += i + f"if ({scan_rel}_size == 0) return;\n\n"

  current_buffer_var = f"{scan_rel}_rowids"
  current_size_var = f"{scan_rel}_size"

  code += i + "// Initial row IDs are just [0, N)\n"
  code += (
    i + f"thrust::device_vector<uint32_t> {current_buffer_var}({current_size_var});\n"
  )
  code += (
    i + f"thrust::sequence({current_buffer_var}.begin(), "
    f"{current_buffer_var}.end());\n\n"
  )

  # Phase 2: Probe joins
  buffer_counter = 0
  for probe_op in probe_join_ops:
    buffer_counter += 1
    probe_rel = probe_op.probe_rel
    join_key = probe_op.join_key
    output_buf = probe_op.output_buffer
    probe_idx_join_under = "_".join(str(c) for c in probe_op.probe_index)
    probe_idx_join_comma = ", ".join(str(c) for c in probe_op.probe_index)

    code += (
      i + f"// Phase 2.{buffer_counter}: Probe {probe_rel} on {join_key}\n"
    )

    probe_view_var = f"view_{probe_rel}_{probe_idx_join_under}"
    code += (
      i + f"auto {probe_view_var} = db.get_view<{probe_rel}, "
      f"{probe_idx_join_comma}>();\n"
    )

    count_var = f"{output_buf}_counts"
    input_n = current_size_var

    code += i + "// Count matches per input row (for load balancing)\n"
    code += i + f"thrust::device_vector<uint32_t> {count_var}({input_n});\n"
    code += (
      i + f"probe_count_matches({current_buffer_var}, {join_key}_gather, "
      f"{probe_view_var}, {count_var}.data());\n"
    )

    offset_var = f"{output_buf}_offsets"
    total_var = f"{output_buf}_total"

    code += (
      i + f"thrust::device_vector<uint32_t> {offset_var}({input_n} + 1);\n"
    )
    code += (
      i + f"thrust::exclusive_scan({count_var}.begin(), {count_var}.end(), "
      f"{offset_var}.begin(), 0u);\n"
    )
    code += (
      i + f"{offset_var}[{input_n}] = {offset_var}[{input_n}-1] + "
      f"{count_var}[{input_n}-1];\n"
    )
    code += i + f"uint32_t {total_var} = {offset_var}[{input_n}];\n"
    code += i + f"if ({total_var} == 0) return;\n\n"

    pair_buf_left = f"{output_buf}_left"
    pair_buf_right = f"{output_buf}_right"

    code += i + "// Materialize row ID pairs\n"
    code += (
      i + f"thrust::device_vector<uint32_t> {pair_buf_left}({total_var});\n"
    )
    code += (
      i + f"thrust::device_vector<uint32_t> {pair_buf_right}({total_var});\n"
    )
    code += (
      i + f"probe_materialize_pairs({current_buffer_var}, {join_key}_gather, "
      f"{probe_view_var}, {offset_var}.data(), {pair_buf_left}.data(), "
      f"{pair_buf_right}.data());\n\n"
    )

    current_buffer_var = pair_buf_left
    current_size_var = total_var

  # Phase 3: Gather columns
  if gather_ops:
    code += i + "// Phase 3: Gather output columns\n"
    for gather_op in gather_ops:
      gather_rel = gather_op.rel_name
      gather_col = gather_op.column
      output_var = gather_op.output_var
      input_buf = gather_op.input_buffer

      code += (
        i + f"thrust::device_vector<int64_t> {output_var}_gather({current_size_var});\n"
      )
      code += (
        i + f"gather_column<{gather_rel}, {gather_col}>({input_buf}"
        f"_right, {output_var}_gather.data());\n"
      )
    code += "\n"

  # Phase 4: Insert results
  if insert_ops:
    code += i + "// Phase 4: Insert results\n"
    for insert_op in insert_ops:
      dest_rel = insert_op.rel_name
      dest_vars = insert_op.vars
      code += i + f"// Insert into {dest_rel}\n"
      code += i + f"auto {dest_rel}_dest = db.get_new<{dest_rel}>();\n"
      gather_args = ", ".join(f"{v}_gather" for v in dest_vars)
      code += (
        i + f"insert_gathered_tuples({dest_rel}_dest, {gather_args});\n"
      )

  return code


# -----------------------------------------------------------------------------
# Materialized runner (the main entry)
# -----------------------------------------------------------------------------

def gen_materialized_runner(
  node: m.ExecutePipeline, db_type_name: str,
) -> str:
  '''Emit a host-side Thrust executor for a materialized-join pipeline.

  Produces `struct JitRunner_<rule_name>` with an `execute(DB&, iter)`
  static method that walks Scan → ProbeJoin+ → GatherColumn* → InsertInto
  using Thrust primitives.

  Matches Nim's `genMaterializedRunner` byte-for-byte (modulo the usual
  clang-format whitespace).
  '''
  assert isinstance(node, m.ExecutePipeline)
  rule_name = node.rule_name
  pipeline = list(node.pipeline)

  scan_op: Optional[m.Scan] = None
  probe_join_ops: list[m.ProbeJoin] = []
  gather_ops: list[m.GatherColumn] = []
  insert_ops: list[m.InsertInto] = []

  for op in pipeline:
    if isinstance(op, m.Scan):
      scan_op = op
    elif isinstance(op, m.ProbeJoin):
      probe_join_ops.append(op)
    elif isinstance(op, m.GatherColumn):
      gather_ops.append(op)
    elif isinstance(op, m.InsertInto):
      insert_ops.append(op)

  if scan_op is None:
    return "// ERROR: Materialized join requires moScan as first op\n"

  first_schema = scan_op.rel_name
  first_version = _version_cpp(scan_op.version)

  result = ""
  result += "// =============================================================\n"
  result += f"// JIT-Generated Materialized Runner: {rule_name}\n"
  result += "// Host-side Thrust executor (no CUDA kernels)\n"
  result += "// =============================================================\n\n"

  result += f"struct JitRunner_{rule_name} {{\n"
  result += f"  using DB = {db_type_name};\n"
  result += f"  using FirstSchema = {first_schema};\n"
  if node.dest_specs:
    first_dest = node.dest_specs[0]
    if isinstance(first_dest, m.InsertInto):
      result += f"  using DestSchema = {first_dest.rel_name};\n"
  result += "  using ValueType = typename FirstSchema::intern_value_type;\n"
  result += (
    "  using RelType = std::decay_t<decltype(get_relation_by_schema<FirstSchema, "
    f"{first_version}>(std::declval<DB&>()))>;\n"
  )
  result += "  using IndexType = typename RelType::IndexTypeInst;\n"
  result += "  using ViewType = typename IndexType::NodeView;\n\n"

  # execute()
  result += "  static void execute(DB& db, uint32_t iteration = 0) {\n"
  result += f'    nvtxRangePushA("{rule_name}");\n\n'
  result += "    using namespace SRDatalog::GPU::JIT;\n\n"

  # Phase 1
  scan_idx_comma = ", ".join(str(c) for c in scan_op.index)
  result += "    // Phase 1: Get initial data from first relation\n"
  result += (
    f"    auto& rel_0 = get_relation_by_schema<{first_schema}, {first_version}>(db);\n"
  )
  result += (
    f"    auto& idx_0 = rel_0.ensure_index(SRDatalog::IndexSpec{{{{{scan_idx_comma}}}}}, false);\n"
  )
  result += "    auto view_0 = idx_0.view();\n"
  result += "    size_t n_0 = view_0.num_rows_;\n"
  result += "    if (n_0 == 0) { nvtxRangePop(); return; }\n\n"

  result += "    // Current row ID buffer (starts as simple [0..N) sequence)\n"
  result += "    thrust::device_vector<uint32_t> rowids_0(n_0);\n"
  result += "    thrust::sequence(rowids_0.begin(), rowids_0.end());\n"
  result += "    size_t current_n = n_0;\n\n"

  current_left_rowids = "rowids_0"
  current_right_rowids = ""
  prev_rel_idx = 0

  # Phase 2: probe joins
  for buffer_idx, probe_op in enumerate(probe_join_ops):
    probe_rel = probe_op.probe_rel
    probe_idx_comma = ", ".join(str(c) for c in probe_op.probe_index)
    join_key = probe_op.join_key
    probe_version = _version_cpp(probe_op.probe_version)

    rel_idx = buffer_idx + 1
    pair_left = f"left_{rel_idx}"
    pair_right = f"right_{rel_idx}"

    result += (
      f"    // Phase 2.{buffer_idx + 1}: Probe {probe_rel} on {join_key}\n"
    )
    result += (
      f"    auto& rel_{rel_idx} = get_relation_by_schema<{probe_rel}, "
      f"{probe_version}>(db);\n"
    )
    result += (
      f"    auto& idx_{rel_idx} = rel_{rel_idx}.ensure_index("
      f"SRDatalog::IndexSpec{{{{{probe_idx_comma}}}}}, false);\n"
    )
    result += f"    auto view_{rel_idx} = idx_{rel_idx}.view();\n\n"

    # Gather join key from current rowids.
    result += f"    // Gather join key '{join_key}' from current buffer\n"
    result += (
      f"    thrust::device_vector<ValueType> keys_{rel_idx}(current_n);\n"
    )
    result += "    // Access column 0 via col_data_ (col_data_ + 0 * stride_)\n"
    result += (
      f"    thrust::gather(thrust::device, {current_left_rowids}.begin(), "
      f"{current_left_rowids}.end(),\n"
    )
    result += (
      f"                   view_{prev_rel_idx}.col_data_, keys_{rel_idx}.begin());\n\n"
    )

    # Count matches.
    result += "    // Count matches per input row\n"
    result += (
      f"    thrust::device_vector<uint32_t> counts_{rel_idx}(current_n);\n"
    )
    result += (
      f"    probe_count_matches({current_left_rowids}, keys_{rel_idx}, "
      f"view_{rel_idx}, counts_{rel_idx});\n\n"
    )

    # Offsets.
    result += "    // Compute output offsets\n"
    result += (
      f"    thrust::device_vector<uint32_t> offsets_{rel_idx}(current_n);\n"
    )
    result += (
      f"    uint32_t total_{rel_idx} = compute_output_offsets(counts_{rel_idx}, "
      f"offsets_{rel_idx});\n"
    )
    result += (
      f"    if (total_{rel_idx} == 0) {{ nvtxRangePop(); return; }}\n\n"
    )

    # Materialize pairs.
    result += "    // Materialize (left, right) row ID pairs\n"
    result += (
      f"    thrust::device_vector<uint32_t> {pair_left}(total_{rel_idx});\n"
    )
    result += (
      f"    thrust::device_vector<uint32_t> {pair_right}(total_{rel_idx});\n"
    )
    result += (
      f"    probe_materialize_pairs({current_left_rowids}, keys_{rel_idx}, "
      f"view_{rel_idx},\n"
    )
    result += (
      f"                            offsets_{rel_idx}, {pair_left}, {pair_right});\n\n"
    )

    result += "    // Update current state\n"
    result += f"    current_n = total_{rel_idx};\n\n"

    current_left_rowids = pair_left
    current_right_rowids = pair_right
    prev_rel_idx = rel_idx

  result += "\n"

  # Phase 3: gather columns (stubbed — matches Nim which has TODO comments)
  if gather_ops:
    result += "    // Phase 3: Gather output columns\n"
    for gather_op in gather_ops:
      gather_var = gather_op.output_var
      gather_rel = gather_op.rel_name
      gather_col = gather_op.column
      result += (
        f"    thrust::device_vector<ValueType> {gather_var}_data(current_n);\n"
      )
      result += (
        f"    // Note: Gathering {gather_var} from {gather_rel} column {gather_col}\n"
      )
      result += "    // Using simplified gather from last right-side row IDs\n"
      result += "    // TODO: Track which buffer corresponds to which relation\n\n"
    result += "\n"

  # Phase 4: insert results.
  if insert_ops:
    first_insert = insert_ops[0]
    dest_rel = first_insert.rel_name
    dest_vars = list(first_insert.vars)
    arity = len(dest_vars)

    result += f"    // Phase 4: Insert into {dest_rel}\n"
    result += (
      f"    auto& dest = get_relation_by_schema<{dest_rel}, NEW_VER>(db);\n"
    )
    result += "    size_t old_size = dest.size();\n"
    result += "    size_t new_size = old_size + current_n;\n"
    result += "    dest.resize_interned_columns(new_size);\n\n"

    result += "    // Gather output columns into device vectors\n"
    for col_idx, _dest_var in enumerate(dest_vars):
      result += (
        f"    thrust::device_vector<ValueType> out_col_{col_idx}(current_n);\n"
      )
    result += "\n"

    # First output column: chain back through the left buffers to rowids_0,
    # then gather from view_0 col 0.
    if len(dest_vars) >= 1:
      result += (
        "    // Output column 0 (e.g. invocation): gather from view_0 col 0 via chained left row IDs\n"
      )
      result += "    // Chained gather: left_N -> left_N-1 -> ... -> rowids_0\n"

      num_probes = len(probe_join_ops)
      if num_probes == 0:
        result += (
          "    thrust::gather(thrust::device, rowids_0.begin(), rowids_0.begin() + current_n,\n"
        )
        result += "                   view_0.col_data_, out_col_0.begin());\n"
      else:
        result += "    // Final left buffer chains back to view_0\n"
        result += "    thrust::device_vector<uint32_t> chained_rowids(current_n);\n"
        result += (
          f"    thrust::copy(left_{num_probes}.begin(), left_{num_probes}.end(), "
          "chained_rowids.begin());\n"
        )
        for back_idx in range(num_probes - 1, 0, -1):
          result += (
            "    thrust::gather(thrust::device, chained_rowids.begin(), chained_rowids.end(),\n"
          )
          result += (
            f"                   left_{back_idx}.begin(), chained_rowids.begin());\n"
          )
        result += (
          "    thrust::gather(thrust::device, chained_rowids.begin(), chained_rowids.end(),\n"
        )
        result += "                   rowids_0.begin(), chained_rowids.begin());\n"
        result += (
          "    thrust::gather(thrust::device, chained_rowids.begin(), chained_rowids.end(),\n"
        )
        result += "                   view_0.col_data_, out_col_0.begin());\n"
      result += "\n"

    # Second output column: gather from last view's col 2 via right rowids.
    if len(dest_vars) >= 2:
      last_rel_idx = len(probe_join_ops)
      result += (
        "    // Output column 1 (e.g. toMeth): from last view via right row IDs\n"
      )
      result += (
        f"    thrust::gather(thrust::device, right_{last_rel_idx}.begin(), right_{last_rel_idx}.end(),\n"
      )
      result += (
        f"                   view_{last_rel_idx}.col_data_ + 2 * view_{last_rel_idx}.stride_,\n"
      )
      result += "                   out_col_1.begin());\n\n"

    # Copy gathered columns to destination.
    result += "    // Copy to destination relation columns\n"
    for col_idx, _dest_var in enumerate(dest_vars):
      result += (
        f"    thrust::copy(out_col_{col_idx}.begin(), out_col_{col_idx}.end(),\n"
      )
      result += (
        f"                 dest.template interned_column<{col_idx}>() + old_size);\n"
      )
    result += "\n"

  result += "\n    nvtxRangePop();\n"
  result += "  }\n"
  result += "};\n\n"
  return result
