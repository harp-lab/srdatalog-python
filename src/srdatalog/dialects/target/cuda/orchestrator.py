'''JIT orchestrator — step_N() method bodies.

Port of src/srdatalog/codegen/target_jit/jit_orchestrator.nim. Named
`orchestrator_jit` to avoid colliding with the existing legacy
`codegen/orchestrator.py` (mhk's python-api-notemplate design, which
this module replaces for byte-match purposes).

Top-level entry:
  gen_step_body(plan, db_type_name, is_recursive, step_num, count_only_rels)
    = the full `template <typename DB>\\n  static void step_N(...)`
      function, including fixpoint loop for recursive steps and
      straight-line sequence for non-recursive.

Internal structure (mirrors Nim procs):
  gen_index_spec_type(rel, ver, cols)  — `IndexSpecT<R, integer_sequence<int,...>, V>`
  version_string(ver)                   — MIR version -> C++ const
  extract_source_info(src_spec)         — (rel, ver, idx) from any source node
  build_dest_stream_map(exec_ops)       — dest rel -> [stream indices]
  all_dests_independent(dest_streams)   — every dest written by ≤1 stream
  is_count_only_pipeline(instr, cors)   — dest in count-only set
  gen_instruction_code(instr, indent, iter_var, dest_stream_map, count_only_rels)
                                        — BIG dispatcher
  collect_canonical_specs(instrs)       — canonical index per rel
  gen_fixpoint_body(plan, db, indent, cors)          — recursive path
  gen_non_recursive_block(plan, db, indent, cors)    — non-recursive path

Scope: baseline. WS / BG / dedup / fan-out variants of ExecutePipeline
emission use the same dispatcher paths Nim does (the feature flags are
read off the ExecutePipeline node itself, not the context), so the
generated C++ is correct for those rules too — the difference is that
our kernel-side emit doesn't yet produce the matching WS / BG kernel
bodies. The orchestrator half is self-contained.
'''

from __future__ import annotations

import srdatalog.mir.types as m
from srdatalog.hir.types import Version

# -----------------------------------------------------------------------------
# C++ type expression generators
# -----------------------------------------------------------------------------


def gen_index_spec_type(
  rel_name: str,
  version: str,
  cols: list[int],
) -> str:
  '''`SRDatalog::mir::IndexSpecT<Rel, std::integer_sequence<int, ...>, VER>`'''
  col_str = ", ".join(str(c) for c in cols)
  col_seq = f"std::integer_sequence<int, {col_str}>"
  return f"SRDatalog::mir::IndexSpecT<{rel_name}, {col_seq}, {version}>"


def version_string(ver: str) -> str:
  '''Convert MIR version name to C++ constant. Nim accepts both
  forms (`"FULL_VER"` and `"full"`); we accept both to stay compatible
  with whatever shape upstream callers produce.'''
  if ver in ("DELTA_VER", "delta", "DELTA"):
    return "DELTA_VER"
  if ver in ("FULL_VER", "full", "FULL"):
    return "FULL_VER"
  if ver in ("NEW_VER", "newt", "NEW"):
    return "NEW_VER"
  return ver


def _version_str_of(node: m.MirNode) -> str:
  '''Version code (`"FULL_VER"` etc.) for a node carrying a Version.'''
  ver = getattr(node, "version", None)
  if isinstance(ver, Version):
    return ver.code
  return version_string(str(ver)) if ver else ""


def extract_source_info(src_spec: m.MirNode) -> tuple[str, str, list[int]]:
  '''Pull `(rel_name, version-as-C++-code, index)` from any source node.'''
  if isinstance(src_spec, (m.ColumnSource, m.Scan, m.Negation, m.Aggregate)):
    return src_spec.rel_name, src_spec.version.code, list(src_spec.index)
  return "", "", []


# -----------------------------------------------------------------------------
# Dest / count-only helpers
# -----------------------------------------------------------------------------


def build_dest_stream_map(exec_ops: list[m.ExecutePipeline]) -> dict[str, list[int]]:
  '''dest rel → list of stream indices (one per pipeline) that write to it.'''
  out: dict[str, list[int]] = {}
  for idx, op in enumerate(exec_ops):
    for dest in op.dest_specs:
      if not isinstance(dest, m.InsertInto):
        continue
      out.setdefault(dest.rel_name, []).append(idx)
  return out


def all_dests_independent(dest_streams: dict[str, list[int]]) -> bool:
  return all(len(streams) <= 1 for streams in dest_streams.values())


def is_count_only_pipeline(
  instr: m.MirNode,
  count_only_rels: set[str],
) -> bool:
  if not isinstance(instr, m.ExecutePipeline):
    return False
  if not count_only_rels:
    return False
  return any(
    isinstance(dest, m.InsertInto) and dest.rel_name in count_only_rels for dest in instr.dest_specs
  )


# -----------------------------------------------------------------------------
# Canonical spec collection
# -----------------------------------------------------------------------------


def collect_canonical_specs(
  instrs: list[m.MirNode],
) -> list[tuple[str, list[int]]]:
  '''First (rel, cols) per relation appearing in ComputeDelta /
  ComputeDeltaIndex / MergeIndex ops. Used for `using R_canonical_spec_t`
  declarations and the fixpoint convergence check.
  '''
  out: list[tuple[str, list[int]]] = []
  seen: list[str] = []
  for instr in instrs:
    if isinstance(instr, m.ComputeDelta) and instr.rel_name not in seen:
      out.append((instr.rel_name, list(instr.index)))
      seen.append(instr.rel_name)
    elif isinstance(instr, m.ComputeDeltaIndex) and instr.rel_name not in seen:
      out.append((instr.rel_name, list(instr.canonical_index)))
      seen.append(instr.rel_name)
    elif isinstance(instr, m.MergeIndex) and instr.rel_name not in seen:
      out.append((instr.rel_name, list(instr.index)))
      seen.append(instr.rel_name)
  return out


# -----------------------------------------------------------------------------
# gen_instruction_code — big per-instruction dispatcher
# -----------------------------------------------------------------------------


def _gen_execute_pipeline(
  instr: m.ExecutePipeline,
  indent: str,
  iter_var: str,
  count_only_rels: set[str],
) -> str:
  runner_name = f"JitRunner_{instr.rule_name}"
  if instr.count or is_count_only_pipeline(instr, count_only_rels):
    out = indent + "// Count-only query mode\n"
    out += indent + "{\n"
    out += indent + f"  auto p = {runner_name}::setup(db, {iter_var}, 0);\n"
    out += indent + f"  {runner_name}::launch_count(p, 0);\n"
    out += indent + "  GPU_DEVICE_SYNCHRONIZE();\n"
    out += (
      indent + "  uint64_t total = thrust::reduce(thrust::device, "
      "p.thread_counts_ptr, p.thread_counts_ptr + p.num_threads, "
      "uint64_t(0), thrust::plus<uint64_t>());\n"
    )
    out += (
      indent + f"  std::cout << \" >>>>>>>>>>>>>>>>> {instr.rule_name}"
      " count: \" << total << std::endl;\n"
    )
    out += indent + "}\n"
    return out

  has_fused = (
    not instr.dedup_hash
    and not instr.work_stealing
    and not instr.block_group
    and len(instr.dest_specs) == 1
  )
  if has_fused:
    return (
      indent + f"if (_tail_mode) {runner_name}::execute_fused(db, {iter_var}); "
      f"else {runner_name}::execute(db, {iter_var});\n"
    )
  return indent + f"{runner_name}::execute(db, {iter_var});\n"


def _gen_parallel_group(
  instr: m.ParallelGroup,
  indent: str,
  iter_var: str,
  dest_stream_map_out: dict[str, list[int]],
  count_only_rels: set[str],
) -> str:
  '''Port of Nim's moParallelGroup branch — ~250 LOC of phased stream-
  parallel dispatch. Modifies dest_stream_map_out in place so downstream
  maintenance ops can emit the correct stream waits.
  '''
  exec_ops: list[m.ExecutePipeline] = []
  other_ops: list[m.MirNode] = []
  for op in instr.ops:
    if isinstance(op, m.ExecutePipeline):
      exec_ops.append(op)
    else:
      other_ops.append(op)

  has_dedup = any(op.dedup_hash for op in exec_ops)

  if len(exec_ops) <= 1:
    out = indent + "// === ParallelGroup (single rule, sequential) ===\n"
    for op in instr.ops:
      out += gen_instruction_code(
        op,
        indent,
        iter_var,
        dest_stream_map_out,
        count_only_rels,
      )
    return out

  if has_dedup:
    out = indent + "// === ParallelGroup (sequential, dedup_hash present) ===\n"
    for op in exec_ops:
      runner = f"JitRunner_{op.rule_name}"
      out += indent + f"{runner}::execute(db, {iter_var});\n"
    for op in other_ops:
      out += gen_instruction_code(
        op,
        indent,
        iter_var,
        dest_stream_map_out,
        count_only_rels,
      )
    return out

  # Stream-parallel phased path.
  new_map = build_dest_stream_map(exec_ops)
  dest_stream_map_out.clear()
  dest_stream_map_out.update(new_map)

  N = len(exec_ops)
  i = indent
  i2 = indent + "  "

  all_fused_eligible = all(
    not op.dedup_hash and not op.work_stealing and not op.block_group and len(op.dest_specs) == 1
    for op in exec_ops
  )

  out = i + "if (_tail_mode" + ("" if all_fused_eligible else " && false") + ") {\n"
  if all_fused_eligible:
    for op in exec_ops:
      runner = f"JitRunner_{op.rule_name}"
      out += i2 + f"{runner}::execute_fused(db, {iter_var});\n"
  for op in other_ops:
    out += gen_instruction_code(
      op,
      i2,
      iter_var,
      dest_stream_map_out,
      count_only_rels,
    )
  out += i + "} else {\n"

  out += i + f"// === ParallelGroup (stream-parallel, {N} rules, shared dests) ===\n"
  out += i + "{\n"
  out += i2 + 'nvtxRangePushA("join_pipeline");\n'

  # Phase 1: setup.
  out += i2 + "// Phase 1: Setup all rules\n"
  for idx, op in enumerate(exec_ops):
    runner = f"JitRunner_{op.rule_name}"
    out += i2 + f"auto p_{idx} = {runner}::setup(db, {iter_var}, _stream_pool.get({idx}));\n"
  out += "\n"

  # Classify single-dest vs multi-head.
  single_dest_indices: list[int] = []
  multi_head_indices: list[int] = []
  for idx, op in enumerate(exec_ops):
    if len(op.dest_specs) == 1 and not op.work_stealing and not op.dedup_hash:
      single_dest_indices.append(idx)
    else:
      multi_head_indices.append(idx)

  dest_single_rules: dict[str, list[int]] = {}
  for idx in single_dest_indices:
    dest = exec_ops[idx].dest_specs[0]
    if isinstance(dest, m.InsertInto):
      dest_single_rules.setdefault(dest.rel_name, []).append(idx)

  # Phase 2a: shared count buffers.
  out += i2 + "// Phase 2a: Allocate shared count buffers per dest\n"
  for dest_rel, rule_indices in sorted(dest_single_rules.items()):
    if len(rule_indices) < 2:
      continue
    buf = "shared_buf_" + dest_rel
    size_expr = " + ".join(f"p_{r}.num_threads" for r in rule_indices) + " + 1"
    out += i2 + f"uint32_t {buf}_size = {size_expr};\n"
    out += i2 + f"SRDatalog::GPU::DeviceArray<uint32_t> {buf}({buf}_size);\n"
    for k, rule_idx in enumerate(rule_indices):
      off = f"off_{dest_rel}_{k}"
      if k == 0:
        out += i2 + f"uint32_t {off} = 0;\n"
      else:
        prev_off = f"off_{dest_rel}_{k - 1}"
        out += i2 + f"uint32_t {off} = {prev_off} + p_{rule_indices[k - 1]}.num_threads;\n"
      out += i2 + f"p_{rule_idx}.thread_counts_ptr = {buf}.data() + {off};\n"
  out += "\n"

  # Phase 2b: launch count kernels.
  out += i2 + "// Phase 2b: Launch count kernels (parallel streams)\n"
  for idx, op in enumerate(exec_ops):
    runner = f"JitRunner_{op.rule_name}"
    out += i2 + f"{runner}::launch_count(p_{idx}, _stream_pool.get({idx}));\n"
  out += i2 + "_stream_pool.sync_all();\n\n"

  # Phase 3a: scan shared buffers + multi-head scans.
  out += i2 + "// Phase 3a: Scan shared buffers (one per dest) + multi-head scans\n"
  for dest_rel, rule_indices in sorted(dest_single_rules.items()):
    if len(rule_indices) < 2:
      idx = rule_indices[0]
      runner = f"JitRunner_{exec_ops[idx].rule_name}"
      out += i2 + f"{runner}::scan_only(p_{idx}, _stream_pool.get({idx}));\n"
    else:
      buf = "shared_buf_" + dest_rel
      out += (
        i2 + f"thrust::exclusive_scan(rmm::exec_policy("
        f"_stream_pool.get({rule_indices[0]})), {buf}.data(), "
        f"{buf}.data() + {buf}_size, {buf}.data(), 0, thrust::plus<uint32_t>());\n"
      )
  for idx in multi_head_indices:
    runner = f"JitRunner_{exec_ops[idx].rule_name}"
    out += i2 + f"{runner}::scan_only(p_{idx}, _stream_pool.get({idx}));\n"
  out += "\n"

  # Phase 3b: sync + readback.
  out += i2 + "// Phase 3b: Single sync + readback all totals\n"
  out += i2 + "GPU_DEVICE_SYNCHRONIZE();\n"

  for dest_rel, rule_indices in sorted(dest_single_rules.items()):
    if len(rule_indices) < 2:
      idx = rule_indices[0]
      runner = f"JitRunner_{exec_ops[idx].rule_name}"
      out += i2 + f"uint32_t total_{idx} = {runner}::read_total(p_{idx});\n"
    else:
      buf = "shared_buf_" + dest_rel
      K = len(rule_indices)
      out += i2 + f"// Read boundary values for {dest_rel} ({K} rules)\n"
      out += i2 + f"uint32_t bnd_{dest_rel}[{K + 1}];\n"
      for k, rule_idx in enumerate(rule_indices):
        off = f"off_{dest_rel}_{k}"
        out += (
          i2 + f"GPU_MEMCPY_ASYNC(&bnd_{dest_rel}[{k}], {buf}.data() + {off}, "
          "sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);\n"
        )
      out += (
        i2 + f"GPU_MEMCPY_ASYNC(&bnd_{dest_rel}[{K}], "
        f"{buf}.data() + {buf}_size - 1, "
        "sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);\n"
      )
      for k, rule_idx in enumerate(rule_indices):
        out += i2 + f"uint32_t total_{rule_idx} = bnd_{dest_rel}[{k + 1}] - bnd_{dest_rel}[{k}];\n"

  for idx in multi_head_indices:
    runner = f"JitRunner_{exec_ops[idx].rule_name}"
    out += i2 + f"uint32_t total_{idx} = {runner}::read_total(p_{idx});\n"
  out += "\n"

  # Count-only prints.
  for idx, op in enumerate(exec_ops):
    if op.count or is_count_only_pipeline(op, count_only_rels):
      out += (
        i2 + f"std::cout << \" >>>>>>>>>>>>>>>>> {op.rule_name}"
        f" count: \" << total_{idx} << std::endl;\n"
      )

  # Phase 3c: resize + offsets.
  out += i2 + "// Phase 3c: Resize once per unique dest + assign per-rule offsets\n"
  for dest_rel, rule_indices in sorted(dest_stream_map_out.items()):
    all_count_only = all(
      exec_ops[r].count or is_count_only_pipeline(exec_ops[r], count_only_rels)
      for r in rule_indices
    )
    if all_count_only:
      out += i2 + f"// skip resize for count_only dest {dest_rel}\n"
      for rule_idx in rule_indices:
        op = exec_ops[rule_idx]
        for d_idx, dest in enumerate(op.dest_specs):
          if isinstance(dest, m.InsertInto) and dest.rel_name == dest_rel:
            out += i2 + f"p_{rule_idx}.old_size_{d_idx} = 0;\n"
      continue

    sum_expr = " + ".join(f"total_{r}" for r in rule_indices)
    out += i2 + f"{{  // Resize {dest_rel}\n"
    out += i2 + f"  uint32_t sum_{dest_rel} = {sum_expr};\n"
    out += i2 + f"  auto& dest_{dest_rel} = get_relation_by_schema<{dest_rel}, NEW_VER>(db);\n"
    out += (
      i2 + f"  uint32_t base_offset_{dest_rel} = static_cast<uint32_t>(dest_{dest_rel}.size());\n"
    )
    out += (
      i2 + f"  if (sum_{dest_rel} > 0) dest_{dest_rel}"
      f".resize_interned_columns(base_offset_{dest_rel} + sum_{dest_rel});\n"
    )

    single_in_dest: list[int] = []
    multi_in_dest: list[int] = []
    for rule_idx in rule_indices:
      op = exec_ops[rule_idx]
      if len(op.dest_specs) == 1 and not op.work_stealing and not op.dedup_hash:
        single_in_dest.append(rule_idx)
      else:
        multi_in_dest.append(rule_idx)

    has_concat = dest_rel in dest_single_rules and len(dest_single_rules[dest_rel]) >= 2

    if has_concat:
      for rule_idx in single_in_dest:
        exec_ops[rule_idx].concurrent_write = True
      for rule_idx in single_in_dest:
        op = exec_ops[rule_idx]
        for local_idx, dest in enumerate(op.dest_specs):
          if isinstance(dest, m.InsertInto) and dest.rel_name == dest_rel:
            out += i2 + f"  p_{rule_idx}.old_size_{local_idx} = base_offset_{dest_rel};\n"
            break
      if multi_in_dest:
        single_sum = " + ".join(f"total_{r}" for r in single_in_dest)
        out += (
          i2 + f"  uint32_t running_offset_{dest_rel} = base_offset_{dest_rel} + {single_sum};\n"
        )
        for rule_idx in multi_in_dest:
          op = exec_ops[rule_idx]
          for local_idx, dest in enumerate(op.dest_specs):
            if isinstance(dest, m.InsertInto) and dest.rel_name == dest_rel:
              out += i2 + f"  p_{rule_idx}.old_size_{local_idx} = running_offset_{dest_rel};\n"
              out += i2 + f"  running_offset_{dest_rel} += total_{rule_idx};\n"
              break
    else:
      out += i2 + f"  uint32_t running_offset_{dest_rel} = base_offset_{dest_rel};\n"
      for rule_idx in rule_indices:
        op = exec_ops[rule_idx]
        for local_idx, dest in enumerate(op.dest_specs):
          if isinstance(dest, m.InsertInto) and dest.rel_name == dest_rel:
            out += i2 + f"  p_{rule_idx}.old_size_{local_idx} = running_offset_{dest_rel};\n"
            out += i2 + f"  running_offset_{dest_rel} += total_{rule_idx};\n"
            break
    out += i2 + "}\n"
  out += "\n"

  # Phase 4: materialize.
  for idx, op in enumerate(exec_ops):
    if op.count or is_count_only_pipeline(op, count_only_rels):
      out += i2 + f"// skip materialize for count_only rule {op.rule_name}\n"
    else:
      runner = f"JitRunner_{op.rule_name}"
      out += (
        i2 + f"if (total_{idx} > 0) {runner}::launch_materialize("
        f"db, p_{idx}, total_{idx}, _stream_pool.get({idx}));\n"
      )
  for idx in range(len(exec_ops)):
    out += i2 + f"_stream_pool.record_event({idx});\n"
  out += i2 + "nvtxRangePop();  // join_pipeline\n"

  for op in other_ops:
    out += gen_instruction_code(
      op,
      i2,
      iter_var,
      dest_stream_map_out,
      count_only_rels,
    )

  out += i + "}\n"
  out += i + "}\n"  # close tail-mode else
  return out


def gen_instruction_code(
  instr: m.MirNode,
  indent: str,
  iter_var: str,
  dest_stream_map: dict[str, list[int]],
  count_only_rels: set[str] | None = None,
) -> str:
  '''Emit imperative C++ for one fixpoint-level MIR instruction.'''
  if count_only_rels is None:
    count_only_rels = set()

  if isinstance(instr, m.ExecutePipeline):
    return _gen_execute_pipeline(instr, indent, iter_var, count_only_rels)

  if isinstance(instr, m.ParallelGroup):
    return _gen_parallel_group(
      instr,
      indent,
      iter_var,
      dest_stream_map,
      count_only_rels,
    )

  if isinstance(instr, m.ComputeDelta):
    if instr.rel_name in count_only_rels:
      return indent + f"// skip compute_delta for count_only rel {instr.rel_name}\n"
    spec_type = gen_index_spec_type(instr.rel_name, "NEW_VER", list(instr.index))
    out = indent + 'nvtxRangePushA("merge");\n'
    out += (
      indent + "SRDatalog::GPU::mir_helpers::compute_delta_fn<"
      f"{instr.rel_name}, {spec_type}>(db);\n"
    )
    out += indent + 'nvtxRangePop();  // merge\n'
    return out

  if isinstance(instr, m.ComputeDeltaIndex):
    if instr.rel_name in count_only_rels:
      return indent + f"// skip compute_delta_index for count_only rel {instr.rel_name}\n"
    spec_new = gen_index_spec_type(instr.rel_name, "NEW_VER", list(instr.canonical_index))
    spec_full = gen_index_spec_type(instr.rel_name, "FULL_VER", list(instr.canonical_index))
    spec_delta = gen_index_spec_type(instr.rel_name, "DELTA_VER", list(instr.canonical_index))
    out = indent + 'nvtxRangePushA("merge");\n'
    out += (
      indent + "SRDatalog::GPU::mir_helpers::compute_delta_index_fn<"
      f"{spec_new}, {spec_full}, {spec_delta}>(db);\n"
    )
    out += indent + 'nvtxRangePop();  // merge\n'
    return out

  if isinstance(instr, m.MergeIndex):
    if instr.rel_name in count_only_rels:
      return indent + f"// skip merge_index for count_only rel {instr.rel_name}\n"
    spec_type = gen_index_spec_type(instr.rel_name, "FULL_VER", list(instr.index))
    out = indent + 'nvtxRangePushA("merge");\n'
    out += indent + f"SRDatalog::GPU::mir_helpers::merge_index_fn<{spec_type}>(db);\n"
    out += indent + 'nvtxRangePop();  // merge\n'
    return out

  if isinstance(instr, m.CheckSize):
    if instr.rel_name in count_only_rels:
      return indent + f"// skip check_size for count_only rel {instr.rel_name}\n"
    ver = version_string(instr.version.code)
    return (
      indent + "SRDatalog::GPU::mir_helpers::check_size_fn<"
      f"{instr.rel_name}, {ver}, {instr.rel_name}_canonical_spec_t>(db);\n"
    )

  if isinstance(instr, m.RebuildIndex):
    if instr.rel_name in count_only_rels:
      return indent + f"// skip rebuild_index for count_only rel {instr.rel_name}\n"
    ver = version_string(instr.version.code)
    spec_type = gen_index_spec_type(instr.rel_name, ver, list(instr.index))
    return indent + f"SRDatalog::GPU::mir_helpers::rebuild_index_fn<{spec_type}>(db);\n"

  if isinstance(instr, m.RebuildIndexFromIndex):
    ver = version_string(instr.version.code)
    source_spec = gen_index_spec_type(instr.rel_name, ver, list(instr.source_index))
    target_spec = gen_index_spec_type(instr.rel_name, ver, list(instr.target_index))
    return (
      indent + "SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn<"
      f"{source_spec}, {target_spec}>(db);\n"
    )

  if isinstance(instr, m.CreateFlatView):
    ver = version_string(instr.version.code)
    spec_type = gen_index_spec_type(instr.rel_name, ver, list(instr.index))
    return indent + f"SRDatalog::GPU::mir_helpers::create_flat_view_fn<{spec_type}>(db);\n"

  if isinstance(instr, m.ClearRelation):
    if instr.rel_name in count_only_rels:
      return indent + f"// skip clear_relation for count_only rel {instr.rel_name}\n"
    ver = version_string(instr.version.code)
    return (
      indent + f"SRDatalog::GPU::mir_helpers::clear_relation_fn<{instr.rel_name}, {ver}>(db);\n"
    )

  if isinstance(instr, m.MergeRelation):
    out = indent + 'nvtxRangePushA("merge");\n'
    out += indent + f"SRDatalog::GPU::mir_helpers::merge_relation_fn<{instr.rel_name}>(db);\n"
    out += indent + 'nvtxRangePop();  // merge\n'
    return out

  if isinstance(instr, m.InjectCppHook):
    return indent + "{\n" + instr.code + "\n" + indent + "}\n"

  if isinstance(instr, m.PostStratumReconstructInternCols):
    if instr.rel_name in count_only_rels:
      return indent + f"// skip reconstruct for count_only rel {instr.rel_name}\n"
    spec_type = gen_index_spec_type(
      instr.rel_name,
      "FULL_VER",
      list(instr.canonical_index),
    )
    return indent + f"SRDatalog::GPU::mir_helpers::reconstruct_fn<{spec_type}>(db);\n"

  return indent + f"// TODO: unhandled MIR op kind {type(instr).__name__}\n"


# -----------------------------------------------------------------------------
# gen_fixpoint_body — recursive step path
# -----------------------------------------------------------------------------


def gen_fixpoint_body(
  plan: m.FixpointPlan,
  db_type_name: str,
  indent: str,
  count_only_rels: set[str] | None = None,
) -> str:
  '''Recursive path: fixpoint loop with convergence check + maintenance
  ops + post-fixpoint reconstruct for each canonical rel.'''
  if count_only_rels is None:
    count_only_rels = set()

  instrs = list(plan.instructions)
  i = indent
  i2 = indent + "  "
  i3 = indent + "    "

  canonical_specs = collect_canonical_specs(instrs)

  out = ""
  for rel_name, cols in canonical_specs:
    spec_type = gen_index_spec_type(rel_name, "FULL_VER", cols)
    out += i + f"using {rel_name}_canonical_spec_t = {spec_type};\n"
  out += "\n"

  out += i + "// Build all required indexes\n"
  for instr in instrs:
    exec_pipelines: list[m.ExecutePipeline] = []
    if isinstance(instr, m.ExecutePipeline):
      exec_pipelines.append(instr)
    elif isinstance(instr, m.ParallelGroup):
      for op in instr.ops:
        if isinstance(op, m.ExecutePipeline):
          exec_pipelines.append(op)

    for ep in exec_pipelines:
      for src_spec in ep.source_specs:
        if isinstance(src_spec, m.ColumnSource):
          ver = version_string(src_spec.version.code)
          spec_type = gen_index_spec_type(src_spec.rel_name, ver, list(src_spec.index))
          out += i + f"mir_helpers::create_index_fn<{spec_type}>(db, 0);\n"
          if ver == "DELTA_VER":
            full_spec_type = gen_index_spec_type(
              src_spec.rel_name,
              "FULL_VER",
              list(src_spec.index),
            )
            out += i + f"mir_helpers::create_index_fn<{full_spec_type}>(db, 0);\n"

    if isinstance(instr, m.RebuildIndex):
      ver = version_string(instr.version.code)
      spec_type = gen_index_spec_type(instr.rel_name, ver, list(instr.index))
      out += i + f"mir_helpers::create_index_fn<{spec_type}>(db, 0);\n"

  out += "\n\n"

  # Fixpoint loop.
  out += i + "bool _tail_mode = false;\n"
  out += i + "for (std::size_t iter = 0; iter < max_iterations; ++iter) {\n"

  if canonical_specs:
    out += i2 + "if (iter > 0) {\n"
    out += i3 + "std::size_t total_new_facts = 0;\n"
    for rel_name, cols in canonical_specs:
      out += i3 + "{\n"
      out += i3 + f"  auto& delta_rel = get_relation_by_schema<{rel_name}, DELTA_VER>(db);\n"
      col_str = ", ".join(str(c) for c in cols)
      out += i3 + "  SRDatalog::IndexSpec spec{{" + col_str + "}};\n"
      out += i3 + "  try {\n"
      out += i3 + "    auto& delta_idx = delta_rel.get_index(spec);\n"
      out += i3 + "    total_new_facts += delta_idx.root().degree();\n"
      out += i3 + "  } catch (...) {}\n"
      out += i3 + "}\n"

    out += i3 + 'if (std::getenv("SRDATALOG_PRINT_DELTA_DETAIL")) {\n'
    out += i3 + '  std::cerr << "[iter " << iter << "]";\n'
    for rel_name, cols in canonical_specs:
      out += i3 + "  {\n"
      out += i3 + f"    auto& _dr = get_relation_by_schema<{rel_name}, DELTA_VER>(db);\n"
      col_str = ", ".join(str(c) for c in cols)
      out += i3 + "    SRDatalog::IndexSpec _sp{{" + col_str + "}};\n"
      out += (
        i3 + f'    try {{ auto& _di = _dr.get_index(_sp); std::cerr '
        f'<< " {rel_name}=" << _di.root().degree(); }} catch (...) {{ '
        f'std::cerr << " {rel_name}=?"; }}\n'
      )
      out += i3 + "  }\n"
    out += i3 + "  std::cerr << std::endl;\n"
    out += i3 + '} else if (std::getenv("SRDATALOG_PRINT_DELTA")) {\n'
    out += i3 + '  std::cerr << "[iter " << iter << "] delta=" << total_new_facts << std::endl;\n'
    out += i3 + "}\n"
    out += i3 + "if (total_new_facts == 0) break;\n"
    out += (
      i3 + '_tail_mode = (std::getenv("SRDATALOG_NO_TAIL") == nullptr) '
      '&& (total_new_facts < 1000);\n'
    )
    out += i2 + "}\n\n"

  max_stream_count = 0
  for instr in instrs:
    if isinstance(instr, m.ParallelGroup):
      num_exec = sum(1 for op in instr.ops if isinstance(op, m.ExecutePipeline))
      max_stream_count = max(max_stream_count, num_exec)

  if max_stream_count > 1:
    out += i2 + "static SRDatalog::GPU::StreamPool _stream_pool;\n"
    out += i2 + f"_stream_pool.ensure({max_stream_count});\n\n"

  dest_stream_map: dict[str, list[int]] = {}
  synced_after_parallel = False
  maint_kinds = (
    m.RebuildIndex,
    m.ComputeDelta,
    m.ComputeDeltaIndex,
    m.ClearRelation,
    m.CheckSize,
    m.MergeIndex,
    m.RebuildIndexFromIndex,
    m.CreateFlatView,
  )
  for instr in instrs:
    if not synced_after_parallel and dest_stream_map and isinstance(instr, maint_kinds):
      out += i2 + "GPU_DEVICE_SYNCHRONIZE(); // sync all materialize streams\n"
      synced_after_parallel = True
    out += gen_instruction_code(
      instr,
      i2,
      "static_cast<uint32_t>(iter)",
      dest_stream_map,
      count_only_rels,
    )

  out += i + "}\n"
  out += i + "GPU_DEVICE_SYNCHRONIZE();\n\n"

  out += i + "// Reconstruct intern columns from canonical index\n"
  for rel_name, cols in canonical_specs:
    spec_type = gen_index_spec_type(rel_name, "FULL_VER", cols)
    out += i + f"mir_helpers::reconstruct_fn<{spec_type}>(db);\n"

  return out


# -----------------------------------------------------------------------------
# gen_non_recursive_block — non-recursive step path
# -----------------------------------------------------------------------------


def gen_non_recursive_block(
  plan: m.MirNode,
  db_type_name: str,
  indent: str,
  count_only_rels: set[str] | None = None,
) -> str:
  '''Non-recursive path. Handles Block / FixpointPlan / ExecutePipeline /
  PostStratumReconstructInternCols.'''
  if count_only_rels is None:
    count_only_rels = set()

  i = indent
  dest_stream_map: dict[str, list[int]] = {}
  out = ""

  if isinstance(plan, m.Block):
    out += i + "bool _tail_mode = false;\n"
    for instr in plan.instructions:
      out += gen_instruction_code(
        instr,
        i,
        "0",
        dest_stream_map,
        count_only_rels,
      )
    out += i + "GPU_DEVICE_SYNCHRONIZE();\n"
    return out

  if isinstance(plan, m.ExecutePipeline):
    out += gen_instruction_code(
      plan,
      i,
      "0",
      dest_stream_map,
      count_only_rels,
    )
    out += i + "GPU_DEVICE_SYNCHRONIZE();\n"
    return out

  if isinstance(plan, m.FixpointPlan):
    instrs = list(plan.instructions)

    out += i + "// Build output indexes\n"
    for instr in instrs:
      if isinstance(instr, m.RebuildIndex) and instr.rel_name not in count_only_rels:
        ver = version_string(instr.version.code)
        spec_type = gen_index_spec_type(instr.rel_name, ver, list(instr.index))
        out += i + f"mir_helpers::create_index_fn<{spec_type}>(db, 0);\n"
    out += "\n"

    canonical_specs = collect_canonical_specs(instrs)
    for rel_name, cols in canonical_specs:
      spec_type = gen_index_spec_type(rel_name, "FULL_VER", cols)
      out += i + f"using {rel_name}_canonical_spec_t = {spec_type};\n"
    out += i + "bool _tail_mode = false;\n"
    out += "\n"

    max_stream_count = 0
    for instr in instrs:
      if isinstance(instr, m.ParallelGroup):
        num_exec = sum(1 for op in instr.ops if isinstance(op, m.ExecutePipeline))
        max_stream_count = max(max_stream_count, num_exec)
    if max_stream_count > 1:
      out += i + "static SRDatalog::GPU::StreamPool _stream_pool;\n"
      out += i + f"_stream_pool.ensure({max_stream_count});\n\n"

    for instr in instrs:
      if isinstance(instr, (m.ExecutePipeline, m.ParallelGroup)):
        exec_pipelines: list[m.ExecutePipeline] = []
        if isinstance(instr, m.ExecutePipeline):
          exec_pipelines.append(instr)
        else:
          for op in instr.ops:
            if isinstance(op, m.ExecutePipeline):
              exec_pipelines.append(op)
        for ep in exec_pipelines:
          for src_spec in ep.source_specs:
            rel_name, raw_ver, idx = extract_source_info(src_spec)
            if rel_name:
              ver = version_string(raw_ver)
              spec_type = gen_index_spec_type(rel_name, ver, idx)
              out += i + f"mir_helpers::create_index_fn<{spec_type}>(db, 0);\n"
              if ver == "DELTA_VER":
                full_spec_type = gen_index_spec_type(rel_name, "FULL_VER", idx)
                out += i + f"mir_helpers::create_index_fn<{full_spec_type}>(db, 0);\n"
      out += gen_instruction_code(
        instr,
        i,
        "0",
        dest_stream_map,
        count_only_rels,
      )
    return out

  if isinstance(plan, m.PostStratumReconstructInternCols):
    if plan.rel_name in count_only_rels:
      out += i + f"// skip reconstruct for count_only rel {plan.rel_name}\n"
    else:
      spec_type = gen_index_spec_type(
        plan.rel_name,
        "FULL_VER",
        list(plan.canonical_index),
      )
      out += i + f"mir_helpers::reconstruct_fn<{spec_type}>(db);\n"
      out += i + "GPU_DEVICE_SYNCHRONIZE();\n"
    return out

  out += i + f"// ERROR: unexpected plan kind for non-recursive step: {type(plan).__name__}\n"
  return out


# -----------------------------------------------------------------------------
# gen_step_body (top-level)
# -----------------------------------------------------------------------------


def gen_step_body(
  plan: m.MirNode,
  db_type_name: str,
  is_recursive: bool,
  step_num: int,
  count_only_rels: set[str] | None = None,
) -> str:
  '''Matches Nim's `genStepBody`. Returns the full `template <typename DB>
  static void step_N(...) { ... }` function. What the fixture-dumping
  tool captures for each (node, is_recursive) pair in the MIR program.
  '''
  if count_only_rels is None:
    count_only_rels = set()

  indent = "    "
  body = ""
  body += "  template <typename DB>\n"
  body += f"  static void step_{step_num}(DB& db, std::size_t max_iterations) {{\n"

  if is_recursive:
    assert isinstance(plan, m.FixpointPlan), "recursive step must be a FixpointPlan"
    body += gen_fixpoint_body(plan, db_type_name, indent, count_only_rels)
  else:
    body += gen_non_recursive_block(plan, db_type_name, indent, count_only_rels)

  body += "  }\n"
  return body
