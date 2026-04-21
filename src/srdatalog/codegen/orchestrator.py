'''Orchestrator-level C++ code generation.

Ported from mhk's `python-api-notemplate` branch, adapted to consume our
`mir_types.Program` directly instead of mhk's `Block(body, recursive)`
wrapper. A program step (our `(node, is_recursive)` tuple) becomes one
`step_N(DB& db, size_t max_iterations)` function; `SRDatalogProgram`
drives the full orchestrator file assembly.

Structural note: our HIR→MIR pipeline emits `PostStratumReconstructInternCols`
as its own step after the fixpoint step, whereas mhk's codegen tacked the
reconstruct call onto the tail of the recursive runner. Both produce the
same net C++ behavior; here we keep the steps separate so the orchestrator
mirrors the MIR one-to-one.
'''

from __future__ import annotations

from dataclasses import dataclass

import srdatalog.mir.types as m
from srdatalog.codegen import cpp_emit
from srdatalog.codegen.schema import Pragma, SchemaDefinition
from srdatalog.hir.types import Version

# -----------------------------------------------------------------------------
# Preamble / header blocks (verbatim from nt orchestrator)
# -----------------------------------------------------------------------------

PRELUDE = '''
#include "srdatalog.h"
#include "runtime/io.h"
using namespace SRDatalog;
using namespace SRDatalog::AST::Literals;
using string = std::string;
using Arena = boost::container::pmr::monotonic_buffer_resource;

#include "gpu/runtime/query.h"
#include "gpu/gpu_api.h"
#include "gpu/init.h"
#include "gpu/runtime/gpu_fixpoint_executor.h"
#include <chrono>
'''

PRELUDE_2 = '''
#include "gpu/runtime/jit/materialized_join.h"
#include "gpu/runtime/jit/ws_infrastructure.h"
#include "gpu/runtime/gpu_mir_helpers.h"
#include "gpu/runtime/stream_pool.h"
using namespace SRDatalog::GPU;
'''

FFI_HEADER = '''
  /* Defines API for cffi functions */

typedef void *DBHandle;

DBHandle db_new();
void db_free(DBHandle h);
void load(DBHandle h, const char *root_dir);
void run(DBHandle h, size_t max_iters);
'''


# -----------------------------------------------------------------------------
# Plan inspection helpers
# -----------------------------------------------------------------------------


def get_canonical_specs(instrs: list[m.MirNode]) -> list[tuple[str, list[int]]]:
  '''Walk `instrs` and return `(rel_name, canonical_index)` pairs — one per
  distinct relation, keyed off the canonical-tagging ops
  (`ComputeDelta`, `ComputeDeltaIndex`, `MergeIndex`). First occurrence
  per relation wins, matching mhk's `get_canonical_specs` order.
  '''
  seen: list[str] = []
  result: list[tuple[str, list[int]]] = []
  for op in instrs:
    if isinstance(op, m.ComputeDelta):
      rel, idx = op.rel_name, list(op.index)
    elif isinstance(op, m.ComputeDeltaIndex):
      rel, idx = op.rel_name, list(op.canonical_index)
    elif isinstance(op, m.MergeIndex):
      rel, idx = op.rel_name, list(op.index)
    else:
      continue
    if rel not in seen:
      seen.append(rel)
      result.append((rel, idx))
  return result


def _split_pipelines(plan: m.FixpointPlan) -> tuple[list[m.ExecutePipeline], list[m.MirNode]]:
  '''Partition FixpointPlan.instructions into (pipelines, non-pipeline ops).

  Handles both flat layouts (single ExecutePipeline followed by maintenance)
  and nested (a ParallelGroup wrapping the pipelines, then maintenance).
  '''
  pipelines: list[m.ExecutePipeline] = []
  non_pipelines: list[m.MirNode] = []
  for instr in plan.instructions:
    if isinstance(instr, m.ExecutePipeline):
      pipelines.append(instr)
    elif isinstance(instr, m.ParallelGroup):
      for op in instr.ops:
        if isinstance(op, m.ExecutePipeline):
          pipelines.append(op)
        else:
          non_pipelines.append(op)
    else:
      non_pipelines.append(instr)
  return pipelines, non_pipelines


def generate_dest_stream_map(pipelines: list[m.ExecutePipeline]) -> dict[str, list[int]]:
  '''Dest-relation → list of pipeline indices that write to it. Used to emit
  `_stream_pool.wait_event(idx)` before scalar ops that read those dests.
  '''
  result: dict[str, list[int]] = {}
  for idx, ep in enumerate(pipelines):
    for dest in ep.dest_specs:
      rel = _dest_rel_name(dest)
      if rel:
        result.setdefault(rel, []).append(idx)
  return result


def _dest_rel_name(dest: m.MirNode) -> str:
  '''Dest spec may be an InsertInto or a ColumnSource-like spec; pull out rel_name.'''
  if isinstance(dest, (m.InsertInto, m.ColumnSource, m.Scan, m.Negation, m.Aggregate)):
    return dest.rel_name
  return ""


# -----------------------------------------------------------------------------
# Pipeline source/index helpers
# -----------------------------------------------------------------------------


def _collect_source_specs(
  pipelines: list[m.ExecutePipeline],
) -> list[tuple[str, Version, tuple[int, ...]]]:
  '''Union of (rel, ver, index) specs read by any pipeline — driving the
  `mir_helpers::create_index_fn<...>` calls at the top of the runner.
  DELTA specs also pull in their FULL counterpart (DELTA dispatches to
  FULL on the first iteration).
  '''
  seen: set[tuple[str, Version, tuple[int, ...]]] = set()
  ordered: list[tuple[str, Version, tuple[int, ...]]] = []

  def add(rel: str, ver: Version, idx: list[int]) -> None:
    key = (rel, ver, tuple(idx))
    if key not in seen:
      seen.add(key)
      ordered.append(key)

  for ep in pipelines:
    for src in ep.source_specs:
      if isinstance(src, (m.ColumnSource, m.Scan, m.Negation, m.Aggregate)):
        add(src.rel_name, src.version, src.index)
        if src.version == Version.DELTA:
          add(src.rel_name, Version.FULL, src.index)
    for dest in ep.dest_specs:
      if isinstance(dest, m.InsertInto):
        add(dest.rel_name, dest.version, dest.index)
  return ordered


# -----------------------------------------------------------------------------
# Non-parallel pipeline runner (single stream, used by non-recursive strata
# and by recursive strata with exactly one pipeline)
# -----------------------------------------------------------------------------


def _emit_prebuild(canon_specs: list[tuple[str, list[int]]]) -> str:
  '''`using Rel_canonical_spec_t = IndexSpec<Rel, cols, FULL_VER>;` per relation.'''
  out: list[str] = []
  for rel, idx in canon_specs:
    out.append(f"using {rel}_canonical_spec_t = {cpp_emit.index_spec_t(rel, Version.FULL, idx)};")
  return "\n".join(out) + ("\n" if out else "")


def _emit_create_indexes(pipelines: list[m.ExecutePipeline], non_pipelines: list[m.MirNode]) -> str:
  '''Emit one `mir_helpers::create_index_fn<...>(db, 0);` per unique
  (rel, ver, index) touched by the stratum. Specs from pipelines come
  first (per mhk's order); explicit `RebuildIndex` nodes contribute
  their own indexes so create-index happens even when there's no pipeline.
  '''
  seen: set[tuple[str, str, tuple[int, ...]]] = set()
  lines: list[str] = []

  def add(rel: str, ver: Version, idx: list[int]) -> None:
    key = (rel, ver.code, tuple(idx))
    if key in seen:
      return
    seen.add(key)
    lines.append(cpp_emit.index_create_call(rel, ver, idx))

  # Pipeline source/dest indexes
  for rel, ver, idx in _collect_source_specs(pipelines):
    add(rel, ver, list(idx))

  # Standalone RebuildIndex ops in the non-pipeline tail
  for op in non_pipelines:
    if isinstance(op, m.RebuildIndex):
      add(op.rel_name, op.version, op.index)

  return "\n".join(lines) + ("\n" if lines else "")


def _emit_setup_streams(pipelines: list[m.ExecutePipeline]) -> str:
  '''Emit the StreamPool declaration when there's more than one pipeline
  (single-pipeline strata don't need one).'''
  if len(pipelines) <= 1:
    return ""
  return (
    f"static SRDatalog::GPU::StreamPool _stream_pool;\n_stream_pool.ensure({len(pipelines)});\n\n"
  )


def _emit_fixpoint_break_check(canon_specs: list[tuple[str, list[int]]]) -> str:
  '''`if (iter > 0) { ... break if no new delta ... }` preamble.'''
  if not canon_specs:
    return "for (std::size_t iter = 0; iter < max_iterations; ++iter) {\n"
  out = ["for (std::size_t iter = 0; iter < max_iterations; ++iter) {"]
  out.append("if (iter > 0) {")
  out.append("std::size_t total_new_facts = 0;")
  for rel, idx in canon_specs:
    out.append("{")
    out.append(f"auto& delta_rel = get_relation_by_schema<{rel}, DELTA_VER>(db);")
    cols = ", ".join(str(c) for c in idx)
    out.append("SRDatalog::IndexSpec spec{{" + cols + "}};")
    out.append("try {")
    out.append("auto& delta_idx = delta_rel.get_index(spec);")
    out.append("total_new_facts += delta_idx.root().degree();")
    out.append("} catch (...) {}")
    out.append("}")
  out.append("if (total_new_facts == 0) break;")
  out.append("}\n")
  return "\n".join(out) + "\n"


def _emit_nonpipeline_op(op: m.MirNode, dest_stream_map: dict[str, list[int]] | None) -> str:
  '''Emit a single non-pipeline op (scalar maintenance, InjectCppHook, etc.).'''
  return cpp_emit.emit_orchestrator(op, iter_var=9999, dest_stream_map=dest_stream_map)


# -----------------------------------------------------------------------------
# ParallelGroup 4-phase emitter (recursive, multi-pipeline case)
# -----------------------------------------------------------------------------


def _emit_parallel_pipelines(
  pipelines: list[m.ExecutePipeline],
  non_pipelines: list[m.MirNode],
  iter_var: int,
) -> str:
  '''Emit the setup→count→scan-resize→materialize sequence for a recursive
  stratum with multiple pipelines running concurrently.

  Ported ~line-for-line from `ParallelGroup._generate_parallel_pipelines` in
  mhk's `nt_commands.py`; only the field accesses changed to our MIR shape.
  '''
  if not pipelines:
    return "\n//No pipelines found\n"

  if len(pipelines) == 1:
    ep = pipelines[0]
    return (
      "// === ParallelGroup (single rule, sequential) ===\n"
      f"JitRunner_{ep.rule_name}::execute(db, 0);\n"
    )

  phase_1 = "{ // Phase 1: Setup all rules\n"
  for idx, ep in enumerate(pipelines):
    phase_1 += (
      f'nvtxRangePushA("{ep.rule_name}::setup");\n'
      f'auto p_{idx} = JitRunner_{ep.rule_name}::setup'
      f'(db, static_cast<uint32_t>(iter), _stream_pool.get({idx}));\n'
      f"nvtxRangePop();\n"
    )

  # Partition pipelines by single-dest-no-ws vs multi-head/ws
  single_idxs: list[int] = []
  multi_idxs: list[int] = []
  for idx, ep in enumerate(pipelines):
    if len(ep.dest_specs) == 1 and not ep.work_stealing:
      single_idxs.append(idx)
    else:
      multi_idxs.append(idx)

  # Build per-dest groups of single-dest rules (for shared count buffers)
  dest_single: dict[str, list[int]] = {}
  for idx in single_idxs:
    rel = _dest_rel_name(pipelines[idx].dest_specs[0])
    if rel:
      dest_single.setdefault(rel, []).append(idx)

  phase_2a = "// Phase 2a: Allocate shared count buffers per dest\n"
  for rel, idxs in dest_single.items():
    if len(idxs) < 2:
      continue
    buf = "shared_buf_" + rel
    size_expr = " + ".join(f"p_{i}.num_threads" for i in idxs) + " + 1"
    phase_2a += f"uint32_t {buf}_size = {size_expr};\n"
    phase_2a += f"SRDatalog::GPU::DeviceArray<uint32_t> {buf}({buf}_size);\n"
    for k, rule_idx in enumerate(idxs):
      off_name = f"off_{rel}_{k}"
      if k == 0:
        phase_2a += f"uint32_t {off_name} = 0;\n"
      else:
        prev_off = f"off_{rel}_{k - 1}"
        phase_2a += f"uint32_t {off_name} = {prev_off} + p_{idxs[k - 1]}.num_threads;\n"
      phase_2a += f"p_{rule_idx}.thread_counts_ptr = {buf}.data() + {off_name};\n"

  phase_2b = "// Phase 2b: Launch count kernels (parallel streams)\n"
  for idx, ep in enumerate(pipelines):
    runner = f"JitRunner_{ep.rule_name}"
    phase_2b += (
      f'nvtxRangePushA("{ep.rule_name}::count");\n'
      f"{runner}::launch_count(p_{idx}, _stream_pool.get({idx}));\n"
      "nvtxRangePop();\n"
    )
  phase_2b += "_stream_pool.sync_all();\n\n"

  phase_3a = "// Phase 3a: Scan shared buffers (one per dest) + multi-head scans\n"
  phase_3a += 'nvtxRangePushA("scan_all");\n'
  for rel, idxs in dest_single.items():
    if len(idxs) < 2:
      idx = idxs[0]
      runner = f"JitRunner_{pipelines[idx].rule_name}"
      phase_3a += f"{runner}::scan_only(p_{idx}, _stream_pool.get({idx}));\n"
    else:
      buf = f"shared_buf_{rel}"
      phase_3a += (
        f"thrust::exclusive_scan(rmm::exec_policy(_stream_pool.get({idxs[0]})), "
        f"{buf}.data(), {buf}.data() + {buf}_size, "
        f"{buf}.data(), 0, thrust::plus<uint32_t>());\n"
      )
  for idx in multi_idxs:
    runner = f"JitRunner_{pipelines[idx].rule_name}"
    phase_3a += f"{runner}::scan_only(p_{idx}, _stream_pool.get({idx}));\n"
  phase_3a += "nvtxRangePop();\n\n"

  phase_3b = "// Phase 3b: Single sync + readback all totals\n"
  phase_3b += "GPU_DEVICE_SYNCHRONIZE();\n"
  for rel, idxs in dest_single.items():
    if len(idxs) < 2:
      idx = idxs[0]
      runner = f"JitRunner_{pipelines[idx].rule_name}"
      phase_3b += f"uint32_t total_{idx} = {runner}::read_total(p_{idx});\n"
    else:
      buf = f"shared_buf_{rel}"
      phase_3b += f"// Read boundary values for {rel} ({len(idxs)} rules)\n"
      phase_3b += f"uint32_t bnd_{rel}[{len(idxs) + 1}];\n"
      for k, rule_idx in enumerate(idxs):
        off_name = f"off_{rel}_{k}"
        phase_3b += (
          f"GPU_MEMCPY_ASYNC(&bnd_{rel}[{k}], {buf}.data() + {off_name}, "
          "sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);\n"
        )
      phase_3b += (
        f"GPU_MEMCPY_ASYNC(&bnd_{rel}[{len(idxs)}], "
        f"{buf}.data() + {buf}_size - 1, "
        "sizeof(uint32_t), GPU_DEVICE_TO_HOST, 0);\n"
      )
      for k, rule_idx in enumerate(idxs):
        phase_3b += f"uint32_t total_{rule_idx} = bnd_{rel}[{k + 1}] - bnd_{rel}[{k}];\n"
  for idx in multi_idxs:
    runner = f"JitRunner_{pipelines[idx].rule_name}"
    phase_3b += f"uint32_t total_{idx} = {runner}::read_total(p_{idx});\n"

  phase_3c = "// Phase 3c: Resize once per unique dest + assign per-rule offsets\n"
  dest_stream_map = generate_dest_stream_map(pipelines)
  for rel, idxs in dest_stream_map.items():
    sum_expr = " + ".join(f"total_{i}" for i in idxs)
    phase_3c += (
      f"{{  // Resize {rel}\n"
      f"uint32_t sum_{rel} = {sum_expr};\n"
      f"auto& dest_{rel} = get_relation_by_schema<{rel}, NEW_VER>(db);\n"
      f"uint32_t base_offset_{rel} = static_cast<uint32_t>(dest_{rel}.size());\n"
      f"if (sum_{rel} > 0) dest_{rel}.resize_interned_columns(base_offset_{rel} + sum_{rel});\n"
    )

    single_in_dest = [
      i for i in idxs if len(pipelines[i].dest_specs) == 1 and not pipelines[i].work_stealing
    ]
    multi_in_dest = [i for i in idxs if i not in single_in_dest]
    has_concat = rel in dest_single and len(dest_single[rel]) >= 2

    if has_concat:
      for rule_idx in single_in_dest:
        ep = pipelines[rule_idx]
        for local_idx, dest_spec in enumerate(ep.dest_specs):
          if _dest_rel_name(dest_spec) == rel:
            phase_3c += f"  p_{rule_idx}.old_size_{local_idx} = base_offset_{rel};\n"
            break
      if multi_in_dest:
        single_sum = " + ".join(f"total_{i}" for i in single_in_dest)
        phase_3c += f"uint32_t running_offset_{rel} = base_offset_{rel} + {single_sum};\n"
        for rule_idx in multi_in_dest:
          ep = pipelines[rule_idx]
          for local_idx, dest_spec in enumerate(ep.dest_specs):
            if _dest_rel_name(dest_spec) == rel:
              phase_3c += f"p_{rule_idx}.old_size_{local_idx} = running_offset_{rel};\n"
              phase_3c += f"running_offset_{rel} += total_{rule_idx};\n"
              break
    else:
      phase_3c += f"uint32_t running_offset_{rel} = base_offset_{rel};\n"
      for rule_idx in idxs:
        ep = pipelines[rule_idx]
        for local_idx, dest_spec in enumerate(ep.dest_specs):
          if _dest_rel_name(dest_spec) == rel:
            phase_3c += f"p_{rule_idx}.old_size_{local_idx} = running_offset_{rel};\n"
            phase_3c += f"running_offset_{rel} += total_{rule_idx};\n"
            break
    phase_3c += "}\n"

  phase_4 = "// Phase 4: Launch all materialize kernels on separate streams\n"
  phase_4 += 'nvtxRangePushA("materialize");\n'
  for idx, ep in enumerate(pipelines):
    runner = f"JitRunner_{ep.rule_name}"
    phase_4 += (
      f"if (total_{idx} > 0) {runner}::launch_materialize"
      f"(db, p_{idx}, total_{idx}, _stream_pool.get({idx}));\n"
    )
  for idx, _ep in enumerate(pipelines):
    phase_4 += f"_stream_pool.record_event({idx});\n"
  phase_4 += "nvtxRangePop();\n}\n"

  # Non-pipeline ops (InjectCppHook, RebuildIndex with stream-wait, maintenance)
  for op in non_pipelines:
    phase_4 += _emit_nonpipeline_op(op, dest_stream_map)

  return phase_1 + phase_2a + phase_2b + phase_3a + phase_3b + phase_3c + phase_4


# -----------------------------------------------------------------------------
# Non-parallel (single-pipeline) body emission
# -----------------------------------------------------------------------------


def _emit_single_pipeline_body(
  pipelines: list[m.ExecutePipeline],
  non_pipelines: list[m.MirNode],
) -> str:
  '''Non-recursive runner body: one pipeline kick + the maintenance ops.'''
  parts: list[str] = []
  for ep in pipelines:
    parts.append(f"JitRunner_{ep.rule_name}::execute(db, 0);")
  for op in non_pipelines:
    parts.append(_emit_nonpipeline_op(op, None))
  return "\n".join(parts) + "\n"


# -----------------------------------------------------------------------------
# Top-level step generation
# -----------------------------------------------------------------------------


def generate_fixpoint_runner(plan: m.FixpointPlan, is_recursive: bool) -> str:
  pipelines, non_pipelines = _split_pipelines(plan)
  canon_specs = get_canonical_specs(plan.instructions)
  prebuild = _emit_prebuild(canon_specs)
  indexes = _emit_create_indexes(pipelines, non_pipelines)
  streams = _emit_setup_streams(pipelines)

  if not is_recursive:
    body = _emit_single_pipeline_body(pipelines, non_pipelines)
    return prebuild + indexes + streams + body

  fixpoint_open = _emit_fixpoint_break_check(canon_specs)
  if len(pipelines) > 1:
    pipeline_body = _emit_parallel_pipelines(pipelines, non_pipelines, iter_var=0)
  else:
    pipeline_body = _emit_single_pipeline_body(pipelines, non_pipelines)
  close = "}\nGPU_DEVICE_SYNCHRONIZE();\n"
  return prebuild + indexes + fixpoint_open + streams + pipeline_body + close


def generate_step_body(node: m.MirNode, is_recursive: bool) -> str:
  '''Emit the body of one step_N function. Dispatches on our top-level
  MIR step shapes: FixpointPlan, PostStratumReconstructInternCols, and
  any of the scalar maintenance ops that occasionally appear as their
  own step (e.g. a pre-reconstruct RebuildIndex).
  '''
  if isinstance(node, m.FixpointPlan):
    return generate_fixpoint_runner(node, is_recursive)
  if isinstance(node, m.PostStratumReconstructInternCols):
    return cpp_emit.emit_post_stratum_reconstruct(node) + "\nGPU_DEVICE_SYNCHRONIZE();\n"
  if isinstance(node, m.RebuildIndex):
    return cpp_emit.emit_rebuild_index(node)
  if isinstance(node, m.InjectCppHook):
    return cpp_emit.emit_inject_cpp_hook(node)
  raise NotImplementedError(
    f"orchestrator.generate_step_body: unsupported step kind {type(node).__name__}"
  )


def generate_step(number: int, node: m.MirNode, is_recursive: bool) -> str:
  body = generate_step_body(node, is_recursive)
  return (
    "\n  template <typename DB>\n"
    f"  static void step_{number}(DB& db, std::size_t max_iterations) {{\n"
    f"    {body}\n"
    "  }\n"
  )


# -----------------------------------------------------------------------------
# Program driver
# -----------------------------------------------------------------------------


@dataclass
class SRDatalogProgram:
  '''Pairs a compiled `mir_types.Program` with a schema and drives full
  orchestrator emission.'''

  name: str
  database: SchemaDefinition
  program: m.Program

  def generate_orchestrator(self, include_ffi: bool = True) -> str:
    parts: list[str] = [
      PRELUDE,
      str(self.database),
      "\nusing namespace SRDatalog::mir::dsl;",
      PRELUDE_2,
    ]
    for i, (node, is_rec) in enumerate(self.program.steps):
      parts.append(generate_step(i, node, is_rec))
    if include_ffi:
      parts.append(FFI_HEADER)
    parts.append(self._generate_load_instruction())
    return "".join(parts)

  def _generate_load_instruction(self) -> str:
    pragmas = ""
    for f in self.database.facts:
      if Pragma.INPUT in f.pragmas:
        pragmas += (
          f'\nSRDatalog::load_from_file<{f.name}>(db, root_dir + "/{f.pragmas[Pragma.INPUT]}");'
        )
    return f"\nstatic void load_data(DB& db, std::string root_dir) {{{pragmas}\n}}\n"
