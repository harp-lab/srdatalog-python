'''Free-function C++ emitters for MIR nodes.

Ported from the method-based style in mhk's `nt_commands.py` on
`python-api-notemplate`: each `__str__` / `generate_orchestrator`
has become a stand-alone function that takes one of our `mir_types`
nodes directly, so the backend shares IR with the HIR→MIR pipeline.

This commit covers the scalar / maintenance MIR ops — the ones with
no children. Pipeline ops (ExecutePipeline, ColumnJoin, ...) are
handled in `batchfile.py` / `orchestrator.py` in later commits.

Entry points:
  emit(node)                    — render the C++ for one node (no stream-sync)
  emit_orchestrator(node, ...)  — same, but adds `_stream_pool.wait_event(...)`
                                   calls when the node writes a dest that was
                                   produced by a prior parallel pipeline
  fact_index_str(rel, idx)      — building block: `"Rel, 0, 1"`
  index_spec_t(rel, ver, idx)   — `SRDatalog::mir::IndexSpecT<Rel, ..., FULL_VER>`
  index_create_call(...)        — `mir_helpers::create_index_fn<...>(db, 0);`

Semantic gap (carried forward): mhk's RebuildIndexFromIndex supported
cross-relation + cross-version (source_fact/source_version vs dest_fact/
dest_version). Ours is same-rel same-version (as produced by Nim
lowering), so `emit_rebuild_index_from_index` uses a single rel_name and
a single version; the cross-relation variant is out of scope until our
MIR model extends to match.
'''
from __future__ import annotations
from typing import Optional

import srdatalog.mir_types as m
from srdatalog.hir_types import Version


# -----------------------------------------------------------------------------
# Building blocks (pure string helpers)
# -----------------------------------------------------------------------------

def fact_index_str(rel_name: str, index: list[int]) -> str:
  '''`"PointsTo, 0, 1"` — the "fact index" form used inside C++ templates.'''
  return f"{rel_name}, {', '.join(str(c) for c in index)}"


def index_spec_t(rel_name: str, version: Version, index: list[int]) -> str:
  '''`SRDatalog::mir::IndexSpecT<PointsTo, std::integer_sequence<int, 0, 1>, FULL_VER>`.

  The primary template used by every C++ helper we emit.
  '''
  idx_str = ", ".join(str(c) for c in index)
  return (
    f"SRDatalog::mir::IndexSpecT<{rel_name}, "
    f"std::integer_sequence<int, {idx_str}>, {version.code}>"
  )


def index_create_call(rel_name: str, version: Version, index: list[int]) -> str:
  '''`mir_helpers::create_index_fn<...>(db, 0);` — the call emitted for each
  source / dest index spec at the top of a stratum's runner.
  '''
  return f"mir_helpers::create_index_fn<{index_spec_t(rel_name, version, index)}>(db, 0);"


# -----------------------------------------------------------------------------
# Scalar / maintenance ops
# -----------------------------------------------------------------------------

def emit_rebuild_index(node: m.RebuildIndex) -> str:
  return (
    f"SRDatalog::GPU::mir_helpers::rebuild_index_fn<"
    f"{index_spec_t(node.rel_name, node.version, node.index)}>(db);"
  )


def emit_merge_index(node: m.MergeIndex) -> str:
  '''MergeIndex in our MIR is implicitly into FULL (no version field). Nim's
  lowering only ever generates FULL merges — matches the non-template C++
  helper `merge_index_fn<IndexSpec<..., FULL_VER>>`.
  '''
  return (
    f"SRDatalog::GPU::mir_helpers::merge_index_fn<"
    f"{index_spec_t(node.rel_name, Version.FULL, node.index)}>(db);"
  )


def emit_check_size(node: m.CheckSize) -> str:
  return (
    f"SRDatalog::GPU::mir_helpers::check_size_fn<"
    f"{node.rel_name}, {node.version.code}, {node.rel_name}_canonical_spec_t>(db);"
  )


def emit_compute_delta(node: m.ComputeDelta) -> str:
  '''ComputeDelta is implicitly on NEW (Nim MIR emits it only at that
  version). The C++ helper is templated `<Rel, IndexSpec<Rel, cols, NEW_VER>>`.
  '''
  return (
    f"SRDatalog::GPU::mir_helpers::compute_delta_fn<"
    f"{node.rel_name}, {index_spec_t(node.rel_name, Version.NEW, node.index)}>(db);"
  )


def emit_compute_delta_index(node: m.ComputeDeltaIndex) -> str:
  '''compute_delta_index_fn takes three IndexSpecs (NEW, FULL, DELTA) over
  the same canonical index.
  '''
  canon = node.canonical_index
  new_spec = index_spec_t(node.rel_name, Version.NEW, canon)
  full_spec = index_spec_t(node.rel_name, Version.FULL, canon)
  delta_spec = index_spec_t(node.rel_name, Version.DELTA, canon)
  return (
    "SRDatalog::GPU::mir_helpers::compute_delta_index_fn<\n"
    f"      {new_spec}, \n"
    f"      {full_spec}, \n"
    f"      {delta_spec}\n"
    "    >(db);"
  )


def emit_clear_relation(node: m.ClearRelation) -> str:
  return (
    f"SRDatalog::GPU::mir_helpers::clear_relation_fn<"
    f"{node.rel_name}, {node.version.code}>(db);"
  )


def emit_merge_relation(node: m.MergeRelation) -> str:
  return (
    f"SRDatalog::GPU::mir_helpers::merge_relation_fn<{node.rel_name}>(db);"
  )


def emit_rebuild_index_from_index(node: m.RebuildIndexFromIndex) -> str:
  '''Our MIR models rebuild-index-from-index as same-relation same-version
  (matches the Nim lowering). `source_index` / `target_index` give the two
  column orders; both share `rel_name` and `version`.
  '''
  source_spec = index_spec_t(node.rel_name, node.version, node.source_index)
  target_spec = index_spec_t(node.rel_name, node.version, node.target_index)
  return (
    "SRDatalog::GPU::mir_helpers::rebuild_index_from_index_fn< "
    f"{source_spec}, {target_spec} >(db);"
  )


def emit_inject_cpp_hook(node: m.InjectCppHook) -> str:
  return "{" + node.code + "}"


def emit_post_stratum_reconstruct(node: m.PostStratumReconstructInternCols) -> str:
  '''`Reconstruct` in mhk's naming — rebuild the canonical index in FULL
  after a stratum's fixpoint loop.
  '''
  spec = index_spec_t(node.rel_name, Version.FULL, node.canonical_index)
  return f"mir_helpers::reconstruct_fn<{spec}>(db);"


# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------

_SCALAR_EMITTERS = {
  m.RebuildIndex: emit_rebuild_index,
  m.MergeIndex: emit_merge_index,
  m.CheckSize: emit_check_size,
  m.ComputeDelta: emit_compute_delta,
  m.ComputeDeltaIndex: emit_compute_delta_index,
  m.ClearRelation: emit_clear_relation,
  m.MergeRelation: emit_merge_relation,
  m.RebuildIndexFromIndex: emit_rebuild_index_from_index,
  m.InjectCppHook: emit_inject_cpp_hook,
  m.PostStratumReconstructInternCols: emit_post_stratum_reconstruct,
}


def emit(node: m.MirNode) -> str:
  '''Emit C++ for one scalar / maintenance MIR node. Raises for node
  types the scalar emitters don't cover (pipeline ops, fixpoint plans,
  structural containers — handled by other codegen modules).
  '''
  fn = _SCALAR_EMITTERS.get(type(node))
  if fn is None:
    raise NotImplementedError(
      f"cpp_emit.emit: no scalar emitter for {type(node).__name__}; "
      "pipeline/structural nodes are handled elsewhere"
    )
  return fn(node)


def emit_orchestrator(
  node: m.MirNode,
  iter_var: int = 9999,
  dest_stream_map: Optional[dict[str, list[int]]] = None,
) -> str:
  '''Orchestrator-level emit: prepends `_stream_pool.wait_event(...)` for
  every prior parallel-pipeline stream that wrote to this node's relation.

  Currently only `RebuildIndex` needs the wait (matches mhk's original
  placement); everything else delegates to `emit()`. `iter_var` is
  accepted for interface compatibility and isn't consumed yet.
  '''
  if isinstance(node, m.RebuildIndex):
    prefix = ""
    if dest_stream_map is not None and node.rel_name in dest_stream_map:
      prefix += f"// Wait for streams that wrote to {node.rel_name}\n"
      for stream_idx in dest_stream_map[node.rel_name]:
        prefix += f"_stream_pool.wait_event({stream_idx});\n"
    return prefix + emit_rebuild_index(node)
  return emit(node)
