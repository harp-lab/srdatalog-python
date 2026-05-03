'''Pipeline-shape utilities used by `complete_runner`.

Originally ported from `src/srdatalog/codegen/target_jit/
jit_emit_helpers.nim` — the legacy Filter / ConstantBind / InsertInto
emit procs (`jit_filter`, `jit_constant_bind`, `jit_insert_into`) used
to live here too. They've been retired alongside the rest of the
legacy `ir/dialects/target/cuda/pipeline.py` chain; the dialect now owns those
emits via `dialects.relation.sorted_array.lowerings._lower_inner_chain`
and `_lower_insert_into`.

What remains:
  - `has_balanced_scan` / `get_balanced_scan_info` — used to decide
    whether to emit a balanced-scan kernel variant.
  - `has_tiled_cartesian_eligible` — runner uses this to decide
    whether to enable the tiled-Cartesian materialize path.
  - `assign_handle_positions` / `count_handles_in_pipeline` — pipeline
    pre-pass + view-slot counting. Note: `dialects.target.cuda.
    envelope` has its own `assign_handle_positions` for the dialect
    path; this one stays for the runner-side prepass that mutates
    `mutable_pipe` before kernel emission.
'''

from __future__ import annotations

from dataclasses import dataclass, field

import srdatalog.ir.mir.types as m

# -----------------------------------------------------------------------------
# Balanced-partitioning detection
# -----------------------------------------------------------------------------


def has_balanced_scan(ops: list[m.MirNode]) -> bool:
  '''True if the pipeline's first op is a BalancedScan (root level).'''
  return len(ops) > 0 and isinstance(ops[0], m.BalancedScan)


def has_tiled_cartesian_eligible(ops: list[m.MirNode]) -> bool:
  '''Pipeline contains a 2-source CartesianJoin where each source binds
  exactly one variable — eligible for the atomic-free tiled/coalesced
  write optimization.
  '''
  for op in ops:
    if isinstance(op, m.CartesianJoin):
      if (
        len(op.sources) == 2
        and len(op.var_from_source) == 2
        and len(op.var_from_source[0]) == 1
        and len(op.var_from_source[1]) == 1
      ):
        return True
  return False


@dataclass
class BalancedScanInfo:
  '''Lightweight struct returned by get_balanced_scan_info.'''

  group_var: str = ""
  src1_rel_name: str = ""
  src1_index: list[int] = field(default_factory=list)
  src1_handle_idx: int = -1
  src2_rel_name: str = ""
  src2_index: list[int] = field(default_factory=list)
  src2_handle_idx: int = -1


def get_balanced_scan_info(ops: list[m.MirNode]) -> BalancedScanInfo:
  '''Extract group var + per-source (rel, index, handle_idx) from the root
  BalancedScan. Returns an all-empty BalancedScanInfo when the root op
  isn't a BalancedScan (matching Nim's sentinel return).
  '''
  if has_balanced_scan(ops):
    bs = ops[0]
    assert isinstance(bs, m.BalancedScan)
    s1, s2 = bs.source1, bs.source2
    return BalancedScanInfo(
      group_var=bs.group_var,
      src1_rel_name=s1.rel_name,
      src1_index=list(s1.index),
      src1_handle_idx=getattr(s1, "handle_start", -1),
      src2_rel_name=s2.rel_name,
      src2_index=list(s2.index),
      src2_handle_idx=getattr(s2, "handle_start", -1),
    )
  return BalancedScanInfo()


# -----------------------------------------------------------------------------
# Pipeline handle counting
# -----------------------------------------------------------------------------


def _assign_handle_positions_rec(node: m.MirNode, offset_box: list[int]) -> None:
  '''Recursively assign `handle_start` to this node and any children.

  `offset_box` is a single-element list used as a mutable counter
  (Python closures can't reassign captured ints cleanly in a loop).
  '''
  if (
    isinstance(node, m.ColumnSource)
    or isinstance(node, m.Scan)
    or isinstance(node, m.Aggregate)
    or isinstance(node, m.Negation)
  ):
    node.handle_start = offset_box[0]
    offset_box[0] += 1
  elif isinstance(node, m.ColumnJoin) or isinstance(node, m.CartesianJoin):
    node.handle_start = offset_box[0]
    for src in node.sources:
      _assign_handle_positions_rec(src, offset_box)
  elif isinstance(node, m.BalancedScan):
    node.handle_start = offset_box[0]
    _assign_handle_positions_rec(node.source1, offset_box)
    _assign_handle_positions_rec(node.source2, offset_box)
  elif isinstance(node, m.PositionedExtract):
    for src in node.sources:
      _assign_handle_positions_rec(src, offset_box)


def assign_handle_positions(ops: list[m.MirNode]) -> None:
  '''Assign `handle_start` to every source-bearing node in pipeline
  order starting from 0. Mutates `ops` in place. Mirrors Nim's
  assignHandlePositions.'''
  offset_box = [0]
  for op in ops:
    _assign_handle_positions_rec(op, offset_box)


def count_handles_in_pipeline(ops: list[m.MirNode]) -> int:
  '''Max `handle_start + 1` seen across the pipeline — the number of view
  slots the kernel's `views[]` array needs. Zero when no op carries a
  handle_start (caller should still allocate 1 slot in that case; this
  function faithfully returns 0 to match Nim).
  '''
  result = 0
  for op in ops:
    if isinstance(op, m.ColumnJoin) or isinstance(op, m.CartesianJoin):
      for src in op.sources:
        h = getattr(src, "handle_start", -1)
        result = max(result, h + 1)
    elif isinstance(op, m.Scan) or isinstance(op, m.Negation) or isinstance(op, m.Aggregate):
      result = max(result, getattr(op, "handle_start", -1) + 1)
  return result
