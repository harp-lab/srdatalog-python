'''MIR optimization passes. Mirror src/srdatalog/mir/{pre_reconstruct_rebuild,
clause_order_reorder, prefix_source_reorder}.nim. Each operates on a
seq[LoweredStep] (represented in Python as list[tuple[MirNode, bool]]).

Order (matches Nim's registerMirOptimizePass priorities):
  0. insert_pre_reconstruct_rebuilds
  1. apply_clause_order_reordering
  2. apply_prefix_source_reordering
  3. balanced_scan_pass — DEFERRED (DSL lacks balanced pragma)
'''

from __future__ import annotations

import srdatalog.mir.types as mir
from srdatalog.hir.types import Version

# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _has_prefix(source) -> bool:
  '''Source node has a non-empty prefix. Mirrors Nim's hasPrefix.'''
  if isinstance(source, (mir.ColumnSource, mir.Scan, mir.Negation)):
    return len(source.prefix_vars) > 0
  return False


def _regenerate_source_specs(ep: mir.ExecutePipeline) -> None:
  '''After any in-place source reordering, the ExecutePipeline's
  source_specs list must be rebuilt from the pipeline body. Otherwise
  the generated C++ source_specs type drifts out of sync with the actual
  MIR source order and handles point at the wrong views.
  '''
  from srdatalog.hir.lower import _extract_pipeline_sources

  specs: list[mir.MirNode] = []
  for op in ep.pipeline:
    _extract_pipeline_sources(op, specs)
  ep.source_specs = specs


# -----------------------------------------------------------------------------
# Pass 0: insert_pre_reconstruct_rebuilds
# -----------------------------------------------------------------------------


def _extract_merged_indices(step: mir.MirNode, rel_name: str) -> set[tuple[int, ...]]:
  '''Indices of `rel_name` that were merged to FULL in this FixpointPlan step.'''
  out: set[tuple[int, ...]] = set()
  if not isinstance(step, mir.FixpointPlan):
    return out
  for instr in step.instructions:
    if isinstance(instr, mir.MergeIndex) and instr.rel_name == rel_name:
      out.add(tuple(instr.index))
    elif isinstance(instr, mir.ParallelGroup):
      for op in instr.ops:
        if isinstance(op, mir.MergeIndex) and op.rel_name == rel_name:
          out.add(tuple(op.index))
  return out


def _extract_modified_relations(step: mir.MirNode) -> set[str]:
  '''Relations this FixpointPlan step writes to (via InsertInto).'''
  out: set[str] = set()
  if not isinstance(step, mir.FixpointPlan):
    return out
  for instr in step.instructions:
    if isinstance(instr, mir.ExecutePipeline):
      for op in instr.pipeline:
        if isinstance(op, mir.InsertInto):
          out.add(op.rel_name)
    elif isinstance(instr, mir.ParallelGroup):
      for pg_op in instr.ops:
        if isinstance(pg_op, mir.ExecutePipeline):
          for op in pg_op.pipeline:
            if isinstance(op, mir.InsertInto):
              out.add(op.rel_name)
  return out


def _collect_needed_indices(node: mir.MirNode, rel_name: str, out: set[tuple[int, ...]]) -> None:
  '''Recursive walk collecting FULL- or DELTA-version ColumnSource index
  tuples targeting `rel_name`. DELTA dispatches through FULL on the first
  fixpoint iteration, so both count.
  '''
  if isinstance(node, mir.ColumnSource):
    if node.rel_name == rel_name and node.version in (Version.FULL, Version.DELTA):
      out.add(tuple(node.index))
  elif isinstance(node, mir.ColumnJoin) or isinstance(node, mir.CartesianJoin):
    for s in node.sources:
      _collect_needed_indices(s, rel_name, out)
  elif isinstance(node, mir.ExecutePipeline):
    for op in node.pipeline:
      _collect_needed_indices(op, rel_name, out)
  elif isinstance(node, mir.ParallelGroup):
    for op in node.ops:
      _collect_needed_indices(op, rel_name, out)
  elif isinstance(node, mir.FixpointPlan):
    for instr in node.instructions:
      _collect_needed_indices(instr, rel_name, out)


def insert_pre_reconstruct_rebuilds(
  steps: list[tuple[mir.MirNode, bool]],
) -> list[tuple[mir.MirNode, bool]]:
  '''After every PostStratumReconstructInternCols step, insert any
  RebuildIndex(FULL) ops for indices of this relation that subsequent
  strata will read but that this stratum didn't merge to FULL.
  '''
  out: list[tuple[mir.MirNode, bool]] = []
  for i, (node, is_rec) in enumerate(steps):
    out.append((node, is_rec))
    if not isinstance(node, mir.PostStratumReconstructInternCols):
      continue

    rel_name = node.rel_name

    # Most recent prior FixpointPlan that wrote to this relation.
    merged: set[tuple[int, ...]] = set()
    for j in range(i - 1, -1, -1):
      prior_node, _ = steps[j]
      if isinstance(prior_node, mir.FixpointPlan):
        if rel_name in _extract_modified_relations(prior_node):
          merged = _extract_merged_indices(prior_node, rel_name)
          break

    # Union of all FULL-or-DELTA index accesses in subsequent steps.
    needed: set[tuple[int, ...]] = set()
    for j in range(i + 1, len(steps)):
      _collect_needed_indices(steps[j][0], rel_name, needed)

    # Sort for determinism (Nim's HashSet iteration order is hash-dependent;
    # we sort to ensure reproducible output regardless of hash seed).
    for idx in sorted(needed - merged):
      out.append(
        (
          mir.RebuildIndex(
            rel_name=rel_name,
            version=Version.FULL,
            index=list(idx),
          ),
          False,
        )
      )
  return out


# -----------------------------------------------------------------------------
# Pass 1: clause_order_reorder
# -----------------------------------------------------------------------------


def _position_in(clause_order: list[int], clause_idx: int) -> int:
  '''Position of clause_idx in clause_order, or len(clause_order) if absent.'''
  try:
    return clause_order.index(clause_idx)
  except ValueError:
    return len(clause_order)


def _reorder_column_join_by_clause_order(cj: mir.ColumnJoin, clause_order: list[int]) -> None:
  if not clause_order:
    return
  cj.sources.sort(key=lambda s: _position_in(clause_order, s.clause_idx))


def _reorder_cartesian_join_by_clause_order(
  cart: mir.CartesianJoin, clause_order: list[int]
) -> None:
  if not clause_order:
    return
  pairs = list(zip(cart.sources, cart.var_from_source))
  pairs.sort(key=lambda p: _position_in(clause_order, p[0].clause_idx))
  cart.sources = [p[0] for p in pairs]
  cart.var_from_source = [p[1] for p in pairs]


def _apply_clause_order_reorder(ep: mir.ExecutePipeline) -> None:
  for op in ep.pipeline:
    if isinstance(op, mir.ColumnJoin):
      _reorder_column_join_by_clause_order(op, ep.clause_order)
    elif isinstance(op, mir.CartesianJoin):
      _reorder_cartesian_join_by_clause_order(op, ep.clause_order)
  _regenerate_source_specs(ep)


def apply_clause_order_reordering(
  steps: list[tuple[mir.MirNode, bool]],
) -> list[tuple[mir.MirNode, bool]]:
  '''Reorder every ColumnJoin/CartesianJoin's sources by the enclosing
  ExecutePipeline's clause_order. Mutates in place; returns `steps` for
  chain convenience.
  '''
  for node, _ in steps:
    if isinstance(node, mir.FixpointPlan):
      for instr in node.instructions:
        if isinstance(instr, mir.ParallelGroup):
          for op in instr.ops:
            if isinstance(op, mir.ExecutePipeline):
              _apply_clause_order_reorder(op)
        elif isinstance(instr, mir.ExecutePipeline):
          _apply_clause_order_reorder(instr)
    elif isinstance(node, mir.ExecutePipeline):
      _apply_clause_order_reorder(node)
  return steps


# -----------------------------------------------------------------------------
# Pass 2: prefix_source_reorder
# -----------------------------------------------------------------------------


def _reorder_column_join_by_prefix(cj: mir.ColumnJoin) -> None:
  '''Put prefixed sources first, but only if the first source isn't
  already prefixed (short-circuit avoids disrupting already-good orders).
  '''
  if len(cj.sources) < 2:
    return
  if _has_prefix(cj.sources[0]):
    return
  if not any(_has_prefix(s) for s in cj.sources[1:]):
    return
  cj.sources.sort(key=lambda s: 0 if _has_prefix(s) else 1)


def _reorder_cartesian_join_by_prefix(cart: mir.CartesianJoin) -> None:
  if len(cart.sources) < 2:
    return
  if _has_prefix(cart.sources[0]):
    return
  if not any(_has_prefix(s) for s in cart.sources[1:]):
    return
  pairs = list(zip(cart.sources, cart.var_from_source))
  pairs.sort(key=lambda p: 0 if _has_prefix(p[0]) else 1)
  cart.sources = [p[0] for p in pairs]
  cart.var_from_source = [p[1] for p in pairs]


def _apply_prefix_reorder(ep: mir.ExecutePipeline) -> None:
  for op in ep.pipeline:
    if isinstance(op, mir.ColumnJoin):
      _reorder_column_join_by_prefix(op)
    elif isinstance(op, mir.CartesianJoin):
      _reorder_cartesian_join_by_prefix(op)
  _regenerate_source_specs(ep)


def apply_prefix_source_reordering(
  steps: list[tuple[mir.MirNode, bool]],
) -> list[tuple[mir.MirNode, bool]]:
  '''Move prefixed sources to the front of every ColumnJoin/CartesianJoin
  (avoids "galloping from root" on unprefixed sources). Mutates in place.
  '''
  for node, _ in steps:
    if isinstance(node, mir.FixpointPlan):
      for instr in node.instructions:
        if isinstance(instr, mir.ParallelGroup):
          for op in instr.ops:
            if isinstance(op, mir.ExecutePipeline):
              _apply_prefix_reorder(op)
        elif isinstance(instr, mir.ExecutePipeline):
          _apply_prefix_reorder(instr)
    elif isinstance(node, mir.ExecutePipeline):
      _apply_prefix_reorder(node)
  return steps


# -----------------------------------------------------------------------------
# Pass 3: balanced_scan_pass
# -----------------------------------------------------------------------------


def _transform_balanced_pipeline(pipeline: list[mir.MirNode]) -> list[mir.MirNode]:
  '''If the first op is a BalancedScan, convert any subsequent ColumnJoin
  for one of its balanced vars into a PositionedExtract (point-lookup
  instead of iteration). Otherwise return the pipeline unchanged.

  Mirrors transformBalancedPipeline in balanced_scan_pass.nim. Non-ColumnJoin
  ops pass through unchanged.
  '''
  if not pipeline or not isinstance(pipeline[0], mir.BalancedScan):
    return pipeline
  bs = pipeline[0]
  balanced_vars = set(bs.vars1) | set(bs.vars2)
  out: list[mir.MirNode] = [bs]
  for op in pipeline[1:]:
    if isinstance(op, mir.ColumnJoin) and op.var_name in balanced_vars:
      out.append(
        mir.PositionedExtract(
          sources=list(op.sources),
          var_name=op.var_name,
          bind_vars=[],
        )
      )
    else:
      out.append(op)
  return out


def _apply_balanced_scan_pass_recursive(node: mir.MirNode) -> mir.MirNode:
  '''Walk the MIR tree, transforming each ExecutePipeline in place (via
  pipeline replacement) when its body starts with BalancedScan.
  '''
  if isinstance(node, mir.ExecutePipeline):
    node.pipeline = _transform_balanced_pipeline(node.pipeline)
    return node
  if isinstance(node, mir.FixpointPlan):
    node.instructions = [_apply_balanced_scan_pass_recursive(instr) for instr in node.instructions]
    return node
  if isinstance(node, mir.ParallelGroup):
    node.ops = [_apply_balanced_scan_pass_recursive(op) for op in node.ops]
    return node
  return node


def apply_balanced_scan_pass(
  steps: list[tuple[mir.MirNode, bool]],
) -> list[tuple[mir.MirNode, bool]]:
  '''Apply balanced-scan -> positioned-extract transform to every
  ExecutePipeline. No-op when the Python DSL hasn't emitted a BalancedScan
  (current default: never, since balanced-scan lowering isn't wired in).
  '''
  for node, _ in steps:
    _apply_balanced_scan_pass_recursive(node)
  return steps


# -----------------------------------------------------------------------------
# Chain
# -----------------------------------------------------------------------------


def apply_all_mir_passes(steps: list[tuple[mir.MirNode, bool]]) -> list[tuple[mir.MirNode, bool]]:
  '''Run the ported MIR optimization passes in Nim order.'''
  steps = insert_pre_reconstruct_rebuilds(steps)
  steps = apply_clause_order_reordering(steps)
  steps = apply_prefix_source_reordering(steps)
  steps = apply_balanced_scan_pass(steps)
  return steps
