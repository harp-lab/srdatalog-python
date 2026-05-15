'''MIR optimization passes. Mirror src/srdatalog/mir/{pre_reconstruct_rebuild,
clause_order_reorder, prefix_source_reorder}.nim. Each operates on a
seq[LoweredStep] (represented in Python as list[tuple[MirNode, bool]]).

Order (matches Nim's registerMirOptimizePass priorities):
  0. insert_pre_reconstruct_rebuilds
  1. apply_clause_order_reordering
  2. apply_prefix_source_reordering
  3. balanced_scan_pass — DEFERRED (DSL lacks balanced pragma)
  4. apply_concurrent_write_marking — Phase A2.2 (compute-once
     concurrent_write decision; replaces orchestrator's mutation shim)
'''

from __future__ import annotations

import dataclasses

import srdatalog.ir.mir.types as mir
from srdatalog.ir.hir.types import Version

# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _has_prefix(source) -> bool:
  '''Source node has a non-empty prefix. Mirrors Nim's hasPrefix.'''
  if isinstance(source, (mir.ColumnSource, mir.Scan, mir.Negation)):
    return len(source.prefix_vars) > 0
  return False


def _regenerate_source_specs(ep: mir.ExecutePipeline) -> None:
  '''After source reordering, regenerate `source_specs` from the
  (possibly reordered) pipeline body. Mutates `ep.source_specs` in
  place via `object.__setattr__` shim (MIR is frozen post-Phase-A;
  A2/A3 will thread this properly).
  '''
  from srdatalog.ir.hir.lower import _extract_pipeline_sources

  specs: list[mir.ColumnSource | mir.Scan | mir.Negation | mir.Aggregate] = []
  for op in ep.pipeline:
    _extract_pipeline_sources(op, specs)
  object.__setattr__(ep, 'source_specs', specs)


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


def _reorder_column_join_by_clause_order(cj: mir.ColumnJoin, clause_order: tuple[int, ...]) -> None:
  '''Reorder cj.sources in place via `object.__setattr__` shim
  (transition tech-debt; A2/A3 threads properly).'''
  if not clause_order:
    return
  new_sources = tuple(sorted(cj.sources, key=lambda s: _position_in(clause_order, s.clause_idx)))
  object.__setattr__(cj, 'sources', list(new_sources))


def _reorder_cartesian_join_by_clause_order(
  cart: mir.CartesianJoin, clause_order: tuple[int, ...]
) -> None:
  '''Reorder cart.sources + cart.var_from_source in place.'''
  if not clause_order:
    return
  pairs = list(zip(cart.sources, cart.var_from_source))
  pairs.sort(key=lambda p: _position_in(clause_order, p[0].clause_idx))
  object.__setattr__(cart, 'sources', [p[0] for p in pairs])
  object.__setattr__(cart, 'var_from_source', [p[1] for p in pairs])


def _apply_clause_order_reorder(ep: mir.ExecutePipeline) -> None:
  '''Reorder ColumnJoin/CartesianJoin sources in place + regenerate
  source_specs.'''
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
  ExecutePipeline's clause_order. Mutates in place via shim; returns
  `steps` for chain convenience.
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
  '''Move prefixed sources to front in place via shim.'''
  if len(cj.sources) < 2:
    return
  if _has_prefix(cj.sources[0]):
    return
  if not any(_has_prefix(s) for s in cj.sources[1:]):
    return
  new_sources = sorted(cj.sources, key=lambda s: 0 if _has_prefix(s) else 1)
  object.__setattr__(cj, 'sources', new_sources)


def _reorder_cartesian_join_by_prefix(cart: mir.CartesianJoin) -> None:
  '''Move prefixed sources to front + paired-reorder var_from_source.'''
  if len(cart.sources) < 2:
    return
  if _has_prefix(cart.sources[0]):
    return
  if not any(_has_prefix(s) for s in cart.sources[1:]):
    return
  pairs = list(zip(cart.sources, cart.var_from_source))
  pairs.sort(key=lambda p: 0 if _has_prefix(p[0]) else 1)
  object.__setattr__(cart, 'sources', [p[0] for p in pairs])
  object.__setattr__(cart, 'var_from_source', [p[1] for p in pairs])


def _apply_prefix_reorder(ep: mir.ExecutePipeline) -> None:
  '''Apply prefix reorder + regenerate source_specs in place.'''
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
  (avoids "galloping from root" on unprefixed sources). Mutates in
  place via shim; returns `steps` for chain convenience.
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
  '''Walk the MIR tree; transform each ExecutePipeline in place via
  shim (MIR is frozen; A2/A3 will thread properly).
  '''
  if isinstance(node, mir.ExecutePipeline):
    object.__setattr__(node, 'pipeline', _transform_balanced_pipeline(node.pipeline))
    return node
  if isinstance(node, mir.FixpointPlan):
    object.__setattr__(
      node,
      'instructions',
      [_apply_balanced_scan_pass_recursive(instr) for instr in node.instructions],
    )
    return node
  if isinstance(node, mir.ParallelGroup):
    object.__setattr__(node, 'ops', [_apply_balanced_scan_pass_recursive(op) for op in node.ops])
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
# Pass 4: concurrent_write marking (Phase A2.2)
# -----------------------------------------------------------------------------


def _ep_is_single_dest_concat_eligible(ep: mir.ExecutePipeline) -> bool:
  '''Same predicate `_gen_parallel_group` uses to filter `single_in_dest`:
  exactly one dest, no work-stealing, no dedup-hash. Mirrors the
  upstream Nim orchestrator decision so byte-equivalence holds.
  '''
  return len(ep.dest_specs) == 1 and not ep.work_stealing and not ep.dedup_hash


def _mark_concurrent_writes_in_parallel_group(
  pg: mir.ParallelGroup,
) -> mir.ParallelGroup:
  '''Compute the concurrent_write decision for a ParallelGroup and
  return a new ParallelGroup whose ExecutePipeline children carry
  `concurrent_write=True` where appropriate. Mirrors the legacy
  orchestrator branch:

    has_concat = (dest_rel has >= 2 single-dest writers in this group)
    if has_concat:
        for each single-dest writer in this dest: concurrent_write=True

  Skips groups that take the dedup-sequential or single-rule path
  in `_gen_parallel_group` — those never set the flag.
  '''
  exec_pairs: list[tuple[int, mir.ExecutePipeline]] = [
    (idx, op) for idx, op in enumerate(pg.ops) if isinstance(op, mir.ExecutePipeline)
  ]

  if len(exec_pairs) <= 1:
    return pg

  exec_indices = [i for i, _ in exec_pairs]
  exec_ops = [op for _, op in exec_pairs]
  if any(op.dedup_hash for op in exec_ops):
    # Dedup-sequential path in `_gen_parallel_group`: no marking.
    return pg

  # Build dest_rel -> [exec_indices] for single-dest concat-eligible EPs.
  dest_single_rules: dict[str, list[int]] = {}
  for ep_pos, op in enumerate(exec_ops):
    if not _ep_is_single_dest_concat_eligible(op):
      continue
    dest = op.dest_specs[0]
    if isinstance(dest, mir.InsertInto):
      dest_single_rules.setdefault(dest.rel_name, []).append(ep_pos)

  # Identify which exec_ops positions need concurrent_write=True.
  to_mark: set[int] = set()
  for _dest_rel, positions in dest_single_rules.items():
    if len(positions) >= 2:
      to_mark.update(positions)

  if not to_mark:
    return pg

  # Rebuild ops list with replacements. Only the marked
  # ExecutePipelines are replaced; everything else is identity.
  new_ops: list[mir.MirNode] = list(pg.ops)
  for ep_pos in to_mark:
    src_idx = exec_indices[ep_pos]
    new_ops[src_idx] = dataclasses.replace(exec_ops[ep_pos], concurrent_write=True)

  return dataclasses.replace(pg, ops=new_ops)


def _mark_concurrent_writes_in_node(node: mir.MirNode) -> mir.MirNode:
  '''Walk a top-level step node, returning a new node whose enclosed
  ParallelGroups carry the concurrent_write decision. Recurses into
  FixpointPlan + Block instruction lists.
  '''
  if isinstance(node, mir.ParallelGroup):
    return _mark_concurrent_writes_in_parallel_group(node)
  if isinstance(node, mir.FixpointPlan):
    new_instructions: list[mir.MirNode] = [
      _mark_concurrent_writes_in_node(instr) for instr in node.instructions
    ]
    if all(a is b for a, b in zip(new_instructions, node.instructions)):
      return node
    return dataclasses.replace(node, instructions=new_instructions)
  if isinstance(node, mir.Block):
    new_instructions = [_mark_concurrent_writes_in_node(instr) for instr in node.instructions]
    if all(a is b for a, b in zip(new_instructions, node.instructions)):
      return node
    return dataclasses.replace(node, instructions=new_instructions)
  return node


def apply_concurrent_write_marking(
  steps: list[tuple[mir.MirNode, bool]],
) -> list[tuple[mir.MirNode, bool]]:
  '''Mark `ExecutePipeline.concurrent_write=True` for any pipeline that
  participates in a ParallelGroup where two or more single-dest
  pipelines share a destination relation. Computes the decision once
  at MIR-pass time and propagates via `dataclasses.replace`, replacing
  the `object.__setattr__` shim previously living in
  `ir/codegen/cuda/orchestrator.py` `_gen_parallel_group` (Phase A2.2,
  per `docs/phase_a_mir_onto_op.md` section 8 + `docs/code_discipline.md`
  D18).

  The orchestrator + `complete_runner.py` both read the flag off the
  same MIR program after this pass, so the decision is consistent
  across orchestrator emit + per-rule runner emit (the
  `tiled_cartesian_eligible` check on the runner side disables tiled
  Cartesian when the flag is set).
  '''
  out: list[tuple[mir.MirNode, bool]] = []
  for node, is_rec in steps:
    out.append((_mark_concurrent_writes_in_node(node), is_rec))
  return out


# -----------------------------------------------------------------------------
# Chain
# -----------------------------------------------------------------------------


def apply_all_mir_passes(steps: list[tuple[mir.MirNode, bool]]) -> list[tuple[mir.MirNode, bool]]:
  '''Run the ported MIR optimization passes in Nim order.'''
  steps = insert_pre_reconstruct_rebuilds(steps)
  steps = apply_clause_order_reordering(steps)
  steps = apply_prefix_source_reordering(steps)
  steps = apply_balanced_scan_pass(steps)
  steps = apply_concurrent_write_marking(steps)
  return steps
