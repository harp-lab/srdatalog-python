'''HIR -> MIR Lowering (Pass 6). Phase 1+2.

Phase 1 (shipped): single-clause variant case + helpers.
Phase 2 (this file): multi-clause lowering (ColumnJoin per join var + one
CartesianJoin for the remaining independent vars), negation patterns, and
the four maintenance generators (rebuild/merge indices, simple + loop
maintenance).

Phase 3 (future): stratum wrapping (ExecutePipeline / Block / FixpointPlan
/ Program, parallel groups, schema arities, before/after hooks) and a
Nim-side tool that emits MIR S-expr so we can do end-to-end byte-diff.

Deliberately NOT ported (no Python DSL equivalent or deferred optimization):
  - Binary-join / materialized-join dispatch (alternate dialects)
  - Balanced-scan (balancedRoot / balancedSources pragmas)
  - Split rule lowering (SplitClause + tempRelName)
  - IfClause (Filter) / LetClause (ConstantBind) / AggClause handling
  - InnerPipeline / debug-hook injection
'''
from __future__ import annotations

from srdatalog.dsl import Atom, Negation, Filter, Let, ArgKind
from srdatalog.hir.types import AccessPattern, HirProgram, HirRuleVariant, HirStratum, Version
from srdatalog.hir.index import complete_index, get_arity
import srdatalog.mir.types as mir


def _prefix_vars(pattern: AccessPattern) -> list[str]:
  '''Extract the prefix-length slice of access_order (bound-var prefix).'''
  return list(pattern.access_order[:pattern.prefix_len])


def generate_column_source(pattern: AccessPattern) -> mir.ColumnSource:
  '''Mirror Nim generateColumnSource. Used by Phase 2 ColumnJoin lowering.'''
  return mir.ColumnSource(
    rel_name=pattern.rel_name,
    version=pattern.version,
    index=list(pattern.index_cols),
    prefix_vars=_prefix_vars(pattern),
    clause_idx=pattern.clause_idx,
  )


def generate_scan(pattern: AccessPattern, bound_vars: list[str]) -> mir.Scan:
  '''Mirror Nim generateScan. Produces the Scan node used as the outer
  iteration driver when a variant has a single body clause.

  `bound_vars` is accepted for API parity with Nim; the Nim implementation
  does not use it either (all binding context comes from the access
  pattern's own `access_order` / `prefix_len`).
  '''
  return mir.Scan(
    vars=list(pattern.access_order),
    rel_name=pattern.rel_name,
    version=pattern.version,
    index=list(pattern.index_cols),
    prefix_vars=_prefix_vars(pattern),
  )


def generate_insert_into(head: Atom, canonical_index: list[int]) -> mir.InsertInto:
  '''Mirror Nim generateInsertInto. Emits to NEW_VER (always) with the
  stratum's canonical index for the head relation.
  '''
  head_vars = [a.var_name for a in head.args if a.kind is ArgKind.LVAR]
  return mir.InsertInto(
    rel_name=head.rel,
    version=Version.NEW,
    vars=head_vars,
    index=list(canonical_index),
  )


def _lower_multi_clause_body(
  variant: HirRuleVariant,
) -> list[mir.MirNode]:
  '''Produce the body of a multi-clause variant: one ColumnJoin per join var,
  then one CartesianJoin over the remaining independent vars. Mirrors
  lowerVariantToPipeline's WCOJ path (lines 495-730 of lowering.nim).

  Key invariant preserved: the ColumnSource's `index` for a given clause is
  decided at its first appearance (in a ColumnJoin) and reused verbatim for
  any later CartesianJoin, so prefixes and independent-var column positions
  line up with what the JIT expects.
  '''
  ops: list[mir.MirNode] = []
  rule = variant.original_rule
  var_order = variant.var_order
  join_vars_set = variant.join_vars
  bound_vars: list[str] = []
  pattern_computed_index: dict[int, list[int]] = {}

  # --- ColumnJoin per join var (in var_order) ---
  for jv in var_order:
    if jv not in join_vars_set:
      continue
    sources: list[mir.MirNode] = []
    for p in variant.access_patterns:
      if jv in p.access_order and p.rel_name:
        prefix = [v for v in p.access_order if v in bound_vars]
        idx = list(p.index_cols)
        pattern_computed_index[p.clause_idx] = idx
        sources.append(mir.ColumnSource(
          rel_name=p.rel_name,
          version=p.version,
          index=idx,
          prefix_vars=prefix,
          clause_idx=p.clause_idx,
        ))
    if sources:
      # TODO: balanced-scan dispatch (balancedRoot/balancedSources) — Phase 4.
      ops.append(mir.ColumnJoin(var_name=jv, sources=sources))
      bound_vars.append(jv)

  # --- Independent vars (non-join, non-negation-only) ---
  positive_vars: set[str] = set()
  for p in variant.access_patterns:
    positive_vars.update(p.access_order)
  negation_only_vars: set[str] = set()
  for p in variant.negation_patterns:
    for v in p.access_order:
      if v not in positive_vars:
        negation_only_vars.add(v)

  independent_vars = [
    v for v in var_order
    if v not in join_vars_set and v not in negation_only_vars
  ]
  indep_set = set(independent_vars)
  # Wildcards (_genN) may not be in var_order but still appear in positive
  # clauses — include them so CartesianJoin fan-out iterates correctly.
  for body in rule.body:
    if isinstance(body, Atom):
      for arg in body.args:
        if arg.kind is ArgKind.LVAR:
          v = arg.var_name
          if (
            v not in join_vars_set
            and v not in indep_set
            and v not in negation_only_vars
          ):
            independent_vars.append(v)
            indep_set.add(v)

  # --- CartesianJoin (if any independent vars) ---
  if independent_vars:
    cart_sources: list[mir.MirNode] = []
    var_from_source: list[list[str]] = []
    for p in variant.access_patterns:
      has_indep = any(v in indep_set for v in p.access_order)
      if not (has_indep and p.rel_name):
        continue

      # Reuse the index computed during ColumnJoin; if the pattern never
      # participated in a ColumnJoin, compute a fresh one that puts
      # bound-var columns first (matches Nim fallback).
      if p.clause_idx in pattern_computed_index:
        computed_index = list(pattern_computed_index[p.clause_idx])
      else:
        used_cols: set[int] = set()
        computed_index = []
        for prefix_var in bound_vars:
          for i, v in enumerate(p.access_order):
            if v == prefix_var and i < len(p.index_cols):
              col = p.index_cols[i]
              if col not in used_cols:
                computed_index.append(col)
                used_cols.add(col)
              break
        for col in p.index_cols:
          if col not in used_cols:
            computed_index.append(col)

      # Prefix: bound vars in COMPUTED INDEX column order (not access_order).
      prefix: list[str] = []
      for col_idx in computed_index:
        for i, v in enumerate(p.access_order):
          if i < len(p.index_cols) and p.index_cols[i] == col_idx:
            if v in bound_vars and v not in prefix:
              prefix.append(v)
            break

      # Independent vars this clause provides, in computed-index column order.
      vars_from_this: list[str] = []
      for col_idx in computed_index:
        for i, v in enumerate(p.access_order):
          if i < len(p.index_cols) and p.index_cols[i] == col_idx:
            if v in indep_set and v not in prefix and v not in vars_from_this:
              vars_from_this.append(v)
            break

      cart_sources.append(mir.ColumnSource(
        rel_name=p.rel_name,
        version=p.version,
        index=computed_index,
        prefix_vars=prefix,
        clause_idx=p.clause_idx,
      ))
      var_from_source.append(vars_from_this)

    if cart_sources:
      ops.append(mir.CartesianJoin(
        vars=independent_vars,
        sources=cart_sources,
        var_from_source=var_from_source,
      ))

  return ops


def _lower_negations(variant: HirRuleVariant) -> list[mir.MirNode]:
  out: list[mir.MirNode] = []
  for p in variant.negation_patterns:
    prefix_vars = list(p.access_order[:p.prefix_len])
    out.append(mir.Negation(
      rel_name=p.rel_name,
      version=p.version,
      index=list(p.index_cols),
      prefix_vars=prefix_vars,
      const_args=list(p.const_args),
    ))
  return out


def _lower_filter_and_let_clauses(variant: HirRuleVariant) -> list[mir.MirNode]:
  '''Lower each Filter / Let body clause to its MIR counterpart, iterated
  in SOURCE order (matches Nim lowering.nim's final per-clause loop).
  '''
  out: list[mir.MirNode] = []
  for b in variant.original_rule.body:
    if isinstance(b, Filter):
      out.append(mir.Filter(vars=list(b.vars), code=b.code))
    elif isinstance(b, Let):
      out.append(mir.ConstantBind(
        var_name=b.var_name, code=b.code, deps=list(b.deps),
      ))
  return out


def _lower_above_filter_and_let(variant: HirRuleVariant) -> list[mir.MirNode]:
  '''Filter / Let clauses whose source body index is strictly below split_at.'''
  out: list[mir.MirNode] = []
  for i in range(variant.split_at):
    b = variant.original_rule.body[i]
    if isinstance(b, Filter):
      out.append(mir.Filter(vars=list(b.vars), code=b.code))
    elif isinstance(b, Let):
      out.append(mir.ConstantBind(
        var_name=b.var_name, code=b.code, deps=list(b.deps),
      ))
  return out


def _lower_below_filter_and_let(variant: HirRuleVariant) -> list[mir.MirNode]:
  '''Filter / Let clauses whose source body index is strictly above split_at.'''
  out: list[mir.MirNode] = []
  for i in range(variant.split_at + 1, len(variant.original_rule.body)):
    b = variant.original_rule.body[i]
    if isinstance(b, Filter):
      out.append(mir.Filter(vars=list(b.vars), code=b.code))
    elif isinstance(b, Let):
      out.append(mir.ConstantBind(
        var_name=b.var_name, code=b.code, deps=list(b.deps),
      ))
  return out


# -----------------------------------------------------------------------------
# Split-rule lowering (Pipeline A = above-split, Pipeline B = below-split)
# -----------------------------------------------------------------------------

def lower_split_above(
  variant: HirRuleVariant, stratum: HirStratum,
) -> list[mir.MirNode]:
  '''Mirror lowerSplitAbove in lowering.nim. Supports single-positive-clause
  above-split (Scan + negations + filter/let), which covers the negation-
  pushdown use case. Returns empty list if multi-positive above-split is
  encountered (caller falls back to full pipeline).
  '''
  ops: list[mir.MirNode] = []

  above_patterns = [
    p for p in variant.access_patterns if p.clause_idx < variant.split_at
  ]
  if len(above_patterns) == 1:
    p = above_patterns[0]
    ops.append(mir.Scan(
      vars=list(p.access_order),
      rel_name=p.rel_name,
      version=p.version,
      index=list(p.index_cols),
      prefix_vars=[],
    ))
  elif len(above_patterns) > 1:
    # Unsupported — caller falls back.
    return []

  # Negations above the split.
  for p in variant.negation_patterns:
    if p.clause_idx < variant.split_at:
      ops.append(mir.Negation(
        rel_name=p.rel_name,
        version=p.version,
        index=list(p.index_cols),
        prefix_vars=list(p.access_order[:p.prefix_len]),
        const_args=list(p.const_args),
      ))

  # Filters / Lets above the split (source order).
  ops.extend(_lower_above_filter_and_let(variant))

  # InsertInto the temp relation (NEW_VER, identity index).
  temp_idx = list(range(len(variant.temp_vars)))
  ops.append(mir.InsertInto(
    rel_name=variant.temp_rel_name,
    version=Version.NEW,
    vars=list(variant.temp_vars),
    index=temp_idx,
  ))
  return ops


def lower_split_below(
  variant: HirRuleVariant,
  stratum: HirStratum,
  temp_version: Version = Version.FULL,
) -> list[mir.MirNode]:
  '''Mirror lowerSplitBelow: Scan(temp) + CartesianJoin(below sources,
  prefix = temp vars that they share) + below negations + below filters
  + InsertInto(head).

  `temp_version` switches between the non-recursive default (FULL — the
  temp is merged into FULL by standard maintenance before the Scan) and
  the recursive inner-loop variant (NEW — temp is repopulated each
  iteration and consumed directly from NEW).
  '''
  rule = variant.original_rule
  ops: list[mir.MirNode] = []

  temp_idx = list(range(len(variant.temp_vars)))
  temp_vars_set = set(variant.temp_vars)

  # Step 1: Scan temp — binds all temp vars.
  ops.append(mir.Scan(
    vars=list(variant.temp_vars),
    rel_name=variant.temp_rel_name,
    version=temp_version,
    index=temp_idx,
    prefix_vars=[],
  ))

  # Step 2: CartesianJoin for below patterns that introduce head vars.
  below_patterns = [
    p for p in variant.access_patterns if p.clause_idx > variant.split_at
  ]
  head_vars: set[str] = set()
  for a in rule.head.args:
    if a.kind is ArgKind.LVAR:
      head_vars.add(a.var_name)

  if below_patterns:
    cart_vars: list[str] = []
    cart_sources: list[mir.MirNode] = []
    cart_var_from_source: list[list[str]] = []
    for p in below_patterns:
      p_vars: list[str] = []
      for v in p.access_order:
        if v not in temp_vars_set and v in head_vars and v not in cart_vars:
          p_vars.append(v)
      if not p_vars:
        continue
      prefix = [v for v in p.access_order if v in temp_vars_set]
      cart_sources.append(mir.ColumnSource(
        rel_name=p.rel_name,
        version=p.version,
        index=list(p.index_cols),
        prefix_vars=prefix,
        clause_idx=p.clause_idx,
      ))
      cart_var_from_source.append(p_vars)
      cart_vars.extend(p_vars)

    if cart_vars:
      ops.append(mir.CartesianJoin(
        vars=cart_vars,
        sources=cart_sources,
        var_from_source=cart_var_from_source,
      ))

  # Negations below the split.
  for p in variant.negation_patterns:
    if p.clause_idx > variant.split_at:
      ops.append(mir.Negation(
        rel_name=p.rel_name,
        version=p.version,
        index=list(p.index_cols),
        prefix_vars=list(p.access_order[:p.prefix_len]),
        const_args=list(p.const_args),
      ))

  # Filters / Lets below the split.
  ops.extend(_lower_below_filter_and_let(variant))

  # InsertInto the head.
  head = rule.head
  canonical = stratum.canonical_index.get(head.rel)
  if canonical is None:
    canonical = list(range(len(head.args)))
  ops.append(generate_insert_into(head, list(canonical)))
  return ops


def lower_variant_to_pipeline(
  variant: HirRuleVariant, stratum: HirStratum
) -> list[mir.MirNode]:
  '''Lower a rule variant to an MIR pipeline.

  For a single-clause variant: Scan (+ Negation*) + InsertInto.
  For a multi-clause variant: ColumnJoin* + CartesianJoin? + Negation* + InsertInto.
  '''
  ops: list[mir.MirNode] = []
  n = len(variant.access_patterns)

  if n == 0:
    # Body is pure filters/lets — not expressible in the Python DSL yet.
    pass
  elif n == 1:
    ops.append(generate_scan(variant.access_patterns[0], bound_vars=[]))
  else:
    ops.extend(_lower_multi_clause_body(variant))

  ops.extend(_lower_negations(variant))
  ops.extend(_lower_filter_and_let_clauses(variant))

  head = variant.original_rule.head
  canonical = stratum.canonical_index.get(head.rel)
  if canonical is None:
    canonical = list(range(len(head.args)))
  ops.append(generate_insert_into(head, list(canonical)))

  return ops


# -----------------------------------------------------------------------------
# Maintenance generators (mirror generateRebuildIndices, generateMergeIndices,
# generateSimpleMaintenance, generateLoopMaintenance in lowering.nim).
# -----------------------------------------------------------------------------

def generate_rebuild_indices(
  rel_name: str, indices: list[list[int]], version: Version
) -> list[mir.MirNode]:
  return [
    mir.RebuildIndex(rel_name=rel_name, version=version, index=list(idx))
    for idx in indices
  ]


def generate_merge_indices(
  rel_name: str, indices: list[list[int]]
) -> list[mir.MirNode]:
  return [
    mir.MergeIndex(rel_name=rel_name, index=list(idx)) for idx in indices
  ]


def generate_simple_maintenance(
  rel_name: str,
  indices: list[list[int]],
  canonical_index: list[int],
  arity: int,
) -> list[mir.MirNode]:
  '''Maintenance for a non-recursive (simple) SCC: build canonical NEW,
  size-check, compute delta, clear NEW, rebuild non-canonical DELTAs,
  merge every index into FULL.
  '''
  assert len(canonical_index) == arity, (
    f"canonical index for {rel_name!r} has {len(canonical_index)} cols, "
    f"expected arity {arity}"
  )
  ops: list[mir.MirNode] = []
  ops.append(mir.RebuildIndex(rel_name=rel_name, version=Version.NEW, index=list(canonical_index)))
  ops.append(mir.CheckSize(rel_name=rel_name, version=Version.NEW))
  ops.append(mir.ComputeDeltaIndex(rel_name=rel_name, canonical_index=list(canonical_index)))
  ops.append(mir.ClearRelation(rel_name=rel_name, version=Version.NEW))
  for idx in indices:
    if list(idx) != list(canonical_index):
      ops.append(mir.RebuildIndexFromIndex(
        rel_name=rel_name,
        source_index=list(canonical_index),
        target_index=list(idx),
        version=Version.DELTA,
      ))
    ops.append(mir.MergeIndex(rel_name=rel_name, index=list(idx)))
  return ops


def generate_loop_maintenance(
  rel_name: str,
  indices: list[list[int]],
  canonical_index: list[int],
  arity: int,
  full_needed: set[tuple[int, ...]] | None = None,
) -> list[mir.MirNode]:
  '''Maintenance at the end of a fixpoint iteration.

  `full_needed` is the set of (completed) indices whose FULL-version is
  actually read by joins in this SCC — only those (plus the canonical
  index) need MergeIndex into FULL. Others just get their DELTA rebuilt.
  '''
  if full_needed is None:
    full_needed = set()
  assert len(canonical_index) == arity, (
    f"canonical index for {rel_name!r} has {len(canonical_index)} cols, "
    f"expected arity {arity}"
  )
  ops: list[mir.MirNode] = []
  ops.append(mir.RebuildIndex(rel_name=rel_name, version=Version.NEW, index=list(canonical_index)))
  ops.append(mir.ClearRelation(rel_name=rel_name, version=Version.DELTA))
  ops.append(mir.CheckSize(rel_name=rel_name, version=Version.NEW))
  ops.append(mir.ComputeDeltaIndex(rel_name=rel_name, canonical_index=list(canonical_index)))
  ops.append(mir.ClearRelation(rel_name=rel_name, version=Version.NEW))
  canon_t = tuple(canonical_index)
  for idx in indices:
    idx_list = list(idx)
    if idx_list != list(canonical_index):
      ops.append(mir.RebuildIndexFromIndex(
        rel_name=rel_name,
        source_index=list(canonical_index),
        target_index=idx_list,
        version=Version.DELTA,
      ))
    idx_t = tuple(idx_list)
    if idx_t == canon_t or idx_t in full_needed:
      ops.append(mir.MergeIndex(rel_name=rel_name, index=idx_list))
  return ops


# -----------------------------------------------------------------------------
# Phase 3: Stratum wrapping (wrapInExecutePipeline + lowerHirToMirSteps +
# lowerHirToMir). Mirrors the top-level pieces of lowering.nim.
# -----------------------------------------------------------------------------

def _extract_pipeline_sources(op: mir.MirNode, out: list[mir.MirNode]) -> None:
  '''Recursively pull source specs out of a pipeline op. Mirrors the
  extractSources inner proc of Nim's wrapInExecutePipeline: joins are
  flattened, leaves are added directly.

  Handles: ColumnSource, Scan, Negation (leaf specs); ColumnJoin,
  CartesianJoin (recurse into their `sources`); BalancedScan (recurse
  into source1 and source2); PositionedExtract (recurse into `sources`).
  Aggregate is deferred.
  '''
  if isinstance(op, (mir.ColumnSource, mir.Scan, mir.Negation, mir.Aggregate)):
    out.append(op)
  elif isinstance(op, mir.ColumnJoin):
    for s in op.sources:
      _extract_pipeline_sources(s, out)
  elif isinstance(op, mir.CartesianJoin):
    for s in op.sources:
      _extract_pipeline_sources(s, out)
  elif isinstance(op, mir.BalancedScan):
    _extract_pipeline_sources(op.source1, out)
    _extract_pipeline_sources(op.source2, out)
  elif isinstance(op, mir.PositionedExtract):
    for s in op.sources:
      _extract_pipeline_sources(s, out)


def wrap_in_execute_pipeline(
  pipeline: list[mir.MirNode],
  clause_order: list[int],
  rule_name: str,
  use_fan_out: bool = False,
  work_stealing: bool = False,
  block_group: bool = False,
  count: bool = False,
  dedup_hash: bool = False,
) -> mir.ExecutePipeline:
  '''Wrap a pipeline body in an ExecutePipeline node, extracting source
  specs (flattened through ColumnJoin/CartesianJoin) and dest specs
  (InsertInto nodes).
  '''
  sources: list[mir.MirNode] = []
  dests: list[mir.MirNode] = []
  for op in pipeline:
    _extract_pipeline_sources(op, sources)
    if isinstance(op, mir.InsertInto):
      dests.append(op)
  return mir.ExecutePipeline(
    pipeline=list(pipeline),
    source_specs=sources,
    dest_specs=dests,
    rule_name=rule_name,
    clause_order=list(clause_order),
    use_fan_out=use_fan_out,
    work_stealing=work_stealing,
    block_group=block_group,
    dedup_hash=dedup_hash,
    count=count,
  )


def _nvtx_rule_name(variant: HirRuleVariant) -> str:
  rn = variant.original_rule.name or ""
  if variant.delta_idx >= 0:
    return f"{rn}_D{variant.delta_idx}"
  return rn


def _collect_full_indices(
  variants: list[HirRuleVariant],
) -> dict[str, set[tuple[int, ...]]]:
  '''rel_name -> set of index_cols tuples accessed as FULL-version.'''
  out: dict[str, set[tuple[int, ...]]] = {}
  for v in variants:
    for pat in v.access_patterns:
      if pat.version is Version.FULL:
        out.setdefault(pat.rel_name, set()).add(tuple(pat.index_cols))
  return out


def _schema_arities(hir: HirProgram) -> list[tuple[str, int]]:
  return [(d.rel_name, len(d.types)) for d in hir.relation_decls]


def lower_hir_to_mir_steps(hir: HirProgram) -> list[tuple[mir.MirNode, bool]]:
  '''Assemble per-stratum FixpointPlan + PostStratumReconstructInternCols
  steps. Returns the flat `[(node, is_recursive)]` sequence consumed by
  `lower_hir_to_mir` (which wraps them in a Program).

  Mirrors lowerHirToMirSteps in lowering.nim. Deliberately not ported
  from Nim (see module docstring): before/after hooks, split-rule
  dispatch, InjectCppHook for debug code, balanced scan.
  '''
  out: list[tuple[mir.MirNode, bool]] = []
  decls = hir.relation_decls

  for stratum in hir.strata:
    if stratum.is_recursive:
      loop_ops: list[mir.MirNode] = []
      parallel_ops: list[mir.MirNode] = []
      split_phase_ops: list[mir.MirNode] = []
      inject_hooks: list[mir.MirNode] = []

      for variant in stratum.recursive_variants:
        nvtx = _nvtx_rule_name(variant)
        if variant.original_rule.debug_code:
          # One hook per variant (matches Nim: for Store rule with 2 delta
          # variants, two inject-cpp-hook nodes are emitted).
          inject_hooks.append(mir.InjectCppHook(
            code=variant.original_rule.debug_code,
            rule_name=variant.original_rule.name or "",
          ))
        if variant.split_at >= 0 and variant.temp_rel_name:
          # Recursive split: ClearRelation temp NEW (per iteration) +
          # Pipeline A + CreateFlatView + Pipeline B consuming temp NEW.
          pipeline_a = lower_split_above(variant, stratum)
          if pipeline_a:
            temp_idx = list(range(len(variant.temp_vars)))
            split_phase_ops.append(mir.ClearRelation(
              rel_name=variant.temp_rel_name, version=Version.NEW,
            ))
            split_phase_ops.append(wrap_in_execute_pipeline(
              pipeline_a, variant.clause_order, nvtx + "_splitA",
            ))
            split_phase_ops.append(mir.CreateFlatView(
              rel_name=variant.temp_rel_name,
              version=Version.NEW,
              index=temp_idx,
            ))
            pipeline_b = lower_split_below(
              variant, stratum, temp_version=Version.NEW,
            )
            split_phase_ops.append(wrap_in_execute_pipeline(
              pipeline_b, variant.clause_order, nvtx + "_splitB",
            ))
          else:
            pipeline = lower_variant_to_pipeline(variant, stratum)
            parallel_ops.append(wrap_in_execute_pipeline(
              pipeline, variant.clause_order, nvtx,
              use_fan_out=variant.fanout,
              work_stealing=variant.work_stealing,
              block_group=variant.block_group,
              count=variant.count,
              dedup_hash=variant.dedup_hash,
            ))
        else:
          pipeline = lower_variant_to_pipeline(variant, stratum)
          parallel_ops.append(wrap_in_execute_pipeline(
            pipeline, variant.clause_order, nvtx,
            use_fan_out=variant.fanout,
            work_stealing=variant.work_stealing,
            block_group=variant.block_group,
            count=variant.count,
            dedup_hash=variant.dedup_hash,
          ))

      # Non-split parallel rules first, then split-phase ops (sequential),
      # then inject-cpp hooks (debug output after pipelines, before maint).
      if len(parallel_ops) > 1:
        loop_ops.append(mir.ParallelGroup(ops=parallel_ops))
      elif len(parallel_ops) == 1:
        loop_ops.append(parallel_ops[0])
      loop_ops.extend(split_phase_ops)
      loop_ops.extend(inject_hooks)

      full_map = _collect_full_indices(stratum.recursive_variants)
      for rel_name in sorted(stratum.scc_members):
        if rel_name in stratum.required_indices:
          canonical_idx = stratum.canonical_index.get(
            rel_name, stratum.required_indices[rel_name][0],
          )
          arity = get_arity(rel_name, decls)
          full_needed: set[tuple[int, ...]] = set()
          for raw_idx in full_map.get(rel_name, set()):
            full_needed.add(tuple(complete_index(list(raw_idx), arity)))
          loop_ops.extend(generate_loop_maintenance(
            rel_name,
            stratum.required_indices[rel_name],
            canonical_idx,
            arity,
            full_needed,
          ))

      if loop_ops:
        out.append((mir.FixpointPlan(
          instructions=loop_ops,
          schema_arities=_schema_arities(hir),
        ), True))
        for rel_name in sorted(stratum.scc_members):
          if rel_name in stratum.canonical_index:
            out.append((mir.PostStratumReconstructInternCols(
              rel_name=rel_name,
              canonical_index=list(stratum.canonical_index[rel_name]),
            ), False))

    else:
      pipeline_ops: list[mir.MirNode] = []
      split_phase_ops: list[mir.MirNode] = []   # split A -> CreateFlatView -> split B
      maintenance_ops: list[mir.MirNode] = []
      modified_rels: list[str] = []  # in variant-appearance order

      for variant in stratum.base_variants:
        nvtx = _nvtx_rule_name(variant)

        if variant.split_at >= 0 and variant.temp_rel_name:
          # Split variant: Pipeline A -> CreateFlatView -> Pipeline B.
          pipeline_a = lower_split_above(variant, stratum)
          if pipeline_a:
            split_phase_ops.append(wrap_in_execute_pipeline(
              pipeline_a,
              variant.clause_order,
              nvtx + "_splitA",
            ))
            temp_idx = list(range(len(variant.temp_vars)))
            split_phase_ops.append(mir.CreateFlatView(
              rel_name=variant.temp_rel_name,
              version=Version.NEW,
              index=temp_idx,
            ))
            pipeline_b = lower_split_below(variant, stratum)
            split_phase_ops.append(wrap_in_execute_pipeline(
              pipeline_b,
              variant.clause_order,
              nvtx + "_splitB",
            ))
          else:
            # Above-split had an unsupported multi-positive shape; fall
            # back to the full unsplit pipeline.
            pipeline = lower_variant_to_pipeline(variant, stratum)
            pipeline_ops.append(wrap_in_execute_pipeline(
              pipeline, variant.clause_order, nvtx,
              use_fan_out=variant.fanout,
              work_stealing=variant.work_stealing,
              block_group=variant.block_group,
              count=variant.count,
              dedup_hash=variant.dedup_hash,
            ))
        else:
          pipeline = lower_variant_to_pipeline(variant, stratum)
          pipeline_ops.append(wrap_in_execute_pipeline(
            pipeline, variant.clause_order, nvtx,
            use_fan_out=variant.fanout,
            work_stealing=variant.work_stealing,
            block_group=variant.block_group,
            count=variant.count,
            dedup_hash=variant.dedup_hash,
          ))

        rel_name = variant.original_rule.head.rel
        if rel_name not in modified_rels:
          modified_rels.append(rel_name)
        if rel_name in stratum.required_indices:
          canonical_idx = stratum.canonical_index.get(
            rel_name, stratum.required_indices[rel_name][0],
          )
          arity = get_arity(rel_name, decls)
          maintenance_ops.extend(generate_simple_maintenance(
            rel_name,
            stratum.required_indices[rel_name],
            canonical_idx,
            arity,
          ))

      ops: list[mir.MirNode] = []
      # Split phase runs first (sequential; depends on temp being populated).
      ops.extend(split_phase_ops)
      if len(pipeline_ops) > 1:
        ops.append(mir.ParallelGroup(ops=pipeline_ops))
      elif len(pipeline_ops) == 1:
        ops.append(pipeline_ops[0])
      ops.extend(maintenance_ops)

      if ops:
        out.append((mir.FixpointPlan(
          instructions=ops,
          schema_arities=_schema_arities(hir),
        ), False))
        for rel_name in modified_rels:
          if rel_name in stratum.canonical_index:
            out.append((mir.PostStratumReconstructInternCols(
              rel_name=rel_name,
              canonical_index=list(stratum.canonical_index[rel_name]),
            ), False))
        # Debug inject-cpp hooks for base variants are separate steps
        # after the FixpointPlan + PostStratumReconstructInternCols.
        for variant in stratum.base_variants:
          if variant.original_rule.debug_code:
            out.append((mir.InjectCppHook(
              code=variant.original_rule.debug_code,
              rule_name=variant.original_rule.name or "",
            ), False))

  return out


def lower_hir_to_mir(hir: HirProgram) -> mir.Program:
  '''Top-level lowering entry: HirProgram -> MIR Program.

  Does NOT run the MIR optimization passes (pre_reconstruct_rebuild,
  clause_order_reorder, etc.) that Nim's compileToMir runs afterwards.
  The Nim-side tool used for golden fixtures also dumps pre-pass MIR,
  so byte-diff lines up.
  '''
  steps = lower_hir_to_mir_steps(hir)
  return mir.Program(steps=[(node, is_rec) for node, is_rec in steps])
