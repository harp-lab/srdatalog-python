'''Recursive pipeline walker for the JIT backend.

Port of src/srdatalog/codegen/target_jit/jit_pipeline.nim.

Two top-level procs:

  jit_nested_pipeline(ops, ctx)
    Dispatches NESTED ops (after the root). Each op kind pre-renders
    the body via a recursive call and passes it to its emitter:

      ColumnJoin     -> pre-register child handles by state key,
                        recurse, jit_nested_column_join
      CartesianJoin  -> set inside_cartesian + cartesian_bound_vars,
                        pre-narrow any following Negation into
                        neg_pre_narrow, detect cartesianAsProduct,
                        recurse, jit_nested_cartesian_join
      Scan           -> recurse, jit_scan
      Negation       -> recurse, jit_negation
      Aggregate      -> recurse, jit_aggregate
      Filter         -> recurse, jit_filter
      ConstantBind   -> recurse, jit_constant_bind
      InsertInto     -> jit_insert_into + recurse
      PositionedExtract -> recurse (with inc_indent if bind_vars), emit

  jit_pipeline(ops, ep_source_specs, ctx)
    TOP-level: emits deduplicated view declarations, pre-registers
    root-CJ child handles when multi-source, sets inside_cartesian for
    Cartesian / BalancedScan roots, then dispatches the first op to
    the root handlers and passes the recursively-rendered rest as body.

Scope (matches Phase B/C baseline):
  * Single-view-per-source (DSAI) only
  * No work-stealing (ws_enabled raises NotImplementedError)
  * No block-group (bg_enabled / bg_histogram_mode raise)
  * No fan-out explore (is_fan_out_explore raises)
  * No tiled-Cartesian dual-body path (raises when detected)
  * Negation pre-narrow is supported (baseline feature, not a flag)
  * cartesianAsProduct short-circuit IS included (counting-only
    straight-line optimization; matches Nim for plain-count rules)

The WS / BG / tiled / fan-out branches will be filled in as the programs
that exercise them come online.
'''
from __future__ import annotations
from typing import Optional

import srdatalog.mir.types as m
from srdatalog.codegen.jit.context import (
  CodeGenContext, ind, inc_indent, gen_unique_name, with_bound_var,
  get_rel_index_type, gen_handle_state_key, gen_index_spec_key,
  gen_view_var_name, NegPreNarrowInfo,
)
from srdatalog.codegen.jit.view_management import (
  collect_unique_view_specs, jit_emit_view_declarations,
)
from srdatalog.codegen.jit.emit_helpers import (
  jit_filter, jit_constant_bind, jit_insert_into,
)
from srdatalog.codegen.jit.scan_negation import jit_scan, jit_negation, jit_aggregate
from srdatalog.codegen.jit.instructions import (
  jit_nested_column_join, jit_nested_cartesian_join, jit_positioned_extract,
)
from srdatalog.codegen.jit.root import (
  jit_root_scan, jit_root_cartesian_join, jit_root_column_join,
)


# -----------------------------------------------------------------------------
# Feature-flag guards shared by the walker
# -----------------------------------------------------------------------------

def _reject_unsupported_flags(ctx: CodeGenContext, where: str) -> None:
  if ctx.ws_enabled:
    raise NotImplementedError(
      f"jit_pipeline.{where}: work-stealing branch not yet ported"
    )
  if ctx.bg_histogram_mode:
    raise NotImplementedError(
      f"jit_pipeline.{where}: block-group histogram branch not yet ported"
    )
  # ctx.bg_enabled is now supported at the root CJ level; nested ops
  # run with bg_enabled=False (handle narrowing already restricted work
  # to this warp's first-level children — no further slicing needed).
  if ctx.is_fan_out_explore:
    raise NotImplementedError(
      f"jit_pipeline.{where}: fan-out explore branch not yet ported"
    )


# -----------------------------------------------------------------------------
# jit_nested_pipeline
# -----------------------------------------------------------------------------

def _register_negation_pre_narrow(
  rest: list[m.MirNode], ctx: CodeGenContext,
) -> None:
  '''Scan `rest` for Negation ops; split each one's prefix vars into
  pre-Cartesian (contiguous leading prefix that are all NOT
  cartesian-bound) and in-Cartesian (the rest). Store into
  ctx.neg_pre_narrow so jit_negation can emit cooperative narrowing
  outside the Cartesian loop and per-thread narrowing inside.
  '''
  cart_bound_set = set(ctx.cartesian_bound_vars)
  for neg_op in rest:
    if not isinstance(neg_op, m.Negation):
      continue

    pre_vars: list[str] = []
    in_vars: list[str] = []
    contiguous = True
    for v in neg_op.prefix_vars:
      if contiguous and v not in cart_bound_set:
        pre_vars.append(v)
      else:
        contiguous = False
        in_vars.append(v)

    if pre_vars or neg_op.const_args:
      neg_spec_key = gen_index_spec_key(
        neg_op.rel_name, list(neg_op.index), neg_op.version.code,
      )
      neg_view_var = ctx.view_vars.get(neg_spec_key, "")
      if not neg_view_var:
        neg_view_var = ctx.view_vars.get(str(neg_op.handle_start), "")
      if not neg_view_var:
        neg_view_var = gen_view_var_name(
          neg_op.rel_name + "_neg", neg_op.handle_start,
        )

      ctx.neg_pre_narrow[neg_op.handle_start] = NegPreNarrowInfo(
        var_name=gen_unique_name(ctx, f"h_{neg_op.rel_name}_neg_pre"),
        pre_vars=pre_vars,
        in_cartesian_vars=in_vars,
        pre_consts=list(neg_op.const_args),
        view_var=neg_view_var,
        rel_name=neg_op.rel_name,
        index_type=get_rel_index_type(ctx, neg_op.rel_name),
      )


def _rest_is_safe_for_cartesian_product(rest: list[m.MirNode]) -> bool:
  '''True iff every op after the Cartesian is just InsertInto — i.e.,
  counting mode can replace the flat loop with add_count(degree product).
  Matches Nim's `safe` check.'''
  return all(isinstance(op, m.InsertInto) for op in rest)


def jit_nested_pipeline(
  ops: list[m.MirNode], ctx: CodeGenContext,
) -> str:
  '''Recursively dispatch NESTED ops. Returns the full body text.

  Empty `ops` closes the tiled-Cartesian ballot block if it's still
  open; otherwise it emits nothing. For every op kind the body is
  rendered first via recursion, then the op's emitter wraps it.
  '''
  if not ops:
    # Close a still-open tiled-Cartesian ballot block (multi-head body).
    if ctx.tiled_cartesian_ballot_done:
      ctx.tiled_cartesian_ballot_done = False
      i = ind(ctx)
      return (
        i + "    warp_local_count += _tc_active;\n"
        + i + "  }\n"      # closes `if (_tc_active > 0) {`
        + i + "}\n"        # closes the outer scope
      )
    return ""

  op = ops[0]
  rest = ops[1:]

  if isinstance(op, m.ColumnJoin):
    _reject_unsupported_flags(ctx, "ColumnJoin")
    # Determine leaf level (no further ColumnJoins ahead) for tile dispatch.
    has_more_cj = any(isinstance(nxt, m.ColumnJoin) for nxt in rest)
    ctx.is_leaf_level = not has_more_cj

    # Pre-register child handles so subsequent ops can find them by state key.
    var_name = op.var_name
    registered_keys: list[str] = []
    for src in op.sources:
      if not isinstance(src, m.ColumnSource):
        continue
      child_prefixes = list(src.prefix_vars) + [var_name]
      state_key = gen_handle_state_key(
        src.rel_name, list(src.index), child_prefixes, src.version.code,
      )
      child_var = f"ch_{src.rel_name}_{src.handle_start}_{var_name}"
      ctx.handle_vars[state_key] = child_var
      registered_keys.append(state_key)

    try:
      body = jit_nested_pipeline(rest, ctx)
      return jit_nested_column_join(op, ctx, body)
    finally:
      for k in registered_keys:
        ctx.handle_vars.pop(k, None)

  if isinstance(op, m.CartesianJoin):
    _reject_unsupported_flags(ctx, "CartesianJoin")
    if ctx.ws_cartesian_valid_var:
      raise NotImplementedError(
        "jit_pipeline.CartesianJoin: ws dual-body path not yet ported"
      )

    # Set Cartesian state BEFORE body renders so nested ops see the flag.
    previous_inside = ctx.inside_cartesian
    previous_bound = list(ctx.cartesian_bound_vars)
    ctx.inside_cartesian = True
    for vars_from_src in op.var_from_source:
      for var_name in vars_from_src:
        ctx.cartesian_bound_vars.append(var_name)

    # Pre-narrow following negations.
    _register_negation_pre_narrow(rest, ctx)

    # Count-as-product short-circuit (matches Nim exactly).
    previous_product_flag = ctx.cartesian_as_product
    if (
      ctx.is_counting
      and not ctx.neg_pre_narrow
      and not ctx.ws_enabled
      and not ctx.is_fan_out_explore
      and not ctx.bg_histogram_mode
      and not ctx.dedup_hash_enabled
      and _rest_is_safe_for_cartesian_product(rest)
    ):
      ctx.cartesian_as_product = True

    # Tiled Cartesian dual-body: when tiled is enabled + shape-eligible
    # + not counting, render two versions of the body — one with
    # tiled_cartesian_valid_var set (ballot-path coalesced writes) and
    # one without (standard emit_direct). jit_nested_cartesian_join
    # emits both inside its `if (total > 32) {...} else {...}` dispatch.
    tiled_eligible = (
      not ctx.ws_enabled
      and ctx.tiled_cartesian_enabled
      and not ctx.is_counting
      and len(op.sources) == 2
      and len(op.var_from_source) == 2
      and len(op.var_from_source[0]) == 1
      and len(op.var_from_source[1]) == 1
    )
    if tiled_eligible:
      # Mirror Nim's value-semantic dual-body in a reference-semantic
      # language. Nim does:
      #   var tiledCtx = ctx                        # counter N
      #   tiledCtx.valid = ctx.genUniqueName(...)   # ctx: N→N+1, tiledCtx stays N
      #   render(tiledCtx)                          # bumps tiledCtx only
      #   var fallbackCtx = ctx                     # counter N+1 (post tc_valid)
      #   render(fallbackCtx)                       # bumps fallbackCtx only
      #   ctx.valid = tiledCtx.valid                # transfer name
      # Key: tiled body sees counter=N (pre tc_valid), fallback sees
      # counter=N+1 (post tc_valid), and neither body's bumps persist.
      saved_tcvv = ctx.tiled_cartesian_valid_var
      saved_tce = ctx.tiled_cartesian_enabled
      saved_name_counter = ctx.name_counter
      tc_valid_name = f"tc_valid_{saved_name_counter + 1}"
      # Render tiled body with counter reset to saved (pre tc_valid bump).
      ctx.tiled_cartesian_valid_var = tc_valid_name
      try:
        tiled_body = jit_nested_pipeline(rest, ctx)
      finally:
        ctx.tiled_cartesian_valid_var = saved_tcvv
      # Render fallback body with counter advanced past tc_valid.
      ctx.tiled_cartesian_enabled = False
      ctx.tiled_cartesian_valid_var = ""
      ctx.name_counter = saved_name_counter + 1
      try:
        fallback_body = jit_nested_pipeline(rest, ctx)
      finally:
        ctx.tiled_cartesian_enabled = saved_tce
      # Restore to Nim's post-dual-body state: only the tc_valid bump
      # persists, body bumps discarded.
      ctx.name_counter = saved_name_counter + 1
      ctx.tiled_cartesian_valid_var = tc_valid_name
      try:
        return jit_nested_cartesian_join(
          op, ctx, fallback_body, tiled_body=tiled_body,
        )
      finally:
        ctx.inside_cartesian = previous_inside
        ctx.cartesian_bound_vars = previous_bound
        ctx.cartesian_as_product = previous_product_flag

    try:
      body = jit_nested_pipeline(rest, ctx)
      return jit_nested_cartesian_join(op, ctx, body)
    finally:
      ctx.inside_cartesian = previous_inside
      ctx.cartesian_bound_vars = previous_bound
      ctx.cartesian_as_product = previous_product_flag

  if isinstance(op, m.Scan):
    body = jit_nested_pipeline(rest, ctx)
    return jit_scan(op, ctx, body)

  if isinstance(op, m.Negation):
    body = jit_nested_pipeline(rest, ctx)
    return jit_negation(op, ctx, body)

  if isinstance(op, m.Aggregate):
    body = jit_nested_pipeline(rest, ctx)
    return jit_aggregate(op, ctx, body)

  if isinstance(op, m.Filter):
    body = jit_nested_pipeline(rest, ctx)
    return jit_filter(op, ctx, body)

  if isinstance(op, m.ConstantBind):
    body = jit_nested_pipeline(rest, ctx)
    return jit_constant_bind(op, ctx, body)

  if isinstance(op, m.InsertInto):
    return jit_insert_into(op, ctx) + jit_nested_pipeline(rest, ctx)

  if isinstance(op, m.PositionedExtract):
    # If bind_vars is non-empty, body sits inside an extra for-loop
    # (per Nim), so bump indent for the recursive body emit.
    if op.bind_vars:
      inc_indent(ctx)
      try:
        body = jit_nested_pipeline(rest, ctx)
      finally:
        from srdatalog.codegen.jit.context import dec_indent
        dec_indent(ctx)
    else:
      body = jit_nested_pipeline(rest, ctx)
    return jit_positioned_extract(op, ctx, body)

  return f"// Unsupported nested op: {type(op).__name__}\n"


# -----------------------------------------------------------------------------
# jit_pipeline (top-level)
# -----------------------------------------------------------------------------

def jit_pipeline(
  ops: list[m.MirNode],
  ep_source_specs: list[m.MirNode],
  ctx: CodeGenContext,
) -> str:
  '''Top-level pipeline emit. Emits dedup view declarations, sets up
  root-specific context (pre-register multi-source CJ handles, toggle
  inside_cartesian for Cartesian / BalancedScan roots), then dispatches
  the first op to the root handlers with the rest rendered recursively.
  '''
  if not ops:
    return ""

  code = ""

  # Deduplicated view declarations.
  view_specs = collect_unique_view_specs(ops)
  code += jit_emit_view_declarations(view_specs, ops, ep_source_specs, ctx)

  first_op = ops[0]
  rest_ops = ops[1:]

  # Multi-source root ColumnJoin: pre-register its narrowed handles so
  # nested ops can resolve by semantic key.
  if isinstance(first_op, m.ColumnJoin) and len(first_op.sources) > 1:
    root_var_name = first_op.var_name
    for src in first_op.sources:
      if not isinstance(src, m.ColumnSource):
        continue
      state_key = gen_handle_state_key(
        src.rel_name, list(src.index), [root_var_name], src.version.code,
      )
      handle_var = f"h_{src.rel_name}_{src.handle_start}_root"
      ctx.handle_vars[state_key] = handle_var

  # Cartesian / BalancedScan roots: set the inside-cartesian flag +
  # extend cartesian_bound_vars so nested code sees per-thread bindings.
  if isinstance(first_op, m.CartesianJoin):
    ctx.inside_cartesian = True
    for vars_from_src in first_op.var_from_source:
      for var_name in vars_from_src:
        ctx.cartesian_bound_vars.append(var_name)
  elif isinstance(first_op, m.BalancedScan):
    ctx.inside_cartesian = True
    ctx.cartesian_bound_vars.append(first_op.group_var)
    for v in first_op.vars1:
      ctx.cartesian_bound_vars.append(v)
    for v in first_op.vars2:
      ctx.cartesian_bound_vars.append(v)

  # Body renders with non-BG context for the inner pipeline (BG only
  # matters at the root dispatch level).
  #
  # Critical: Nim uses `var bodyCtx = ctx` which snapshots name_counter;
  # body's gen_unique_name bumps don't leak back to ctx, so root's
  # subsequent unique names start from the original counter. We mimic
  # that by saving/restoring name_counter across the body emit.
  # Without this, Python produces e.g. `y_idx_9` where Nim emits
  # `y_idx_1` because our shared counter keeps climbing through the
  # body's handle/iterator name generation.
  #
  # Also clear bg_enabled during body render: BG partitioning happens
  # at the root CJ level, and the narrowed handle it produces already
  # restricts work to this warp's slice — the nested pipeline runs
  # normally (matches Nim's `newCtx.bgEnabled = false` before body).
  saved_name_counter = ctx.name_counter
  saved_bg_enabled = ctx.bg_enabled
  ctx.bg_enabled = False
  try:
    body = jit_nested_pipeline(rest_ops, ctx)
  finally:
    ctx.bg_enabled = saved_bg_enabled
  ctx.name_counter = saved_name_counter

  # Dispatch first op to the root handlers.
  if isinstance(first_op, m.ColumnJoin):
    if ctx.bg_enabled:
      from srdatalog.codegen.jit.root import jit_root_column_join_block_group
      code += jit_root_column_join_block_group(first_op, ctx, body)
    else:
      code += jit_root_column_join(first_op, ctx, body)
  elif isinstance(first_op, m.CartesianJoin):
    code += jit_root_cartesian_join(first_op, ctx, body)
  elif isinstance(first_op, m.Scan):
    code += jit_root_scan(first_op, ctx, body)
  elif isinstance(first_op, m.Negation):
    code += jit_negation(first_op, ctx, body)
  elif isinstance(first_op, m.Aggregate):
    code += jit_aggregate(first_op, ctx, body)
  elif isinstance(first_op, m.Filter):
    code += jit_filter(first_op, ctx, body)
  elif isinstance(first_op, m.ConstantBind):
    code += jit_constant_bind(first_op, ctx, body)
  elif isinstance(first_op, m.InsertInto):
    code += jit_insert_into(first_op, ctx) + jit_nested_pipeline(rest_ops, ctx)
  elif isinstance(first_op, m.BalancedScan):
    raise NotImplementedError(
      "jit_pipeline: jit_root_balanced_scan not yet ported"
    )
  else:
    code += f"// Unsupported root op: {type(first_op).__name__}\n"

  return code
