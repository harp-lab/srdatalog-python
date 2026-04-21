'''Nested-op body emitters for the JIT backend.

Port of src/srdatalog/codegen/target_jit/jit_instructions.nim.

These handle SUBSEQUENT operations in a pipeline — after a root op
has set up initial iteration. The handles they operate on are already
narrowed by earlier joins (prefix vars captured at kernel start).

Three top-level emitters here:
  jit_nested_column_join      — intersect / iterate on prefixed handles
  jit_nested_cartesian_join   — Cartesian product over bound vars
  jit_positioned_extract      — balanced-scan follow-up, point lookups

**Scope note**: this first port covers the baseline (non-WS, non-BG,
single-view-per-source DSAI) paths — every current integration fixture
uses only these. The feature-flag branches (work-stealing task donation,
block-group cumulative work tracking, Device2LevelIndex multi-view
segment loops, dedup-hash, fan-out explore) raise NotImplementedError
and will be filled in as the programs that exercise them come online.
'''
from __future__ import annotations
from typing import Optional

import srdatalog.mir.types as m
from srdatalog.codegen.jit.context import (
  CodeGenContext, ind, inc_indent, dec_indent, gen_unique_name,
  with_bound_var, sanitize_var_name, get_view_slot_base, get_rel_index_type,
  gen_view_access, gen_view_var_name, gen_handle_var_name,
  gen_handle_state_key, gen_root_handle, gen_valid, gen_degree,
  gen_get_value_at, gen_child, gen_child_range, gen_iterators,
  gen_chained_prefix_calls,
)
from srdatalog.codegen.jit.plugin import plugin_view_count


# -----------------------------------------------------------------------------
# jit_nested_column_join
# -----------------------------------------------------------------------------

def _multi_view_present(source: m.ColumnSource, ctx: CodeGenContext) -> bool:
  '''True iff this source uses an index plugin that contributes >1 view
  slot (e.g., Device2LevelIndex). DSAI always returns 1.'''
  index_type = get_rel_index_type(ctx, source.rel_name)
  return plugin_view_count(source.version.code, index_type) > 1


def _feature_flags_disabled(ctx: CodeGenContext) -> None:
  '''Guard baseline-only paths. Raises NotImplementedError when the
  caller's context has one of the advanced feature flags set.'''
  if ctx.ws_enabled:
    raise NotImplementedError(
      "jit_instructions: work-stealing ColumnJoin branch not yet ported"
    )
  if ctx.bg_enabled or ctx.bg_histogram_mode:
    # bg_enabled must be cleared by the BG root CJ before descending
    # into nested ops (handle narrowing already restricted work to this
    # warp's slice). Catch leaks here with a helpful message.
    raise NotImplementedError(
      "jit_instructions: block-group ColumnJoin branch not yet ported "
      "(bg_enabled should be cleared by the BG root CJ before nested emit)"
    )
  if ctx.is_fan_out_explore:
    raise NotImplementedError(
      "jit_instructions: fan-out explore branch not yet ported"
    )


def jit_nested_column_join(
  node: m.ColumnJoin, ctx: CodeGenContext, body: str,
) -> str:
  '''Emit a nested ColumnJoin. Single source — iterate the prefixed
  handle; multi-source — emit `intersect_handles(...)` over each
  source's iterators and iterate that, binding child handles for
  subsequent joins.

  Multi-view sources (e.g. Device2LevelIndex where `view_count > 1`
  for FULL reads) trigger segment-loop wrapping that iterates over
  each sorted array. For the single-source path this is a simple
  `for (_seg = 0; _seg < view_count; _seg++)` around the body. For
  the multi-source path segment loops nest only around "fresh" sources
  (prefix_vars empty); prefixed sources stay within the parent's
  segment since child_range from a narrowed handle doesn't cross
  segment boundaries.
  '''
  assert isinstance(node, m.ColumnJoin)
  _feature_flags_disabled(ctx)

  num_sources = len(node.sources)
  code = ""
  if num_sources == 1:
    return code + _nested_column_join_single(node, ctx, body)
  return code + _nested_column_join_multi(node, ctx, body)


def _nested_column_join_single(
  node: m.ColumnJoin, ctx: CodeGenContext, body: str,
) -> str:
  '''Single-source ColumnJoin: iterate the narrowed handle, bind the
  join var, descend into body. When the source is multi-view
  (Device2LevelIndex with FULL version → view_count == 2), wrap the
  whole body in a `for (_seg = 0; _seg < view_count; _seg++)` loop
  that rebinds view_var per segment.
  '''
  i = ind(ctx)
  src = node.sources[0]
  assert isinstance(src, m.ColumnSource)
  src_idx = src.handle_start
  rel_name = src.rel_name
  var_name = node.var_name

  index_type = get_rel_index_type(ctx, rel_name)
  view_count = plugin_view_count(src.version.code, index_type)
  base_slot = get_view_slot_base(ctx, src_idx)
  has_segment_loop = view_count > 1

  code = ""
  if ctx.debug:
    code += (
      i + f"// Nested ColumnJoin (single): bind '{var_name}' from {rel_name}\n"
    )
    code += (
      i + f"// MIR: (column-join :var {var_name} :sources (({rel_name}"
      f" :handle {src_idx} :prefix ({' '.join(src.prefix_vars)}))))\n"
    )

  if has_segment_loop:
    code += i + f"for (int _seg = 0; _seg < {view_count}; _seg++) {{\n"
  si = i + "  " if has_segment_loop else i

  view_var = gen_view_var_name(rel_name, src_idx)
  if has_segment_loop:
    # Multi-view: always re-declare view_var per segment (different
    # sorted array each iteration).
    code += si + f"auto {view_var} = views[{base_slot} + _seg];\n"
  else:
    existing_view = ctx.view_vars.get(str(src_idx), "")
    if existing_view:
      if existing_view != view_var:
        code += si + f"auto& {view_var} = {existing_view};\n"
    else:
      code += si + f"auto {view_var} = {gen_view_access(base_slot)};\n"

  handle_var = gen_handle_var_name(rel_name, src_idx, ctx)

  if src.prefix_vars:
    parent = ctx.handle_vars.get(str(src_idx), "")
    parent_handle = parent if parent else gen_root_handle(view_var, index_type)
    chained = gen_chained_prefix_calls(
      parent_handle, src.prefix_vars, view_var, index_type=index_type,
    )
    code += si + f"auto {handle_var} = {chained};\n"
  else:
    code += si + f"auto {handle_var} = {gen_root_handle(view_var, index_type)};\n"

  code += si + f"if (!{gen_valid(handle_var, index_type)}) continue;\n"

  degree_var = gen_unique_name(ctx, "degree")
  code += si + f"uint32_t {degree_var} = {gen_degree(handle_var, index_type)};\n\n"

  idx_var = gen_unique_name(ctx, "idx")
  code += (
    si + f"for (uint32_t {idx_var} = 0; {idx_var} < {degree_var}; ++{idx_var}) {{\n"
  )

  inc_indent(ctx)
  if has_segment_loop:
    inc_indent(ctx)
  ii = ind(ctx)
  try:
    # Skip the value fetch when counting and the var isn't referenced in body.
    if not (ctx.is_counting and var_name not in body):
      code += (
        ii + f"auto {sanitize_var_name(var_name)} = "
        f"{gen_get_value_at(handle_var, view_var, idx_var, index_type)};\n"
      )

    # Deterministic child name — Cartesian sources look up by this name
    child_var = f"ch_{rel_name}_{src_idx}_{sanitize_var_name(var_name)}"
    code += (
      ii + f"auto {child_var} = {gen_child(handle_var, idx_var, index_type)};\n"
    )
    # Register BOTH semantic keys + numeric key so subsequent ops can resolve.
    child_prefixes = list(src.prefix_vars) + [var_name]
    child_state_key = gen_handle_state_key(
      rel_name, list(src.index), child_prefixes, src.version.code,
    )
    ctx.handle_vars[child_state_key] = child_var
    ctx.handle_vars[str(src_idx)] = child_var
    ctx.bound_vars.append(var_name)

    try:
      code += body
    finally:
      if ctx.bound_vars and ctx.bound_vars[-1] == var_name:
        ctx.bound_vars.pop()
      # Clean up handle_vars mutation to keep ctx re-entrant.
      ctx.handle_vars.pop(child_state_key, None)
      # Numeric key may overlap with a kernel-start registration; only clean
      # up if we wrote over an existing entry (which we did above).
      if ctx.handle_vars.get(str(src_idx)) == child_var:
        del ctx.handle_vars[str(src_idx)]
  finally:
    dec_indent(ctx)
    if has_segment_loop:
      dec_indent(ctx)

  code += si + "}\n"
  if has_segment_loop:
    code += i + "}\n"
  return code


def _nested_column_join_multi(
  node: m.ColumnJoin, ctx: CodeGenContext, body: str,
) -> str:
  '''Multi-source ColumnJoin: set up per-source handles, intersect
  their iterators, iterate the result, register child handles in ctx,
  then descend into body.'''
  i = ind(ctx)
  sources = node.sources
  var_name = node.var_name

  code = ""
  if ctx.debug:
    code += (
      i + f"// Nested ColumnJoin (intersection): bind '{var_name}'"
      f" from {len(sources)} sources\n"
    )
    src_debug = ""
    for src in sources:
      src_debug += (
        f"({src.rel_name} :handle {src.handle_start}"
        f" :prefix ({' '.join(src.prefix_vars)})) "
      )
    code += i + f"// MIR: (column-join :var {var_name} :sources ({src_debug}))\n"

  # Multi-view segment loops for non-prefixed multi-view sources.
  # Sources with prefix_vars descend from a narrowed parent handle that's
  # already pinned to one segment (child_range doesn't cross segment
  # boundaries), so they don't get their own segment loop. Fresh roots
  # (prefix_vars = []) with view_count > 1 do.
  nested_segments: list[dict] = []
  for idx_, src in enumerate(sources):
    if not isinstance(src, m.ColumnSource):
      continue
    idx_type = get_rel_index_type(ctx, src.rel_name)
    vc = plugin_view_count(src.version.code, idx_type)
    if vc > 1 and not src.prefix_vars:
      bs = get_view_slot_base(ctx, src.handle_start)
      sv = f"_nseg_{idx_}"
      fvv = (
        f"view_{src.rel_name}_"
        + "_".join(str(c) for c in src.index)
        + (f"_{src.version.code}" if src.version.code else "")
      )
      nested_segments.append({
        "src_idx_pos": idx_,
        "view_count": vc,
        "seg_var": sv,
        "base_slot": bs,
        "fixed_view_var": fvv,
      })

  # Open segment loops for multi-view non-root sources. Each one bumps
  # the effective emit indent by one level.
  seg_indent = i
  for seg in nested_segments:
    sv = seg["seg_var"]
    vc = seg["view_count"]
    code += (
      seg_indent + f"for (int {sv} = 0; {sv} < {vc}; {sv}++) {{\n"
    )
    seg_indent += "  "
    # Reassign the fixed view variable to the current segment so all
    # nested code referencing it sees the current segment, not just FULL.
    code += (
      seg_indent + f"{seg['fixed_view_var']} = views[{seg['base_slot']} + {sv}];\n"
    )

  # Rest of emit uses seg_indent as the effective `i`.
  i = seg_indent

  iterator_args: list[str] = []
  handle_var_names: list[str] = []
  view_var_names: list[str] = []
  src_indices: list[int] = []
  src_rel_names: list[str] = []
  src_index_specs: list[list[int]] = []
  src_prefix_vars: list[list[str]] = []
  src_versions: list[str] = []

  for src in sources:
    assert isinstance(src, m.ColumnSource)
    src_idx = src.handle_start
    rel_name = src.rel_name
    src_indices.append(src_idx)
    src_rel_names.append(rel_name)
    src_index_specs.append(list(src.index))
    src_prefix_vars.append(list(src.prefix_vars))
    src_versions.append(src.version.code)

    handle_var = gen_handle_var_name(rel_name, src_idx, ctx)
    existing_view = ctx.view_vars.get(str(src_idx), "")
    view_var = existing_view if existing_view else gen_view_var_name(rel_name, src_idx)
    if not existing_view:
      code += (
        i + f"auto {view_var} = "
        f"{gen_view_access(get_view_slot_base(ctx, src_idx))};\n"
      )

    index_type = get_rel_index_type(ctx, rel_name)
    if src.prefix_vars:
      # Check for a fully-prefixed existing handle first (exact match).
      full_state_key = gen_handle_state_key(
        rel_name, list(src.index), list(src.prefix_vars), src.version.code,
      )
      existing_handle = ctx.handle_vars.get(full_state_key, "")
      if existing_handle:
        code += i + f"auto {handle_var} = {existing_handle};\n"
      else:
        # Try parent with prefixes[:-1].
        parent_prefixes = list(src.prefix_vars[:-1])
        parent_state_key = gen_handle_state_key(
          rel_name, list(src.index), parent_prefixes, src.version.code,
        )
        parent_handle = ctx.handle_vars.get(parent_state_key, "")
        if parent_handle:
          remaining = [src.prefix_vars[-1]]
          chained = gen_chained_prefix_calls(
            parent_handle, remaining, view_var, index_type=index_type,
          )
          code += i + f"auto {handle_var} = {chained};\n"
        else:
          # No parent — apply all prefixes from root.
          chained = gen_chained_prefix_calls(
            gen_root_handle(view_var, index_type),
            list(src.prefix_vars), view_var, index_type=index_type,
          )
          code += i + f"auto {handle_var} = {chained};\n"
    else:
      # No prefix vars — prefer pre-registered handle.
      empty_state_key = gen_handle_state_key(
        rel_name, list(src.index), [], src.version.code,
      )
      pre_registered = ctx.handle_vars.get(empty_state_key, "")
      if not pre_registered:
        by_idx = ctx.handle_vars.get(str(src_idx), "")
        if by_idx:
          code += i + f"auto {handle_var} = {by_idx};\n"
        else:
          code += (
            i + f"auto {handle_var} = "
            f"{gen_root_handle(view_var, index_type)};\n"
          )
      else:
        code += i + f"auto {handle_var} = {pre_registered};\n"

    iterator_args.append(gen_iterators(handle_var, view_var, index_type))
    handle_var_names.append(handle_var)
    view_var_names.append(view_var)

  intersect_var = gen_unique_name(ctx, "intersect")
  code += (
    i + f"auto {intersect_var} = intersect_handles("
    f"{ctx.tile_var}, {', '.join(iterator_args)});\n"
  )

  it_var = gen_unique_name(ctx, "it")
  code += (
    i + f"for (auto {it_var} = {intersect_var}.begin(); "
    f"{it_var}.valid(); {it_var}.next()) {{\n"
  )

  inc_indent(ctx)
  ii = ind(ctx)
  try:
    code += ii + f"auto {var_name} = {it_var}.value();\n"
    code += ii + f"auto positions = {it_var}.positions();\n"

    # Register every child handle in ctx before descending into body.
    registered_state_keys: list[str] = []
    registered_numeric: list[int] = []
    for idx_, src in enumerate(sources):
      assert isinstance(src, m.ColumnSource)
      src_idx = src_indices[idx_]
      rel_name = src_rel_names[idx_]
      child_var = f"ch_{rel_name}_{src_idx}_{var_name}"
      src_index_type = get_rel_index_type(ctx, rel_name)
      code += (
        ii + f"auto {child_var} = "
        f"{gen_child_range(handle_var_names[idx_], f'positions[{idx_}]', var_name, ctx.tile_var, view_var_names[idx_], src_index_type)};\n"
      )
      child_prefixes = list(src_prefix_vars[idx_]) + [var_name]
      child_state_key = gen_handle_state_key(
        rel_name, src_index_specs[idx_], child_prefixes, src_versions[idx_],
      )
      ctx.handle_vars[child_state_key] = child_var
      ctx.handle_vars[str(src_idx)] = child_var
      registered_state_keys.append(child_state_key)
      registered_numeric.append(src_idx)

    ctx.bound_vars.append(var_name)
    try:
      code += body
    finally:
      if ctx.bound_vars and ctx.bound_vars[-1] == var_name:
        ctx.bound_vars.pop()
      for k in registered_state_keys:
        ctx.handle_vars.pop(k, None)
      for k_ in registered_numeric:
        # Only remove if we're the owner — a re-entrant outer caller may
        # have registered this numeric key before us.
        ctx.handle_vars.pop(str(k_), None)
  finally:
    dec_indent(ctx)

  code += i + "}\n"

  # Close segment loops for multi-view non-prefixed sources (reverse
  # order to undo nesting).
  close_indent = i
  for _ in range(len(nested_segments)):
    close_indent = close_indent[:-2]
    code += close_indent + "}\n"
  return code


# -----------------------------------------------------------------------------
# jit_nested_cartesian_join (baseline: no WS, no BG, no tiled, no fan-out)
# -----------------------------------------------------------------------------

def jit_nested_cartesian_join(
  node: m.CartesianJoin, ctx: CodeGenContext, body: str,
  tiled_body: str = "",
) -> str:
  '''Nested CartesianJoin: Cartesian product over prefixed handles.

  Emits (matching Nim's jitNestedCartesianJoin baseline path):
    - MIR debug comments
    - `uint32_t lane_N = tile.thread_rank();`
      `uint32_t group_size_N = tile.size();`
    - Per-source view + narrowed handle with state-key reuse.
    - Per-source degree; combined validity `continue` check.
    - `uint32_t total = deg0 * deg1 ...;` then `if (total == 0) continue;`
    - Pre-narrow negation handles from ctx.neg_pre_narrow.
    - Dispatch based on ctx flags + source shape:
        * tiled Cartesian (2 sources, 1 var each, not counting,
          tiled_cartesian_enabled) — smem pre-load tile loop when
          total > 32, fallback otherwise; when tiled_body is non-empty
          the tiled path uses ballot-based coalesced writes
        * standard parallel flat loop with `major_is_1` 2-source
          decomposition (or countdown remainder for N≥3)
    - Per-var binding via `view.get_value(col, handle.begin() + idx)`
    - Body

  `tiled_body` is the pipeline-level pre-rendered body with
  `tiled_cartesian_valid_var` set on the context — used inside the
  tiled path for warp-coalesced ballot writes. When empty, the tiled
  path uses the regular `body` without the ballot machinery.

  WS / BG / fan-out / dedup branches still raise NotImplementedError.
  '''
  assert isinstance(node, m.CartesianJoin)
  _feature_flags_disabled(ctx)
  if ctx.ws_cartesian_valid_var:
    raise NotImplementedError(
      "jit_instructions: ws Cartesian batch branch not yet ported"
    )

  sources = node.sources
  vars_bound = list(node.vars)
  var_from_source = node.var_from_source
  num_sources = len(sources)

  code = ""
  i = ind(ctx)

  if ctx.debug:
    code += (
      i + f"// Nested CartesianJoin: bind {', '.join(vars_bound)}"
      f" from {num_sources} source(s)\n"
    )
    src_debug = ""
    for src in sources:
      src_debug += (
        f"({src.rel_name} :handle {src.handle_start}"
        f" :prefix ({' '.join(src.prefix_vars)})) "
      )
    code += (
      i + f"// MIR: (cartesian-join :vars ({' '.join(vars_bound)})"
      f" :sources ({src_debug}))\n"
    )

  # Note: Cartesian doesn't open its own segment loop — when an outer
  # ColumnJoin has a multi-view source its `_seg` loop is already in
  # scope, and the narrowed parent handles we consume already point at
  # the current segment's view. Child-range lookups from a prefixed
  # handle stay within that segment. So no guard needed.

  # lane + group_size preamble.
  lane_var = gen_unique_name(ctx, "lane")
  group_size_var = gen_unique_name(ctx, "group_size")
  code += i + f"uint32_t {lane_var} = {ctx.tile_var}.thread_rank();\n"
  code += i + f"uint32_t {group_size_var} = {ctx.tile_var}.size();\n\n"

  handle_var_names: list[str] = []
  view_var_names: list[str] = []
  degree_var_names: list[str] = []

  for idx_, src in enumerate(sources):
    assert isinstance(src, m.ColumnSource)
    h_idx = src.handle_start
    rel_name = src.rel_name

    # Degree var is declared first (name used later for assignment).
    degree_var = gen_unique_name(ctx, "degree")

    # Handle state-key reuse: match the full prefix first.
    state_key = gen_handle_state_key(
      rel_name, list(src.index), list(src.prefix_vars), src.version.code,
    )
    existing_handle = ctx.handle_vars.get(state_key, "")

    index_type = get_rel_index_type(ctx, rel_name)

    # View lookup by spec-key or numeric.
    from srdatalog.codegen.jit.context import gen_index_spec_key
    spec_key = gen_index_spec_key(rel_name, list(src.index), src.version.code)
    existing_view = ctx.view_vars.get(spec_key, "")
    if not existing_view:
      existing_view = ctx.view_vars.get(str(h_idx), "")

    view_var = existing_view if existing_view else gen_view_var_name(rel_name, h_idx)
    handle_var = gen_handle_var_name(rel_name, h_idx, ctx)

    if not existing_view:
      code += (
        i + f"auto {view_var} = "
        f"{gen_view_access(get_view_slot_base(ctx, h_idx))};  // {rel_name}\n"
      )

    if existing_handle:
      code += (
        i + f"auto {handle_var} = {existing_handle};"
        "  // reusing narrowed handle\n"
      )
    elif src.prefix_vars:
      # No exact match — try parent with prefixes[:-1].
      parent_prefixes = list(src.prefix_vars[:-1])
      parent_state_key = gen_handle_state_key(
        rel_name, list(src.index), parent_prefixes, src.version.code,
      )
      parent_handle = ctx.handle_vars.get(parent_state_key, "")
      if parent_handle:
        remaining = [src.prefix_vars[-1]]
        chained = gen_chained_prefix_calls(
          parent_handle, remaining, view_var, index_type=index_type,
        )
        code += i + f"auto {handle_var} = {chained};\n"
      else:
        chained = gen_chained_prefix_calls(
          gen_root_handle(view_var, index_type),
          list(src.prefix_vars), view_var, index_type=index_type,
        )
        code += i + f"auto {handle_var} = {chained};\n"
    else:
      code += (
        i + f"auto {handle_var} = {gen_root_handle(view_var, index_type)};\n"
      )

    handle_var_names.append(handle_var)
    view_var_names.append(view_var)
    degree_var_names.append(degree_var)

  code += "\n"

  # Combined validity check.
  validity_checks: list[str] = []
  for idx_, src in enumerate(sources):
    src_index_type = get_rel_index_type(ctx, src.rel_name)
    validity_checks.append(f"!{gen_valid(handle_var_names[idx_], src_index_type)}")
  code += i + f"if ({' || '.join(validity_checks)}) continue;\n\n"

  # Degrees.
  for idx_, src in enumerate(sources):
    src_index_type = get_rel_index_type(ctx, src.rel_name)
    code += (
      i + f"uint32_t {degree_var_names[idx_]} = "
      f"{gen_degree(handle_var_names[idx_], src_index_type)};\n"
    )

  total_var = gen_unique_name(ctx, "total")
  code += i + f"uint32_t {total_var} = {' * '.join(degree_var_names)};\n"
  code += i + f"if ({total_var} == 0) continue;\n\n"

  # Count-as-product short-circuit: in counting mode with a flat rest
  # (pure emit, no filters/negations/joins), the per-thread Cartesian
  # loop is replaced by a closed-form lane-share `add_count`. Skips
  # `flat_idx`, `idx0`, `idx1`, `major_is_1` allocations — matches
  # Nim's `cartesianAsProduct` branch. BG has a separate variant that
  # tracks `bg_cumulative` / `bg_local_{begin,end}`.
  if ctx.is_counting and ctx.cartesian_as_product:
    total_expr = " * (uint64_t)".join(degree_var_names)
    if ctx.bg_enabled and ctx.bg_cumulative_var:
      raise NotImplementedError(
        "jit_nested_cartesian_join: BG count-as-product not yet ported"
      )
    code += i + "// Count-as-product: per-lane share without inner loop\n"
    code += i + "{\n"
    nbII = i + "  "
    code += nbII + f"uint64_t cap_total = (uint64_t){total_expr};\n"
    code += (
      nbII + "uint32_t lane_total = static_cast<uint32_t>(cap_total);\n"
    )
    code += (
      nbII + f"uint32_t lane_share = ({lane_var} < lane_total) ? ("
      f"(lane_total - {lane_var} + {group_size_var} - 1) / "
      f"{group_size_var}) : 0;\n"
    )
    code += nbII + f"{ctx.output_var_name}.add_count(lane_share);\n"
    code += i + "}\n"
    return code

  # Pre-narrow negation handles cooperatively (before Cartesian loop).
  # For each negation that follows this Cartesian, pre-apply the prefix
  # vars that are NOT Cartesian-bound — they're constant within the
  # Cartesian loop and should use cooperative .prefix() once instead
  # of per-thread .prefix_seq() on every iteration. Matches Nim's
  # jitNestedCartesianJoin pre-narrow block.
  for handle_start, info in ctx.neg_pre_narrow.items():
    if ctx.debug:
      code += (
        i + f"// Pre-narrow negation handle for {info.rel_name}"
        f" (pre-Cartesian vars: {', '.join(info.pre_vars)})\n"
      )
    current_handle = gen_root_handle(info.view_var, info.index_type)
    # Const prefixes first (HIR indexCols: const cols before var cols).
    for col_idx, const_val in info.pre_consts:
      const_var = gen_unique_name(ctx, f"h_{info.rel_name}_neg_pre_const")
      code += (
        i + f"auto {const_var} = {current_handle}.prefix("
        f"{const_val}, tile, {info.view_var});\n"
      )
      current_handle = const_var
    # Pre-Cartesian variable prefixes (cooperative via empty
    # cartesian_bound_vars — all use .prefix(), not .prefix_seq()).
    if info.pre_vars:
      chained = gen_chained_prefix_calls(
        current_handle, info.pre_vars, info.view_var,
        index_type=info.index_type,
      )
      code += i + f"auto {info.var_name} = {chained};\n"
    else:
      code += i + f"auto {info.var_name} = {current_handle};\n"
    code += "\n"

  # Tiled Cartesian dual-body: when enabled + shape-eligible, emit
  # `if (total > 32) { tiled smem loop } else { fallback }` and skip
  # the standard flat-loop/decomposition path emitted below. Matches
  # Nim's tiledCartesianEnabled branch in jitNestedCartesianJoin.
  tiled_eligible = (
    ctx.tiled_cartesian_enabled
    and not ctx.is_counting
    and num_sources == 2
    and len(var_from_source) == 2
    and len(var_from_source[0]) == 1
    and len(var_from_source[1]) == 1
  )
  # Nim allocates `flat_idx` BEFORE the dispatch so both tiled and
  # fallback branches share the same name (and the counter bump order
  # matches regardless of which branch is taken). Must allocate here
  # before the tiled path to keep the Nim byte-order.
  flat_idx_var = gen_unique_name(ctx, "flat_idx")

  if tiled_eligible:
    return code + _emit_tiled_cartesian(
      node, ctx, body, tiled_body,
      i=i,
      handle_var_names=handle_var_names,
      view_var_names=view_var_names,
      degree_var_names=degree_var_names,
      lane_var=lane_var,
      group_size_var=group_size_var,
      total_var=total_var,
      flat_idx_var=flat_idx_var,
      var_from_source=var_from_source,
    )

  # Parallel flat loop.
  code += (
    i + f"for (uint32_t {flat_idx_var} = {lane_var}; "
    f"{flat_idx_var} < {total_var}; "
    f"{flat_idx_var} += {group_size_var}) {{\n"
  )

  inc_indent(ctx)
  ii = ind(ctx)
  previous_inside = ctx.inside_cartesian
  previous_cart_bound = list(ctx.cartesian_bound_vars)
  ctx.inside_cartesian = True
  for vfs in var_from_source:
    for v in vfs:
      ctx.cartesian_bound_vars.append(v)

  try:
    # Index decomposition.
    idx_vars = [gen_unique_name(ctx, f"idx{s}") for s in range(num_sources)]

    if num_sources == 1:
      code += ii + f"uint32_t {idx_vars[0]} = {flat_idx_var};\n"
    elif num_sources == 2:
      major_var = gen_unique_name(ctx, "major_is_1")
      code += (
        ii + f"const bool {major_var} = "
        f"({degree_var_names[1]} >= {degree_var_names[0]});\n"
      )
      code += ii + f"uint32_t {idx_vars[0]}, {idx_vars[1]};\n"
      code += ii + f"if ({major_var}) {{\n"
      code += (
        ii + f"  {idx_vars[0]} = {flat_idx_var} / {degree_var_names[1]};\n"
      )
      code += (
        ii + f"  {idx_vars[1]} = {flat_idx_var} % {degree_var_names[1]};\n"
      )
      code += ii + "} else {\n"
      code += (
        ii + f"  {idx_vars[1]} = {flat_idx_var} / {degree_var_names[0]};\n"
      )
      code += (
        ii + f"  {idx_vars[0]} = {flat_idx_var} % {degree_var_names[0]};\n"
      )
      code += ii + "}\n"
    else:
      code += ii + f"uint32_t remaining = {flat_idx_var};\n"
      for src_idx in range(num_sources - 1, -1, -1):
        code += (
          ii + f"uint32_t {idx_vars[src_idx]} = "
          f"remaining % {degree_var_names[src_idx]};\n"
        )
        if src_idx > 0:
          code += ii + f"remaining /= {degree_var_names[src_idx]};\n"

    code += "\n"

    # Bind each source's contributed vars via
    # view.get_value(col, handle.begin() + idx).
    from srdatalog.codegen.jit.context import gen_get_value
    for idx_, src in enumerate(sources):
      prefix_len = len(src.prefix_vars)
      src_index_type = get_rel_index_type(ctx, src.rel_name)
      if idx_ >= len(var_from_source):
        continue
      for v_idx, var_name in enumerate(var_from_source[idx_]):
        col_idx = prefix_len + v_idx
        if not (ctx.is_counting and var_name not in body):
          pos_expr = f"{handle_var_names[idx_]}.begin() + {idx_vars[idx_]}"
          code += (
            ii + f"auto {sanitize_var_name(var_name)} = "
            f"{gen_get_value(view_var_names[idx_], col_idx, pos_expr, src_index_type)};\n"
          )
        ctx.bound_vars.append(var_name)

    code += "\n"
    code += body
  finally:
    ctx.inside_cartesian = previous_inside
    ctx.cartesian_bound_vars = previous_cart_bound
    # Pop bound vars we pushed.
    total_bound = sum(len(vs) for vs in var_from_source)
    for _ in range(total_bound):
      if ctx.bound_vars:
        ctx.bound_vars.pop()
    dec_indent(ctx)

  code += i + "}\n"
  return code


# -----------------------------------------------------------------------------
# Tiled Cartesian — smem pre-load + coalesced writes for 2-source / 1-var
# -----------------------------------------------------------------------------

def _emit_tiled_cartesian(
  node: m.CartesianJoin,
  ctx: CodeGenContext,
  body: str,
  tiled_body: str,
  *,
  i: str,
  handle_var_names: list[str],
  view_var_names: list[str],
  degree_var_names: list[str],
  lane_var: str,
  group_size_var: str,
  total_var: str,
  flat_idx_var: str,
  var_from_source: list[list[str]],
) -> str:
  '''Emit the tiled-Cartesian `if (total > 32) { ... } else { ... }`
  dispatch.

  Tiled path: pre-load each source's values into `s_cart[warp_in_block]
  [side][_ti]` shared memory, sync, then iterate the tile product with
  atomic-free coalesced writes (via the ballot path in `jit_insert_into`
  when `tiled_body` carries `tiled_cartesian_valid_var` set).

  Fallback path: flat loop with `major_is_1` adaptive decomposition.
  When `tiled_body` is non-empty, the fallback also uses the batched-
  valid-flag loop so the outer kernel can still emit coalesced writes.

  Matches Nim's jitNestedCartesianJoin lines 910-1034.
  '''
  code = ""
  src0 = node.sources[0]
  src1 = node.sources[1]
  assert isinstance(src0, m.ColumnSource) and isinstance(src1, m.ColumnSource)
  col0 = len(src0.prefix_vars)
  col1 = len(src1.prefix_vars)
  var_name0 = sanitize_var_name(var_from_source[0][0])
  var_name1 = sanitize_var_name(var_from_source[1][0])

  src0_index_type = get_rel_index_type(ctx, src0.rel_name)
  src1_index_type = get_rel_index_type(ctx, src1.rel_name)

  from srdatalog.codegen.jit.context import gen_get_value, dec_indent as _dec
  # Tiled branch open.
  code += i + f"if ({total_var} > 32) {{\n"
  code += (
    i + "  // Tiled Cartesian: smem pre-load reads, standard emit_direct writes\n"
  )
  t0_base = gen_unique_name(ctx, "t0_base")
  t1_base = gen_unique_name(ctx, "t1_base")
  t0_len = gen_unique_name(ctx, "t0_len")
  t1_len = gen_unique_name(ctx, "t1_len")
  tile_total = gen_unique_name(ctx, "tile_total")

  # Outer tile loop over source 0.
  code += (
    i + f"  for (uint32_t {t0_base} = 0; {t0_base} < {degree_var_names[0]}; "
    f"{t0_base} += kCartTileSize) {{\n"
  )
  code += (
    i + f"    uint32_t {t0_len} = min({t0_base} + (uint32_t)kCartTileSize, "
    f"{degree_var_names[0]}) - {t0_base};\n"
  )
  # Pre-load source 0 values into s_cart[warp_in_block][0][_ti].
  code += (
    i + f"    for (uint32_t _ti = {lane_var}; _ti < {t0_len}; "
    f"_ti += {group_size_var})\n"
  )
  code += (
    i + f"      s_cart[warp_in_block][0][_ti] = {view_var_names[0]}.get_value("
    f"{col0}, {handle_var_names[0]}.begin() + {t0_base} + _ti);\n"
  )
  # Inner tile loop over source 1.
  code += (
    i + f"    for (uint32_t {t1_base} = 0; {t1_base} < {degree_var_names[1]}; "
    f"{t1_base} += kCartTileSize) {{\n"
  )
  code += (
    i + f"      uint32_t {t1_len} = min({t1_base} + (uint32_t)kCartTileSize, "
    f"{degree_var_names[1]}) - {t1_base};\n"
  )
  code += (
    i + f"      for (uint32_t _ti = {lane_var}; _ti < {t1_len}; "
    f"_ti += {group_size_var})\n"
  )
  code += (
    i + f"        s_cart[warp_in_block][1][_ti] = {view_var_names[1]}.get_value("
    f"{col1}, {handle_var_names[1]}.begin() + {t1_base} + _ti);\n"
  )
  code += i + f"      {ctx.tile_var}.sync();\n"
  code += i + f"      uint32_t {tile_total} = {t0_len} * {t1_len};\n"

  # flat_idx_var is allocated by the caller before the tiled/fallback
  # dispatch so both branches share the same counter position (matches
  # Nim's ordering).
  if tiled_body:
    # Batched-valid-flag path (ballot-based coalesced writes).
    batch_var = gen_unique_name(ctx, "tc_batch")
    valid_var = ctx.tiled_cartesian_valid_var or gen_unique_name(ctx, "tc_valid")
    code += (
      i + f"      for (uint32_t {batch_var} = 0; {batch_var} < {tile_total}; "
      f"{batch_var} += {group_size_var}) {{\n"
    )
    code += (
      i + f"        uint32_t {flat_idx_var} = {batch_var} + {lane_var};\n"
    )
    code += i + f"        bool {valid_var} = {flat_idx_var} < {tile_total};\n"
    code += (
      i + f"        auto {var_name0} = {valid_var} ? "
      f"s_cart[warp_in_block][0][{flat_idx_var} / {t1_len}] : ValueType{{0}};\n"
    )
    code += (
      i + f"        auto {var_name1} = {valid_var} ? "
      f"s_cart[warp_in_block][1][{flat_idx_var} % {t1_len}] : ValueType{{0}};\n"
    )
    ctx.bound_vars.append(var_from_source[0][0])
    ctx.bound_vars.append(var_from_source[1][0])
    try:
      code += tiled_body
    finally:
      ctx.bound_vars.pop()
      ctx.bound_vars.pop()
    code += i + "      }\n"
  else:
    # Standard body (no ballot writes).
    code += (
      i + f"      for (uint32_t {flat_idx_var} = {lane_var}; "
      f"{flat_idx_var} < {tile_total}; "
      f"{flat_idx_var} += {group_size_var}) {{\n"
    )
    code += (
      i + f"        auto {var_name0} = s_cart[warp_in_block][0]"
      f"[{flat_idx_var} / {t1_len}];\n"
    )
    code += (
      i + f"        auto {var_name1} = s_cart[warp_in_block][1]"
      f"[{flat_idx_var} % {t1_len}];\n"
    )
    ctx.bound_vars.append(var_from_source[0][0])
    ctx.bound_vars.append(var_from_source[1][0])
    try:
      code += body
    finally:
      ctx.bound_vars.pop()
      ctx.bound_vars.pop()
    code += i + "      }\n"

  code += i + f"      {ctx.tile_var}.sync();\n"
  code += i + "    }\n"  # end t1 loop
  code += i + "  }\n"    # end t0 loop
  code += i + "} else {\n"

  # Fallback: flat loop with major_is_1 decomposition. When tiled_body
  # was provided, the fallback also uses the batched-valid-flag loop
  # so jit_insert_into's ballot path sees the same shape. Reuses the
  # caller-allocated `flat_idx_var` (matches Nim — single flatIdxVar
  # shared between tiled and fallback branches).
  if tiled_body:
    fb_batch_var = gen_unique_name(ctx, "fb_batch")
    valid_var = ctx.tiled_cartesian_valid_var or gen_unique_name(ctx, "tc_valid")
    code += (
      i + f"  for (uint32_t {fb_batch_var} = 0; {fb_batch_var} < {total_var}; "
      f"{fb_batch_var} += {group_size_var}) {{\n"
    )
    code += (
      i + f"    uint32_t {flat_idx_var} = {fb_batch_var} + {lane_var};\n"
    )
    code += i + f"    bool {valid_var} = {flat_idx_var} < {total_var};\n"
    flat_var_inner = flat_idx_var
  else:
    code += (
      i + f"  for (uint32_t {flat_idx_var} = {lane_var}; "
      f"{flat_idx_var} < {total_var}; "
      f"{flat_idx_var} += {group_size_var}) {{\n"
    )
    flat_var_inner = flat_idx_var

  major_var = gen_unique_name(ctx, "major_is_1")
  idx0_var = gen_unique_name(ctx, "idx0")
  idx1_var = gen_unique_name(ctx, "idx1")
  fbii = i + "    "
  code += (
    fbii + f"const bool {major_var} = "
    f"({degree_var_names[1]} >= {degree_var_names[0]});\n"
  )
  code += fbii + f"uint32_t {idx0_var}, {idx1_var};\n"
  code += (
    fbii + f"if ({major_var}) {{ {idx0_var} = {flat_var_inner} / "
    f"{degree_var_names[1]}; {idx1_var} = {flat_var_inner} % "
    f"{degree_var_names[1]}; }}\n"
  )
  code += (
    fbii + f"else {{ {idx1_var} = {flat_var_inner} / {degree_var_names[0]}; "
    f"{idx0_var} = {flat_var_inner} % {degree_var_names[0]}; }}\n"
  )
  # Bind vars from each source.
  for src_idx, src in enumerate(node.sources):
    prefix_len = len(src.prefix_vars)
    src_index_type = get_rel_index_type(ctx, src.rel_name)
    if src_idx >= len(var_from_source):
      continue
    for v_idx, vn in enumerate(var_from_source[src_idx]):
      col_idx = prefix_len + v_idx
      pos_expr = (
        f"{handle_var_names[src_idx]}.begin() + "
        f"{idx0_var if src_idx == 0 else idx1_var}"
      )
      code += (
        fbii + f"auto {sanitize_var_name(vn)} = "
        f"{gen_get_value(view_var_names[src_idx], col_idx, pos_expr, src_index_type)};\n"
      )
  ctx.bound_vars.append(var_from_source[0][0])
  ctx.bound_vars.append(var_from_source[1][0])
  try:
    code += tiled_body if tiled_body else body
  finally:
    ctx.bound_vars.pop()
    ctx.bound_vars.pop()
  code += i + "  }\n"   # close fallback for-loop
  code += i + "}\n"     # close if/else
  return code


# -----------------------------------------------------------------------------
# jit_positioned_extract (balanced-scan follow-up)
# -----------------------------------------------------------------------------

def jit_positioned_extract(
  node: m.PositionedExtract, ctx: CodeGenContext, body: str,
) -> str:
  '''Emit a PositionedExtract — point-lookup on multiple already-bound
  sources. Used as a join level after a BalancedScan when the root
  variable is already bound and we just need to probe each source.

  Baseline only: raises for feature-flag contexts we haven't ported.
  '''
  assert isinstance(node, m.PositionedExtract)
  _feature_flags_disabled(ctx)

  code = ""
  i = ind(ctx)
  extract_var = node.var_name
  bind_vars = list(node.bind_vars)

  if ctx.debug:
    code += (
      i + f"// PositionedExtract: extract {extract_var}"
      f" then bind {', '.join(bind_vars)}\n"
    )

  # Each source contributes a narrowed handle; we look up extract_var
  # on each and verify existence (semijoin behavior).
  valid_var = gen_unique_name(ctx, "pe_valid")
  code += i + f"bool {valid_var} = true;\n"

  handle_narrows: list[tuple[str, str, str]] = []  # (handle_var, view_var, index_type)

  for idx_, src in enumerate(node.sources):
    assert isinstance(src, m.ColumnSource)
    src_idx = src.handle_start
    rel_name = src.rel_name
    existing_view = ctx.view_vars.get(str(src_idx), "")
    view_var = existing_view if existing_view else gen_view_var_name(rel_name, src_idx)
    if not existing_view:
      code += (
        i + f"auto {view_var} = "
        f"{gen_view_access(get_view_slot_base(ctx, src_idx))};\n"
      )
    index_type = get_rel_index_type(ctx, rel_name)
    handle_var = gen_handle_var_name(rel_name, src_idx, ctx)

    parent = ctx.handle_vars.get(str(src_idx), "")
    root = parent if parent else gen_root_handle(view_var, index_type)
    if src.prefix_vars:
      chained = gen_chained_prefix_calls(
        root, list(src.prefix_vars), view_var, index_type=index_type,
      )
      code += i + f"auto {handle_var} = {chained};\n"
    else:
      code += i + f"auto {handle_var} = {root};\n"

    # Probe extract_var cooperatively (per-thread since we're usually inside
    # a Cartesian — match Nim's prefix_seq usage here via the plugin).
    probed = gen_chained_prefix_calls(
      handle_var, [extract_var], view_var,
      ctx.cartesian_bound_vars + [extract_var], index_type=index_type,
    )
    probed_var = gen_unique_name(ctx, f"h_pe_{idx_}")
    code += i + f"auto {probed_var} = {probed};\n"
    code += (
      i + f"{valid_var} = {valid_var} && "
      f"{gen_valid(probed_var, index_type)};\n"
    )
    handle_narrows.append((probed_var, view_var, index_type))

  code += i + f"if ({valid_var}) {{\n"
  inc_indent(ctx)
  ii = ind(ctx)
  try:
    # Bind each additional var by column offset on the first narrow.
    # (Nim's body is small and typically uses the first source for binding.)
    if handle_narrows and bind_vars:
      probed_var, view_var, index_type = handle_narrows[0]
      for col_offset, var_name in enumerate(bind_vars):
        # The extracted key already took one column; bind further cols here.
        col = len(node.sources[0].prefix_vars) + 1 + col_offset
        code += (
          ii + f"auto {sanitize_var_name(var_name)} = "
          f"{view_var}.get_value({col}, {probed_var}.begin());\n"
        )
        ctx.bound_vars.append(var_name)

    try:
      code += body
    finally:
      for _ in bind_vars:
        if ctx.bound_vars:
          ctx.bound_vars.pop()
  finally:
    dec_indent(ctx)
  code += i + "}\n"
  return code
