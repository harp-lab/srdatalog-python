'''Root-level op emitters for the JIT backend.

Port of src/srdatalog/codegen/target_jit/jit_root.nim.

Root operations are the FIRST op in a pipeline. Unlike nested ops
(instructions.py), they iterate over entire relations with parallel
workload partitioning via `warp_id / num_warps` (or a per-warp atomic
counter under "atomic WS" mode).

Procs in this commit:
  gen_grid_stride_loop     — warp/scalar-mode grid-stride iterator setup,
                             plus the lightweight atomic-WS variant
  jit_root_scan            — parallel scan over one relation
  jit_root_cartesian_join  — parallel flat iteration over Cartesian
                             product of N sources
  jit_root_column_join     — single-source or multi-source root ColumnJoin
                             (baseline: single-view-per-source / DSAI only)

Deferred (raise NotImplementedError for now):
  jit_root_column_join_ranged      — BG / ranged variant
  jit_root_balanced_scan           — skewed-workload balanced partition
  jit_root_column_join_block_group — block-group partitioning

Multi-view sources (Device2LevelIndex where view_count > 1) go through
a segment loop wrapping the join body; that path also raises
NotImplementedError in this commit.
'''

from __future__ import annotations

import srdatalog.mir.types as m
from srdatalog.codegen.jit.context import (
  CodeGenContext,
  dec_indent,
  gen_child,
  gen_degree,
  gen_get_value,
  gen_get_value_at,
  gen_handle_state_key,
  gen_handle_var_name,
  gen_root_handle,
  gen_unique_name,
  gen_valid,
  gen_view_access,
  gen_view_var_name,
  get_rel_index_type,
  get_view_slot_base,
  inc_indent,
  ind,
  sanitize_var_name,
)
from srdatalog.codegen.jit.plugin import plugin_view_count

# -----------------------------------------------------------------------------
# Grid-stride loop helper
# -----------------------------------------------------------------------------


def gen_grid_stride_loop(
  i: str,
  idx_var: str,
  bound_var: str,
  ctx: CodeGenContext,
  use_atomic_ws: bool = False,
) -> str:
  '''Emit the outer grid-stride loop preamble + opening brace.

  Default mode (warp-parallel): `for (idx = warp_id; idx < bound; idx += num_warps)`.
  Scalar mode (one thread per row): `for (idx = thread_id; idx < bound; ...)` —
    emitted with a "SCALAR MODE" comment only; the loop structure reuses
    warp_id/num_warps as aliases (set by the kernel prelude in scalar kernels).
  Atomic WS mode: each warp grabs the next key via `atomicAdd` on a shared
    `_d_aws_counter`; used for lightweight load balancing when per-key
    workload is highly skewed.
  '''
  if use_atomic_ws:
    out = (
      i
      + "// ATOMIC WS: lightweight work-stealing via per-warp atomic grab\n"
      + i
      + "while (true) {\n"
      + i
      + f"  uint32_t {idx_var};\n"
      + i
      + f"  if (tile.thread_rank() == 0) {idx_var} "
      f"= atomicAdd(&_d_aws_counter, 1);\n"
      + i
      + f"  {idx_var} = tile.shfl({idx_var}, 0);\n"
      + i
      + f"  if ({idx_var} >= {bound_var}) break;\n"
    )
    return out

  if ctx.scalar_mode:
    comment = i + "// SCALAR MODE: each thread handles one row independently\n"
  else:
    comment = i + "// WARP MODE: 32 threads cooperatively handle one row\n"
  return (
    comment + i + f"for (uint32_t {idx_var} = warp_id; {idx_var} < {bound_var}; "
    f"{idx_var} += num_warps) {{\n"
  )


# -----------------------------------------------------------------------------
# jit_root_scan
# -----------------------------------------------------------------------------


def jit_root_scan(node: m.Scan, ctx: CodeGenContext, body: str) -> str:
  '''Root scan: parallel iteration over an entire relation.

  Lays out:
    - view decl (unless pre-declared at kernel start)
    - root handle, `if (!valid) return;`, degree fetch
    - grid-stride loop
    - per-var get_value(col, idx) bindings (skipping unused vars in count
      phase)
    - body
    - loop close
  '''
  assert isinstance(node, m.Scan)
  handle_idx = node.handle_start
  var_names = list(node.vars)
  rel_name = node.rel_name

  code = ""
  i = ind(ctx)

  if ctx.debug:
    code += i + f"// Root Scan: {rel_name} binding {', '.join(var_names)}\n"
    code += (
      i + f"// MIR: (scan :rel {rel_name} :vars ({' '.join(var_names)}) :handle {handle_idx})\n"
    )

  handle_var = gen_unique_name(ctx, "root_handle")

  existing_view = ctx.view_vars.get(str(handle_idx), "")
  view_var = existing_view if existing_view else gen_unique_name(ctx, "root_view")
  if not existing_view:
    code += i + f"auto {view_var} = {gen_view_access(get_view_slot_base(ctx, handle_idx))};\n"

  index_type = get_rel_index_type(ctx, rel_name)
  code += i + f"auto {handle_var} = {gen_root_handle(view_var, index_type)};\n"
  code += i + f"if (!{gen_valid(handle_var, index_type)}) return;\n"

  degree_var = gen_unique_name(ctx, "degree")
  code += i + f"uint32_t {degree_var} = {gen_degree(handle_var, index_type)};\n\n"

  idx_var = gen_unique_name(ctx, "idx")
  code += gen_grid_stride_loop(i, idx_var, degree_var, ctx)

  inc_indent(ctx)
  ii = ind(ctx)
  try:
    for var_idx, var_name in enumerate(var_names):
      # Counting-phase optimization: skip fetch if var isn't referenced in body.
      if not (ctx.is_counting and var_name not in body):
        code += (
          ii + f"auto {sanitize_var_name(var_name)} = "
          f"{gen_get_value(view_var, var_idx, idx_var, index_type)};\n"
        )
      ctx.bound_vars.append(var_name)
    code += body
  finally:
    for _ in var_names:
      if ctx.bound_vars:
        ctx.bound_vars.pop()
    dec_indent(ctx)

  code += i + "}\n"
  return code


# -----------------------------------------------------------------------------
# jit_root_cartesian_join
# -----------------------------------------------------------------------------


def jit_root_cartesian_join(
  node: m.CartesianJoin,
  ctx: CodeGenContext,
  body: str,
) -> str:
  '''Root CartesianJoin: parallel iteration over the product of N sources.

  Shape:
    - Per-source view + root handle setup
    - Combined validity check (early return if ANY source empty)
    - Per-source degree, total = product, early return if 0
    - Grid-stride loop over flat_idx
    - Index decomposition: 1-source (identity), 2-source (div/mod),
      N-source (countdown remainder)
    - Per-source var bindings via get_value_at(handle, view, idx)
    - Toggles ctx.inside_cartesian + extends cartesian_bound_vars
  '''
  assert isinstance(node, m.CartesianJoin)
  var_names = list(node.vars)
  sources = node.sources
  num_sources = len(sources)

  code = ""
  i = ind(ctx)

  if ctx.debug:
    code += i + f"// Root CartesianJoin: bind {', '.join(var_names)} from {num_sources} source(s)\n"
    src_debug = ""
    for src in sources:
      src_debug += f"({src.rel_name} :handle {src.handle_start}) "
    code += i + f"// MIR: (cartesian-join :vars ({' '.join(var_names)}) :sources ({src_debug}))\n"

  handle_vars: list[str] = []
  view_vars: list[str] = []
  degree_vars: list[str] = []

  # Nim interleaves handle and degree name generation within a single
  # loop — the degree counter is bumped alongside the handle, but its
  # `uint32_t degree_N = ...` assignment is emitted in a later loop.
  # Mirror that so counter values line up for byte-match.
  for src in sources:
    assert isinstance(src, m.ColumnSource)
    h_idx = src.handle_start
    rel_name = src.rel_name
    handle_var = gen_handle_var_name(rel_name, h_idx, ctx)
    deg_var = gen_unique_name(ctx, "degree")
    existing_view = ctx.view_vars.get(str(h_idx), "")
    view_var = existing_view if existing_view else gen_view_var_name(rel_name, h_idx)

    handle_vars.append(handle_var)
    view_vars.append(view_var)
    degree_vars.append(deg_var)

    if not existing_view:
      code += i + f"auto {view_var} = {gen_view_access(get_view_slot_base(ctx, h_idx))};\n"
    index_type = get_rel_index_type(ctx, rel_name)
    code += i + f"auto {handle_var} = {gen_root_handle(view_var, index_type)};\n"

  code += "\n"

  # Combined validity check: `!valid(h0) || !valid(h1) || ...` -> return.
  validity_checks = []
  for src_idx, src in enumerate(sources):
    src_index_type = get_rel_index_type(ctx, src.rel_name)
    validity_checks.append(f"!{gen_valid(handle_vars[src_idx], src_index_type)}")
  code += i + f"if ({' || '.join(validity_checks)}) return;\n\n"

  # Degree assignments (names already allocated above).
  for src_idx, src in enumerate(sources):
    src_index_type = get_rel_index_type(ctx, src.rel_name)
    code += (
      i + f"uint32_t {degree_vars[src_idx]} = {gen_degree(handle_vars[src_idx], src_index_type)};\n"
    )

  total_var = gen_unique_name(ctx, "total")
  code += i + f"uint32_t {total_var} = {' * '.join(degree_vars)};\n"
  code += i + f"if ({total_var} == 0) return;\n\n"

  flat_idx_var = gen_unique_name(ctx, "flat_idx")
  code += gen_grid_stride_loop(i, flat_idx_var, total_var, ctx)

  inc_indent(ctx)
  ii = ind(ctx)
  ctx.inside_cartesian = True
  previous_cart_bound = list(ctx.cartesian_bound_vars)
  ctx.cartesian_bound_vars.extend(var_names)

  try:
    # Index decomposition — row-major.
    idx_vars = [gen_unique_name(ctx, f"idx{s}") for s in range(num_sources)]

    if num_sources == 1:
      code += ii + f"uint32_t {idx_vars[0]} = {flat_idx_var};\n"
    elif num_sources == 2:
      code += ii + f"uint32_t {idx_vars[0]} = {flat_idx_var} / {degree_vars[1]};\n"
      code += ii + f"uint32_t {idx_vars[1]} = {flat_idx_var} % {degree_vars[1]};\n"
    else:
      # N-source: carry `remaining` downward.
      code += ii + f"uint32_t remaining = {flat_idx_var};\n"
      for src_idx in range(num_sources - 1, -1, -1):
        code += ii + f"uint32_t {idx_vars[src_idx]} = remaining % {degree_vars[src_idx]};\n"
        if src_idx > 0:
          code += ii + f"remaining /= {degree_vars[src_idx]};\n"

    code += "\n"

    # Bind vars from each source. `numVarsFromSrc = src.csIndex.len - src.csPrefixVars.len`.
    var_idx_cursor = 0
    for src_idx, src in enumerate(sources):
      assert isinstance(src, m.ColumnSource)
      num_vars_from_src = len(src.index) - len(src.prefix_vars)
      src_index_type = get_rel_index_type(ctx, src.rel_name)
      for _v in range(num_vars_from_src):
        if var_idx_cursor < len(var_names):
          vname = var_names[var_idx_cursor]
          if not (ctx.is_counting and vname not in body):
            code += (
              ii + f"auto {sanitize_var_name(vname)} = "
              f"{gen_get_value_at(handle_vars[src_idx], view_vars[src_idx], idx_vars[src_idx], src_index_type)};\n"
            )
          ctx.bound_vars.append(vname)
          var_idx_cursor += 1

    code += "\n"
    code += body
  finally:
    ctx.inside_cartesian = False
    ctx.cartesian_bound_vars = previous_cart_bound
    for _ in range(var_idx_cursor):
      if ctx.bound_vars:
        ctx.bound_vars.pop()
    dec_indent(ctx)

  code += i + "}\n"
  return code


# -----------------------------------------------------------------------------
# jit_root_column_join — baseline (no BG, no WS, no fan-out, no multi-view)
# -----------------------------------------------------------------------------


def _feature_flags_disabled_for_cj(ctx: CodeGenContext) -> None:
  '''Reject every advanced feature flag the baseline CJ doesn't cover.'''
  if ctx.ws_enabled:
    raise NotImplementedError("jit_root_column_join: work-stealing branch not yet ported")
  if ctx.bg_histogram_mode:
    raise NotImplementedError(
      "jit_root_column_join: BG histogram mode not yet ported here "
      "(emitted directly by complete_runner)"
    )
  if ctx.is_fan_out_explore:
    raise NotImplementedError("jit_root_column_join: fan-out explore branch not yet ported")


def _source_view_count(src: m.ColumnSource, ctx: CodeGenContext) -> int:
  '''`plugin_view_count(version, index_type)` — 1 for DSAI, 2 for
  Device2LevelIndex FULL reads, etc.'''
  index_type = get_rel_index_type(ctx, src.rel_name)
  return plugin_view_count(src.version.code, index_type)


def jit_root_column_join(
  node: m.ColumnJoin,
  ctx: CodeGenContext,
  body: str,
) -> str:
  '''Root ColumnJoin — first op in a pipeline.

  Single source:
    - Warp-parallel iteration over the root handle's degree
    - Binds the join var via `get_value_at(handle, view, idx)`
    - Emits a child handle and registers it under the numeric src_idx key
      and the semantic state key `<Rel>_<cols>_<var>_<VER>` so nested
      ops can find it by either

  Multi source:
    - Parallel iteration over `root_unique_values[...]` (kernel-provided
      sorted-distinct key array produced by a host-side pre-pass)
    - First source uses a hinted-range prefix starting at y_idx (since
      root_unique_values came from its column); remaining sources apply
      a plain `.prefix(root_val, tile, view)` narrowing
    - Early-return on any narrowed handle that isn't valid (`continue`)
    - Binds the root var to `root_val` and registers each handle under
      the numeric + semantic keys before descending into body
  '''
  assert isinstance(node, m.ColumnJoin)
  _feature_flags_disabled_for_cj(ctx)

  sources = node.sources
  num_sources = len(sources)
  var_name = node.var_name

  i = ind(ctx)
  code = ""

  if num_sources == 1:
    return code + _root_cj_single(node, ctx, body)
  return code + _root_cj_multi(node, ctx, body)


def _root_cj_single(
  node: m.ColumnJoin,
  ctx: CodeGenContext,
  body: str,
) -> str:
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
    code += i + f"// Root ColumnJoin (single source): bind '{var_name}' from {rel_name}\n"
    code += i + f"// MIR: (column-join :var {var_name} :sources (({rel_name} :handle {src_idx})))\n"

  if has_segment_loop:
    code += i + f"for (int _seg = 0; _seg < {view_count}; _seg++) {{\n"
  si = i + "  " if has_segment_loop else i

  handle_var = gen_handle_var_name(rel_name, src_idx, ctx)
  degree_var = gen_unique_name(ctx, "degree")

  view_var = gen_view_var_name(rel_name, src_idx)
  if has_segment_loop:
    # Multi-view: fresh view decl per segment.
    code += si + f"auto {view_var} = views[{base_slot} + _seg];\n"
  else:
    existing_view = ctx.view_vars.get(str(src_idx), "")
    if existing_view:
      if existing_view != view_var:
        code += si + f"auto& {view_var} = {existing_view};\n"
    else:
      code += si + f"auto {view_var} = {gen_view_access(base_slot)};\n"

  code += si + f"auto {handle_var} = {gen_root_handle(view_var, index_type)};\n"
  code += si + f"if (!{gen_valid(handle_var, index_type)}) return;\n"
  code += si + f"uint32_t {degree_var} = {gen_degree(handle_var, index_type)};\n\n"

  idx_var = gen_unique_name(ctx, "idx")
  code += gen_grid_stride_loop(si, idx_var, degree_var, ctx)

  inc_indent(ctx)
  if has_segment_loop:
    inc_indent(ctx)
  ii = ind(ctx)
  try:
    if not (ctx.is_counting and var_name not in body):
      code += (
        ii + f"auto {sanitize_var_name(var_name)} = "
        f"{gen_get_value_at(handle_var, view_var, idx_var, index_type)};\n"
      )
    child_var = gen_unique_name(ctx, f"ch_{rel_name}")
    code += ii + f"auto {child_var} = {gen_child(handle_var, idx_var, index_type)};\n"
    ctx.handle_vars[str(src_idx)] = child_var
    state_key = gen_handle_state_key(rel_name, list(src.index), [var_name], src.version.code)
    ctx.handle_vars[state_key] = child_var
    ctx.bound_vars.append(var_name)
    try:
      code += body
    finally:
      if ctx.bound_vars and ctx.bound_vars[-1] == var_name:
        ctx.bound_vars.pop()
      ctx.handle_vars.pop(state_key, None)
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


def _root_cj_multi(
  node: m.ColumnJoin,
  ctx: CodeGenContext,
  body: str,
) -> str:
  i = ind(ctx)
  sources = node.sources
  num_sources = len(sources)
  var_name = node.var_name

  code = ""
  if ctx.debug:
    code += (
      i + f"// Root ColumnJoin (multi-source intersection): bind '{var_name}'"
      f" from {num_sources} sources\n"
    )
    code += i + "// Uses root_unique_values + prefix() pattern (like TMP)\n"
    src_debug = ""
    for src in sources:
      src_debug += f"({src.rel_name} :handle {src.handle_start}) "
    code += i + f"// MIR: (column-join :var {var_name} :sources ({src_debug}))\n"

  y_idx_var = gen_unique_name(ctx, "y_idx")
  code += gen_grid_stride_loop(i, y_idx_var, "num_unique_root_keys", ctx)

  inc_indent(ctx)
  ii = ind(ctx)
  # Stash per-source metadata so the two emission phases share state.
  handle_var_names: list[str] = []
  view_var_names: list[str] = []  # "" placeholder for deferred multi-view
  src_indices: list[int] = []
  src_rel_names: list[str] = []
  src_versions: list[str] = []
  src_objs: list[m.ColumnSource] = []

  # Multi-view non-first sources defer to a segment-loop emission phase
  # that wraps the narrowing + body. Record their metadata and emit a
  # placeholder (blank) view_var that gets filled in below.
  segment_loop_sources: list[dict] = []

  try:
    root_val_var = gen_unique_name(ctx, "root_val")
    code += ii + f"auto {root_val_var} = root_unique_values[{y_idx_var}];\n\n"

    # -------------------- Phase 1: single-view + first source --------------------
    for idx_, src in enumerate(sources):
      assert isinstance(src, m.ColumnSource)
      src_idx = src.handle_start
      rel_name = src.rel_name
      src_indices.append(src_idx)
      src_rel_names.append(rel_name)
      src_versions.append(src.version.code)
      src_objs.append(src)

      index_type = get_rel_index_type(ctx, rel_name)
      base_slot = get_view_slot_base(ctx, src_idx)
      view_count = plugin_view_count(src.version.code, index_type)

      # Deterministic handle name that matches the pre-registration naming.
      handle_var = f"h_{rel_name}_{src_idx}_root"
      handle_var_names.append(handle_var)

      if view_count > 1 and idx_ > 0:
        # Multi-view non-first source: defer handle + narrowing to the
        # segment-loop phase below.
        segment_loop_sources.append(
          {
            "src_idx_pos": idx_,
            "view_count": view_count,
            "seg_var": f"_seg_{idx_}",
            "base_slot": base_slot,
          }
        )
        view_var_names.append("")  # placeholder, filled in inside segment loop
        continue

      # Single-view (or first source): emit handle now.
      existing_view = ctx.view_vars.get(str(src_idx), "")
      view_var = existing_view if existing_view else gen_view_var_name(rel_name, src_idx)
      if not existing_view:
        code += ii + f"auto {view_var} = {gen_view_access(base_slot)};\n"

      if idx_ == 0:
        # First source: root_unique_values came from this relation's column.
        hint_lo = gen_unique_name(ctx, "hint_lo")
        hint_hi = gen_unique_name(ctx, "hint_hi")
        code += ii + f"uint32_t {hint_lo} = {y_idx_var};\n"
        code += (
          ii + f"uint32_t {hint_hi} = {view_var}.num_rows_"
          f" - (num_unique_root_keys - {y_idx_var} - 1);\n"
        )
        code += (
          ii + f"{hint_hi} = ({hint_hi} <= {view_var}.num_rows_) ? "
          f"{hint_hi} : {view_var}.num_rows_;\n"
        )
        code += ii + f"{hint_hi} = ({hint_hi} > {hint_lo}) ? {hint_hi} : {view_var}.num_rows_;\n"
        code += (
          ii + f"auto {handle_var} = HandleType({hint_lo}, {hint_hi}, 0)"
          f".prefix({root_val_var}, tile, {view_var});\n"
        )
      else:
        code += (
          ii + f"auto {handle_var} = {gen_root_handle(view_var, index_type)}"
          f".prefix({root_val_var}, tile, {view_var});\n"
        )

      code += ii + f"if (!{gen_valid(handle_var, index_type)}) continue;\n"
      view_var_names.append(view_var)

    # -------------------- Phase 2: segment loops for multi-view non-first sources --------------------
    # Each multi-view source opens a for(_seg) loop. The body + variable
    # binding goes inside the innermost loop; segment loops nest, so the
    # effective indent grows by 2 spaces per loop.
    seg_indent = ii
    for seg in segment_loop_sources:
      idx_ = seg["src_idx_pos"]
      src = src_objs[idx_]
      rel_name = src_rel_names[idx_]
      handle_var = handle_var_names[idx_]
      index_type = get_rel_index_type(ctx, rel_name)
      seg_var = seg["seg_var"]
      view_count = seg["view_count"]
      base_slot = seg["base_slot"]

      if ctx.debug:
        code += (
          seg_indent + f"// Segment loop: {rel_name} {src.version.code}"
          f" has {view_count} segments (FULL + HEAD)\n"
        )

      code += seg_indent + f"for (int {seg_var} = 0; {seg_var} < {view_count}; {seg_var}++) {{\n"
      seg_indent += "  "

      # Per-segment view var declaration.
      view_var = gen_view_var_name(rel_name, src_indices[idx_])
      code += seg_indent + f"auto {view_var} = views[{base_slot} + {seg_var}];\n"
      # Reassign the fixed view var (declared at kernel start by view
      # decls) to the current segment so nested emitters see the right
      # array, not just FULL.
      fixed_view_var = (
        f"view_{rel_name}_"
        + "_".join(str(c) for c in src.index)
        + (f"_{src.version.code}" if src.version.code else "")
      )
      if fixed_view_var != view_var:
        code += seg_indent + f"{fixed_view_var} = {view_var};\n"

      code += (
        seg_indent + f"auto {handle_var} = {gen_root_handle(view_var, index_type)}"
        f".prefix({root_val_var}, tile, {view_var});\n"
      )
      code += seg_indent + f"if (!{gen_valid(handle_var, index_type)}) continue;\n"

      view_var_names[idx_] = view_var

    # -------------------- Binding + body at innermost segment indent --------------------
    # Bind the root var + register per-source handles in ctx.
    code += seg_indent + f"auto {sanitize_var_name(var_name)} = {root_val_var};\n"

    registered_state_keys: list[str] = []
    registered_numeric: list[int] = []
    for idx_, src in enumerate(sources):
      src_idx = src_indices[idx_]
      rel_name = src_rel_names[idx_]
      ctx.handle_vars[str(src_idx)] = handle_var_names[idx_]
      state_key = gen_handle_state_key(
        rel_name,
        list(src.index),
        [var_name],
        src_versions[idx_],
      )
      ctx.handle_vars[state_key] = handle_var_names[idx_]
      registered_state_keys.append(state_key)
      registered_numeric.append(src_idx)

    ctx.bound_vars.append(var_name)
    # Bump indent so nested emitters emit at the innermost segment's
    # nesting level.
    for _ in segment_loop_sources:
      inc_indent(ctx)
    try:
      code += body
    finally:
      for _ in segment_loop_sources:
        dec_indent(ctx)
      if ctx.bound_vars and ctx.bound_vars[-1] == var_name:
        ctx.bound_vars.pop()
      for k in registered_state_keys:
        ctx.handle_vars.pop(k, None)
      for k_ in registered_numeric:
        ctx.handle_vars.pop(str(k_), None)
  finally:
    dec_indent(ctx)

  # Close segment loops in reverse nesting order.
  for seg_idx in range(len(segment_loop_sources) - 1, -1, -1):
    close_indent = ii + "  " * seg_idx
    close_indent = ii + "  " * seg_idx
    code += close_indent + "}\n"

  code += i + "}\n"
  return code


# -----------------------------------------------------------------------------
# Block-group root ColumnJoin
# -----------------------------------------------------------------------------


def jit_root_column_join_block_group(
  node: m.ColumnJoin,
  ctx: CodeGenContext,
  body: str,
) -> str:
  '''Root ColumnJoin with block-group work-balanced partitioning.

  Each block gets a contiguous slice of the total flat work space
  `[0, bg_total_work)`. Binary search on `bg_cumulative_work[]` finds
  the starting root key; within each key, work is redistributed across
  warps in the block row-proportionally on the first source's degree.

  Only supports multi-source ColumnJoin (single-source falls back to
  the baseline). Port of jit_root.nim:jitRootColumnJoinBlockGroup.
  Baseline emit here handles single-view (DSAI) first sources; the
  dual-pointer 2-level variant is deferred to a future pass.
  '''
  assert isinstance(node, m.ColumnJoin)
  var_name = node.var_name
  sources = node.sources
  num_sources = len(sources)
  if num_sources == 1:
    return jit_root_column_join(node, ctx, body)

  first_src = sources[0]
  assert isinstance(first_src, m.ColumnSource)
  first_index_type = get_rel_index_type(ctx, first_src.rel_name)
  first_view_count = plugin_view_count(first_src.version.code, first_index_type)
  if first_view_count > 1:
    raise NotImplementedError(
      "jit_root_column_join_block_group: 2-level first-source (dual-pointer) variant not yet ported"
    )

  code = ""
  i = ind(ctx)
  if ctx.debug:
    code += i + f"// Root ColumnJoin (BLOCK-GROUP): bind '{var_name}' from {num_sources} sources\n"
    code += i + "// Block-group work-balanced partitioning with inner redistribution\n"

  # Block-level work assignment preamble (fixed template).
  code += i + "static constexpr int kWarpsPerBlock = kBlockSize / kGroupSize;\n"
  code += i + "uint64_t bg_work_per_block = (bg_total_work + gridDim.x - 1) / gridDim.x;\n"
  code += i + "uint64_t bg_block_begin = (uint64_t)blockIdx.x * bg_work_per_block;\n"
  code += i + "uint64_t bg_block_end = bg_block_begin + bg_work_per_block;\n"
  code += i + "if (bg_block_end > bg_total_work) bg_block_end = bg_total_work;\n"
  code += i + "if (bg_block_begin >= bg_total_work) {\n"
  if ctx.is_counting:
    code += i + "  thread_counts[thread_id] = 0;\n"
  code += i + "  return;\n"
  code += i + "}\n\n"

  # Binary search cumulative_work for starting key.
  code += i + "uint32_t bg_key_lo = 0, bg_key_hi = num_unique_root_keys;\n"
  code += i + "while (bg_key_lo < bg_key_hi) {\n"
  code += i + "  uint32_t bg_mid = bg_key_lo + (bg_key_hi - bg_key_lo) / 2;\n"
  code += (
    i + "  if (bg_cumulative_work[bg_mid] <= (uint64_t)bg_block_begin) bg_key_lo = bg_mid + 1;\n"
  )
  code += i + "  else bg_key_hi = bg_mid;\n"
  code += i + "}\n\n"

  code += i + "uint64_t bg_remaining_begin = bg_block_begin;\n"
  code += i + "uint64_t bg_remaining_end = bg_block_end;\n\n"

  # Key loop.
  key_idx_var = gen_unique_name(ctx, "bg_key_idx")
  code += (
    i + f"for (uint32_t {key_idx_var} = bg_key_lo; "
    f"{key_idx_var} < num_unique_root_keys && "
    "bg_remaining_begin < bg_remaining_end; "
    f"{key_idx_var}++) {{\n"
  )

  inc_indent(ctx)
  ii = ind(ctx)

  root_val_var = gen_unique_name(ctx, "root_val")
  code += ii + f"auto {root_val_var} = root_unique_values[{key_idx_var}];\n"

  # Per-key work range.
  code += (
    ii + f"uint64_t bg_key_work_start = ({key_idx_var} > 0) ? "
    f"bg_cumulative_work[{key_idx_var} - 1] : 0;\n"
  )
  code += ii + f"uint64_t bg_key_work_end = bg_cumulative_work[{key_idx_var}];\n"
  code += ii + "if (bg_key_work_end <= bg_remaining_begin) continue;\n"
  code += ii + "if (bg_key_work_start >= bg_remaining_end) break;\n\n"
  code += (
    ii + "uint64_t bg_my_begin_in_key = "
    "(bg_remaining_begin > bg_key_work_start) ? "
    "(bg_remaining_begin - bg_key_work_start) : 0;\n"
  )
  code += (
    ii + "uint64_t bg_my_end_in_key = "
    "(bg_remaining_end < bg_key_work_end) ? "
    "(bg_remaining_end - bg_key_work_start) : "
    "(bg_key_work_end - bg_key_work_start);\n\n"
  )

  # Per-source handle prefix (single-view / DSAI path only).
  handle_var_names: list[str] = []
  view_var_names: list[str] = []
  src_indices: list[int] = []
  src_rel_names: list[str] = []

  for idx_, src in enumerate(sources):
    assert isinstance(src, m.ColumnSource)
    src_idx = src.handle_start
    rel_name = src.rel_name
    src_indices.append(src_idx)
    src_rel_names.append(rel_name)
    handle_var = f"h_{rel_name}_{src_idx}_root"
    handle_var_names.append(handle_var)

    src_index_type = get_rel_index_type(ctx, rel_name)
    src_view_count = plugin_view_count(src.version.code, src_index_type)
    if src_view_count > 1:
      raise NotImplementedError(
        "jit_root_column_join_block_group: 2-level non-first source not yet ported"
      )
    src_base_slot = get_view_slot_base(ctx, src_idx)
    existing_view = ctx.view_vars.get(str(src_idx), "")
    view_var = existing_view if existing_view else gen_view_var_name(rel_name, src_idx)
    view_var_names.append(view_var)

    if idx_ == 0:
      # First source: use key_idx as row hint + narrow with root_val.
      if not existing_view:
        code += ii + f"auto {view_var} = {gen_view_access(src_base_slot)};\n"
      hint_lo = gen_unique_name(ctx, "hint_lo")
      hint_hi = gen_unique_name(ctx, "hint_hi")
      code += ii + f"uint32_t {hint_lo} = {key_idx_var};\n"
      code += (
        ii + f"uint32_t {hint_hi} = {view_var}.num_rows_ - "
        f"(num_unique_root_keys - {key_idx_var} - 1);\n"
      )
      code += (
        ii + f"{hint_hi} = ({hint_hi} <= {view_var}.num_rows_) ? "
        f"{hint_hi} : {view_var}.num_rows_;\n"
      )
      code += ii + f"{hint_hi} = ({hint_hi} > {hint_lo}) ? {hint_hi} : {view_var}.num_rows_;\n"
      code += (
        ii + f"auto {handle_var} = HandleType({hint_lo}, {hint_hi}, 0)"
        f".prefix({root_val_var}, tile, {view_var});\n"
      )
    else:
      if not existing_view:
        code += ii + f"auto {view_var} = {gen_view_access(src_base_slot)};\n"
      code += (
        ii + f"auto {handle_var} = {gen_root_handle(view_var, src_index_type)}"
        f".prefix({root_val_var}, tile, {view_var});\n"
      )
    code += (
      ii + f"if (!{gen_valid(handle_var, src_index_type)}) {{ "
      "bg_remaining_begin = bg_key_work_end; continue; }\n"
    )

  # Warp redistribution within block (row-proportional on first source).
  first_handle = handle_var_names[0]
  code += "\n"
  code += ii + "// Distribute within-key work across warps in block (row-proportional)\n"
  code += ii + "uint32_t bg_warp_in_block = threadIdx.x / kGroupSize;\n"
  code += ii + "uint64_t bg_key_total_work = bg_key_work_end - bg_key_work_start;\n"
  code += (
    ii + f"uint32_t bg_deg_first = (uint32_t)({first_handle}.end() - {first_handle}.begin());\n"
  )
  code += (
    ii + "uint32_t bg_block_row_begin = (uint32_t)"
    "((bg_my_begin_in_key * (uint64_t)bg_deg_first) / bg_key_total_work);\n"
  )
  code += (
    ii + "uint32_t bg_block_row_end = (uint32_t)"
    "((bg_my_end_in_key * (uint64_t)bg_deg_first) / bg_key_total_work);\n"
  )
  code += ii + "if (bg_my_end_in_key >= bg_key_total_work) bg_block_row_end = bg_deg_first;\n"
  code += (
    ii + "if (bg_block_row_begin >= bg_block_row_end) { "
    "bg_remaining_begin = bg_key_work_end; continue; }\n\n"
  )
  code += ii + "uint32_t bg_rows_in_block = bg_block_row_end - bg_block_row_begin;\n"
  code += (
    ii + "uint32_t bg_warp_row_size = (bg_rows_in_block + kWarpsPerBlock - 1) / kWarpsPerBlock;\n"
  )
  code += (
    ii + "uint32_t bg_warp_row_begin = bg_block_row_begin + bg_warp_in_block * bg_warp_row_size;\n"
  )
  code += ii + "uint32_t bg_warp_row_end = bg_warp_row_begin + bg_warp_row_size;\n"
  code += ii + "if (bg_warp_row_end > bg_block_row_end) bg_warp_row_end = bg_block_row_end;\n"
  code += (
    ii + "if (bg_warp_row_begin >= bg_warp_row_end) { "
    "bg_remaining_begin = bg_key_work_end; continue; }\n\n"
  )
  # Narrow first source handle to warp's row range.
  code += ii + "// Narrow first source handle to warp's row range\n"
  code += ii + "{\n"
  code += ii + f"  auto bg_narrow_begin = {first_handle}.begin() + bg_warp_row_begin;\n"
  code += ii + f"  auto bg_narrow_end = {first_handle}.begin() + bg_warp_row_end;\n"
  code += (
    ii + f"  {first_handle} = HandleType(bg_narrow_begin, bg_narrow_end, {first_handle}.depth());\n"
  )
  code += ii + "}\n\n"

  # Inner pipeline runs with BG disabled — narrowed handle already
  # restricts work to this warp's first-level children.
  saved_bg_enabled = ctx.bg_enabled
  ctx.bg_enabled = False

  # Bind root var + register narrowed handles under semantic keys.
  code += ii + f"auto {sanitize_var_name(var_name)} = {root_val_var};\n"
  registered_keys: list[str] = []
  registered_numeric: list[int] = []
  for idx_, src in enumerate(sources):
    assert isinstance(src, m.ColumnSource)
    ctx.handle_vars[str(src.handle_start)] = handle_var_names[idx_]
    registered_numeric.append(src.handle_start)
    state_key = gen_handle_state_key(
      src.rel_name,
      list(src.index),
      [var_name],
      src.version.code,
    )
    ctx.handle_vars[state_key] = handle_var_names[idx_]
    registered_keys.append(state_key)

  ctx.bound_vars.append(var_name)
  try:
    code += body
  finally:
    if ctx.bound_vars and ctx.bound_vars[-1] == var_name:
      ctx.bound_vars.pop()
    for k in registered_keys:
      ctx.handle_vars.pop(k, None)
    for n in registered_numeric:
      ctx.handle_vars.pop(str(n), None)
    ctx.bg_enabled = saved_bg_enabled
    dec_indent(ctx)

  code += ii + "bg_remaining_begin = bg_key_work_end;\n"
  code += i + "}\n"
  return code
