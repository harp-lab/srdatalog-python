'''JIT emitters for Scan, Negation, Aggregate (nested / non-root uses).

Port of src/srdatalog/codegen/target_jit/jit_scan_negation.nim.

These are the body-pass versions of the three source-bearing op kinds.
Root-level Scan / Negation / Aggregate live in jit_root.py (later
commit); what's here wraps its `body` argument in whatever narrowed-
handle iteration the op requires:

  jit_scan       — binds N vars from remaining columns; emits a
                   standard for-loop over the handle's iterators
  jit_negation   — anti-join: handle must be invalid for body to fire.
                   Folds into ws/tiled Cartesian valid flags when in a
                   cooperative batch loop; else uses `if (!valid) { body }`.
                   Applies constant prefixes first (matching HIR indexCols
                   ordering: const cols come before variable cols),
                   then variable prefixes (cooperative outside Cartesian,
                   per-thread seq inside). Uses pre-narrowed handle from
                   ctx.neg_pre_narrow when available (Cartesian setup
                   already applied the pre-Cartesian vars cooperatively).
  jit_aggregate  — binds result_var to aggregate<Func>(handle, view);
                   delegates body with result_var added to bound vars.

Each proc returns the full C++ string (declarations + open block +
body + close block) — it's a continuation-passing port of the Nim
original: the caller pre-renders the body at the next indent level.
'''
from __future__ import annotations

import srdatalog.mir_types as m
from srdatalog.codegen.jit.context import (
  CodeGenContext, ind, inc_indent, with_bound_var, gen_unique_name,
  get_view_slot_base, get_rel_index_type,
  gen_view_access, gen_view_var_name, gen_root_handle, gen_valid,
  gen_chained_prefix_calls, gen_chained_prefix_calls_seq,
  sanitize_var_name,
)


# -----------------------------------------------------------------------------
# jit_scan
# -----------------------------------------------------------------------------

def jit_scan(node: m.Scan, ctx: CodeGenContext, body: str) -> str:
  '''Emit a Scan: iterate the (possibly-narrowed) handle, bind one var
  per remaining column, then emit `body` inside the loop.
  '''
  assert isinstance(node, m.Scan)
  code = ""
  i = ind(ctx)
  src_idx = node.handle_start
  rel_name = node.rel_name

  if ctx.debug:
    code += i + f"// Scan: bind {', '.join(node.vars)} from {rel_name}\n"
    code += (
      i + f"// MIR: (scan :rel {rel_name} :vars ({' '.join(node.vars)})"
      f" :prefix ({' '.join(node.prefix_vars)}) :handle {src_idx})\n"
    )

  existing_view = ctx.view_vars.get(str(src_idx), "")
  view_var = existing_view if existing_view else gen_view_var_name(rel_name, src_idx)

  if not existing_view:
    code += i + f"auto {view_var} = {gen_view_access(get_view_slot_base(ctx, src_idx))};\n"

  index_type = get_rel_index_type(ctx, rel_name)
  parent_handle = ctx.handle_vars.get(str(src_idx), "")
  handle_var = parent_handle if parent_handle else gen_root_handle(view_var, index_type)

  # Narrow via prefix vars if any.
  if node.prefix_vars:
    narrowed_var = gen_unique_name(ctx, f"h_scan_{src_idx}")
    chained = gen_chained_prefix_calls(
      handle_var, node.prefix_vars, view_var,
      ctx.cartesian_bound_vars, index_type=index_type,
    )
    code += i + f"auto {narrowed_var} = {chained};\n"
    narrowed_handle = narrowed_var
  else:
    narrowed_handle = handle_var

  it_var = gen_unique_name(ctx, "scan_it")
  code += (
    i + f"for (auto {it_var} = {narrowed_handle}.begin(); "
    f"{it_var} != {narrowed_handle}.end(); ++{it_var}) {{\n"
  )

  # Body runs at indent+1. Create a new ctx for iteration to avoid
  # polluting the caller's bound_vars.
  new_ctx = ctx
  inc_indent(new_ctx)  # NB: Python's ctx is mutable; safe to bump in-place
  ii = ind(new_ctx)
  for var_idx, var_name in enumerate(node.vars):
    col_offset = len(node.prefix_vars) + var_idx
    code += ii + f"auto {var_name} = {view_var}.get_value({col_offset}, {it_var});\n"
    new_ctx = with_bound_var(new_ctx, var_name)

  code += body
  # Restore indent on the shared ctx — Nim uses a copy; we emulate by
  # dec'ing since we inc'd above.
  from srdatalog.codegen.jit.context import dec_indent
  dec_indent(ctx)
  code += i + "}\n"
  return code


# -----------------------------------------------------------------------------
# jit_negation
# -----------------------------------------------------------------------------

def jit_negation(node: m.Negation, ctx: CodeGenContext, body: str) -> str:
  '''Emit an anti-join: body fires only when the narrowed handle is
  invalid (i.e., the tuple doesn't exist in the negated relation).
  '''
  assert isinstance(node, m.Negation)
  code = ""
  i = ind(ctx)
  src_idx = node.handle_start
  rel_name = node.rel_name

  if ctx.debug:
    code += i + f"// Negation: NOT EXISTS in {rel_name}\n"
    code += (
      i + f"// MIR: (negation :rel {rel_name}"
      f" :prefix ({' '.join(node.prefix_vars)}) :handle {src_idx})\n"
    )

  existing_view = ctx.view_vars.get(str(src_idx), "")
  view_var = (
    existing_view if existing_view else gen_view_var_name(rel_name + "_neg", src_idx)
  )
  if not existing_view:
    code += i + f"auto {view_var} = {gen_view_access(get_view_slot_base(ctx, src_idx))};\n"

  neg_index_type = get_rel_index_type(ctx, rel_name)

  current_handle: str
  # Pre-narrowed handle from Cartesian setup?
  if src_idx in ctx.neg_pre_narrow:
    info = ctx.neg_pre_narrow[src_idx]
    current_handle = info.var_name
    if ctx.debug:
      code += (
        i + "// Using pre-narrowed handle (pre-Cartesian vars: "
        + ", ".join(info.pre_vars) + ")\n"
      )
    if info.in_cartesian_vars:
      narrowed_var = gen_unique_name(ctx, f"h_{rel_name}_neg_{src_idx}")
      chained = gen_chained_prefix_calls_seq(
        current_handle, info.in_cartesian_vars, info.view_var, neg_index_type,
      )
      code += i + f"auto {narrowed_var} = {chained};\n"
      current_handle = narrowed_var
  else:
    # Standard path.
    parent_handle = ctx.handle_vars.get(str(src_idx), "")
    handle_var = parent_handle if parent_handle else gen_root_handle(view_var, neg_index_type)
    current_handle = handle_var

    # Constant prefixes first (HIR indexCols puts const cols first, then vars).
    if node.const_args:
      for col_idx, const_val in node.const_args:
        const_narrowed = gen_unique_name(ctx, f"h_{rel_name}_neg_const")
        if ctx.inside_cartesian:
          code += (
            i + f"auto {const_narrowed} = {current_handle}"
            f".prefix_seq({const_val}, {view_var});\n"
          )
        else:
          code += (
            i + f"auto {const_narrowed} = {current_handle}"
            f".prefix({const_val}, tile, {view_var});\n"
          )
        current_handle = const_narrowed

    # Then variable prefixes.
    if node.prefix_vars:
      narrowed_var = gen_unique_name(ctx, f"h_{rel_name}_neg_{src_idx}")
      if ctx.inside_cartesian:
        chained = gen_chained_prefix_calls_seq(
          current_handle, node.prefix_vars, view_var, neg_index_type,
        )
      else:
        chained = gen_chained_prefix_calls(
          current_handle, node.prefix_vars, view_var,
          ctx.cartesian_bound_vars, index_type=neg_index_type,
        )
      code += i + f"auto {narrowed_var} = {chained};\n"
      current_handle = narrowed_var

  # Emit the guard or fold into active valid flag.
  if ctx.ws_cartesian_valid_var:
    v = ctx.ws_cartesian_valid_var
    code += i + f"{v} = {v} && (!{gen_valid(current_handle, neg_index_type)});\n"
    code += body
  elif ctx.tiled_cartesian_valid_var:
    v = ctx.tiled_cartesian_valid_var
    code += i + f"{v} = {v} && (!{gen_valid(current_handle, neg_index_type)});\n"
    code += body
  else:
    code += i + f"if (!{gen_valid(current_handle, neg_index_type)}) {{\n"
    code += body
    code += i + "}\n"
  return code


# -----------------------------------------------------------------------------
# jit_aggregate
# -----------------------------------------------------------------------------

def jit_aggregate(node: m.Aggregate, ctx: CodeGenContext, body: str) -> str:
  '''Bind `result_var = aggregate<Func>(handle, view);`, then emit body
  with result_var in scope.
  '''
  assert isinstance(node, m.Aggregate)
  code = ""
  i = ind(ctx)
  src_idx = node.handle_start
  rel_name = node.rel_name

  if ctx.debug:
    code += (
      i + f"// Aggregate: {node.result_var} = {node.agg_func} from {rel_name}\n"
    )
    code += (
      i + f"// MIR: (aggregate :rel {rel_name} :result {node.result_var}"
      f" :func {node.agg_func}"
      f" :prefix ({' '.join(node.prefix_vars)}) :handle {src_idx})\n"
    )

  existing_view = ctx.view_vars.get(str(src_idx), "")
  view_var = (
    existing_view if existing_view else gen_view_var_name(rel_name + "_agg", src_idx)
  )
  if not existing_view:
    code += i + f"auto {view_var} = {gen_view_access(get_view_slot_base(ctx, src_idx))};\n"

  agg_index_type = get_rel_index_type(ctx, rel_name)
  parent_handle = ctx.handle_vars.get(str(src_idx), "")
  handle_var = parent_handle if parent_handle else gen_root_handle(view_var, agg_index_type)

  if node.prefix_vars:
    narrowed_var = gen_unique_name(ctx, f"h_{rel_name}_agg_{src_idx}")
    chained = gen_chained_prefix_calls(
      handle_var, node.prefix_vars, view_var,
      ctx.cartesian_bound_vars, index_type=agg_index_type,
    )
    code += i + f"auto {narrowed_var} = {chained};\n"
    narrowed_handle = narrowed_var
  else:
    narrowed_handle = handle_var

  code += (
    i + f"auto {node.result_var} = aggregate<{node.agg_func}>"
    f"({narrowed_handle}, {view_var});\n"
  )

  # Body runs with result_var in scope — but since ctx is shared mutable
  # state in Python we add and then remove to avoid polluting caller.
  ctx.bound_vars.append(node.result_var)
  try:
    code += body
  finally:
    if ctx.bound_vars and ctx.bound_vars[-1] == node.result_var:
      ctx.bound_vars.pop()
  return code
