'''Leaf codegen helpers for pipeline operations.

Port of src/srdatalog/codegen/target_jit/jit_emit_helpers.nim.

Covers:
  - Balanced-partitioning detection + info extraction
  - jit_filter       — wraps the body in `if (cond) { ... }` (or folds
                       the condition into a valid flag for WS / tiled
                       Cartesian batched loops so warp-cooperative ops
                       don't deadlock on divergent branches)
  - jit_constant_bind — emits `auto v = <expr>;` for ConstantBind
  - jit_insert_into   — InsertInto emission: count vs materialize,
                       dedup-hash guard, lane-0 guard, WS coalesced
                       writes, tiled Cartesian ballot path
  - count_handles_in_pipeline — max `handle_start + 1` across the pipeline

The three emit procs return C++ source strings; each takes `ctx` plus
(for jit_filter / jit_constant_bind) a `body` string that will be placed
inside the emitted scope. This mirrors the Nim continuation-passing
style: the body is pre-rendered before being wrapped.
'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import srdatalog.mir.types as m
from srdatalog.codegen.jit.context import (
  CodeGenContext, ind, inc_indent, sanitize_var_name, with_bound_var,
)


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
  src1_index: list[int] = None
  src1_handle_idx: int = -1
  src2_rel_name: str = ""
  src2_index: list[int] = None
  src2_handle_idx: int = -1

  def __post_init__(self):
    if self.src1_index is None:
      self.src1_index = []
    if self.src2_index is None:
      self.src2_index = []


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
# jit_filter
# -----------------------------------------------------------------------------

def jit_filter(node: m.Filter, ctx: CodeGenContext, body: str) -> str:
  '''Emit a Filter: wrap `body` in `if (cond) { ... }` OR fold the
  condition into the active valid flag when inside a warp-cooperative
  batch loop (WS / tiled Cartesian). The latter matters because
  emit_warp_coalesced is a cooperative warp op — all threads must call
  it, so wrapping in `if` would cause divergence deadlock.

  Nim's flCode is `"return <expr>;"`; we strip the `return ` and the
  trailing `;` to turn it into a bare C++ boolean expression.
  '''
  assert isinstance(node, m.Filter)
  i = ind(ctx)

  expr = node.code.strip()
  if expr.startswith("return "):
    expr = expr[len("return "):]
  if expr.endswith(";"):
    expr = expr[:-1]

  code = ""
  if ctx.ws_cartesian_valid_var:
    # WS batch loop — fold into valid flag
    v = ctx.ws_cartesian_valid_var
    code += f"{i}{v} = {v} && ({expr});\n"
    code += body
  elif ctx.tiled_cartesian_valid_var:
    # Tiled Cartesian ballot path — fold into valid flag
    v = ctx.tiled_cartesian_valid_var
    code += f"{i}{v} = {v} && ({expr});\n"
    code += body
  else:
    code += f"{i}if ({expr}) {{\n"
    # Note: Nim increments indent but the body is already rendered at
    # the outer indent. We keep the same behavior — body is pre-indented
    # at whatever indent the caller used.
    code += body
    code += f"{i}}}\n"
  return code


# -----------------------------------------------------------------------------
# jit_constant_bind
# -----------------------------------------------------------------------------

def jit_constant_bind(
  node: m.ConstantBind, ctx: CodeGenContext, body: str,
) -> str:
  '''Emit `auto <var> = <code>;` then the body.'''
  assert isinstance(node, m.ConstantBind)
  i = ind(ctx)
  var = sanitize_var_name(node.var_name)
  code = f"{i}auto {var} = {node.code};\n"
  code += body
  return code


# -----------------------------------------------------------------------------
# jit_insert_into — the meaty one
# -----------------------------------------------------------------------------

def jit_insert_into(node: m.InsertInto, ctx: CodeGenContext) -> str:
  '''Emit an InsertInto — count phase or materialize phase, with the
  right lane-0 guard, WS coalesced-write path, tiled-Cartesian ballot
  path, or dedup-hash try_insert/check_winner wrapper.

  This is a faithful port of Nim's `jitInsertInto`. The control flow
  is intricate; comments inline mirror the Nim rationale verbatim.
  '''
  assert isinstance(node, m.InsertInto)
  code = ""
  i = ind(ctx)
  vars_list = list(node.vars)

  if ctx.debug:
    code += f"{i}// Emit: {node.rel_name}({', '.join(vars_list)})\n"

  # Output var per-relation (legacy name when no override).
  out_var = ctx.output_var_name
  if node.rel_name in ctx.output_vars:
    out_var = ctx.output_vars[node.rel_name]

  # When there is no Cartesian in the pipeline, all threads cooperatively
  # found the same result — only lane 0 should emit, else we 32x overcount.
  need_lane0_guard = not ctx.inside_cartesian

  # Dedup-hash try_insert/winner guard
  dedup_guard_open = False
  if ctx.dedup_hash_enabled and vars_list:
    sanitized = [sanitize_var_name(v) for v in vars_list]
    args_str = ", ".join(sanitized)
    code += f"{i}{{ bool _p = dedup_table.try_insert(thread_id, {args_str});\n"
    code += f"{i}  if (_p) {{\n"
    dedup_guard_open = True

  if ctx.is_counting:
    # Counting phase: one increment / one emit_direct() per body match.
    # Skip duplicate counting for secondary outputs flagged __skip_counting__.
    if out_var == "__skip_counting__":
      if ctx.debug:
        code += f"{i}// Skip counting for secondary output {node.rel_name}\n"
    else:
      if ctx.ws_enabled:
        # WS count: out_var is `local_count` (uint32_t) — just ++.
        if need_lane0_guard:
          code += (
            f"{i}if ({ctx.tile_var}.thread_rank() == 0) {out_var}++;\n"
          )
        else:
          code += f"{i}{out_var}++;\n"
      else:
        if need_lane0_guard:
          code += (
            f"{i}if ({ctx.tile_var}.thread_rank() == 0) "
            f"{out_var}.emit_direct();\n"
          )
        else:
          code += f"{i}{out_var}.emit_direct();\n"
  else:
    # Materialize phase.
    if ctx.ws_cartesian_valid_var:
      # WS Cartesian batch loop — coalesced warp writes.
      sanitized = ", ".join(sanitize_var_name(v) for v in vars_list)
      code += (
        f"{i}{out_var}.emit_warp_coalesced({ctx.tile_var}, "
        f"{ctx.ws_cartesian_valid_var}, {sanitized});\n"
      )
    elif ctx.tiled_cartesian_valid_var:
      # Tiled Cartesian ballot path — atomic-free coalesced writes using
      # warp_write_base captured from the count phase.
      sanitized = [sanitize_var_name(v) for v in vars_list]
      valid_var = ctx.tiled_cartesian_valid_var
      dest_idx = (
        out_var[len("output_ctx_"):] if out_var.startswith("output_ctx_") else "0"
      )
      # First InsertInto in the body does ballot + _tc_off once; subsequent
      # InsertIntos in the same body reuse the already-computed offset.
      if not ctx.tiled_cartesian_ballot_done:
        ctx.tiled_cartesian_ballot_done = True
        code += f"{i}{{\n"
        code += (
          f"{i}  uint32_t _tc_ballot = {ctx.tile_var}.ballot({valid_var});\n"
        )
        code += f"{i}  uint32_t _tc_active = __popc(_tc_ballot);\n"
        code += f"{i}  if (_tc_active > 0) {{\n"
        code += (
          f"{i}    uint32_t _tc_mask = "
          f"(1u << {ctx.tile_var}.thread_rank()) - 1u;\n"
        )
        code += f"{i}    uint32_t _tc_off = __popc(_tc_ballot & _tc_mask);\n"
      # Write this destination (works for first + subsequent dests).
      code += f"{i}    if ({valid_var}) {{\n"
      code += (
        f"{i}      uint32_t _tc_pos_{dest_idx} = old_size_{dest_idx} "
        f"+ warp_write_base + warp_local_count + _tc_off;\n"
      )
      for col, name in enumerate(sanitized):
        code += (
          f"{i}      output_data_{dest_idx}[{col} * "
          f"static_cast<uint32_t>(output_stride_{dest_idx}) + "
          f"_tc_pos_{dest_idx}] = {name};\n"
        )
      code += f"{i}    }}\n"
    else:
      # Baseline path.
      if ctx.dedup_hash_enabled:
        # Dedup materialize — atomicAdd for the write position.
        sanitized = [sanitize_var_name(v) for v in vars_list]
        guard_prefix = (
          f"if ({ctx.tile_var}.thread_rank() == 0) " if need_lane0_guard else ""
        )
        code += f"{i}{guard_prefix}{{\n"
        code += f"{i}  uint32_t pos = atomicAdd(atomic_write_pos, 1u);\n"
        for col, name in enumerate(sanitized):
          code += (
            f"{i}  out_data_0[(pos + out_base_0) + {col} "
            f"* out_stride_0] = {name};\n"
          )
        code += f"{i}}}\n"
      elif need_lane0_guard:
        sanitized = ", ".join(sanitize_var_name(v) for v in vars_list)
        code += (
          f"{i}if ({ctx.tile_var}.thread_rank() == 0) "
          f"{out_var}.emit_direct({sanitized});\n"
        )
      else:
        sanitized = ", ".join(sanitize_var_name(v) for v in vars_list)
        code += f"{i}{out_var}.emit_direct({sanitized});\n"

  if dedup_guard_open:
    code += f"{i}}} }}\n"  # close the if (_p) and the outer { block
  return code


# -----------------------------------------------------------------------------
# Pipeline handle counting
# -----------------------------------------------------------------------------

def _assign_handle_positions_rec(node: m.MirNode, offset_box: list[int]) -> None:
  '''Recursively assign `handle_start` to this node and any children.

  `offset_box` is a single-element list used as a mutable counter
  (Python closures can't reassign captured ints cleanly in a loop).
  '''
  if isinstance(node, m.ColumnSource):
    node.handle_start = offset_box[0]
    offset_box[0] += 1
  elif isinstance(node, m.Scan):
    node.handle_start = offset_box[0]
    offset_box[0] += 1
  elif isinstance(node, m.Aggregate):
    node.handle_start = offset_box[0]
    offset_box[0] += 1
  elif isinstance(node, m.Negation):
    node.handle_start = offset_box[0]
    offset_box[0] += 1
  elif isinstance(node, m.ColumnJoin):
    node.handle_start = offset_box[0]
    for src in node.sources:
      _assign_handle_positions_rec(src, offset_box)
  elif isinstance(node, m.CartesianJoin):
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
    if isinstance(op, m.ColumnJoin):
      for src in op.sources:
        h = getattr(src, "handle_start", -1)
        result = max(result, h + 1)
    elif isinstance(op, m.CartesianJoin):
      for src in op.sources:
        h = getattr(src, "handle_start", -1)
        result = max(result, h + 1)
    elif isinstance(op, m.Scan):
      result = max(result, getattr(op, "handle_start", -1) + 1)
    elif isinstance(op, m.Negation):
      result = max(result, getattr(op, "handle_start", -1) + 1)
    elif isinstance(op, m.Aggregate):
      result = max(result, getattr(op, "handle_start", -1) + 1)
  return result
