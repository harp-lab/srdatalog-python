'''Tests for tiled Cartesian codegen path.

Activated when:
  - ctx.tiled_cartesian_enabled is True
  - not ctx.is_counting
  - exactly 2 sources
  - each source contributes exactly 1 var

Emits `if (total > 32) { <tiled smem loop> } else { <fallback> }`.
When a `tiled_body` is passed (the pipeline renders one with
`tiled_cartesian_valid_var` set), both branches thread the valid
flag through for ballot-based coalesced writes.
'''
import sys
from pathlib import Path


import srdatalog.mir_types as m
from srdatalog.hir_types import Version
from srdatalog.codegen.jit.context import new_code_gen_context
from srdatalog.codegen.jit.instructions import jit_nested_cartesian_join
from srdatalog.codegen.jit.pipeline import jit_nested_pipeline


def _cs(rel, ver, idx, prefix=(), handle_start=0):
  return m.ColumnSource(
    rel_name=rel, version=ver, index=idx,
    prefix_vars=list(prefix), handle_start=handle_start,
  )


def _cart_2src_1var_each(rel0="R", rel1="S"):
  return m.CartesianJoin(
    vars=["x", "y"],
    sources=[
      _cs(rel0, Version.FULL, [0], handle_start=0),
      _cs(rel1, Version.FULL, [0], handle_start=1),
    ],
    var_from_source=[["x"], ["y"]],
  )


# -----------------------------------------------------------------------------
# Eligibility (shape + ctx flags)
# -----------------------------------------------------------------------------

def test_not_eligible_without_flag_skips_tiled_branch():
  cart = _cart_2src_1var_each()
  ctx = new_code_gen_context()
  # Default: tiled_cartesian_enabled = False
  out = jit_nested_cartesian_join(cart, ctx, "body();\n")
  # No tiled `if (total > 32)` dispatch
  assert "if (total_" not in out or "> 32)" not in out
  # Standard flat loop still emitted
  assert "for (uint32_t flat_idx_" in out


def test_not_eligible_counting_mode_skips_tiled_branch():
  cart = _cart_2src_1var_each()
  ctx = new_code_gen_context()
  ctx.tiled_cartesian_enabled = True
  ctx.is_counting = True
  out = jit_nested_cartesian_join(cart, ctx, "body();\n")
  assert "if (total_" not in out or "> 32)" not in out


def test_not_eligible_three_sources_skips_tiled_branch():
  cart = m.CartesianJoin(
    vars=["x", "y", "z"],
    sources=[
      _cs("R", Version.FULL, [0], handle_start=0),
      _cs("S", Version.FULL, [0], handle_start=1),
      _cs("T", Version.FULL, [0], handle_start=2),
    ],
    var_from_source=[["x"], ["y"], ["z"]],
  )
  ctx = new_code_gen_context()
  ctx.tiled_cartesian_enabled = True
  out = jit_nested_cartesian_join(cart, ctx, "body();\n")
  assert "> 32)" not in out


def test_not_eligible_multi_var_per_source_skips_tiled_branch():
  cart = m.CartesianJoin(
    vars=["x", "y"],
    sources=[
      _cs("R", Version.FULL, [0, 1], handle_start=0),
      _cs("S", Version.FULL, [0], handle_start=1),
    ],
    var_from_source=[["x", "y"], []],
  )
  ctx = new_code_gen_context()
  ctx.tiled_cartesian_enabled = True
  out = jit_nested_cartesian_join(cart, ctx, "body();\n")
  assert "> 32)" not in out


# -----------------------------------------------------------------------------
# Tiled emission shape (eligible + no tiled_body)
# -----------------------------------------------------------------------------

def test_tiled_without_tiled_body_uses_standard_body():
  cart = _cart_2src_1var_each()
  ctx = new_code_gen_context()
  ctx.tiled_cartesian_enabled = True
  out = jit_nested_cartesian_join(cart, ctx, "body();\n")

  # Dispatch: if (total > 32) tiled else fallback
  assert "> 32) {" in out
  # Tiled path: smem pre-load loops
  assert "for (uint32_t t0_base_" in out
  assert "for (uint32_t t1_base_" in out
  assert "s_cart[warp_in_block][0][_ti]" in out
  assert "s_cart[warp_in_block][1][_ti]" in out
  # Tile total computed as t0_len * t1_len
  assert "uint32_t tile_total_" in out
  # Explicit tile.sync() between pre-load and body, and after
  assert ".sync();" in out
  # Inner flat loop over the tile product (NO valid flag w/o tiled_body)
  assert "bool tc_valid_" not in out
  # Body bound vars from smem (no valid gating)
  assert "auto x = s_cart[warp_in_block][0][" in out
  assert "auto y = s_cart[warp_in_block][1][" in out
  # Fallback path: flat loop + major_is_1 decomposition
  assert "} else {" in out
  assert "const bool major_is_1_" in out
  # Body emitted twice (tiled + fallback)
  assert out.count("body();") == 2


def test_tiled_with_tiled_body_uses_ballot_valid_flag():
  cart = _cart_2src_1var_each()
  ctx = new_code_gen_context()
  ctx.tiled_cartesian_enabled = True
  # Pipeline would set this when rendering the tiled body — simulate.
  ctx.tiled_cartesian_valid_var = "tc_valid_99"
  out = jit_nested_cartesian_join(
    cart, ctx, "body_fallback();\n", tiled_body="body_ballot();\n",
  )

  # Valid-flag path in tiled branch
  assert "bool tc_valid_99 =" in out
  # Smem reads gate on valid (ternary ? : ValueType{0})
  assert "tc_valid_99 ?" in out
  assert ": ValueType{0};" in out
  # Fallback also uses valid-flag batched loop (ballot-coalesced writes)
  assert "uint32_t fb_batch_" in out
  # The ballot-path tiled_body appears twice (tiled + fallback variants)
  assert out.count("body_ballot();") == 2
  # The non-ballot body was unused
  assert "body_fallback();" not in out


# -----------------------------------------------------------------------------
# Pipeline-level integration
# -----------------------------------------------------------------------------

def test_pipeline_renders_both_bodies_when_tiled_eligible():
  '''End-to-end via jit_nested_pipeline: with tiled_cartesian_enabled +
  eligible Cartesian, the pipeline renders two versions of the rest
  (one tiled_cartesian_valid_var-set, one without). When the tiled
  body is non-empty, BOTH the tiled branch and the fallback branch
  use the ballot-coalesced-write path — matches Nim.'''
  cart = _cart_2src_1var_each()
  insert = m.InsertInto(
    rel_name="Out", version=Version.NEW, vars=["x", "y"], index=[0, 1],
  )
  ctx = new_code_gen_context()
  ctx.tiled_cartesian_enabled = True
  ctx.output_vars["Out"] = "output_ctx_0"
  out = jit_nested_pipeline([cart, insert], ctx)

  # Tiled dispatch emitted
  assert "> 32) {" in out
  # Ballot path used by both branches (jit_insert_into's
  # tiled_cartesian path emits warp_write_base + _tc_pos_0).
  assert "_tc_ballot =" in out
  assert "warp_write_base" in out
  # _tc_ballot appears twice — once per branch (tiled + fallback).
  assert out.count("_tc_ballot =") == 2


def test_pipeline_not_eligible_uses_standard_cartesian():
  '''When tiled_cartesian_enabled=False, the pipeline falls back to
  single-body rendering (the standard flat-loop Cartesian).'''
  cart = _cart_2src_1var_each()
  insert = m.InsertInto(
    rel_name="Out", version=Version.NEW, vars=["x", "y"], index=[0, 1],
  )
  ctx = new_code_gen_context()
  # ctx.tiled_cartesian_enabled = False (default)
  ctx.output_vars["Out"] = "output_ctx_0"
  out = jit_nested_pipeline([cart, insert], ctx)
  assert "> 32) {" not in out
  assert "s_cart[warp_in_block]" not in out
  # Standard flat loop with major_is_1 decomposition.
  assert "const bool major_is_1_" in out


if __name__ == "__main__":
  import inspect
  this = sys.modules[__name__]
  passed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if name.startswith("test_"):
      fn()
      passed += 1
  print(f"{passed} passed")
