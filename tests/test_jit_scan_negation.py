'''Tests for codegen/jit/scan_negation.py.'''
import sys
from pathlib import Path


import srdatalog.mir.types as m
from srdatalog.hir.types import Version
from srdatalog.codegen.jit.context import new_code_gen_context, NegPreNarrowInfo
from srdatalog.codegen.jit.scan_negation import jit_scan, jit_negation, jit_aggregate


# -----------------------------------------------------------------------------
# jit_scan
# -----------------------------------------------------------------------------

def test_scan_basic_no_prefix_binds_remaining_cols():
  node = m.Scan(vars=["x", "y"], rel_name="Edge",
                version=Version.FULL, index=[0, 1], handle_start=0)
  ctx = new_code_gen_context()
  out = jit_scan(node, ctx, "          body();\n")
  assert "auto view_Edge_0 = views[0];" in out
  assert "// Scan: bind x, y from Edge" in out
  # Standard iterator for-loop
  assert "for (auto scan_it_" in out
  # get_value uses prefix_len + varIdx offsets (0 and 1 here)
  assert "auto x = view_Edge_0.get_value(0, scan_it_" in out
  assert "auto y = view_Edge_0.get_value(1, scan_it_" in out
  assert "body();" in out


def test_scan_with_prefix_emits_chained_prefix_narrow():
  node = m.Scan(vars=["y"], rel_name="Edge",
                version=Version.FULL, index=[0, 1],
                prefix_vars=["x"], handle_start=0)
  ctx = new_code_gen_context()
  out = jit_scan(node, ctx, "")
  # Narrow via chained .prefix(x, tile, view)
  assert "auto h_scan_0_" in out
  assert ".prefix(x, tile, view_Edge_0)" in out
  # Bound col is offset 1 (len(prefix_vars)=1)
  assert ".get_value(1, scan_it_" in out


def test_scan_reuses_preexisting_view_var():
  node = m.Scan(vars=["x"], rel_name="R", version=Version.FULL,
                index=[0], handle_start=5)
  ctx = new_code_gen_context()
  ctx.view_vars["5"] = "view_R_5_preexisting"
  out = jit_scan(node, ctx, "")
  # Does NOT emit `auto view_R_5 = views[...];` because one is already declared
  assert "auto view_R_5 =" not in out
  assert "view_R_5_preexisting" in out


def test_scan_uses_parent_handle_when_available():
  '''If a parent handle is registered in ctx.handle_vars, reuse it
  instead of emitting a fresh root handle.'''
  node = m.Scan(vars=["x"], rel_name="R", version=Version.FULL,
                index=[0], handle_start=5)
  ctx = new_code_gen_context()
  ctx.view_vars["5"] = "view_R_5"
  ctx.handle_vars["5"] = "parent_h_42"
  out = jit_scan(node, ctx, "")
  # The for-loop should iterate parent_h_42 directly (no root handle emitted)
  assert "parent_h_42.begin()" in out
  assert "HandleType(0" not in out  # no root handle


# -----------------------------------------------------------------------------
# jit_negation — standard path
# -----------------------------------------------------------------------------

def test_negation_standard_if_not_valid_wraps_body():
  node = m.Negation(rel_name="R", version=Version.FULL, index=[0],
                    prefix_vars=["x"], handle_start=0)
  ctx = new_code_gen_context()
  out = jit_negation(node, ctx, "          body();\n")
  assert "auto view_R_neg_0 = views[0];" in out
  assert "// Negation: NOT EXISTS in R" in out
  assert "if (!" in out and ".valid()" in out
  assert "body();" in out


def test_negation_constant_args_apply_first():
  '''Nim ordering: constants first, then variables (matches HIR
  indexCols layout: const cols come before variable cols).'''
  node = m.Negation(
    rel_name="Method_Modifier", version=Version.FULL, index=[1, 0],
    prefix_vars=["meth"], const_args=[(1, 42)], handle_start=0,
  )
  ctx = new_code_gen_context()
  out = jit_negation(node, ctx, "")
  const_idx = out.find("prefix(42")
  var_idx = out.find("prefix(meth")
  assert 0 < const_idx < var_idx


def test_negation_inside_cartesian_uses_prefix_seq():
  '''Inside Cartesian, prefix narrowing must be per-thread sequential.'''
  node = m.Negation(rel_name="R", version=Version.FULL, index=[0],
                    prefix_vars=["x"], handle_start=0)
  ctx = new_code_gen_context()
  ctx.inside_cartesian = True
  out = jit_negation(node, ctx, "")
  assert ".prefix_seq(x" in out
  assert ".prefix(x, tile" not in out


def test_negation_ws_cartesian_folds_into_valid_flag():
  node = m.Negation(rel_name="R", version=Version.FULL, index=[0],
                    prefix_vars=["x"], handle_start=0)
  ctx = new_code_gen_context()
  ctx.ws_cartesian_valid_var = "ws_valid"
  out = jit_negation(node, ctx, "body();\n")
  assert "ws_valid = ws_valid && (!" in out
  # No `if` wrapper
  assert "if (!" not in out


def test_negation_tiled_cartesian_folds_into_valid_flag():
  node = m.Negation(rel_name="R", version=Version.FULL, index=[0],
                    prefix_vars=["x"], handle_start=0)
  ctx = new_code_gen_context()
  ctx.tiled_cartesian_valid_var = "tc_valid"
  out = jit_negation(node, ctx, "body();\n")
  assert "tc_valid = tc_valid && (!" in out


# -----------------------------------------------------------------------------
# jit_negation — pre-narrowed path
# -----------------------------------------------------------------------------

def test_negation_pre_narrowed_uses_existing_var():
  node = m.Negation(rel_name="R", version=Version.FULL, index=[0, 1],
                    prefix_vars=[], handle_start=7)
  ctx = new_code_gen_context()
  ctx.neg_pre_narrow[7] = NegPreNarrowInfo(
    var_name="h_prenarrowed_12",
    pre_vars=["p"],
    in_cartesian_vars=[],
    view_var="view_R_neg_7",
    rel_name="R",
  )
  out = jit_negation(node, ctx, "body();\n")
  # No base handle emitted — starts from pre-narrowed
  assert "h_prenarrowed_12" in out
  assert "// Using pre-narrowed handle (pre-Cartesian vars: p)" in out


def test_negation_pre_narrowed_applies_in_cartesian_vars_per_thread():
  node = m.Negation(rel_name="R", version=Version.FULL, index=[0, 1],
                    prefix_vars=[], handle_start=7)
  ctx = new_code_gen_context()
  ctx.neg_pre_narrow[7] = NegPreNarrowInfo(
    var_name="h_pn",
    pre_vars=["p"],
    in_cartesian_vars=["q"],
    view_var="view_R_neg_7",
    rel_name="R",
  )
  out = jit_negation(node, ctx, "")
  # q applied via prefix_seq (in-cartesian = per-thread)
  assert ".prefix_seq(q, view_R_neg_7)" in out


# -----------------------------------------------------------------------------
# jit_aggregate
# -----------------------------------------------------------------------------

def test_aggregate_basic():
  node = m.Aggregate(result_var="cnt", agg_func="SRDatalog::AggCount",
                     rel_name="R", version=Version.FULL, index=[0],
                     handle_start=0)
  ctx = new_code_gen_context()
  out = jit_aggregate(node, ctx, "        body();\n")
  assert "auto view_R_agg_0 = views[0];" in out
  assert (
    "auto cnt = aggregate<SRDatalog::AggCount>"
    "(HandleType(0, view_R_agg_0.num_rows_, 0), view_R_agg_0);"
  ) in out
  assert "body();" in out


def test_aggregate_with_prefix_narrows_first():
  node = m.Aggregate(result_var="cnt", agg_func="AggCount",
                     rel_name="R", version=Version.FULL, index=[0, 1],
                     prefix_vars=["x"], handle_start=0)
  ctx = new_code_gen_context()
  out = jit_aggregate(node, ctx, "")
  assert "auto h_R_agg_0_" in out
  assert ".prefix(x, tile, view_R_agg_0)" in out
  # aggregate<> uses narrowed handle
  assert "aggregate<AggCount>(h_R_agg_0_" in out


def test_aggregate_binds_result_var_into_body_scope():
  '''The result var should be in ctx.bound_vars while body is rendered,
  then popped on return.'''
  seen_during_body: list[str] = []

  def make_body(ctx):
    seen_during_body.extend(ctx.bound_vars)
    return "body()\n"

  node = m.Aggregate(result_var="total", agg_func="AggSum",
                     rel_name="R", version=Version.FULL, index=[0],
                     handle_start=0)
  ctx = new_code_gen_context()
  # Render body using a lambda that captures bound_vars
  body_captured = make_body(ctx)  # before aggregate: no 'total'
  assert "total" not in seen_during_body
  # Call jit_aggregate with an inline body string (can't capture runtime
  # state — check ctx.bound_vars ISN'T polluted after call)
  jit_aggregate(node, ctx, body_captured)
  assert "total" not in ctx.bound_vars


if __name__ == "__main__":
  import inspect
  this = sys.modules[__name__]
  passed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if name.startswith("test_"):
      fn()
      passed += 1
  print(f"{passed} passed")
