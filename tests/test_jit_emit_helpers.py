'''Tests for codegen/jit/emit_helpers.py.

Covers the balanced-scan helpers, jit_filter / jit_constant_bind /
jit_insert_into emission across every branch (count / materialize,
lane-0 guard, WS coalesced, tiled Cartesian ballot, dedup-hash), and
handle counting.
'''
import sys
from pathlib import Path


import srdatalog.mir.types as m
from srdatalog.hir.types import Version
from srdatalog.codegen.jit.context import new_code_gen_context
from srdatalog.codegen.jit.emit_helpers import (
  has_balanced_scan,
  has_tiled_cartesian_eligible,
  get_balanced_scan_info,
  BalancedScanInfo,
  jit_filter,
  jit_constant_bind,
  jit_insert_into,
  count_handles_in_pipeline,
)


def _cs(rel, ver, idx, prefix=(), handle_start=-1):
  return m.ColumnSource(
    rel_name=rel, version=ver, index=idx,
    prefix_vars=list(prefix), handle_start=handle_start,
  )


# -----------------------------------------------------------------------------
# Balanced-scan detection
# -----------------------------------------------------------------------------

def test_has_balanced_scan_when_root():
  bs = m.BalancedScan(
    group_var="z",
    source1=_cs("R", Version.DELTA, [0, 1]),
    source2=_cs("S", Version.FULL, [1, 0]),
    vars1=["x"],
    vars2=["y"],
  )
  assert has_balanced_scan([bs]) is True


def test_has_balanced_scan_false_when_not_root():
  # BS buried second position -> False (only root counts)
  scan = m.Scan(vars=["x"], rel_name="R", version=Version.FULL, index=[0])
  assert has_balanced_scan([scan]) is False
  assert has_balanced_scan([]) is False


def test_has_tiled_cartesian_eligible_two_sources_one_var_each():
  cart = m.CartesianJoin(
    vars=["x", "y"],
    sources=[_cs("R", Version.FULL, [0, 1]), _cs("S", Version.FULL, [0, 1])],
    var_from_source=[["x"], ["y"]],
  )
  assert has_tiled_cartesian_eligible([cart]) is True


def test_has_tiled_cartesian_not_eligible_three_sources():
  cart = m.CartesianJoin(
    vars=["x", "y", "z"],
    sources=[_cs("R", Version.FULL, [0]), _cs("S", Version.FULL, [0]),
              _cs("T", Version.FULL, [0])],
    var_from_source=[["x"], ["y"], ["z"]],
  )
  assert has_tiled_cartesian_eligible([cart]) is False


def test_has_tiled_cartesian_not_eligible_multi_var_per_source():
  cart = m.CartesianJoin(
    vars=["x", "y"],
    sources=[_cs("R", Version.FULL, [0, 1]), _cs("S", Version.FULL, [0, 1])],
    var_from_source=[["x", "y"], []],
  )
  assert has_tiled_cartesian_eligible([cart]) is False


def test_get_balanced_scan_info_populated():
  bs = m.BalancedScan(
    group_var="z",
    source1=_cs("R", Version.DELTA, [0, 1], handle_start=3),
    source2=_cs("S", Version.FULL, [1, 0], handle_start=5),
    vars1=["x"],
    vars2=["y"],
  )
  info = get_balanced_scan_info([bs])
  assert info.group_var == "z"
  assert info.src1_rel_name == "R"
  assert info.src1_index == [0, 1]
  assert info.src1_handle_idx == 3
  assert info.src2_rel_name == "S"
  assert info.src2_index == [1, 0]
  assert info.src2_handle_idx == 5


def test_get_balanced_scan_info_sentinel_when_not_root():
  info = get_balanced_scan_info([])
  assert info == BalancedScanInfo()


# -----------------------------------------------------------------------------
# jit_filter
# -----------------------------------------------------------------------------

def test_filter_baseline_wraps_in_if():
  node = m.Filter(vars=["x", "y"], code="return x != y;")
  ctx = new_code_gen_context()
  out = jit_filter(node, ctx, "        body();\n")
  assert "if (x != y) {" in out
  assert "body();" in out
  assert out.strip().endswith("}")


def test_filter_strips_return_and_semicolon():
  node = m.Filter(vars=["x"], code="  return (x > 0);  ")
  ctx = new_code_gen_context()
  out = jit_filter(node, ctx, "")
  assert "if ((x > 0))" in out


def test_filter_ws_cartesian_folds_into_valid():
  node = m.Filter(vars=["x", "y"], code="return x != y;")
  ctx = new_code_gen_context()
  ctx.ws_cartesian_valid_var = "ws_valid"
  out = jit_filter(node, ctx, "body();\n")
  assert "ws_valid = ws_valid && (x != y);" in out
  assert "if (" not in out  # no branching inside batch loop


def test_filter_tiled_cartesian_folds_into_valid():
  node = m.Filter(vars=["x"], code="return x > 0;")
  ctx = new_code_gen_context()
  ctx.tiled_cartesian_valid_var = "tc_valid"
  out = jit_filter(node, ctx, "body();\n")
  assert "tc_valid = tc_valid && (x > 0);" in out


# -----------------------------------------------------------------------------
# jit_constant_bind
# -----------------------------------------------------------------------------

def test_constant_bind_emits_auto_decl():
  node = m.ConstantBind(var_name="my_var", code="x + 1", deps=["x"])
  ctx = new_code_gen_context()
  out = jit_constant_bind(node, ctx, "body();\n")
  assert "auto my_var = x + 1;" in out
  assert "body();" in out


def test_constant_bind_sanitizes_keyword():
  node = m.ConstantBind(var_name="int", code="42", deps=[])
  ctx = new_code_gen_context()
  out = jit_constant_bind(node, ctx, "")
  assert "auto int_val = 42;" in out


# -----------------------------------------------------------------------------
# jit_insert_into
# -----------------------------------------------------------------------------

def _make_insert(rel="PointsTo", vars_=("y", "x"), idx=(0, 1)):
  return m.InsertInto(
    rel_name=rel, version=Version.NEW, vars=list(vars_), index=list(idx),
  )


def test_insert_count_lane0_guard_no_cartesian():
  '''Count phase, no Cartesian: lane-0 guard emits emit_direct().'''
  ii = _make_insert()
  ctx = new_code_gen_context()
  ctx.is_counting = True
  out = jit_insert_into(ii, ctx)
  assert "// Emit: PointsTo(y, x)" in out
  assert f"if ({ctx.tile_var}.thread_rank() == 0) output.emit_direct();" in out


def test_insert_count_no_lane_guard_inside_cartesian():
  ii = _make_insert()
  ctx = new_code_gen_context()
  ctx.is_counting = True
  ctx.inside_cartesian = True
  out = jit_insert_into(ii, ctx)
  assert "output.emit_direct();" in out
  assert "thread_rank() == 0" not in out


def test_insert_count_skip_counting_secondary_dest():
  ii = _make_insert()
  ctx = new_code_gen_context()
  ctx.is_counting = True
  ctx.output_vars["PointsTo"] = "__skip_counting__"
  out = jit_insert_into(ii, ctx)
  assert "Skip counting for secondary output PointsTo" in out
  assert "emit_direct" not in out


def test_insert_count_ws_enabled_increments_local_counter():
  ii = _make_insert()
  ctx = new_code_gen_context()
  ctx.is_counting = True
  ctx.ws_enabled = True
  ctx.output_vars["PointsTo"] = "local_count"
  out = jit_insert_into(ii, ctx)
  assert f"if ({ctx.tile_var}.thread_rank() == 0) local_count++;" in out


def test_insert_materialize_lane0_emit_direct_with_args():
  ii = _make_insert()
  ctx = new_code_gen_context()
  ctx.output_vars["PointsTo"] = "ctx0"
  out = jit_insert_into(ii, ctx)
  assert f"if ({ctx.tile_var}.thread_rank() == 0) ctx0.emit_direct(y, x);" in out


def test_insert_materialize_inside_cartesian_no_guard():
  ii = _make_insert()
  ctx = new_code_gen_context()
  ctx.inside_cartesian = True
  ctx.output_vars["PointsTo"] = "ctx0"
  out = jit_insert_into(ii, ctx)
  assert "ctx0.emit_direct(y, x);" in out
  assert "thread_rank() == 0" not in out


def test_insert_materialize_ws_cartesian_emit_warp_coalesced():
  ii = _make_insert()
  ctx = new_code_gen_context()
  ctx.ws_cartesian_valid_var = "ws_valid"
  ctx.output_vars["PointsTo"] = "ctx0"
  out = jit_insert_into(ii, ctx)
  assert (
    f"ctx0.emit_warp_coalesced({ctx.tile_var}, ws_valid, y, x);" in out
  )


def test_insert_materialize_tiled_cartesian_ballot_first_and_subsequent():
  '''First InsertInto sets up ballot + _tc_off; second reuses, doesn't
  re-ballot.'''
  ii = _make_insert()
  ctx = new_code_gen_context()
  ctx.tiled_cartesian_valid_var = "tc_valid"
  ctx.output_vars["PointsTo"] = "output_ctx_0"

  out1 = jit_insert_into(ii, ctx)
  assert "_tc_ballot = tile.ballot(tc_valid)" in out1
  assert "_tc_pos_0 = old_size_0 + warp_write_base + warp_local_count + _tc_off" in out1

  # Second call — ballot already done
  out2 = jit_insert_into(ii, ctx)
  assert "_tc_ballot" not in out2
  # Still emits the _tc_pos_0 write
  assert "_tc_pos_0" in out2


def test_insert_materialize_dedup_hash_wraps_in_try_insert():
  ii = _make_insert()
  ctx = new_code_gen_context()
  ctx.dedup_hash_enabled = True
  ctx.output_vars["PointsTo"] = "ctx0"
  out = jit_insert_into(ii, ctx)
  # Outer dedup wrapper
  assert "bool _p = dedup_table.try_insert(thread_id, y, x);" in out
  assert "if (_p) {" in out
  # Inner materialize uses atomicAdd write path
  assert "uint32_t pos = atomicAdd(atomic_write_pos, 1u);" in out
  assert "out_data_0[(pos + out_base_0) + 0 * out_stride_0] = y;" in out
  assert "out_data_0[(pos + out_base_0) + 1 * out_stride_0] = x;" in out


def test_insert_sanitizes_keyword_vars():
  ii = _make_insert(vars_=("int", "y"))
  ctx = new_code_gen_context()
  ctx.output_vars["PointsTo"] = "ctx0"
  out = jit_insert_into(ii, ctx)
  assert "ctx0.emit_direct(int_val, y);" in out


# -----------------------------------------------------------------------------
# count_handles_in_pipeline
# -----------------------------------------------------------------------------

def test_count_handles_empty_pipeline():
  assert count_handles_in_pipeline([]) == 0


def test_count_handles_single_scan():
  sc = m.Scan(vars=["x"], rel_name="R", version=Version.FULL, index=[0])
  sc.handle_start = 3
  assert count_handles_in_pipeline([sc]) == 4


def test_count_handles_column_join_picks_max():
  cj = m.ColumnJoin(
    var_name="z",
    sources=[
      _cs("R", Version.DELTA, [0, 1], handle_start=2),
      _cs("S", Version.FULL, [1, 0], handle_start=7),
    ],
  )
  assert count_handles_in_pipeline([cj]) == 8


def test_count_handles_across_pipeline_takes_overall_max():
  sc = m.Scan(vars=["x"], rel_name="R", version=Version.FULL, index=[0])
  sc.handle_start = 1
  neg = m.Negation(rel_name="N", version=Version.FULL, index=[0])
  neg.handle_start = 5
  agg = m.Aggregate(result_var="c", agg_func="AggCount", rel_name="A",
                    version=Version.FULL, index=[0])
  agg.handle_start = 3
  assert count_handles_in_pipeline([sc, neg, agg]) == 6


if __name__ == "__main__":
  import inspect
  this = sys.modules[__name__]
  passed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if name.startswith("test_"):
      fn()
      passed += 1
  print(f"{passed} passed")
