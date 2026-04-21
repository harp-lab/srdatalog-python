'''Tests for codegen/jit/materialized.py — host-side Thrust executor
for binary-join (materialized) pipelines.
'''

import sys

import srdatalog.mir.types as m
from srdatalog.codegen.jit.context import new_code_gen_context
from srdatalog.codegen.jit.materialized import (
  _version_cpp,
  gen_materialized_join_helpers,
  gen_materialized_join_kernel,
  gen_materialized_runner,
  is_materialized_pipeline,
)
from srdatalog.hir.types import Version

# -----------------------------------------------------------------------------
# is_materialized_pipeline
# -----------------------------------------------------------------------------


def test_is_materialized_pipeline_with_probe_join():
  ops = [
    m.Scan(vars=["x"], rel_name="R", version=Version.FULL, index=[0]),
    m.ProbeJoin(
      probe_rel="S",
      probe_version=Version.FULL,
      probe_index=[0],
      join_key="x",
      output_buffer="buf_1",
    ),
  ]
  assert is_materialized_pipeline(ops) is True


def test_is_materialized_pipeline_without_probe_join():
  ops = [
    m.Scan(vars=["x"], rel_name="R", version=Version.FULL, index=[0]),
    m.InsertInto(rel_name="T", version=Version.NEW, vars=["x"], index=[0]),
  ]
  assert is_materialized_pipeline(ops) is False


# -----------------------------------------------------------------------------
# gen_materialized_join_helpers (static string)
# -----------------------------------------------------------------------------


def test_helpers_contains_three_kernels():
  helpers = gen_materialized_join_helpers()
  assert "__global__ void probe_count_matches_kernel" in helpers
  assert "__global__ void probe_materialize_pairs_kernel" in helpers
  assert "__global__ void gather_column_kernel" in helpers


def test_helpers_uses_thrust_view_patterns():
  helpers = gen_materialized_join_helpers()
  assert "view.get_range(key)" in helpers
  assert "counts[tid] = range.second - range.first;" in helpers


# -----------------------------------------------------------------------------
# _version_cpp normalizer
# -----------------------------------------------------------------------------


def test_version_cpp_enum():
  assert _version_cpp(Version.FULL) == "FULL_VER"
  assert _version_cpp(Version.DELTA) == "DELTA_VER"
  assert _version_cpp(Version.NEW) == "NEW_VER"


def test_version_cpp_strings():
  assert _version_cpp("DELTA") == "DELTA_VER"
  assert _version_cpp("FullVer") == "FULL_VER"
  assert _version_cpp("NEW_VER") == "NEW_VER"
  assert _version_cpp("unknown") == "FULL_VER"


# -----------------------------------------------------------------------------
# gen_materialized_runner — missing scan fallback
# -----------------------------------------------------------------------------


def test_runner_errors_on_missing_scan():
  ep = m.ExecutePipeline(
    pipeline=[
      m.InsertInto(rel_name="T", version=Version.NEW, vars=["x"], index=[0]),
    ],
    source_specs=[],
    dest_specs=[
      m.InsertInto(rel_name="T", version=Version.NEW, vars=["x"], index=[0]),
    ],
    rule_name="Broken",
  )
  out = gen_materialized_runner(ep, "Plan_DB_DeviceDB")
  assert "ERROR: Materialized join requires moScan as first op" in out


# -----------------------------------------------------------------------------
# gen_materialized_runner — Scan-only (no probes)
# -----------------------------------------------------------------------------


def test_runner_scan_only_no_probes():
  scan = m.Scan(vars=["x"], rel_name="AddressOf", version=Version.FULL, index=[0, 1])
  insert = m.InsertInto(rel_name="PointsTo", version=Version.NEW, vars=["x"], index=[0])
  ep = m.ExecutePipeline(
    pipeline=[scan, insert],
    source_specs=[scan],
    dest_specs=[insert],
    rule_name="Base",
  )
  out = gen_materialized_runner(ep, "Andersen_DB_DeviceDB")

  # Runner struct + standard aliases
  assert "struct JitRunner_Base {" in out
  assert "using DB = Andersen_DB_DeviceDB;" in out
  assert "using FirstSchema = AddressOf;" in out
  assert "using DestSchema = PointsTo;" in out
  assert "using ValueType = typename FirstSchema::intern_value_type;" in out
  assert "using ViewType = typename IndexType::NodeView;" in out
  # execute() + NVTX range
  assert "static void execute(DB& db, uint32_t iteration = 0) {" in out
  assert 'nvtxRangePushA("Base");' in out
  # Phase 1: initial data
  assert "get_relation_by_schema<AddressOf, FULL_VER>(db)" in out
  assert "SRDatalog::IndexSpec{{0, 1}}" in out
  assert "thrust::device_vector<uint32_t> rowids_0(n_0);" in out
  # No probe joins -> no left_/right_ buffers
  assert "left_1" not in out
  # Phase 4 insert path
  assert "get_relation_by_schema<PointsTo, NEW_VER>(db);" in out
  assert "dest.resize_interned_columns(new_size);" in out
  # No-probes chain-gather branch (uses rowids_0 directly).
  assert "thrust::gather(thrust::device, rowids_0.begin()," in out
  assert "view_0.col_data_, out_col_0.begin());" in out


# -----------------------------------------------------------------------------
# gen_materialized_runner — Scan + ProbeJoin
# -----------------------------------------------------------------------------


def test_runner_scan_plus_two_probes():
  scan = m.Scan(vars=["a", "b"], rel_name="R1", version=Version.FULL, index=[0, 1])
  probe1 = m.ProbeJoin(
    probe_rel="R2",
    probe_version=Version.FULL,
    probe_index=[0, 1],
    join_key="b",
    output_buffer="buf_1",
  )
  probe2 = m.ProbeJoin(
    probe_rel="R3",
    probe_version=Version.DELTA,
    probe_index=[0, 1],
    join_key="c",
    output_buffer="buf_2",
  )
  insert = m.InsertInto(
    rel_name="Out",
    version=Version.NEW,
    vars=["a", "d"],
    index=[0, 1],
  )
  ep = m.ExecutePipeline(
    pipeline=[scan, probe1, probe2, insert],
    source_specs=[scan],
    dest_specs=[insert],
    rule_name="JoinRule",
  )
  out = gen_materialized_runner(ep, "Plan_DB_DeviceDB")

  # Both probes emitted with rel_1 / rel_2 naming
  assert "// Phase 2.1: Probe R2 on b" in out
  assert "// Phase 2.2: Probe R3 on c" in out
  assert "auto& rel_1 = get_relation_by_schema<R2, FULL_VER>(db);" in out
  assert "auto& rel_2 = get_relation_by_schema<R3, DELTA_VER>(db);" in out
  # Per-probe keys/counts/offsets/pair buffers
  assert "thrust::device_vector<ValueType> keys_1(current_n);" in out
  assert "thrust::device_vector<uint32_t> counts_1(current_n);" in out
  assert "uint32_t total_1 = compute_output_offsets(counts_1, offsets_1);" in out
  assert "thrust::device_vector<uint32_t> left_1(total_1);" in out
  assert "thrust::device_vector<uint32_t> right_1(total_1);" in out
  assert "probe_materialize_pairs(rowids_0, keys_1, view_1," in out
  # Second probe reuses left_1 (from prior iteration) as input
  assert "thrust::device_vector<uint32_t> left_2(total_2);" in out
  assert "thrust::device_vector<uint32_t> right_2(total_2);" in out
  assert "probe_materialize_pairs(left_1, keys_2, view_2," in out
  # current_n threaded through
  assert "current_n = total_1;" in out
  assert "current_n = total_2;" in out


def test_runner_multi_probe_chained_gather_for_first_column():
  '''Two probes + 2-column dest -> Python emits the chained-gather
  block that walks left_2 -> left_1 -> rowids_0 -> view_0.col_data_
  to recover the first output column.'''
  scan = m.Scan(vars=["a"], rel_name="R1", version=Version.FULL, index=[0])
  probe1 = m.ProbeJoin(
    probe_rel="R2",
    probe_version=Version.FULL,
    probe_index=[0, 1],
    join_key="a",
    output_buffer="buf_1",
  )
  probe2 = m.ProbeJoin(
    probe_rel="R3",
    probe_version=Version.FULL,
    probe_index=[0, 1],
    join_key="b",
    output_buffer="buf_2",
  )
  insert = m.InsertInto(
    rel_name="Out",
    version=Version.NEW,
    vars=["a", "c"],
    index=[0, 1],
  )
  ep = m.ExecutePipeline(
    pipeline=[scan, probe1, probe2, insert],
    source_specs=[scan],
    dest_specs=[insert],
    rule_name="Chain2",
  )
  out = gen_materialized_runner(ep, "Plan_DB_DeviceDB")

  # Chained rowids declared + starts from the final left_ buffer
  assert "thrust::device_vector<uint32_t> chained_rowids(current_n);" in out
  assert "thrust::copy(left_2.begin(), left_2.end(), chained_rowids.begin());" in out
  # One walk-back step (from left_2 back through left_1)
  assert "left_1.begin(), chained_rowids.begin());" in out
  # Final gather from rowids_0 -> view_0 col_data_
  assert "rowids_0.begin(), chained_rowids.begin());" in out
  assert "view_0.col_data_, out_col_0.begin());" in out
  # Second output column from last view via right_2
  assert "right_2.begin(), right_2.end()," in out
  assert "view_2.col_data_ + 2 * view_2.stride_," in out
  # Per-column copy into the interned columns
  assert "dest.template interned_column<0>() + old_size);" in out
  assert "dest.template interned_column<1>() + old_size);" in out


# -----------------------------------------------------------------------------
# gen_materialized_join_kernel (legacy in-kernel variant)
# -----------------------------------------------------------------------------


def test_legacy_kernel_missing_scan():
  ctx = new_code_gen_context()
  out = gen_materialized_join_kernel(
    [m.InsertInto(rel_name="T", version=Version.NEW, vars=["x"], index=[0])],
    "Busted",
    ctx,
  )
  assert "ERROR: Materialized join requires moScan as first op" in out


def test_legacy_kernel_scan_plus_probe():
  scan = m.Scan(vars=["x"], rel_name="R", version=Version.FULL, index=[0])
  probe = m.ProbeJoin(
    probe_rel="S",
    probe_version=Version.FULL,
    probe_index=[0, 1],
    join_key="x",
    output_buffer="buf_1",
  )
  ctx = new_code_gen_context()
  out = gen_materialized_join_kernel([scan, probe], "InKernel", ctx)
  assert "// ===== Materialized Join: InKernel =====" in out
  assert "auto view_R_0 = db.get_view<R, 0>();" in out
  assert "// Phase 2.1: Probe S on x" in out
  assert "auto view_S_0_1 = db.get_view<S, 0, 1>();" in out
  assert "thrust::device_vector<uint32_t> buf_1_counts(R_size);" in out


if __name__ == "__main__":
  import inspect

  this = sys.modules[__name__]
  passed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if name.startswith("test_"):
      fn()
      passed += 1
  print(f"{passed} passed")
