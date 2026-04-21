'''Unit tests for the MIR S-expression emitter.

Each test hand-constructs a small MIR node (or tree) and asserts the
emitted S-expression matches the format produced by
src/srdatalog/mir/printer.nim. The goal is to lock the format down now
so the future HIR->MIR lowering produces byte-identical output.

Full end-to-end byte-diff against a Nim-generated MIR fixture is
deferred until a Nim-side tool can emit tc's MIR S-expr; that lands
alongside the HIR->MIR lowering pass.
'''

import srdatalog.mir.types as m
from srdatalog.hir.types import Version
from srdatalog.mir.emit import print_mir_sexpr


def test_column_source():
  n = m.ColumnSource(rel_name="Edge", version=Version.FULL, index=[0, 1], prefix_vars=[])
  assert print_mir_sexpr(n) == ("(column-source :index (Edge 0 1) :ver FULL :prefix ())")


def test_column_source_with_prefix():
  n = m.ColumnSource(rel_name="Path", version=Version.DELTA, index=[1, 0], prefix_vars=["x"])
  assert print_mir_sexpr(n) == ("(column-source :index (Path 1 0) :ver DELTA :prefix (x))")


def test_scan():
  n = m.Scan(
    vars=["x", "y"], rel_name="ArcInput", version=Version.FULL, index=[0, 1], prefix_vars=[]
  )
  assert print_mir_sexpr(n) == ("(scan :vars (x y) :index (ArcInput 0 1) :ver FULL :prefix ())")


def test_insert_into():
  n = m.InsertInto(rel_name="Path", version=Version.NEW, vars=["x", "z"], index=[0, 1])
  assert print_mir_sexpr(n) == (
    "(insert-into :schema Path :ver NEW :dedup-index (0 1) :terms (x z))"
  )


def test_rebuild_index():
  n = m.RebuildIndex(rel_name="Path", version=Version.FULL, index=[1, 0])
  assert print_mir_sexpr(n) == ("(rebuild-index :index (Path 1 0) :ver FULL)")


def test_clear_relation():
  assert print_mir_sexpr(m.ClearRelation(rel_name="Path", version=Version.NEW)) == (
    "(clear-relation :schema Path :ver NEW)"
  )


def test_check_size():
  assert print_mir_sexpr(m.CheckSize(rel_name="Path", version=Version.NEW)) == (
    "(check-size :schema Path :ver NEW)"
  )


def test_compute_delta():
  assert print_mir_sexpr(m.ComputeDelta(rel_name="Path")) == ("(compute-delta :schema Path)")


def test_compute_delta_index():
  n = m.ComputeDeltaIndex(rel_name="Path", canonical_index=[1, 0])
  assert print_mir_sexpr(n) == ("(compute-delta-index :schema Path :canonical-index (1 0))")


def test_merge_index():
  n = m.MergeIndex(rel_name="Path", index=[1, 0])
  assert print_mir_sexpr(n) == "(merge-index :index (Path 1 0))"


def test_merge_relation():
  assert print_mir_sexpr(m.MergeRelation(rel_name="Path")) == ("(merge-relation :schema Path)")


def test_rebuild_index_from_index():
  n = m.RebuildIndexFromIndex(
    rel_name="Path", source_index=[1, 0], target_index=[0, 1], version=Version.DELTA
  )
  assert print_mir_sexpr(n) == (
    "(rebuild-index-from-index :source (Path 1 0) :target (Path 0 1) :ver DELTA)"
  )


def test_column_join_nested():
  '''ColumnJoin emits multi-line with nested source indent +2 per level.'''
  src1 = m.ColumnSource(rel_name="Path", version=Version.DELTA, index=[1, 0], prefix_vars=[])
  src2 = m.ColumnSource(rel_name="Edge", version=Version.FULL, index=[0, 1], prefix_vars=["y"])
  n = m.ColumnJoin(var_name="y", sources=[src1, src2])
  out = print_mir_sexpr(n)
  # Spot-check shape: header, two indented sources, closing
  assert out.startswith("(column-join :var y\n")
  assert "  :sources (\n" in out
  assert "    (column-source :index (Path 1 0) :ver DELTA :prefix ())" in out
  assert "    (column-source :index (Edge 0 1) :ver FULL :prefix (y))" in out
  assert out.endswith("  ))")


def test_execute_pipeline_shape():
  '''ExecutePipeline wraps sources/dests as (tuple ...) and inlines the body.'''
  cs = m.ColumnSource(rel_name="Edge", version=Version.FULL, index=[0, 1], prefix_vars=[])
  insert = m.InsertInto(rel_name="Path", version=Version.NEW, vars=["x", "y"], index=[0, 1])
  scan = m.Scan(
    vars=["x", "y"], rel_name="Edge", version=Version.FULL, index=[0, 1], prefix_vars=[]
  )
  ep = m.ExecutePipeline(
    pipeline=[scan, insert],
    source_specs=[cs],
    dest_specs=[insert],
    rule_name="TCBase",
  )
  out = print_mir_sexpr(ep)
  assert out.startswith("(execute-pipeline :rule TCBase\n")
  assert "(index-spec :schema Edge :index (0 1) :ver FULL)" in out
  assert "(index-spec :schema Path :index (0 1) :ver FULL)" in out  # InsertInto forces FULL
  assert "(scan :vars (x y)" in out
  assert "(insert-into :schema Path :ver NEW :dedup-index (0 1) :terms (x y))" in out


def test_fixpoint_plan_and_block_and_program():
  '''Structural wrappers nest cleanly. Program uses indent+4 for its nested
  plan (matches Nim) and emits lowercase bool for :recursive.
  '''
  cd = m.ComputeDelta(rel_name="Path")
  fp = m.FixpointPlan(instructions=[cd])
  blk = m.Block(instructions=[fp])
  prog = m.Program(steps=[(blk, True)])
  out = print_mir_sexpr(prog)
  assert out.startswith("(program\n")
  assert "  (step :recursive true\n" in out  # lowercase
  assert "        (block\n" in out  # indent+4 = 8 spaces
  assert "          (fixpoint-plan\n" in out  # block body at +2 from block = 10
  assert "            (compute-delta :schema Path)" in out  # fp body at +2 = 12
  assert out.endswith(")")


def test_program_false_recursive_step():
  '''Cover the false branch of the bool printing.'''
  blk = m.Block(instructions=[])
  out = print_mir_sexpr(m.Program(steps=[(blk, False)]))
  assert "  (step :recursive false\n" in out


def test_parallel_group():
  c1 = m.ComputeDelta(rel_name="A")
  c2 = m.ComputeDelta(rel_name="B")
  out = print_mir_sexpr(m.ParallelGroup(ops=[c1, c2]))
  assert out.startswith("(parallel-group  ;; 2 independent ops\n")
  assert "  (compute-delta :schema A)\n" in out
  assert "  (compute-delta :schema B)\n" in out
  assert out.endswith(")")


def test_inject_cpp_hook_with_rule_name():
  out = print_mir_sexpr(m.InjectCppHook(code="/*body*/", rule_name="MyRule"))
  assert out == "(inject-cpp-hook :rule MyRule :code \"...\")"


def test_inject_cpp_hook_no_rule_name():
  out = print_mir_sexpr(m.InjectCppHook(code="/*body*/"))
  assert out == "(inject-cpp-hook :code \"...\")"


def test_post_stratum_reconstruct_intern_cols():
  n = m.PostStratumReconstructInternCols(rel_name="Path", canonical_index=[1, 0])
  assert print_mir_sexpr(n) == (
    "(post-stratum-reconstruct-intern-cols :rel Path :canonical-index (1 0))"
  )


def test_balanced_scan_emit():
  s1 = m.ColumnSource(rel_name="A", version=Version.FULL, index=[0, 1], prefix_vars=[])
  s2 = m.ColumnSource(rel_name="B", version=Version.FULL, index=[0, 1], prefix_vars=[])
  bs = m.BalancedScan(
    group_var="k",
    source1=s1,
    source2=s2,
    vars1=["a"],
    vars2=["b"],
  )
  out = print_mir_sexpr(bs)
  assert out.startswith("(balanced-scan :group-var k\n")
  assert "(column-source :index (A 0 1)" in out
  assert "(column-source :index (B 0 1)" in out
  assert " :vars1 (a)" in out
  assert " :vars2 (b)" in out


def test_positioned_extract_emit():
  s1 = m.ColumnSource(rel_name="A", version=Version.FULL, index=[0])
  s2 = m.ColumnSource(rel_name="B", version=Version.FULL, index=[0])
  pe = m.PositionedExtract(sources=[s1, s2], var_name="k", bind_vars=[])
  out = print_mir_sexpr(pe)
  assert out.startswith("(positioned-extract :var k :sources (")
  assert "(column-source :index (A 0)" in out
  assert "(column-source :index (B 0)" in out
  assert out.endswith(" :bind ())")


def test_aggregate_emit():
  '''Matches the Nim printer format exactly (Nim has an explicit moAggregate case).'''
  agg = m.Aggregate(
    result_var="cnt",
    agg_func="AggCount",
    rel_name="Rel",
    version=Version.FULL,
    index=[0, 1],
    prefix_vars=["x", "y"],
  )
  assert print_mir_sexpr(agg) == (
    "(aggregate :var cnt :func AggCount :index (Rel 0 1) :ver FULL :prefix (x y))"
  )


def test_create_flat_view_emit():
  cfv = m.CreateFlatView(rel_name="_tmp_R", version=Version.NEW, index=[0, 1, 2])
  assert print_mir_sexpr(cfv) == ("(create-flat-view :index (_tmp_R 0 1 2) :ver NEW)")


def test_inner_pipeline_emit_nests_handles_and_ops():
  '''Structure check: header + two indented sections for :handles and :ops.'''
  cs = m.ColumnSource(rel_name="A", version=Version.FULL, index=[0])
  ins = m.InsertInto(rel_name="R", version=Version.NEW, vars=["x"], index=[0])
  ip = m.InnerPipeline(
    rule_name="Inner",
    input_handles=[cs],
    inner_ops=[ins],
    bound_vars=["x"],
  )
  out = print_mir_sexpr(ip)
  assert out.startswith("(inner-pipeline :rule Inner :bound-vars (x)\n")
  assert "  :handles (\n" in out
  assert "    (column-source :index (A 0)" in out
  assert "  :ops (\n" in out
  assert "    (insert-into :schema R" in out
  assert out.endswith("  ))")


def test_probe_join_emit():
  pj = m.ProbeJoin(
    probe_rel="R",
    probe_version=Version.FULL,
    probe_index=[0, 1],
    join_key="k",
    input_buffer="buf0",
    output_buffer="buf1",
  )
  assert print_mir_sexpr(pj) == (
    "(probe-join :input-buffer buf0 :output-buffer buf1 :probe (R 0 1) :ver FULL :key k)"
  )


def test_probe_join_emit_without_input_buffer():
  '''First probe in a pipeline has no input buffer — flag omitted.'''
  pj = m.ProbeJoin(
    probe_rel="R",
    probe_version=Version.DELTA,
    probe_index=[0],
    join_key="k",
    output_buffer="buf0",
  )
  out = print_mir_sexpr(pj)
  assert ":input-buffer" not in out
  assert " :output-buffer buf0" in out


def test_gather_column_emit():
  gc = m.GatherColumn(
    rel_name="R",
    rel_version=Version.FULL,
    column=2,
    output_var="z",
    input_buffer="buf1",
  )
  assert print_mir_sexpr(gc) == (
    "(gather-column :input-buffer buf1 :schema R :ver FULL :col 2 :out z)"
  )


if __name__ == "__main__":
  tests = [
    test_column_source,
    test_column_source_with_prefix,
    test_scan,
    test_insert_into,
    test_rebuild_index,
    test_clear_relation,
    test_check_size,
    test_compute_delta,
    test_compute_delta_index,
    test_merge_index,
    test_merge_relation,
    test_rebuild_index_from_index,
    test_column_join_nested,
    test_execute_pipeline_shape,
    test_fixpoint_plan_and_block_and_program,
    test_program_false_recursive_step,
    test_parallel_group,
    test_inject_cpp_hook_with_rule_name,
    test_inject_cpp_hook_no_rule_name,
    test_post_stratum_reconstruct_intern_cols,
    test_balanced_scan_emit,
    test_positioned_extract_emit,
    test_aggregate_emit,
    test_create_flat_view_emit,
    test_inner_pipeline_emit_nests_handles_and_ops,
    test_probe_join_emit,
    test_probe_join_emit_without_input_buffer,
    test_gather_column_emit,
  ]
  for t in tests:
    t()
  print(f"OK ({len(tests)} tests)")
