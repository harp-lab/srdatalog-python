'''Unit tests for python/mir_passes.py. Each pass gets a hand-constructed
steps list exercising the relevant transformation.

End-to-end byte-match against Nim is in test_hir_mir_tc_e2e.py /
test_hir_user_plan.py; this file verifies each pass in isolation.
'''
import sys
from pathlib import Path


from srdatalog.hir_types import Version
import srdatalog.mir.types as mir
from srdatalog.mir.passes import (
  insert_pre_reconstruct_rebuilds,
  apply_clause_order_reordering,
  apply_prefix_source_reordering,
  apply_all_mir_passes,
)


# -----------------------------------------------------------------------------
# Pass 0: insert_pre_reconstruct_rebuilds
# -----------------------------------------------------------------------------

def _stratum_step(
  rel: str, merge_index: list[int], insert_index: list[int],
) -> tuple[mir.FixpointPlan, bool]:
  '''Mini FixpointPlan: InsertInto + MergeIndex for a single relation.'''
  ep = mir.ExecutePipeline(
    pipeline=[
      mir.InsertInto(rel_name=rel, version=Version.NEW, vars=["x", "y"], index=insert_index),
    ],
    source_specs=[],
    dest_specs=[],
    rule_name="R",
  )
  fp = mir.FixpointPlan(instructions=[
    ep,
    mir.MergeIndex(rel_name=rel, index=merge_index),
  ])
  return (fp, False)


def test_pre_reconstruct_noop_when_needed_subset_of_merged():
  '''If subsequent strata only need indices already merged, no rebuilds added.'''
  s0 = _stratum_step("R", merge_index=[0, 1], insert_index=[0, 1])
  psr = (mir.PostStratumReconstructInternCols(rel_name="R", canonical_index=[0, 1]), False)

  # Subsequent step reads R[0,1] as FULL.
  ep2 = mir.ExecutePipeline(
    pipeline=[
      mir.ColumnSource(rel_name="R", version=Version.FULL, index=[0, 1]),
    ],
    source_specs=[], dest_specs=[], rule_name="Rec",
  )
  s1 = (mir.FixpointPlan(instructions=[ep2]), True)

  out = insert_pre_reconstruct_rebuilds([s0, psr, s1])
  # No new steps inserted.
  assert len(out) == 3


def test_pre_reconstruct_inserts_missing_full_rebuild():
  '''If subsequent strata need an index that wasn't merged, insert a
  RebuildIndex(FULL) for it after PostStratumReconstructInternCols.
  '''
  s0 = _stratum_step("R", merge_index=[0, 1], insert_index=[0, 1])
  psr = (mir.PostStratumReconstructInternCols(rel_name="R", canonical_index=[0, 1]), False)

  # Subsequent step reads R[1,0] as DELTA (dispatches through FULL).
  ep2 = mir.ExecutePipeline(
    pipeline=[
      mir.ColumnSource(rel_name="R", version=Version.DELTA, index=[1, 0]),
    ],
    source_specs=[], dest_specs=[], rule_name="Rec",
  )
  s1 = (mir.FixpointPlan(instructions=[ep2]), True)

  out = insert_pre_reconstruct_rebuilds([s0, psr, s1])
  assert len(out) == 4
  inserted = out[2][0]
  assert isinstance(inserted, mir.RebuildIndex)
  assert inserted.rel_name == "R"
  assert inserted.index == [1, 0]
  assert inserted.version is Version.FULL


# -----------------------------------------------------------------------------
# Pass 1: apply_clause_order_reordering
# -----------------------------------------------------------------------------

def test_clause_order_reorder_column_join():
  '''ColumnJoin sources with clause_idx [0, 1] under clause_order=[1, 0]
  should get reordered to [clause_idx=1, clause_idx=0].
  '''
  s0 = mir.ColumnSource(rel_name="A", version=Version.FULL, index=[0], clause_idx=0)
  s1 = mir.ColumnSource(rel_name="B", version=Version.FULL, index=[0], clause_idx=1)
  cj = mir.ColumnJoin(var_name="x", sources=[s0, s1])
  ins = mir.InsertInto(rel_name="R", version=Version.NEW, vars=["x"], index=[0])
  ep = mir.ExecutePipeline(
    pipeline=[cj, ins], source_specs=[s0, s1], dest_specs=[ins],
    rule_name="Test", clause_order=[1, 0],
  )
  apply_clause_order_reordering([(ep, False)])
  assert [s.clause_idx for s in cj.sources] == [1, 0]
  # source_specs should be regenerated in the new order.
  assert [s.clause_idx for s in ep.source_specs] == [1, 0]


def test_clause_order_reorder_cartesian_join_keeps_var_from_source_aligned():
  '''Reordering a CartesianJoin must drag var_from_source along.'''
  s0 = mir.ColumnSource(rel_name="A", version=Version.FULL, index=[0, 1], clause_idx=0)
  s1 = mir.ColumnSource(rel_name="B", version=Version.FULL, index=[0, 1], clause_idx=1)
  cart = mir.CartesianJoin(
    vars=["x", "z"], sources=[s0, s1], var_from_source=[["x"], ["z"]],
  )
  ins = mir.InsertInto(rel_name="R", version=Version.NEW, vars=["x", "z"], index=[0, 1])
  ep = mir.ExecutePipeline(
    pipeline=[cart, ins], source_specs=[s0, s1], dest_specs=[ins],
    rule_name="Test", clause_order=[1, 0],
  )
  apply_clause_order_reordering([(ep, False)])
  assert [s.clause_idx for s in cart.sources] == [1, 0]
  assert cart.var_from_source == [["z"], ["x"]]


def test_clause_order_reorder_empty_clause_order_is_noop():
  s0 = mir.ColumnSource(rel_name="A", version=Version.FULL, index=[0], clause_idx=0)
  s1 = mir.ColumnSource(rel_name="B", version=Version.FULL, index=[0], clause_idx=1)
  cj = mir.ColumnJoin(var_name="x", sources=[s0, s1])
  ins = mir.InsertInto(rel_name="R", version=Version.NEW, vars=["x"], index=[0])
  ep = mir.ExecutePipeline(
    pipeline=[cj, ins], source_specs=[s0, s1], dest_specs=[ins],
    rule_name="Test", clause_order=[],
  )
  apply_clause_order_reordering([(ep, False)])
  assert [s.clause_idx for s in cj.sources] == [0, 1]


# -----------------------------------------------------------------------------
# Pass 2: apply_prefix_source_reordering
# -----------------------------------------------------------------------------

def test_prefix_reorder_moves_prefixed_to_front():
  '''Unprefixed source first, prefixed second -> swap.'''
  s0 = mir.ColumnSource(rel_name="A", version=Version.FULL, index=[0], prefix_vars=[], clause_idx=0)
  s1 = mir.ColumnSource(rel_name="B", version=Version.FULL, index=[0], prefix_vars=["x"], clause_idx=1)
  cj = mir.ColumnJoin(var_name="y", sources=[s0, s1])
  ins = mir.InsertInto(rel_name="R", version=Version.NEW, vars=["y"], index=[0])
  ep = mir.ExecutePipeline(pipeline=[cj, ins], source_specs=[], dest_specs=[], rule_name="T")
  apply_prefix_source_reordering([(ep, False)])
  assert cj.sources[0].rel_name == "B"   # prefixed moved to front
  assert cj.sources[1].rel_name == "A"


def test_prefix_reorder_noop_when_first_already_prefixed():
  '''First source already prefixed — short-circuit, no reorder.'''
  s0 = mir.ColumnSource(rel_name="A", version=Version.FULL, index=[0], prefix_vars=["x"], clause_idx=0)
  s1 = mir.ColumnSource(rel_name="B", version=Version.FULL, index=[0], prefix_vars=[], clause_idx=1)
  cj = mir.ColumnJoin(var_name="y", sources=[s0, s1])
  ins = mir.InsertInto(rel_name="R", version=Version.NEW, vars=["y"], index=[0])
  ep = mir.ExecutePipeline(pipeline=[cj, ins], source_specs=[], dest_specs=[], rule_name="T")
  apply_prefix_source_reordering([(ep, False)])
  assert cj.sources[0].rel_name == "A"
  assert cj.sources[1].rel_name == "B"


def test_prefix_reorder_noop_when_no_prefixed_source():
  '''All unprefixed — no reorder.'''
  s0 = mir.ColumnSource(rel_name="A", version=Version.FULL, index=[0], clause_idx=0)
  s1 = mir.ColumnSource(rel_name="B", version=Version.FULL, index=[0], clause_idx=1)
  cj = mir.ColumnJoin(var_name="y", sources=[s0, s1])
  ins = mir.InsertInto(rel_name="R", version=Version.NEW, vars=["y"], index=[0])
  ep = mir.ExecutePipeline(pipeline=[cj, ins], source_specs=[], dest_specs=[], rule_name="T")
  apply_prefix_source_reordering([(ep, False)])
  assert [s.rel_name for s in cj.sources] == ["A", "B"]


# -----------------------------------------------------------------------------
# Chain
# -----------------------------------------------------------------------------

def test_apply_all_chains_passes():
  '''Empty steps list chains through all four passes without error.'''
  out = apply_all_mir_passes([])
  assert out == []


# -----------------------------------------------------------------------------
# Pass 3: apply_balanced_scan_pass
# -----------------------------------------------------------------------------

def test_balanced_scan_pass_converts_column_join_to_positioned_extract():
  '''A pipeline starting with BalancedScan gets its subsequent ColumnJoin
  for a balanced var turned into a PositionedExtract.
  '''
  from srdatalog.mir.passes import apply_balanced_scan_pass
  s1 = mir.ColumnSource(rel_name="A", version=Version.FULL, index=[0, 1])
  s2 = mir.ColumnSource(rel_name="B", version=Version.FULL, index=[0, 1])
  bs = mir.BalancedScan(
    group_var="k", source1=s1, source2=s2, vars1=["a"], vars2=[],
  )
  cj_src = mir.ColumnSource(rel_name="C", version=Version.FULL, index=[0],
                            clause_idx=2)
  # cjVarName "a" is in bsVars1 → should convert to PositionedExtract.
  cj = mir.ColumnJoin(var_name="a", sources=[cj_src])
  ins = mir.InsertInto(rel_name="Out", version=Version.NEW, vars=["a"], index=[0])
  ep = mir.ExecutePipeline(
    pipeline=[bs, cj, ins], source_specs=[], dest_specs=[ins], rule_name="T",
  )
  apply_balanced_scan_pass([(ep, False)])
  # bs stays; cj turned into positioned-extract; ins stays.
  assert isinstance(ep.pipeline[0], mir.BalancedScan)
  assert isinstance(ep.pipeline[1], mir.PositionedExtract)
  assert ep.pipeline[1].var_name == "a"
  assert ep.pipeline[1].sources == [cj_src]
  assert isinstance(ep.pipeline[2], mir.InsertInto)


def test_balanced_scan_pass_leaves_non_balanced_var_alone():
  '''A ColumnJoin for a var NOT in the balanced set passes through.'''
  from srdatalog.mir.passes import apply_balanced_scan_pass
  s1 = mir.ColumnSource(rel_name="A", version=Version.FULL, index=[0, 1])
  s2 = mir.ColumnSource(rel_name="B", version=Version.FULL, index=[0, 1])
  bs = mir.BalancedScan(
    group_var="k", source1=s1, source2=s2, vars1=["a"], vars2=[],
  )
  cj_src = mir.ColumnSource(rel_name="C", version=Version.FULL, index=[0])
  cj = mir.ColumnJoin(var_name="z", sources=[cj_src])
  ins = mir.InsertInto(rel_name="Out", version=Version.NEW, vars=["z"], index=[0])
  ep = mir.ExecutePipeline(
    pipeline=[bs, cj, ins], source_specs=[], dest_specs=[ins], rule_name="T",
  )
  apply_balanced_scan_pass([(ep, False)])
  assert isinstance(ep.pipeline[1], mir.ColumnJoin)


def test_balanced_scan_pass_noop_when_no_balanced_scan():
  '''Pipeline without BalancedScan at position 0 is untouched.'''
  from srdatalog.mir.passes import apply_balanced_scan_pass
  scan = mir.Scan(vars=["x"], rel_name="R", version=Version.FULL, index=[0])
  ins = mir.InsertInto(rel_name="S", version=Version.NEW, vars=["x"], index=[0])
  ep = mir.ExecutePipeline(
    pipeline=[scan, ins], source_specs=[scan], dest_specs=[ins], rule_name="T",
  )
  apply_balanced_scan_pass([(ep, False)])
  assert ep.pipeline == [scan, ins]


if __name__ == "__main__":
  tests = [
    test_pre_reconstruct_noop_when_needed_subset_of_merged,
    test_pre_reconstruct_inserts_missing_full_rebuild,
    test_clause_order_reorder_column_join,
    test_clause_order_reorder_cartesian_join_keeps_var_from_source_aligned,
    test_clause_order_reorder_empty_clause_order_is_noop,
    test_prefix_reorder_moves_prefixed_to_front,
    test_prefix_reorder_noop_when_first_already_prefixed,
    test_prefix_reorder_noop_when_no_prefixed_source,
    test_apply_all_chains_passes,
    test_balanced_scan_pass_converts_column_join_to_positioned_extract,
    test_balanced_scan_pass_leaves_non_balanced_var_alone,
    test_balanced_scan_pass_noop_when_no_balanced_scan,
  ]
  for t in tests:
    t()
  print(f"OK ({len(tests)} tests)")
