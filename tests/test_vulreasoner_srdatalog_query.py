import srdatalog.ir.mir.types as mir
from minimal_vulreasoner.srdatalog_query import CONNECTORS, build_analyst_program
from srdatalog.ir.hir import compile_to_hir, compile_to_mir


def test_query_keeps_delay_and_intervals_at_datalog_hir_level():
  hir = compile_to_hir(build_analyst_program())
  rules = [rule for stratum in hir.strata for rule in stratum.stratum_rules]
  recursive = [rule for rule in rules if rule.name and rule.name.startswith("Analyst")]

  assert len(recursive) == len(CONNECTORS) + 1  # seed plus six connector rules
  delayed = [rule for rule in recursive if rule.name != "AnalystSeed"]
  assert all(any(getattr(body, "rel", None) == "Successor" for body in rule.body) for rule in delayed)
  assert all(any(getattr(body, "rel", None) == "StepFrom" for body in rule.body) for rule in delayed)

  analyst_decl = next(decl for decl in hir.relation_decls if decl.rel_name == "AnalystAt")
  assert analyst_decl.value_spec is not None
  assert analyst_decl.value_spec.key_columns == (0, 1)
  assert analyst_decl.value_spec.value_columns == (2, 3)


def test_query_lowers_six_rules_to_ordinary_pipelines_and_one_lattice_delta():
  lowered = compile_to_mir(build_analyst_program(), apply_mir_passes=False)
  recursive_plans = [
    step
    for step, is_recursive in lowered.steps
    if is_recursive and isinstance(step, mir.FixpointPlan)
  ]
  assert len(recursive_plans) == 1
  plan = recursive_plans[0]

  parallel = next(op for op in plan.instructions if isinstance(op, mir.ParallelGroup))
  assert len(parallel.ops) == 2 * len(CONNECTORS)
  assert all(isinstance(op, mir.ExecutePipeline) for op in parallel.ops)
  connector_pipelines = [
    pipeline for pipeline in parallel.ops if pipeline.rule_name.startswith("Analyst")
  ]
  promotion_pipelines = [
    pipeline for pipeline in parallel.ops if pipeline.rule_name.startswith("Promote")
  ]
  assert len(connector_pipelines) == len(CONNECTORS)
  assert len(promotion_pipelines) == len(CONNECTORS)
  for pipeline in connector_pipelines:
    cartesian = next(op for op in pipeline.pipeline if isinstance(op, mir.CartesianJoin))
    assert "result_lower" not in cartesian.vars
    assert "result_upper" not in cartesian.vars
    assert [
      op.var_name for op in pipeline.pipeline if isinstance(op, mir.ConstantBind)
    ] == ["result_lower", "result_upper"]

  lattice_merges = [op for op in plan.instructions if isinstance(op, mir.LatticeMergeDelta)]
  assert len(lattice_merges) == len(CONNECTORS) + 1
  analyst_merge = next(op for op in lattice_merges if op.rel_name == "AnalystAt")
  assert analyst_merge.join.value == "interval-intersection"
  assert all(
    op.join.value == "max-lower-select"
    for op in lattice_merges
    if op.rel_name != "AnalystAt"
  )
  assert not any(
    isinstance(op, (mir.ComputeDeltaIndex, mir.MergeIndex)) and op.rel_name == "AnalystAt"
    for op in plan.instructions
  )
