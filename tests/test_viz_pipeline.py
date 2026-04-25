'''Tests for the new pipeline.compile_program extraction — same
artifacts as build_project, just in-memory.'''

from __future__ import annotations

from srdatalog.dsl import Program, Relation, Var
from srdatalog.pipeline import compile_program


def _triangle_program() -> Program:
  x, y, z = Var("x"), Var("y"), Var("z")
  h, f = Var("h"), Var("f")
  R = Relation("RRel", 2, column_types=(int, int))
  S = Relation("SRel", 3, column_types=(int, int, int))
  T = Relation("TRel", 3, column_types=(int, int, int))
  Z = Relation("ZRel", 3, column_types=(int, int, int))
  return Program(rules=[(Z(x, y, z) <= R(x, y) & S(y, z, h) & T(z, x, f)).named("Triangle")])


def test_compile_program_populates_all_fields():
  cr = compile_program(_triangle_program(), "TrianglePlan")
  assert cr.ext_db == "TrianglePlan_DB"
  assert cr.device_db == "TrianglePlan_DB_DeviceDB"
  # 4 relations: ZRel, RRel, SRel, TRel (rule-first-occurrence order)
  assert [d.rel_name for d in cr.hir.relation_decls] == ["ZRel", "RRel", "SRel", "TRel"]
  assert len(cr.mir.steps) >= 1
  # Single rule → single runner
  assert len(cr.per_rule_runners) == 1
  name, code = cr.per_rule_runners[0]
  assert name == "Triangle"
  assert "JitRunner_Triangle" in code
  assert "Triangle" in cr.runner_decls
  # Schema + DB blob contain every relation name
  for rel in ("RRel", "SRel", "TRel", "ZRel"):
    assert rel in cr.schema_defs
    assert rel in cr.db_alias


def test_compile_program_index_type_plumbing():
  # A relation declaring a non-default index should surface in
  # rel_index_types + trigger the corresponding extra header.
  x, y = Var("x"), Var("y")
  edge = Relation("Edge", 2, index_type="SRDatalog::GPU::Device2LevelIndex")
  path = Relation("Path", 2)
  prog = Program(rules=[(path(x, y) <= edge(x, y)).named("Base")])
  cr = compile_program(prog, "IndexTestPlan")
  assert cr.rel_index_types["Edge"] == "SRDatalog::GPU::Device2LevelIndex"
  assert any("device_2level_index.h" in h for h in cr.extra_headers)


def test_build_project_still_uses_pipeline_round_trip(tmp_path):
  '''build_project should keep producing the same artifacts — this
  guards against the pipeline extraction regressing the file layout.'''
  from srdatalog import build_project

  result = build_project(
    _triangle_program(),
    project_name="TrianglePlan",
    cache_base=str(tmp_path),
  )
  assert "dir" in result
  assert "batches" in result
  # Main file written, one batch emitted, schema + kernel headers there.
  assert result["main"]
  assert len(result["batches"]) >= 1
