'''Triangle-counting example.

Run with:

    python examples/triangle.py

or via the CLI:

    srdatalog emit    examples/triangle.py --project TrianglePlan
    srdatalog compile examples/triangle.py --project TrianglePlan --out /tmp/tri.so

Shows the full pipeline: DSL → HIR → MIR → emitted C++ tree written
to disk. Compilation is only attempted if a C++ toolchain is on PATH.
'''
from srdatalog import (
  Var, Relation, Program,
  compile_to_hir, compile_to_mir,
  gen_step_body, gen_complete_runner, gen_main_file_content,
  write_jit_project,
)
from srdatalog.codegen.batchfile import _collect_pipelines


def program() -> Program:
  '''Z(x, y, z) :- R(x, y), S(x, y, z), T(y, x, z) — triangle join
  with one shared key `y`.'''
  x, y, z = Var("x"), Var("y"), Var("z")
  h, f = Var("h"), Var("f")
  R = Relation("RRel", 2)
  S = Relation("SRel", 3)
  T = Relation("TRel", 3)
  Z = Relation("ZRel", 3)
  return Program(
    relations=[R, S, T, Z],
    rules=[
      (Z(x, y, z) <= R(x, y) & S(y, z, h) & T(x, z, f)).named("Triangle"),
    ],
  )


def main() -> None:
  prog = program()
  project_name = "TrianglePlan"
  db_type = f"{project_name}_DB_DeviceDB"

  hir = compile_to_hir(prog)
  mir = compile_to_mir(prog)

  step_bodies = [
    gen_step_body(step, db_type, is_rec, i)
    for i, (step, is_rec) in enumerate(mir.steps)
  ]
  per_rule = []
  runner_decls = {}
  for ep in _collect_pipelines(mir):
    decl, full = gen_complete_runner(ep, db_type)
    per_rule.append((ep.rule_name, full))
    runner_decls[ep.rule_name] = decl

  main_cpp = gen_main_file_content(
    project_name, hir.relation_decls, mir, step_bodies, runner_decls,
    cache_dir_hint="<cache>", jit_batch_count=1,
  )

  result = write_jit_project(
    f"{project_name}_DB",
    main_file_content=main_cpp,
    per_rule_runners=per_rule,
    cache_base="./build",
  )
  print("Wrote:")
  print(f"  main  : {result['main']}")
  for b in result["batches"]:
    print(f"  batch : {b}")
  print(f"  dir   : {result['dir']}")
  print()
  print("To compile: `srdatalog compile examples/triangle.py --project TrianglePlan`")


if __name__ == "__main__":
  main()
