'''Assembly tests for ir/codegen/cuda/main_file.py.'''

import sys

from srdatalog.ir.codegen.cuda.batchfile import _collect_pipelines
from srdatalog.ir.codegen.cuda.complete_runner import gen_complete_runner
from srdatalog.ir.codegen.cuda.main_file import (
  _extract_computed_relations,
  gen_main_file_content,
  gen_relation_typedefs,
  gen_runner_struct,
)
from srdatalog.ir.codegen.cuda.orchestrator import gen_step_body
from srdatalog.ir.hir import compile_to_hir, compile_to_mir

# -----------------------------------------------------------------------------
# Smoke: relation typedefs
# -----------------------------------------------------------------------------


def test_gen_relation_typedefs_shape():
  from srdatalog.ir.hir.types import RelationDecl

  decls = [
    RelationDecl(rel_name="Edge", types=["int", "int"], semiring="NoProvenance"),
    RelationDecl(rel_name="Path", types=["int", "int"], semiring="NoProvenance"),
  ]
  out = gen_relation_typedefs(decls)
  assert (
    'using Edge = AST::RelationSchema<decltype("Edge"_s), '
    'NoProvenance, std::tuple<int, int>>;' in out
  )
  assert (
    'using Path = AST::RelationSchema<decltype("Path"_s), '
    'NoProvenance, std::tuple<int, int>>;' in out
  )


# -----------------------------------------------------------------------------
# _extract_computed_relations matches Nim's extractComputedRelations
# -----------------------------------------------------------------------------


def test_extract_computed_relations_from_triangle():
  from test_integration_triangle import build_triangle

  mir = compile_to_mir(build_triangle())
  # Triangle pipeline: step 0 is the recursive fixpoint, step 1 is
  # post-stratum reconstruct. step 0 should yield ["ZRel"].
  step0 = mir.steps[0][0]
  rels = _extract_computed_relations(step0)
  assert rels == ["ZRel"], f"got {rels}"


# -----------------------------------------------------------------------------
# gen_runner_struct — shape checks
# -----------------------------------------------------------------------------


def test_runner_struct_triangle_shape():
  from test_integration_triangle import build_triangle

  prog = build_triangle()
  hir = compile_to_hir(prog)
  mir = compile_to_mir(prog)
  # Generate per-step bodies via existing orchestrator.
  step_bodies = [
    gen_step_body(step, "TrianglePlan_DB_DeviceDB", is_rec, i)
    for i, (step, is_rec) in enumerate(mir.steps)
  ]
  out = gen_runner_struct(
    "TrianglePlan",
    hir.relation_decls,
    mir,
    step_bodies,
  )
  assert "struct TrianglePlan_Runner {" in out
  assert "using DB = TrianglePlan_DB;" in out
  assert "static void load_data(DB& db, std::string root_dir)" in out
  assert "static void run(DB& db, std::size_t max_iterations =" in out
  assert 'std::cout << "[Step 0 (simple)] "' in out
  assert '<< "Relations: ZRel"' in out
  assert "step_0(db, max_iterations);" in out
  assert "step_1(db, max_iterations);" in out
  assert out.rstrip().endswith("};")


# -----------------------------------------------------------------------------
# gen_main_file_content — full assembly
# -----------------------------------------------------------------------------


def test_main_file_content_triangle_assembly():
  from test_integration_triangle import build_triangle

  prog = build_triangle()
  hir = compile_to_hir(prog)
  mir = compile_to_mir(prog)
  decls = hir.relation_decls
  step_bodies = [
    gen_step_body(step, "TrianglePlan_DB_DeviceDB", is_rec, i)
    for i, (step, is_rec) in enumerate(mir.steps)
  ]
  runner_decls: dict[str, str] = {}
  for ep in _collect_pipelines(mir):
    decl, _full = gen_complete_runner(ep, "TrianglePlan_DB_DeviceDB")
    runner_decls[ep.rule_name] = decl
  out = gen_main_file_content(
    "TrianglePlan",
    decls,
    mir,
    step_bodies,
    runner_decls,
    cache_dir_hint="<jit-cache>",
    jit_batch_count=1,
  )
  # Relation typedefs
  assert 'using RRel = AST::RelationSchema<decltype("RRel"_s)' in out
  # DB alias
  assert "using TrianglePlan_DB = AST::Database<" in out
  # Device DB aliases
  assert "using TrianglePlan_DB_Blueprint = " in out
  assert "using TrianglePlan_DB_DeviceDB = " in out
  # GPU includes
  assert '#include "gpu/runtime/jit/materialized_join.h"' in out
  # Forward decl for JitRunner_Triangle
  assert "struct JitRunner_Triangle {" in out
  # Namespace placeholder
  assert "namespace TrianglePlan_Plans {" in out
  # Runner struct
  assert "struct TrianglePlan_Runner {" in out
  # JIT summary footer
  assert "JIT kernels in 1 batch files" in out


if __name__ == "__main__":
  import inspect

  this = sys.modules[__name__]
  passed = 0
  failed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if not name.startswith("test_"):
      continue
    try:
      fn()
      print(f"OK  {name}")
      passed += 1
    except AssertionError as e:
      print(f"FAIL {name}")
      print(str(e)[:2000])
      failed += 1
    except Exception as e:
      print(f"ERROR {name}: {type(e).__name__}: {e}")
      failed += 1
  print(f"\n{passed} pass / {failed} fail")
