'''Byte-match + smoke tests for codegen/jit/main_file.py.

Phase 6: the main-file emitter ties the per-rule JIT batch files into
a single compilable compile unit. The golden for this module is the
output of Nim's `codegenFixpointRuleSets` (codegen.nim:150-600), which
gets embedded into the Nim-compiled .cpp via `{.emit:}` pragma.

We don't have an automated fixture-dump for this yet (the Nim codegen
is invoked via macro at compile time). Instead this test compares the
middle section of the Python output (the part that corresponds to
`mir_cpp_str`) against a hand-extracted slice of the Nim-compiled
`@mtriangle.nim.cpp` intermediate — committed as
`tests/fixtures/main_file/triangle_main_expected.cpp`.
'''
import sys
from pathlib import Path


from integration_helpers import _cpp_norm
from srdatalog.hir import compile_to_hir, compile_to_mir
from srdatalog.codegen.batchfile import _collect_pipelines
from srdatalog.codegen.jit.complete_runner import gen_complete_runner
from srdatalog.codegen.jit.orchestrator_jit import gen_step_body
from srdatalog.codegen.jit.main_file import (
  gen_main_file_content, gen_relation_typedefs, gen_runner_struct,
  _extract_computed_relations,
)


# -----------------------------------------------------------------------------
# Smoke: relation typedefs
# -----------------------------------------------------------------------------

def test_gen_relation_typedefs_shape():
  from srdatalog.hir.types import RelationDecl
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
    "TrianglePlan", hir.relation_decls, mir, step_bodies,
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
    "TrianglePlan", decls, mir, step_bodies, runner_decls,
    cache_dir_hint="<jit-cache>", jit_batch_count=1,
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


# -----------------------------------------------------------------------------
# Byte-match: compare the `mir_cpp_str`-equivalent slice against the
# hand-extracted Nim output (fixture).
# -----------------------------------------------------------------------------

FIXTURE_MAIN = Path(__file__).resolve().parent / "fixtures" / "main_file"


def test_tc_main_middle_matches_nim_extract():
  '''Same byte-match as triangle, but on transitive closure — which
  has a recursive stratum. Validates step-type "recursive" formatting
  in the run() method. Exercises the `load_data` codegen once the DSL
  plumbs `input_file` through to `RelationDecl` — currently skipped
  because `Relation(...)` in dsl.py doesn't expose an `input_file`
  field (tracked as a separate follow-up).'''
  fixture_path = FIXTURE_MAIN / "tc_main_expected.cpp"
  if not fixture_path.exists():
    print(f"[SKIP] fixture missing: {fixture_path}")
    return

  from test_hir_mir_tc_e2e import build_tc
  prog = build_tc()
  hir = compile_to_hir(prog)
  mir = compile_to_mir(prog)

  # DSL doesn't plumb input_file / print_size yet; inject here so the
  # fixture can validate `load_from_file<ArcInput>(...)` emission and
  # the print_size stats block for Path. When the DSL gains first-
  # class support, these patches can go.
  for d in hir.relation_decls:
    if d.rel_name == "ArcInput":
      d.input_file = "Arc.csv"
    if d.rel_name == "Path":
      d.print_size = True
  canonical = {"Path": [1, 0]}

  step_bodies = [
    gen_step_body(step, "TCPlan_DB_DeviceDB", is_rec, i)
    for i, (step, is_rec) in enumerate(mir.steps)
  ]
  runner_decls: dict[str, str] = {}
  for ep in _collect_pipelines(mir):
    decl, _full = gen_complete_runner(ep, "TCPlan_DB_DeviceDB")
    runner_decls[ep.rule_name] = decl
  full = gen_main_file_content(
    "TCPlan", hir.relation_decls, mir, step_bodies, runner_decls,
    cache_dir_hint="/home/stargazermiao/.cache/nim/jit/TCPlan_DB_2C07",
    jit_batch_count=1,
    canonical_indices=canonical,
  )
  anchor = "using TCPlan_DB ="
  idx = full.find(anchor)
  assert idx != -1
  actual = full[idx:]
  golden = fixture_path.read_text()
  if _cpp_norm(actual) != _cpp_norm(golden):
    a, g = _cpp_norm(actual), _cpp_norm(golden)
    for k, (x, y) in enumerate(zip(a, g)):
      if x != y:
        print(f"First diff at char {k}")
        print(f"  ACTUAL: {a[max(0,k-80):k+80]!r}")
        print(f"  GOLDEN: {g[max(0,k-80):k+80]!r}")
        break
    raise AssertionError(
      f"tc main-file middle mismatch "
      f"(actual len={len(a)}, golden len={len(g)})"
    )


def test_triangle_main_middle_matches_nim_extract():
  '''The part of our Python main file starting from `using TrianglePlan_DB`
  should match the Nim extract (which doesn't include user-declared
  relation typedefs or the user's `datalog_db` blueprint alias).'''
  fixture_path = FIXTURE_MAIN / "triangle_main_expected.cpp"
  if not fixture_path.exists():
    # Fixture not yet committed; skip softly but fail loud enough to notice.
    print(f"[SKIP] fixture missing: {fixture_path}")
    return

  from test_integration_triangle import build_triangle
  prog = build_triangle()
  hir = compile_to_hir(prog)
  mir = compile_to_mir(prog)
  step_bodies = [
    gen_step_body(step, "TrianglePlan_DB_DeviceDB", is_rec, i)
    for i, (step, is_rec) in enumerate(mir.steps)
  ]
  runner_decls: dict[str, str] = {}
  for ep in _collect_pipelines(mir):
    decl, _full = gen_complete_runner(ep, "TrianglePlan_DB_DeviceDB")
    runner_decls[ep.rule_name] = decl
  full = gen_main_file_content(
    "TrianglePlan", hir.relation_decls, mir, step_bodies, runner_decls,
    cache_dir_hint="/home/stargazermiao/.cache/nim/jit/TrianglePlan_DB_FC06",
    jit_batch_count=1,
  )
  # Trim Python output to start at `using TrianglePlan_DB` — the Nim
  # extract doesn't include user-declared relation typedefs.
  anchor = "using TrianglePlan_DB ="
  idx = full.find(anchor)
  assert idx != -1, "missing anchor in Python output"
  actual = full[idx:]

  golden = fixture_path.read_text()
  if _cpp_norm(actual) != _cpp_norm(golden):
    # Show first semantic diff for quick debugging.
    a, g = _cpp_norm(actual), _cpp_norm(golden)
    for k, (x, y) in enumerate(zip(a, g)):
      if x != y:
        print(f"First diff at char {k}")
        print(f"  ACTUAL: {a[max(0,k-80):k+80]!r}")
        print(f"  GOLDEN: {g[max(0,k-80):k+80]!r}")
        break
    raise AssertionError(
      f"triangle main-file middle mismatch "
      f"(actual len={len(a)}, golden len={len(g)})"
    )


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
