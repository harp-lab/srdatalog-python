'''Tests for codegen/jit/cache.py — cache layout, batch sharding,
header generation, write idempotence, and the one-shot project writer.
'''
import os
import sys
import tempfile
from pathlib import Path


from srdatalog.codegen.jit.cache import (
  JitBatchManager,
  JIT_COMMON_INCLUDES,
  JIT_FILE_FOOTER,
  ensure_jit_cache_dir,
  get_batch_file_name,
  get_jit_cache_dir,
  write_jit_project,
)


# -----------------------------------------------------------------------------
# Cache dir + project hash
# -----------------------------------------------------------------------------

def test_cache_dir_uses_srdatalog_scope():
  '''Python's cache lives under `<base>/jit/<project>_<hash>/` to
  avoid clobbering Nim's `.cache/nim/jit/...` layout.'''
  with tempfile.TemporaryDirectory() as base:
    d = get_jit_cache_dir("TrianglePlan_DB", base=base)
    assert d.startswith(base + "/jit/TrianglePlan_DB_")
    # Hash is 4 hex uppercase chars.
    assert len(os.path.basename(d).split("_")[-1]) == 4


def test_project_hash_is_stable():
  '''Hash must not depend on Python's randomized `hash()` — re-running
  in a fresh interpreter gives the same result.'''
  d1 = get_jit_cache_dir("MyProj", base="/tmp/base")
  d2 = get_jit_cache_dir("MyProj", base="/tmp/base")
  assert d1 == d2


def test_ensure_creates_cache_dir():
  with tempfile.TemporaryDirectory() as base:
    d = ensure_jit_cache_dir("X", base=base)
    assert os.path.isdir(d)


def test_batch_file_name_format():
  assert get_batch_file_name(0) == "jit_batch_0.cpp"
  assert get_batch_file_name(7) == "jit_batch_7.cpp"


# -----------------------------------------------------------------------------
# Batch sharding
# -----------------------------------------------------------------------------

def test_add_kernel_shards_at_rules_per_batch():
  mgr = JitBatchManager("X", rules_per_batch=3)
  for i in range(7):
    mgr.add_kernel(f"// kernel {i}\n", rule_name=f"R{i}")
  # 7 rules, 3 per batch → batches 0 (3), 1 (3), 2 (1)
  assert mgr.batch_count() == 3
  assert len(mgr.batches[0]) == 3
  assert len(mgr.batches[1]) == 3
  assert len(mgr.batches[2]) == 1
  assert mgr.rule_names == [f"R{i}" for i in range(7)]


def test_generate_batch_file_contents():
  mgr = JitBatchManager("X", rules_per_batch=4)
  mgr.set_schema_definitions('using Edge = AST::RelationSchema<...>;')
  mgr.set_db_type_alias("using X_DB = ...;")
  mgr.add_kernel("struct JitRunner_Base { };\n", rule_name="Base")
  mgr.add_kernel("struct JitRunner_TC { };\n", rule_name="TC")

  out = mgr.generate_batch_file(0, extra_headers=["gpu/device_2level_index.h"])
  # Preamble
  assert out.startswith(JIT_COMMON_INCLUDES)
  # Extra headers block
  assert '#include "gpu/device_2level_index.h"' in out
  # Schema + DB alias blocks
  assert "// Project-specific schema definitions (inlined)\n" in out
  assert "using Edge = AST::RelationSchema<...>;" in out
  assert "using X_DB = ...;" in out
  # Batch banner + kernels
  assert "// Batch 0 - 2 rules\n" in out
  assert "struct JitRunner_Base" in out
  assert "struct JitRunner_TC" in out
  # Footer
  assert out.endswith(JIT_FILE_FOOTER)


def test_generate_batch_file_empty_idx_returns_empty():
  mgr = JitBatchManager("X")
  assert mgr.generate_batch_file(42) == ""


# -----------------------------------------------------------------------------
# Schema + kernel-decl headers
# -----------------------------------------------------------------------------

def test_schema_header_shape():
  mgr = JitBatchManager("TC")
  mgr.set_schema_definitions("using Edge = ...;\nusing Path = ...;\n")
  out = mgr.generate_schema_header()
  assert out.startswith("// Auto-generated schema definitions for TC\n")
  assert "#pragma once" in out
  assert '#include "srdatalog.h"' in out
  assert "using Edge = ...;" in out


def test_kernel_decl_header_shape():
  mgr = JitBatchManager("TC")
  mgr.add_kernel_declaration("struct JitRunner_TC { };\n")
  out = mgr.generate_kernel_decl_header()
  assert "#pragma once" in out
  assert '#include "gpu/runtime/jit/ws_infrastructure.h"' in out
  assert "struct JitRunner_TC" in out


def test_empty_schema_or_kernel_headers_skip_write():
  with tempfile.TemporaryDirectory() as base:
    mgr = JitBatchManager("Empty", cache_base=base)
    assert mgr.write_schema_header() == ""
    assert mgr.write_kernel_decl_header() == ""


# -----------------------------------------------------------------------------
# write_batch_files — end-to-end I/O
# -----------------------------------------------------------------------------

def test_write_batch_files_lays_out_cache():
  with tempfile.TemporaryDirectory() as base:
    mgr = JitBatchManager("TC", rules_per_batch=2, cache_base=base)
    mgr.set_schema_definitions("using Edge = ...;")
    mgr.add_kernel("struct A {};\n", rule_name="A")
    mgr.add_kernel("struct B {};\n", rule_name="B")
    mgr.add_kernel("struct C {};\n", rule_name="C")

    paths = mgr.write_batch_files(extra_headers=["plugin.h"])
    # 3 rules / 2 per batch → 2 batch files
    assert len(paths) == 2
    for p in paths:
      assert Path(p).exists()
      content = Path(p).read_text()
      assert "struct" in content
      assert '#include "plugin.h"' in content
    # Schema header was written alongside.
    cache_dir = get_jit_cache_dir("TC", base=base)
    assert Path(cache_dir, "TC_schemas.h").exists()


def test_write_is_idempotent_on_unchanged_content():
  '''Second call doesn't touch mtime if the file's content matches.
  Builds depending on timestamps won't re-run the compiler for free.'''
  with tempfile.TemporaryDirectory() as base:
    mgr = JitBatchManager("IdTest", cache_base=base)
    mgr.add_kernel("struct R {};\n", rule_name="R")
    paths = mgr.write_batch_files()
    first_mtime = os.path.getmtime(paths[0])
    # Second invocation with SAME content should not rewrite.
    mgr2 = JitBatchManager("IdTest", cache_base=base)
    mgr2.add_kernel("struct R {};\n", rule_name="R")
    mgr2.write_batch_files()
    assert os.path.getmtime(paths[0]) == first_mtime, \
      "unchanged content must not update mtime"


def test_skip_regen_env_flag():
  '''SRDATALOG_SKIP_JIT_REGEN=1 should leave the existing file as-is
  even if the content changed — debugging mode for hand-editing cached
  JIT code.'''
  with tempfile.TemporaryDirectory() as base:
    mgr1 = JitBatchManager("Skip", cache_base=base)
    mgr1.add_kernel("struct Original {};\n", rule_name="Original")
    paths = mgr1.write_batch_files()
    original = Path(paths[0]).read_text()

    os.environ["SRDATALOG_SKIP_JIT_REGEN"] = "1"
    try:
      mgr2 = JitBatchManager("Skip", cache_base=base)
      mgr2.add_kernel("struct Edited {};\n", rule_name="Edited")
      mgr2.write_batch_files()
      assert Path(paths[0]).read_text() == original, \
        "skip-regen must preserve existing file"
    finally:
      del os.environ["SRDATALOG_SKIP_JIT_REGEN"]


# -----------------------------------------------------------------------------
# One-shot write_jit_project
# -----------------------------------------------------------------------------

def test_write_jit_project_lays_out_full_tree():
  with tempfile.TemporaryDirectory() as base:
    result = write_jit_project(
      "TrianglePlan_DB",
      main_file_content="// main.cpp content\nint foo = 1;\n",
      per_rule_runners=[
        ("Triangle", "struct JitRunner_Triangle { };\n"),
        ("Dummy", "struct JitRunner_Dummy { };\n"),
      ],
      schema_definitions='using RRel = AST::RelationSchema<...>;',
      db_type_alias="using TrianglePlan_DB = AST::Database<...>;",
      extra_headers=["gpu/device_2level_index.h"],
      cache_base=base,
    )
    assert Path(result["dir"]).is_dir()
    assert Path(result["main"]).exists()
    assert "int foo = 1;" in Path(result["main"]).read_text()
    assert len(result["batches"]) >= 1
    assert Path(result["schema_header"]).exists()
    # Kernel header empty (no add_kernel_declaration calls)
    assert result["kernel_header"] == ""
    # Batch file mentions both rules
    batch_content = Path(result["batches"][0]).read_text()
    assert "struct JitRunner_Triangle" in batch_content
    assert "struct JitRunner_Dummy" in batch_content


def test_write_jit_project_end_to_end_with_real_program():
  '''Full pipeline: compile triangle → emit main + runners → write to
  disk → verify everything loads back. Doesn't invoke compiler —
  just checks the on-disk layout matches expectation.'''
  from srdatalog.hir import compile_to_hir, compile_to_mir
  from srdatalog.codegen.batchfile import _collect_pipelines
  from srdatalog.codegen.jit.complete_runner import gen_complete_runner
  from srdatalog.codegen.jit.orchestrator_jit import gen_step_body
  from srdatalog.codegen.jit.main_file import gen_main_file_content
  from test_integration_triangle import build_triangle

  prog = build_triangle()
  hir = compile_to_hir(prog)
  mir = compile_to_mir(prog)
  db_type = "TrianglePlan_DB_DeviceDB"

  step_bodies = [
    gen_step_body(step, db_type, is_rec, i)
    for i, (step, is_rec) in enumerate(mir.steps)
  ]
  per_rule: list[tuple[str, str]] = []
  runner_decls: dict[str, str] = {}
  for ep in _collect_pipelines(mir):
    decl, full = gen_complete_runner(ep, db_type)
    per_rule.append((ep.rule_name, full))
    runner_decls[ep.rule_name] = decl

  main_cpp = gen_main_file_content(
    "TrianglePlan", hir.relation_decls, mir, step_bodies, runner_decls,
    cache_dir_hint="<cache>", jit_batch_count=1,
  )

  with tempfile.TemporaryDirectory() as base:
    result = write_jit_project(
      "TrianglePlan_DB",
      main_file_content=main_cpp,
      per_rule_runners=per_rule,
      db_type_alias="using TrianglePlan_DB_DeviceDB = ...;",
      cache_base=base,
    )
    # Verify everything landed.
    assert Path(result["main"]).exists()
    assert "TrianglePlan_Runner" in Path(result["main"]).read_text()
    assert len(result["batches"]) == 1
    batch = Path(result["batches"][0]).read_text()
    assert "struct JitRunner_Triangle {" in batch
    assert "__global__" in batch  # kernel bodies present


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
