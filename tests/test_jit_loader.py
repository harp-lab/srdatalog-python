'''Tests for codegen/jit/loader.py.

Strategy: compile a tiny real `extern "C"` library with clang++/g++
(not CUDA — any host compiler works) and exercise the full round-trip
through JitRuntime. This validates ctypes bindings, errcheck defaults,
and build_and_load without needing the generalized_datalog runtime.

Skipped on hosts with no C++ compiler on PATH (rare in dev).
'''
import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


from srdatalog.codegen.jit.loader import (
  EntryPoint,
  JitRuntime,
  _apply_errcheck_default,
  build_and_load,
  gen_runtime_shim_template,
)
from srdatalog.codegen.jit.compiler import CompilerConfig, compile_cpp, link_shared


def _cxx() -> str | None:
  return os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")


# -----------------------------------------------------------------------------
# Shim template
# -----------------------------------------------------------------------------

def test_shim_template_shape():
  shim = gen_runtime_shim_template(
    "TrianglePlan",
    "TriangleDBBlueprint",
    dest_relations=[("ZRel", "ZRel")],
  )
  assert 'extern "C"' in shim
  assert "int srdatalog_init()" in shim
  assert "int srdatalog_run(const char* data_dir)" in shim
  assert "int srdatalog_shutdown()" in shim
  assert "uint64_t srdatalog_size_ZRel()" in shim
  assert "TrianglePlan_Runner::load_data" in shim
  assert "TrianglePlan_Runner::run" in shim
  assert "TriangleDBBlueprint" in shim


def test_shim_template_no_dest_relations_omits_size_queries():
  shim = gen_runtime_shim_template("R", "RBlueprint")
  assert "srdatalog_size_" not in shim


# -----------------------------------------------------------------------------
# errcheck default
# -----------------------------------------------------------------------------

def test_errcheck_default_passes_on_zero():
  assert _apply_errcheck_default(0, lambda: None, ()) == 0


def test_errcheck_default_raises_on_nonzero():
  fake_fn = type("F", (), {"__name__": "my_fn"})()
  try:
    _apply_errcheck_default(42, fake_fn, ())
  except RuntimeError as e:
    assert "nonzero status 42" in str(e)
    assert "my_fn" in str(e)
  else:
    raise AssertionError("expected RuntimeError")


# -----------------------------------------------------------------------------
# JitRuntime file checks
# -----------------------------------------------------------------------------

def test_runtime_raises_on_missing_file():
  try:
    JitRuntime("/nonexistent/path/libx.so")
  except FileNotFoundError as e:
    assert "not found" in str(e)
  else:
    raise AssertionError("expected FileNotFoundError")


# -----------------------------------------------------------------------------
# End-to-end: compile + load + call a real .so
# -----------------------------------------------------------------------------

_TEST_LIB_SRC = r"""
#include <cstdint>
#include <cstring>

// Simple state for roundtrip tests.
static int32_t g_value = 0;

extern "C" {

int srdatalog_init() { g_value = 100; return 0; }

int srdatalog_fail() { return 7; }

int srdatalog_add(int32_t x) { g_value += x; return 0; }

int32_t srdatalog_get_value() { return g_value; }

int srdatalog_echo_len(const char* s) {
  return static_cast<int>(std::strlen(s));
}

}  // extern "C"
"""


def _build_test_lib(td: str) -> str:
  cxx = _cxx()
  if not cxx:
    return ""
  src = os.path.join(td, "testlib.cpp")
  Path(src).write_text(_TEST_LIB_SRC)
  obj = os.path.join(td, "testlib.o")
  cfg = CompilerConfig(cxx=cxx, shared=True)
  r = compile_cpp(src, obj, cfg)
  if r.returncode != 0:
    raise RuntimeError(f"test compile failed: {r.stderr}")
  lib = os.path.join(td, "libtest.so")
  lr = link_shared([obj], lib, cfg)
  if lr.returncode != 0:
    raise RuntimeError(f"test link failed: {lr.stderr}")
  return lib


def test_runtime_binds_and_calls_entry_points():
  if not _cxx():
    print("[SKIP] no C++ compiler on PATH")
    return
  with tempfile.TemporaryDirectory() as td:
    lib = _build_test_lib(td)
    rt = JitRuntime(
      lib,
      entry_points=[
        EntryPoint("srdatalog_init"),
        EntryPoint("srdatalog_add", argtypes=[ctypes.c_int32]),
        EntryPoint(
          "srdatalog_get_value",
          restype=ctypes.c_int32,
          # explicitly no errcheck for a non-int-status function
          errcheck=False,
        ),
        EntryPoint(
          "srdatalog_echo_len",
          argtypes=[ctypes.c_char_p],
          restype=ctypes.c_int,
          errcheck=False,
        ),
      ],
    )
    # init → g_value = 100
    assert rt.srdatalog_init() == 0
    # get initial value
    assert rt.srdatalog_get_value() == 100
    # add 5
    assert rt.srdatalog_add(5) == 0
    assert rt.srdatalog_get_value() == 105
    # string roundtrip
    assert rt.srdatalog_echo_len(b"hello") == 5


def test_runtime_errcheck_raises_on_nonzero_default():
  '''Default errcheck for int-returning entries raises on nonzero.'''
  if not _cxx():
    print("[SKIP] no C++ compiler on PATH")
    return
  with tempfile.TemporaryDirectory() as td:
    lib = _build_test_lib(td)
    rt = JitRuntime(lib, entry_points=[EntryPoint("srdatalog_fail")])
    try:
      rt.srdatalog_fail()
    except RuntimeError as e:
      assert "nonzero status 7" in str(e)
    else:
      raise AssertionError("expected RuntimeError from default errcheck")


def test_runtime_bind_missing_symbol_raises():
  if not _cxx():
    print("[SKIP] no C++ compiler on PATH")
    return
  with tempfile.TemporaryDirectory() as td:
    lib = _build_test_lib(td)
    try:
      JitRuntime(lib, entry_points=[EntryPoint("nonexistent_symbol")])
    except AttributeError as e:
      assert "nonexistent_symbol" in str(e)
    else:
      raise AssertionError("expected AttributeError")


# -----------------------------------------------------------------------------
# build_and_load — full chain
# -----------------------------------------------------------------------------

def test_build_and_load_runs_compile_then_dlopen():
  if not _cxx():
    print("[SKIP] no C++ compiler on PATH")
    return
  with tempfile.TemporaryDirectory() as td:
    project_dir = os.path.join(td, "proj")
    os.makedirs(project_dir)
    main_cpp = os.path.join(project_dir, "main.cpp")
    Path(main_cpp).write_text(_TEST_LIB_SRC)
    # No batch files — the "project" is just main.cpp.
    project_result = {
      "dir": project_dir,
      "main": main_cpp,
      "batches": [],
      "schema_header": "",
      "kernel_header": "",
    }
    cfg = CompilerConfig(cxx=_cxx(), shared=True)
    rt = build_and_load(
      project_result,
      entry_points=[
        EntryPoint("srdatalog_init"),
        EntryPoint(
          "srdatalog_get_value",
          restype=ctypes.c_int32,
          errcheck=False,
        ),
      ],
      compiler_config=cfg,
    )
    assert rt.srdatalog_init() == 0
    assert rt.srdatalog_get_value() == 100


def test_build_and_load_reports_compile_errors():
  if not _cxx():
    print("[SKIP] no C++ compiler on PATH")
    return
  with tempfile.TemporaryDirectory() as td:
    project_dir = os.path.join(td, "proj")
    os.makedirs(project_dir)
    main = os.path.join(project_dir, "main.cpp")
    Path(main).write_text("this is not valid c++\n")
    project_result = {
      "dir": project_dir, "main": main, "batches": [],
      "schema_header": "", "kernel_header": "",
    }
    try:
      build_and_load(
        project_result,
        entry_points=[],
        compiler_config=CompilerConfig(cxx=_cxx()),
      )
    except RuntimeError as e:
      assert "build failed" in str(e)
    else:
      raise AssertionError("expected RuntimeError on bad source")


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
      import traceback
      print(f"ERROR {name}: {type(e).__name__}: {e}")
      traceback.print_exc()
      failed += 1
  print(f"\n{passed} pass / {failed} fail")
