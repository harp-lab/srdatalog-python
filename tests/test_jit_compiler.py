'''Tests for codegen/jit/compiler.py.

Pure logic is unit-tested without invoking a real compiler:
  - command assembly (includes, defines, std, -fPIC, link order)
  - auto-detection (CXX env → clang++ → g++)
  - stamp cache (source/arg hash skipped recompile)
  - parallel resolution
  - BuildResult.ok aggregates compile + link returncodes

One integration test invokes a real clang++ on a tiny hello-world to
validate end-to-end compile → link → shared library. Guarded behind
`SRDATALOG_JIT_RUN_COMPILE_TESTS=1` because many dev machines don't
have the generalized_datalog headers to compile a real JIT project.
'''
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock


from srdatalog.codegen.jit.compiler import (
  BuildResult,
  CompilerConfig,
  CompileResult,
  _base_cxx_flags,
  _build_compile_cmd,
  _build_link_cmd,
  _check_stamp,
  _detect_cxx,
  _stamp_digest,
  _stamp_path,
  _write_stamp,
  compile_cpp,
  compile_jit_project,
  link_shared,
)


# -----------------------------------------------------------------------------
# Config + auto-detect
# -----------------------------------------------------------------------------

def test_base_flags_order():
  '''std → fPIC → defines → includes → extra cxx_flags — matches
  how most people expect to read a compile line.'''
  cfg = CompilerConfig(
    cxx_std="c++23",
    defines=["FOO=1", "BAR"],
    include_paths=["/inc1", "/inc2"],
    cxx_flags=["-O2", "-g"],
  )
  flags = _base_cxx_flags(cfg)
  assert flags[0] == "-std=c++23"
  assert flags[1] == "-fPIC"
  assert "-DFOO=1" in flags
  assert "-DBAR" in flags
  assert "-I/inc1" in flags
  assert "-I/inc2" in flags
  assert flags[-2:] == ["-O2", "-g"]


def test_shared_false_drops_fpic():
  cfg = CompilerConfig(shared=False)
  assert "-fPIC" not in _base_cxx_flags(cfg)


def test_detect_cxx_prefers_env():
  with mock.patch.dict(os.environ, {"CXX": "/custom/clang++"}):
    assert _detect_cxx() == "/custom/clang++"


def test_detect_cxx_falls_back_to_clangpp():
  with mock.patch.dict(os.environ, {}, clear=False):
    os.environ.pop("CXX", None)
    with mock.patch("srdatalog.codegen.jit.compiler.shutil.which") as w:
      w.side_effect = lambda c: "/usr/bin/clang++" if c == "clang++" else None
      assert _detect_cxx() == "clang++"


def test_detect_cxx_raises_when_none_available():
  with mock.patch.dict(os.environ, {}, clear=False):
    os.environ.pop("CXX", None)
    with mock.patch("srdatalog.codegen.jit.compiler.shutil.which", return_value=None):
      try:
        _detect_cxx()
      except RuntimeError as e:
        assert "no C++ compiler" in str(e)
      else:
        raise AssertionError("expected RuntimeError")


def test_resolved_jobs_uses_env_override():
  cfg = CompilerConfig()
  with mock.patch.dict(os.environ, {"SRDATALOG_JIT_COMPILE_JOBS": "7"}):
    assert cfg.resolved_jobs() == 7


def test_resolved_jobs_respects_explicit_config():
  cfg = CompilerConfig(jobs=3)
  # Even with env set, explicit `jobs` wins.
  with mock.patch.dict(os.environ, {"SRDATALOG_JIT_COMPILE_JOBS": "9"}):
    assert cfg.resolved_jobs() == 3


# -----------------------------------------------------------------------------
# Command assembly
# -----------------------------------------------------------------------------

def test_compile_cmd_shape():
  cfg = CompilerConfig(
    cxx="clang++",
    include_paths=["/inc"],
    defines=["NDEBUG"],
    cxx_flags=["-O3"],
  )
  cmd = _build_compile_cmd("src.cpp", "out.o", cfg)
  assert cmd[0] == "clang++"
  assert "-c" in cmd
  assert "src.cpp" in cmd
  assert cmd.index("-o") < cmd.index("out.o")
  assert "-std=c++23" in cmd
  assert "-DNDEBUG" in cmd
  assert "-I/inc" in cmd
  assert "-O3" in cmd


def test_link_cmd_shape():
  cfg = CompilerConfig(
    cxx="g++",
    link_flags=["-L/lib"],
    libs=["boost_system", "rmm"],
    extra_sources=["/opt/runtime.a"],
  )
  cmd = _build_link_cmd(["a.o", "b.o"], "libx.so", cfg)
  assert cmd[0] == "g++"
  assert "-shared" in cmd
  # Objects come BEFORE -o
  assert cmd.index("a.o") < cmd.index("-o")
  assert cmd.index("b.o") < cmd.index("-o")
  # extra_sources linked after objects
  assert cmd.index("/opt/runtime.a") > cmd.index("b.o")
  # Libs at the end (link order matters — libs should resolve
  # against earlier objects).
  assert cmd.index("-lboost_system") > cmd.index("libx.so")
  assert cmd.index("-lrmm") > cmd.index("libx.so")
  assert "-L/lib" in cmd


def test_link_static_drops_shared_flag():
  cfg = CompilerConfig(shared=False, cxx="clang++")
  cmd = _build_link_cmd(["a.o"], "libx.a", cfg)
  assert "-shared" not in cmd


# -----------------------------------------------------------------------------
# Stamp cache
# -----------------------------------------------------------------------------

def test_stamp_digest_depends_on_source_content():
  with tempfile.TemporaryDirectory() as td:
    src = os.path.join(td, "a.cpp")
    Path(src).write_text("int main() { return 0; }\n")
    d1 = _stamp_digest(src, ["clang++", "-c", src])
    Path(src).write_text("int main() { return 1; }\n")
    d2 = _stamp_digest(src, ["clang++", "-c", src])
    assert d1 != d2, "digest must change when source changes"


def test_stamp_digest_depends_on_argv():
  with tempfile.TemporaryDirectory() as td:
    src = os.path.join(td, "a.cpp")
    Path(src).write_text("int x;\n")
    d1 = _stamp_digest(src, ["clang++", "-O0"])
    d2 = _stamp_digest(src, ["clang++", "-O3"])
    assert d1 != d2, "digest must change when flags change"


def test_check_stamp_false_when_object_missing():
  with tempfile.TemporaryDirectory() as td:
    obj = os.path.join(td, "a.o")
    _write_stamp(obj, "deadbeef")
    # Stamp exists but object doesn't — must return False.
    assert not _check_stamp(obj, "deadbeef")


def test_check_stamp_true_on_match():
  with tempfile.TemporaryDirectory() as td:
    obj = os.path.join(td, "a.o")
    Path(obj).write_text("")
    _write_stamp(obj, "deadbeef")
    assert _check_stamp(obj, "deadbeef")


def test_check_stamp_false_on_digest_mismatch():
  with tempfile.TemporaryDirectory() as td:
    obj = os.path.join(td, "a.o")
    Path(obj).write_text("")
    _write_stamp(obj, "deadbeef")
    assert not _check_stamp(obj, "cafef00d")


# -----------------------------------------------------------------------------
# compile_cpp returns cached when stamp matches, doesn't spawn proc
# -----------------------------------------------------------------------------

def test_compile_cpp_cached_path_doesnt_spawn_subprocess():
  with tempfile.TemporaryDirectory() as td:
    src = os.path.join(td, "a.cpp")
    Path(src).write_text("int x;\n")
    obj = os.path.join(td, "a.o")
    Path(obj).write_text("")                 # fake existing object
    cfg = CompilerConfig(cxx="clang++")
    # Pre-write a matching stamp.
    cmd = _build_compile_cmd(src, obj, cfg)
    _write_stamp(obj, _stamp_digest(src, cmd))

    with mock.patch(
      "srdatalog.codegen.jit.compiler.subprocess.run",
    ) as run_mock:
      result = compile_cpp(src, obj, cfg)
    assert result.cached
    assert result.returncode == 0
    assert run_mock.call_count == 0


def test_compile_cpp_skip_env_forces_cached():
  with tempfile.TemporaryDirectory() as td:
    src = os.path.join(td, "a.cpp")
    Path(src).write_text("int x;\n")
    obj = os.path.join(td, "a.o")
    cfg = CompilerConfig(cxx="clang++")
    with mock.patch.dict(
      os.environ, {"SRDATALOG_JIT_SKIP_COMPILE": "1"},
    ):
      with mock.patch(
        "srdatalog.codegen.jit.compiler.subprocess.run",
      ) as run_mock:
        result = compile_cpp(src, obj, cfg)
    assert result.cached
    assert run_mock.call_count == 0


def test_compile_cpp_invokes_subprocess_on_miss():
  '''When stamp is absent the compiler is invoked. We mock
  subprocess.run so this doesn't need a real clang++.'''
  with tempfile.TemporaryDirectory() as td:
    src = os.path.join(td, "a.cpp")
    Path(src).write_text("int x;\n")
    obj = os.path.join(td, "a.o")
    cfg = CompilerConfig(cxx="/fake/clang++")
    fake = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch(
      "srdatalog.codegen.jit.compiler.subprocess.run", return_value=fake,
    ) as run_mock:
      # Fake "object" materializing by compile.
      Path(obj).write_text("")
      result = compile_cpp(src, obj, cfg)
    assert run_mock.call_count == 1
    assert result.returncode == 0
    assert not result.cached
    # Stamp was written for future cache hits.
    assert os.path.exists(_stamp_path(obj))


def test_compile_cpp_failure_does_not_write_stamp():
  '''Broken compile must NOT leave a stamp; otherwise we'd skip the
  retry after the user fixes the source.'''
  with tempfile.TemporaryDirectory() as td:
    src = os.path.join(td, "a.cpp")
    Path(src).write_text("int x;\n")
    obj = os.path.join(td, "a.o")
    cfg = CompilerConfig(cxx="/fake/clang++")
    fake = mock.Mock(returncode=1, stdout="", stderr="bang")
    with mock.patch(
      "srdatalog.codegen.jit.compiler.subprocess.run", return_value=fake,
    ):
      result = compile_cpp(src, obj, cfg)
    assert result.returncode == 1
    assert not os.path.exists(_stamp_path(obj))


# -----------------------------------------------------------------------------
# compile_jit_project — end-to-end without a real compiler
# -----------------------------------------------------------------------------

def test_compile_jit_project_aggregates_compile_and_link():
  '''Mock the subprocess so we exercise the full compile-all → link
  path without needing real toolchain artifacts.'''
  with tempfile.TemporaryDirectory() as td:
    # Lay out a fake "project" with main + 2 batches.
    project_dir = os.path.join(td, "proj")
    os.makedirs(project_dir)
    main = os.path.join(project_dir, "main.cpp")
    batches = [
      os.path.join(project_dir, "jit_batch_0.cpp"),
      os.path.join(project_dir, "jit_batch_1.cpp"),
    ]
    for p in [main, *batches]:
      Path(p).write_text("// placeholder\n")

    project_result = {
      "dir": project_dir, "main": main, "batches": batches,
      "schema_header": "", "kernel_header": "",
    }
    cfg = CompilerConfig(cxx="/fake/clang++", jobs=2)

    # Both compile AND link succeed — record calls for inspection.
    calls: list[list[str]] = []
    def fake_run(cmd, *args, **kwargs):
      calls.append(list(cmd))
      # Simulate output-file creation so link has something to chew.
      if "-o" in cmd:
        out_idx = cmd.index("-o") + 1
        Path(cmd[out_idx]).write_text("")
      return mock.Mock(returncode=0, stdout="", stderr="")

    with mock.patch(
      "srdatalog.codegen.jit.compiler.subprocess.run", side_effect=fake_run,
    ):
      result = compile_jit_project(project_result, cfg)

    assert isinstance(result, BuildResult)
    assert result.ok()
    # 3 compile invocations (main + 2 batches) + 1 link.
    assert len(result.compile_results) == 3
    assert result.link_result is not None
    assert len(calls) == 4
    # Artifact path ends with the project-dir stem.
    assert result.artifact.endswith(".so")
    assert os.path.basename(project_dir) in result.artifact


def test_compile_jit_project_reports_compile_failure():
  '''Compile error on one batch: link is skipped, result.ok() is False.'''
  with tempfile.TemporaryDirectory() as td:
    project_dir = os.path.join(td, "proj")
    os.makedirs(project_dir)
    main = os.path.join(project_dir, "main.cpp")
    Path(main).write_text("// main\n")
    batch = os.path.join(project_dir, "jit_batch_0.cpp")
    Path(batch).write_text("// batch\n")
    project_result = {
      "dir": project_dir, "main": main, "batches": [batch],
      "schema_header": "", "kernel_header": "",
    }
    cfg = CompilerConfig(cxx="/fake/clang++")

    # First compile succeeds, second fails.
    outcomes = iter([
      mock.Mock(returncode=0, stdout="", stderr=""),
      mock.Mock(returncode=1, stdout="", stderr="broken\n"),
    ])
    def fake_run(*a, **kw):
      return next(outcomes)
    with mock.patch(
      "srdatalog.codegen.jit.compiler.subprocess.run", side_effect=fake_run,
    ):
      result = compile_jit_project(project_result, cfg)

    assert not result.ok()
    # Link skipped because compile failed.
    assert result.link_result is None
    assert result.artifact == ""


# -----------------------------------------------------------------------------
# Real end-to-end smoke — only when explicitly opted in.
# -----------------------------------------------------------------------------

def test_real_compile_hello_world_roundtrip():
  '''Use an actual clang++/g++ to compile+link a trivial hello-world
  source, confirming the subprocess wiring works on this host.

  Opt in via SRDATALOG_JIT_RUN_COMPILE_TESTS=1 — default CI / dev
  machines often lack CUDA headers so we can't compile a real JIT
  project by default.'''
  if os.environ.get("SRDATALOG_JIT_RUN_COMPILE_TESTS", "") != "1":
    print("[SKIP] set SRDATALOG_JIT_RUN_COMPILE_TESTS=1 to enable")
    return

  cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
  if not cxx:
    print("[SKIP] no C++ compiler on PATH")
    return

  with tempfile.TemporaryDirectory() as td:
    src = os.path.join(td, "main.cpp")
    Path(src).write_text(
      "#include <cstdint>\n"
      "extern \"C\" int32_t add(int32_t a, int32_t b) { return a + b; }\n"
    )
    obj = os.path.join(td, "main.o")
    cfg = CompilerConfig(cxx=cxx, shared=True)
    r = compile_cpp(src, obj, cfg)
    assert r.returncode == 0, r.stderr
    lib = os.path.join(td, "libtest.so")
    lr = link_shared([obj], lib, cfg)
    assert lr.returncode == 0, lr.stderr
    assert os.path.exists(lib)
    # Second compile should hit the cache.
    r2 = compile_cpp(src, obj, cfg)
    assert r2.cached


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
