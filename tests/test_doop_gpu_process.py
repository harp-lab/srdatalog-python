"""A native report is not success until the fresh GPU worker exits cleanly."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
  "doop_gpu_runner", Path(__file__).parents[1] / "examples" / "doop_suite" / "gpu.py"
)
assert _SPEC is not None and _SPEC.loader is not None
gpu = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gpu)


def test_worker_failure_after_writing_report_is_not_success(tmp_path):
  report = tmp_path / "native.json"
  command = [
    sys.executable,
    "-c",
    "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("
    "'{\"status\":\"native_completed\"}'); print('teardown failed', flush=True); sys.exit(17)",
    str(report),
  ]
  log = tmp_path / "worker.log"
  with pytest.raises(RuntimeError, match="exited 17"):
    gpu._run_process(command, log, timeout=10)
  assert report.read_text() == '{"status":"native_completed"}'
  assert "teardown failed" in log.read_text()


def test_worker_timeout_is_an_error_with_preserved_diagnostics(tmp_path):
  log = tmp_path / "timeout.log"
  command = [
    sys.executable,
    "-c",
    "import time; print('entered native fixedpoint', flush=True); time.sleep(60)",
  ]
  with pytest.raises(subprocess.TimeoutExpired):
    gpu._run_process(command, log, timeout=2)
  assert "entered native fixedpoint" in log.read_text()


def test_failed_worker_keeps_relative_artifacts_outside_checkout(tmp_path, monkeypatch):
  checkout, output = tmp_path / "checkout", tmp_path / "output"
  checkout.mkdir()
  output.mkdir()
  monkeypatch.setattr(gpu, "_ROOT", checkout)
  command = [
    sys.executable,
    "-c",
    "from pathlib import Path; Path('allocator_failure.log').write_text('pool exhausted'); "
    "raise SystemExit(17)",
  ]
  with pytest.raises(RuntimeError, match="exited 17"):
    gpu._run_process(command, output / "worker.log", timeout=10)
  assert (output / "allocator_failure.log").read_text() == "pool exhausted"
  assert not (checkout / "allocator_failure.log").exists()


@pytest.mark.parametrize("index_only_outputs", [False, True])
def test_native_block_group_scratch_survives_normal_process_teardown(tmp_path, index_only_outputs):
  import os

  if os.environ.get("SRDATALOG_JIT_RUN_COMPILE_TESTS") != "1":
    pytest.skip("Opt in to native CUDA compilation with SRDATALOG_JIT_RUN_COMPILE_TESTS=1")

  from srdatalog import CompilerConfig, Program, Relation, Var, build_project, compile_jit_project
  from srdatalog.runtime import (
    cuda_compile_flags,
    cuda_include_paths,
    cuda_libs,
    cuda_link_flags,
    runtime_defines,
    runtime_include_paths,
  )

  key, left, right = Var("key"), Var("left"), Var("right")
  start, middle, end = Var("start"), Var("middle"), Var("end")
  lhs = Relation("Left", 2, input_file="Left.csv")
  rhs = Relation("Right", 2, input_file="Right.csv")
  joined = Relation("Joined", 2)
  projected = Relation("Projected", 1)
  chain = Relation("Chain", 2, input_file="Chain.csv")
  closure = Relation("Closure", 2)
  program = Program(
    rules=[
      (joined(left, right) <= lhs(key, left) & rhs(key, right))
      .named("Join")
      .with_plan(block_group=True, var_order=["key", "left", "right"]),
      (projected(left) <= joined(left, right)).named("ReadJoined"),
      (closure(start, end) <= chain(start, end)).named("ClosureSeed"),
      (closure(start, end) <= closure(start, middle) & chain(middle, end)).named("ClosureStep"),
    ]
  )
  project = build_project(
    program, "ScratchLifetime", cache_base=str(tmp_path / "cache"),
    index_only_outputs=index_only_outputs,
  )
  compiled = compile_jit_project(
    project,
    CompilerConfig(
      include_paths=runtime_include_paths() + cuda_include_paths(),
      defines=runtime_defines(),
      cxx_flags=cuda_compile_flags() + ["-fPIC"],
      link_flags=cuda_link_flags(),
      libs=cuda_libs() + ["boost_container"],
      shared=True,
      jobs=2,
    ),
  )
  assert compiled.ok(), compiled
  facts = tmp_path / "facts"
  facts.mkdir()
  # More than 256 root keys exercises persistent block-group histogram scratch.
  for filename, offset in (("Left.csv", 0), ("Right.csv", 10000)):
    (facts / filename).write_text(
      "".join(f"{k}\t{offset + 2 * k + v}\n" for k in range(512) for v in range(2))
    )
  (facts / "Chain.csv").write_text("0\t1\n1\t2\n2\t3\n3\t4\n")
  exported = tmp_path / "Joined.tsv"
  worker = r"""
import ctypes as c
import os
import sys
from pathlib import Path

lib = c.CDLL(sys.argv[1], mode=c.RTLD_GLOBAL)
signatures = {
    "init": ([], c.c_int),
    "load_all": ([c.c_char_p], c.c_int),
    "load_csv": ([c.c_char_p, c.c_char_p], c.c_int),
    "prepare": ([], c.c_int),
    "run": ([c.c_ulonglong], c.c_int),
    "get_size": ([c.c_char_p, c.POINTER(c.c_ulonglong)], c.c_int),
    "size": ([c.c_char_p], c.c_ulonglong),
    "export_tsv": ([c.c_char_p, c.c_char_p], c.c_int),
    "shutdown": ([], c.c_int),
}
for name, (arguments, result) in signatures.items():
    function = getattr(lib, "srdatalog_" + name)
    function.argtypes = arguments
    function.restype = result
assert lib.srdatalog_init() == 0
assert lib.srdatalog_prepare() == 1  # No host database exists yet.
try:
    assert lib.srdatalog_load_all(os.fsencode(sys.argv[2])) == 0
    assert lib.srdatalog_prepare() == 0
    count = c.c_ulonglong()
    assert lib.srdatalog_get_size(b"Left", c.byref(count)) == 0
    assert count.value == 1024
    # Canonical IDB indexes are not built until run; the legacy size probe is
    # valid here and must observe empty outputs after input-only preparation.
    assert lib.srdatalog_size(b"Joined") == 0
    assert lib.srdatalog_run(0) == 0
    assert lib.srdatalog_get_size(b"Closure", c.byref(count)) == 0
    assert count.value == 10
    assert lib.srdatalog_get_size(b"Joined", c.byref(count)) == 0
    assert count.value == lib.srdatalog_size(b"Joined") == 2048
    assert lib.srdatalog_export_tsv(b"Joined", os.fsencode(sys.argv[3])) == 0
    assert lib.srdatalog_get_size(b"Projected", c.byref(count)) == 0
    assert count.value == lib.srdatalog_size(b"Projected") == 1024
    assert lib.srdatalog_export_tsv(b"Projected", os.fsencode(sys.argv[3] + ".projected")) == 0
    # A capped rerun must start from empty IDB, not reuse the completed closure.
    assert lib.srdatalog_run(1) == 0
    assert lib.srdatalog_get_size(b"Closure", c.byref(count)) == 0
    capped_count = count.value
    assert 0 < capped_count < 10
    assert lib.srdatalog_run(1) == 0
    assert lib.srdatalog_get_size(b"Closure", c.byref(count)) == 0
    assert count.value == capped_count
    # Both loading APIs invalidate an already prepared snapshot. Loading appends
    # host inputs, so the subsequent run must include the newly staged chains.
    assert lib.srdatalog_prepare() == 0
    chain_path = Path(sys.argv[2]) / "Chain.csv"
    chain_path.write_text("10\t11\n20\t21\n")
    assert lib.srdatalog_load_csv(b"Chain", os.fsencode(chain_path)) == 0
    assert lib.srdatalog_run(0) == 0
    assert lib.srdatalog_get_size(b"Closure", c.byref(count)) == 0
    assert count.value == 12
    assert lib.srdatalog_prepare() == 0
    chain_path.write_text("30\t31\n31\t32\n")
    assert lib.srdatalog_load_all(os.fsencode(sys.argv[2])) == 0
    assert lib.srdatalog_run(0) == 0
    assert lib.srdatalog_get_size(b"Closure", c.byref(count)) == 0
    assert count.value == 15
finally:
    assert lib.srdatalog_shutdown() == 0
"""
  gpu._run_process(
    [sys.executable, "-c", worker, str(compiled.artifact), str(facts), str(exported)],
    tmp_path / "worker.log",
    timeout=120,
  )
  observed = {
    tuple(map(int, line.split("\t")))
    for line in exported.read_text().splitlines()
  }
  assert observed == {
    (2 * k + i, 10000 + 2 * k + j) for k in range(512) for i in range(2) for j in range(2)
  }
  assert {
    int(line) for line in Path(str(exported) + ".projected").read_text().splitlines()
  } == set(range(1024))
