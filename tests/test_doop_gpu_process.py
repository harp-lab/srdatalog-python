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
