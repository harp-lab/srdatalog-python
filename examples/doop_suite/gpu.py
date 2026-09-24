"""Build and measure the canonical DOOP program through the generated CUDA ABI.

Every warmup/repetition runs in a fresh process and must exit normally after
checked shutdown. Build, CSV loading and exact IDB exports are not timed as
fixedpoint work. A successful report does not assert cross-engine equality.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "examples" / "doop.py"


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _save(path: Path, value: dict) -> None:
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  temporary.replace(path)


def _program(meta: dict, plan: str):
  # Resolve both compiler and logical program from this checkout, not an
  # installed historical version or a workstation-specific module directory.
  sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "examples")]
  from doop import build_doopdb_program

  from srdatalog import Program

  program = build_doopdb_program(meta)
  if plan == "baseline":
    return program
  if plan != "bitmap":
    raise ValueError(f"Unknown GPU plan: {plan}")
  if sum(rule.name == "VPT_Assign" for rule in program.rules) != 1:
    raise ValueError("Expected exactly one canonical VPT_Assign rule")
  return Program(
    rules=[
      rule.with_plan(delta=0, dedup_bitmap=True).with_plan(delta=1, dedup_bitmap=True)
      if rule.name == "VPT_Assign"
      else rule
      for rule in program.rules
    ]
  )


def _check_prepared(facts: Path, program) -> dict:
  from doop_suite.prepare import SCHEMA

  manifest = json.loads((facts / "manifest.json").read_text(encoding="utf-8"))
  if manifest["program"]["sha256"] != _sha256(_SOURCE):
    raise ValueError("Prepared facts target a different logical DOOP program")
  for key, filename in (("metadata", "meta.json"), ("symbols", "str2num.json")):
    if manifest[key]["sha256"] != _sha256(facts / filename):
      raise ValueError(f"Prepared {filename} checksum mismatch")
  inputs = {relation.name: relation for relation in program.relations if relation.input_file}
  if set(manifest["relations"]) != set(SCHEMA) or not inputs.keys() <= SCHEMA.keys():
    raise ValueError("Prepared input relation schema differs from the declared input contract")
  for name, relation in inputs.items():
    entry = manifest["relations"][name]
    if entry["path"] != relation.input_file or entry["arity"] != relation.arity:
      raise ValueError(f"Prepared input schema mismatch: {name}")
    if entry["sha256"] != _sha256(facts / relation.input_file):
      raise ValueError(f"Prepared input checksum mismatch: {name}")
  return manifest


def _run_process(command: list[str], log: Path, timeout: int) -> float:
  """Gate on actual process exit, killing its entire process group on timeout."""
  started = time.perf_counter()
  with log.open("xb") as stream:
    child = subprocess.Popen(
      command,
      stdout=stream,
      stderr=subprocess.STDOUT,
      start_new_session=True,
      cwd=_ROOT,
    )
    try:
      code = child.wait(timeout=timeout)
    except BaseException:
      # The group is ours (start_new_session=True), including compiler children.
      # Do not leave a compiler/GPU worker alive after its caller gives up.
      with contextlib.suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
      child.wait()
      raise
  if code != 0:
    raise RuntimeError(f"GPU worker exited {code}; see {log}")
  return time.perf_counter() - started


def _build(facts: Path, output: Path, plan: str, jobs: int) -> None:
  started = time.perf_counter()
  program = _program(json.loads((facts / "meta.json").read_text(encoding="utf-8")), plan)
  manifest = _check_prepared(facts, program)
  preparation_seconds = time.perf_counter() - started
  from srdatalog import CompilerConfig, build_project, compile_jit_project
  from srdatalog.runtime import (
    cuda_compile_flags,
    cuda_include_paths,
    cuda_libs,
    cuda_link_flags,
    runtime_defines,
    runtime_include_paths,
  )

  started = time.perf_counter()
  project = build_project(program, "DoopSuite", cache_base=str(output / "build-cache"))
  emit_seconds = time.perf_counter() - started
  config = CompilerConfig(
    include_paths=runtime_include_paths() + cuda_include_paths(),
    defines=runtime_defines(),
    cxx_flags=cuda_compile_flags() + ["-fPIC"],
    link_flags=cuda_link_flags(),
    libs=cuda_libs() + ["boost_container"],
    shared=True,
    jobs=jobs,
  )
  started = time.perf_counter()
  build = compile_jit_project(project, config)
  compile_seconds = time.perf_counter() - started
  for result in [*build.compile_results, *([build.link_result] if build.link_result else [])]:
    print(json.dumps({"command": result.command, "returncode": result.returncode}), flush=True)
    if result.stdout:
      print(result.stdout, flush=True)
    if result.stderr:
      print(result.stderr, file=sys.stderr, flush=True)
  if not build.ok() or not build.artifact or not Path(build.artifact).is_file():
    raise RuntimeError("DOOP GPU compilation/linking failed")
  _save(
    output / "build.json",
    {
      "status": "built",
      "library": str(Path(build.artifact).resolve()),
      "library_sha256": _sha256(Path(build.artifact)),
      "source_sha256": _sha256(_SOURCE),
      "metadata_sha256": _sha256(facts / "meta.json"),
      "input_source_sha256": manifest["source_sha256"],
      "input_manifest_sha256": _sha256(facts / "manifest.json"),
      "preparation_seconds": preparation_seconds,
      "emit_seconds": emit_seconds,
      "compile_seconds": compile_seconds,
      "relations": [
        {"name": relation.name, "arity": relation.arity, "input_file": relation.input_file}
        for relation in program.relations
      ],
      "input_rows": {
        relation.name: manifest["relations"][relation.name]["rows"]
        for relation in program.relations
        if relation.input_file
      },
    },
  )


def _execute(build: dict, facts: Path, report_path: Path, export: Path | None) -> None:
  # The library remains loaded until normal process exit: do not dlclose while
  # generated function-local GPU scratch or CUDA stream pools still exist.
  library = ctypes.CDLL(build["library"], mode=ctypes.RTLD_GLOBAL)
  signatures = {
    "init": [],
    "load_all": [ctypes.c_char_p],
    "run": [ctypes.c_ulonglong],
    "synchronize": [],
    "shutdown": [],
    "get_size": [ctypes.c_char_p, ctypes.POINTER(ctypes.c_ulonglong)],
    "export_tsv": [ctypes.c_char_p, ctypes.c_char_p],
  }
  for name, args in signatures.items():
    function = getattr(library, "srdatalog_" + name)
    function.argtypes = args
    function.restype = ctypes.c_int

  def checked(name: str, *arguments) -> None:
    code = getattr(library, "srdatalog_" + name)(*arguments)
    if code != 0:
      raise RuntimeError(f"srdatalog_{name} returned {code}")

  report = {"status": "running", "stages_seconds": {}, "outputs": {}}
  initialized = False
  _save(report_path, report)
  try:
    started = time.perf_counter()
    checked("init")
    initialized = True
    checked("synchronize")
    report["stages_seconds"]["init"] = time.perf_counter() - started
    started = time.perf_counter()
    checked("load_all", os.fsencode(facts))
    checked("synchronize")
    report["stages_seconds"]["load"] = time.perf_counter() - started
    # Includes fresh host-to-device DB construction and *every* fixedpoint step.
    # Zero means no iteration cap, not a single-iteration smoke benchmark.
    started = time.perf_counter()
    checked("run", 0)
    checked("synchronize")
    report["fixedpoint_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    counts = {}
    for relation in build["relations"]:
      count = ctypes.c_ulonglong()
      checked("get_size", relation["name"].encode(), ctypes.byref(count))
      counts[relation["name"]] = count.value
    report["relation_counts"] = counts
    report["stages_seconds"]["counts"] = time.perf_counter() - started
    for name, rows in build["input_rows"].items():
      if counts[name] != rows:
        raise RuntimeError(
          f"Loaded input cardinality mismatch for {name}: {counts[name]} != {rows}"
        )
    if export is not None:
      export.mkdir()
      started = time.perf_counter()
      for relation in build["relations"]:
        if relation["input_file"]:
          continue
        name = relation["name"]
        path = export / f"{name}.tsv"
        checked("export_tsv", name.encode(), os.fsencode(path))
        if not path.is_file():
          raise RuntimeError(f"Native export did not create {path}")
        report["outputs"][name] = str(path)
      checked("synchronize")
      report["stages_seconds"]["export"] = time.perf_counter() - started
  except BaseException as error:
    report["status"] = "failed"
    report["error"] = repr(error)
    raise
  finally:
    try:
      if initialized:
        started = time.perf_counter()
        checked("shutdown")
        report["stages_seconds"]["shutdown"] = time.perf_counter() - started
    except BaseException as error:
      report["status"] = "failed"
      report["shutdown_error"] = repr(error)
      raise
    finally:
      _save(report_path, report)
  report["status"] = "native_completed"
  _save(report_path, report)


def run_gpu(
  facts: Path,
  output: Path,
  *,
  plan: str = "baseline",
  jobs: int = 2,
  timeout: int = 900,
  warmups: int = 1,
  repeats: int = 3,
) -> dict:
  """Return a process-gated report; timeout applies to build and each fresh run.

  Requires the checkout's CUDA compiler/runtime dependencies and an available
  GPU. Export borrows canonical index columns and uses bounded host buffers.
  Neither a child-written report nor native return alone is success:
  normal process teardown must also finish within the timeout.
  """
  if plan not in ("baseline", "bitmap"):
    raise ValueError("plan must be 'baseline' or 'bitmap'")
  if jobs < 1 or timeout < 1 or warmups < 0 or repeats < 1:
    raise ValueError("jobs/timeout/repeats must be positive; warmups must be nonnegative")
  facts, output = Path(facts).resolve(), Path(output).resolve()
  if not facts.is_dir():
    raise NotADirectoryError(facts)
  if output.is_relative_to(_ROOT) or output.is_relative_to(facts):
    raise ValueError("GPU output must be outside the repository and prepared facts")
  output.mkdir(parents=True, exist_ok=False)
  report = {
    "status": "running",
    "backend": "gpu",
    "dataset": facts.name,
    "plan": plan,
    "facts": str(facts),
    "source_sha256": _sha256(_SOURCE),
    "timings_seconds": [],
    "relation_counts": {},
    "outputs": {},
    "runs": [],
    "jobs": jobs,
    "timeout_seconds_per_process": timeout,
    "warmups": warmups,
    "repeats": repeats,
    "timing_boundary": "fresh H2D database + entire unlimited fixedpoint + checked device synchronization",
    "process_isolation": "one fresh process per warmup/measured run; normal exit required",
  }
  report_path = output / "report.json"
  _save(report_path, report)
  worker = [sys.executable, "-B", str(Path(__file__).resolve())]
  try:
    report["build_process_seconds"] = _run_process(
      [*worker, "_build", str(facts), str(output), plan, str(jobs)],
      output / "build.log",
      timeout,
    )
    build = json.loads((output / "build.json").read_text(encoding="utf-8"))
    if build["status"] != "built" or build["source_sha256"] != report["source_sha256"]:
      raise RuntimeError("Build report incomplete or logical program changed during build")
    for field in (
      "metadata_sha256",
      "input_source_sha256",
      "input_manifest_sha256",
      "library",
      "library_sha256",
      "preparation_seconds",
      "emit_seconds",
      "compile_seconds",
    ):
      report[field] = build[field]
    reference_counts = None
    for index in range(-warmups, repeats):
      label = f"warmup-{index + warmups:03d}" if index < 0 else f"run-{index:03d}"
      run_path = output / f"{label}.json"
      export = output / "relations" if index == repeats - 1 else None
      command = [*worker, "_run", str(output / "build.json"), str(facts), str(run_path)]
      if export is not None:
        command.append(str(export))
      elapsed = _run_process(command, output / f"{label}.log", timeout)
      run = json.loads(run_path.read_text(encoding="utf-8"))
      if run["status"] != "native_completed":
        raise RuntimeError(f"{label} exited without completing every native stage")
      counts = run["relation_counts"]
      if set(counts) != {relation["name"] for relation in build["relations"]}:
        raise RuntimeError(f"{label} did not report the complete relation schema")
      if reference_counts is not None and counts != reference_counts:
        raise RuntimeError(f"Relation cardinalities changed in fresh repetition {label}")
      reference_counts = counts
      run.update({"index": index, "warmup": index < 0, "process_seconds": elapsed})
      report["runs"].append(run)
      if index >= 0:
        report["timings_seconds"].append(run["fixedpoint_seconds"])
      report["relation_counts"] = counts
      if export is not None:
        expected = {
          relation["name"] for relation in build["relations"] if not relation["input_file"]
        }
        if set(run["outputs"]) != expected:
          raise RuntimeError("Final repetition did not export every IDB relation")
        report["outputs"] = run["outputs"]
        report["export_seconds"] = run["stages_seconds"]["export"]
      _save(report_path, report)
    report["load_seconds"] = [
      run["stages_seconds"]["load"] for run in report["runs"] if not run["warmup"]
    ]
    report["status"] = "passed"
    _save(report_path, report)
    return report
  except BaseException as error:
    report["status"] = "failed"
    report["error"] = repr(error)
    _save(report_path, report)
    raise


def _main() -> None:
  if len(sys.argv) > 1 and sys.argv[1] == "_build":
    _build(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], int(sys.argv[5]))
    return
  if len(sys.argv) > 1 and sys.argv[1] == "_run":
    build = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    _execute(
      build, Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]) if len(sys.argv) > 5 else None
    )
    return
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("facts", type=Path)
  parser.add_argument("output", type=Path)
  parser.add_argument("--plan", choices=("baseline", "bitmap"), default="baseline")
  parser.add_argument("--jobs", type=int, default=2)
  parser.add_argument("--timeout", type=int, default=900)
  parser.add_argument("--warmups", type=int, default=1)
  parser.add_argument("--repeats", type=int, default=3)
  args = parser.parse_args()
  print(json.dumps(run_gpu(**vars(args)), indent=2))


if __name__ == "__main__":
  _main()
