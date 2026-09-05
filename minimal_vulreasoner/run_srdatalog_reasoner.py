#!/usr/bin/env python3
'''Run the existing minimal VulReasoner inputs end to end on SRDatalog.'''

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
  sys.path.insert(0, str(REPO_SRC))

from srdatalog import build_project, compile_jit_project, float32_to_u32

try:
  from .benchmark_srdatalog import _bind, _compiler_config, _copy_analyst_rows
  from .example_config import END_TIME, INITIAL_NODE, KG_FILE, WORKFLOW
  from .graphml_ingest import emit_minimal_dataset
  from .srdatalog_query import RULE_SPECS, build_analyst_program
except ImportError:
  from benchmark_srdatalog import _bind, _compiler_config, _copy_analyst_rows
  from example_config import END_TIME, INITIAL_NODE, KG_FILE, WORKFLOW
  from graphml_ingest import emit_minimal_dataset
  from srdatalog_query import RULE_SPECS, build_analyst_program


def _analyst_digest(rows: dict[str, list[float]]) -> str:
  digest = hashlib.sha256()
  for key, (lower, upper) in sorted(rows.items()):
    digest.update(
      f"{key}:{float32_to_u32(lower)}:{float32_to_u32(upper)}\n".encode()
    )
  return digest.hexdigest()


def _artifact(project: dict[str, object], *, compile_project: bool, jobs: int):
  if compile_project:
    started = time.perf_counter()
    build = compile_jit_project(project, _compiler_config(jobs))
    seconds = time.perf_counter() - started
    if not build.ok():
      failure = next(result for result in build.compile_results if result.returncode)
      raise RuntimeError((failure.stderr or failure.stdout)[-12000:])
    return str(Path(build.artifact).resolve()), seconds
  artifacts = list(Path(project["dir"]).glob("*.so"))
  if not artifacts:
    raise RuntimeError(f"no cached shared library in {project['dir']}; omit --no-compile")
  return str(artifacts[0].resolve()), 0.0


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--cache-base", default="./build")
  parser.add_argument("--data-dir", default="./build/minimal_vulreasoner_graphml")
  parser.add_argument("--jobs", type=int, default=8)
  parser.add_argument("--no-compile", action="store_true")
  args = parser.parse_args()

  ingest_started = time.perf_counter()
  data_dir = emit_minimal_dataset(
    KG_FILE,
    args.data_dir,
    workflow=WORKFLOW,
    initial_node=INITIAL_NODE,
    end_time=END_TIME,
    connector_predicates=tuple(spec.connector_predicate for spec in RULE_SPECS),
  )
  ingest_seconds = time.perf_counter() - ingest_started
  manifest = json.loads((data_dir / "manifest.json").read_text())
  symbols = json.loads((data_dir / "symbols.json").read_text())
  symbol_names = {int(identifier): symbol for symbol, identifier in symbols.items()}

  emit_started = time.perf_counter()
  project = build_project(
    build_analyst_program(),
    "VulReasonerPlan",
    cache_base=args.cache_base,
  )
  emit_seconds = time.perf_counter() - emit_started
  artifact, compile_seconds = _artifact(
    project,
    compile_project=not args.no_compile,
    jobs=args.jobs,
  )

  lib = _bind(artifact)
  if lib.srdatalog_init() != 0:
    raise RuntimeError("srdatalog_init failed")
  load_started = time.perf_counter()
  if lib.srdatalog_load_all(str(data_dir.resolve()).encode()) != 0:
    raise RuntimeError("srdatalog_load_all failed")
  load_seconds = time.perf_counter() - load_started

  run_started = time.perf_counter()
  if lib.srdatalog_run(0) != 0:
    raise RuntimeError("srdatalog_run failed")
  run_seconds = time.perf_counter() - run_started

  query_started = time.perf_counter()
  raw_rows = _copy_analyst_rows(lib)
  analyst_rows = {
    f"{symbol_names[node]}@{time_}": [lower, upper]
    for node, time_, lower, upper in raw_rows
  }
  analyst_rows = dict(sorted(analyst_rows.items()))
  query_seconds = time.perf_counter() - query_started
  result = {
    "engine": "srdatalog",
    "graph_nodes": manifest["graph_nodes"],
    "graph_edges": manifest["graph_edges"],
    "selected_connector_facts": manifest["selected_connector_facts"],
    "analyst_rows": analyst_rows,
    "analyst_rows_sha256": _analyst_digest(analyst_rows),
    "ingest_seconds": ingest_seconds,
    "emit_seconds": emit_seconds,
    "compile_seconds": compile_seconds,
    "load_seconds": load_seconds,
    "run_seconds": run_seconds,
    "query_seconds": query_seconds,
  }
  print("Expected temporal chain: b1@0, b1@1, b2@2, b3@3, b4@4")
  print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
  lib.srdatalog_shutdown()
  sys.stdout.flush()
  sys.stderr.flush()

  # The current ctypes/CUDA DSO has a teardown-order issue after explicit DB
  # shutdown.  Exit without running dlclose/static destructors.
  os._exit(0)


if __name__ == "__main__":
  main()
