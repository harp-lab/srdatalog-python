#!/usr/bin/env python3
"""Run deterministic VulReasoner stress workloads through the PyReason oracle."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import networkx as nx
from annotation_fn import paired_minimum_bounds_ann_fn
from stress_workload import (
  emit_csv_dataset,
  generate_stress_workload,
  summarize_target_bounds,
  workload_summary,
)

HERE = Path(__file__).resolve().parent
RULES_FILE = HERE / "rules" / "analyst_rules.csv"


def _load_pyreason(checkout: str | None):
  if checkout:
    root = Path(checkout).expanduser().resolve()
  else:
    root = HERE.parent.parent / "pyreason"
  if not (root / "pyreason" / "__init__.py").exists():
    raise RuntimeError(f"PyReason checkout not found at {root}; pass --pyreason-checkout")
  sys.path.insert(0, str(root))
  import pyreason as pr

  return pr


def _bounded_fact(predicate: str, left: str, right: str | None, lower: float, upper: float) -> str:
  component = left if right is None else f"{left},{right}"
  return f"{predicate}({component}):[{lower:.8f},{upper:.8f}]"


def _configure(pr, workload, *, parallel: bool) -> None:
  pr.reset()
  pr.load_graph(nx.DiGraph())
  pr.settings.verbose = False
  pr.settings.atom_trace = False
  pr.settings.save_graph_attributes_to_trace = False
  pr.settings.store_interpretation_changes = False
  pr.settings.allow_ground_rules = True
  pr.settings.persistent = False
  pr.settings.parallel_computing = parallel
  pr.add_annotation_function(paired_minimum_bounds_ann_fn)
  pr.add_rule_from_csv(str(RULES_FILE), raise_errors=False)
  pr.add_closed_world_predicate("analystAt")

  for i, item in enumerate(workload.labels):
    pr.add_fact(
      pr.Fact(
        _bounded_fact("hasLabel", item.block, item.label, item.lower, item.upper),
        f"label_{i}",
        static=True,
      )
    )
  for i, item in enumerate(workload.connectors):
    pr.add_fact(
      pr.Fact(
        _bounded_fact(
          item.predicate,
          item.source_label,
          item.target_label,
          item.lower,
          item.upper,
        ),
        f"connector_{i}",
        static=True,
      )
    )
  for i, step in enumerate(workload.steps):
    pr.add_fact(
      pr.Fact(
        f"stepFrom({step.source},{step.target})",
        f"step_{i}",
        step.time,
        step.time,
      )
    )
  for i, seed in enumerate(workload.seeds):
    pr.add_fact(
      pr.Fact(
        _bounded_fact("analystAt", seed.node, None, seed.lower, seed.upper),
        f"seed_{i}",
        seed.time,
        seed.time,
      )
    )


def _run_once(pr, workload, *, parallel: bool) -> dict[str, object]:
  setup_start = time.perf_counter()
  _configure(pr, workload, parallel=parallel)
  setup_seconds = time.perf_counter() - setup_start

  reason_start = time.perf_counter()
  interpretation = pr.reason(timesteps=workload.depth + 1)
  reason_seconds = time.perf_counter() - reason_start

  query_start = time.perf_counter()
  reached = 0
  target_bounds: dict[str, tuple[float, float]] = {}
  for target in workload.target_nodes:
    query = pr.Query(f"analystAt({target}):[0,1]")
    bounds = interpretation.query(query, return_bool=False)
    target_bounds[target] = (float(bounds[0]), float(bounds[1]))
    if bounds != (0, 1) and bounds != (0, 0):
      reached += 1
  query_seconds = time.perf_counter() - query_start

  return {
    **workload_summary(workload),
    "parallel": parallel,
    "setup_seconds": setup_seconds,
    "reason_seconds": reason_seconds,
    "query_seconds": query_seconds,
    "total_seconds": setup_seconds + reason_seconds + query_seconds,
    "reached_targets": reached,
    **summarize_target_bounds(target_bounds),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--depth", type=int, default=4)
  parser.add_argument("--width", type=int, default=1)
  parser.add_argument("--fanout", type=int, default=1)
  parser.add_argument("--repeat", type=int, default=1)
  parser.add_argument(
    "--case",
    action="append",
    default=[],
    metavar="DEPTH,WIDTH,FANOUT",
    help="Run multiple shapes in one warm process; overrides depth/width/fanout",
  )
  parser.add_argument(
    "--warmup",
    action="store_true",
    help="Compile PyReason on a 1x1x1 case before reporting requested cases",
  )
  parser.add_argument("--parallel", action="store_true")
  parser.add_argument("--pyreason-checkout")
  parser.add_argument("--emit-dir")
  args = parser.parse_args()

  pr = _load_pyreason(args.pyreason_checkout)
  if args.warmup:
    _run_once(pr, generate_stress_workload(1, 1, 1), parallel=args.parallel)

  if args.case:
    shapes: list[tuple[int, int, int]] = []
    for raw in args.case:
      try:
        shape = tuple(int(value) for value in raw.split(","))
      except ValueError as exc:
        raise SystemExit(f"invalid --case {raw!r}; expected DEPTH,WIDTH,FANOUT") from exc
      if len(shape) != 3:
        raise SystemExit(f"invalid --case {raw!r}; expected DEPTH,WIDTH,FANOUT")
      shapes.append(shape)
  else:
    shapes = [(args.depth, args.width, args.fanout)]

  for depth, width, fanout in shapes:
    workload = generate_stress_workload(depth, width, fanout)
    if args.emit_dir:
      case_dir = Path(args.emit_dir)
      if len(shapes) > 1:
        case_dir /= f"d{depth}_w{width}_f{fanout}"
      emit_csv_dataset(workload, case_dir)
    for repeat in range(args.repeat):
      result = _run_once(pr, workload, parallel=args.parallel)
      result["repeat"] = repeat
      print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
  main()
