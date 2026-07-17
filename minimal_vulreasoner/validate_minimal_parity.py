#!/usr/bin/env python3
'''Differentially validate the real minimal example against PyReason.'''

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from minimal_vulreasoner.example_config import WORKFLOW

RESULT_PREFIX = "RESULT_JSON="


def _pyreason_rows(checkout: str | None) -> dict[str, list[float]]:
  if checkout:
    root = Path(checkout).expanduser().resolve()
  else:
    root = Path(__file__).resolve().parents[2] / "pyreason"
  if not (root / "pyreason" / "__init__.py").exists():
    raise RuntimeError(f"PyReason checkout not found at {root}; pass --pyreason-checkout")
  sys.path.insert(0, str(root))

  import pyreason as pr

  from minimal_vulreasoner import run_minimal_reasoner as oracle

  oracle.configure_engine()
  oracle.load_rules()
  oracle.add_facts()
  interpretation = pr.reason(timesteps=oracle.END_TIME)
  history = interpretation.get_dict()
  workflow_nodes = {node for node, _ in WORKFLOW}
  rows = {}
  for time_, components in history.items():
    for node, predicates in components.items():
      if node not in workflow_nodes or "analystAt" not in predicates:
        continue
      lower, upper = predicates["analystAt"]
      if (lower, upper) != (0.0, 0.0):
        rows[f"{node}@{time_}"] = [float(lower), float(upper)]
  return dict(sorted(rows.items()))


def _srdatalog_result(args) -> dict[str, object]:
  runner = Path(__file__).with_name("run_srdatalog_reasoner.py")
  command = [
    sys.executable,
    str(runner),
    "--cache-base",
    args.cache_base,
    "--data-dir",
    args.data_dir,
    "--jobs",
    str(args.jobs),
  ]
  if args.no_compile:
    command.append("--no-compile")
  completed = subprocess.run(command, text=True, capture_output=True, check=False)
  if completed.returncode != 0:
    raise RuntimeError(
      f"SRDatalog runner failed ({completed.returncode})\n"
      + completed.stdout
      + completed.stderr
    )
  print(completed.stdout, end="")
  result_line = next(
    (line for line in completed.stdout.splitlines() if line.startswith(RESULT_PREFIX)),
    None,
  )
  if result_line is None:
    raise RuntimeError("SRDatalog runner did not emit RESULT_JSON")
  return json.loads(result_line.removeprefix(RESULT_PREFIX))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--pyreason-checkout")
  parser.add_argument("--cache-base", default="./build")
  parser.add_argument("--data-dir", default="./build/minimal_vulreasoner_graphml")
  parser.add_argument("--jobs", type=int, default=8)
  parser.add_argument("--no-compile", action="store_true")
  args = parser.parse_args()

  pyreason_rows = _pyreason_rows(args.pyreason_checkout)
  srdatalog = _srdatalog_result(args)
  srdatalog_rows = srdatalog["analyst_rows"]
  if srdatalog_rows != pyreason_rows:
    print("FAIL: PyReason and SRDatalog temporal AnalystAt rows differ")
    print("PYREASON=" + json.dumps(pyreason_rows, sort_keys=True))
    print("SRDATALOG=" + json.dumps(srdatalog_rows, sort_keys=True))
    raise SystemExit(1)
  print("PASS: complete temporal AnalystAt rows match PyReason")
  print("ANALYST_ROWS=" + json.dumps(pyreason_rows, sort_keys=True))


if __name__ == "__main__":
  main()
