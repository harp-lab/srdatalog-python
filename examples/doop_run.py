'''Doop points-to analysis — canonical end-to-end driver.

Thin wrapper around `run_benchmark.py doop` with sensible defaults for
batik on this box. Runs the full pipeline:

    DSL program (from auto-translated doop.py)
      → HIR → MIR → .cpp tree (build_project)
      → .so (ninja + ccache compile)
      → dlopen + extern "C" shim
      → load batik CSVs + run fixpoint + read print_size results

Usage:

    python examples/doop_run.py                     # batik, full run
    python examples/doop_run.py --max-iter 3        # quick sanity
    python examples/doop_run.py --data <dir> --meta <json>   # other dataset

For any benchmark besides doop, see `examples/run_benchmark.py`.
'''

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Defer to the generic runner — this file exists as a named entry point
# users expect for doop specifically, but we don't want the driver logic
# duplicated.
from run_benchmark import main as _run_main

DEFAULT_DATA = "/home/stargazermiao/workspace/SRDatalog/benchmarks/data/doop/batik_interned"
DEFAULT_META = (
  "/home/stargazermiao/workspace/SRDatalog/integration_tests/examples/doop/batik_meta.json"
)


def main() -> int:
  # Inject benchmark name + batik defaults when the caller didn't override.
  argv = ["doop"]
  argv_user = sys.argv[1:]
  if "--data" not in argv_user:
    argv += ["--data", DEFAULT_DATA]
  if "--meta" not in argv_user:
    argv += ["--meta", DEFAULT_META]
  argv += argv_user
  sys.argv = [sys.argv[0]] + argv
  return _run_main()


if __name__ == "__main__":
  raise SystemExit(main())
