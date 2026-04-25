'''Command-line entry for the VS Code extension.

Two subcommands:

    python -m srdatalog.viz dump FILE [--entry NAME] [--meta JSON]
    python -m srdatalog.viz patch FILE RULE [--var-order a,b,c] [--clause-order 1,0,2] [--delta N]

`dump` writes JSON (the visualization bundle + source locations) to
stdout. `patch` rewrites FILE in place (or stdout with `--stdout`).
'''

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from srdatalog.viz.bundle import get_visualization_bundle
from srdatalog.viz.introspect import ProgramDiscoveryError, load_program
from srdatalog.viz.patch import PlanPatchError, patch_rule_plan
from srdatalog.viz.source import find_rule_locations


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(
    prog="python -m srdatalog.viz",
    description="Visualization / plan-edit helper for srdatalog-viz",
  )
  sub = p.add_subparsers(dest="cmd", required=True)

  d = sub.add_parser("dump", help="emit JSON bundle + source locations")
  d.add_argument("file", type=Path)
  d.add_argument("--entry", help="build_*_program function name (auto-discovers if omitted)")
  d.add_argument("--meta", help="dataset_const meta.json path")
  d.add_argument(
    "--project-name", default="VizProject", help="used in C++ type names inside the bundle"
  )

  pp = sub.add_parser("patch", help="rewrite a rule's with_plan kwargs")
  pp.add_argument("file", type=Path)
  pp.add_argument("rule", help="rule name (the string in .named(...))")
  pp.add_argument("--var-order", help='comma list, e.g. "x,y,z"')
  pp.add_argument("--clause-order", help='comma list, e.g. "1,0,2"')
  pp.add_argument("--delta", type=int, default=-1)
  pp.add_argument(
    "--stdout", action="store_true", help="write to stdout instead of editing in place"
  )

  args = p.parse_args(argv)

  if args.cmd == "dump":
    return _cmd_dump(args)
  if args.cmd == "patch":
    return _cmd_patch(args)
  return 2


def _cmd_dump(args) -> int:
  try:
    prog = load_program(args.file, entry=args.entry, meta=args.meta)
  except ProgramDiscoveryError as e:
    print(f"error: {e}", file=sys.stderr)
    return 1
  bundle = get_visualization_bundle(prog, project_name=args.project_name)
  source_text = Path(args.file).read_text()
  bundle["source_locations"] = [
    {
      "name": loc.name,
      "start_line": loc.start_line,
      "end_line": loc.end_line,
      "start": loc.start,
      "end": loc.end,
      "plan_calls": [
        {
          "start": pc.start,
          "end": pc.end,
          "kwargs": [{"kwarg": k.kwarg, "start": k.start, "end": k.end} for k in pc.kwargs],
        }
        for pc in loc.plan_calls
      ],
    }
    for loc in find_rule_locations(source_text)
  ]
  json.dump(bundle, sys.stdout, indent=2, ensure_ascii=False)
  sys.stdout.write("\n")
  return 0


def _cmd_patch(args) -> int:
  var_order = args.var_order.split(",") if args.var_order else None
  clause_order = [int(x) for x in args.clause_order.split(",")] if args.clause_order else None
  source = Path(args.file).read_text()
  try:
    new = patch_rule_plan(
      source,
      args.rule,
      var_order=var_order,
      clause_order=clause_order,
      delta=args.delta,
    )
  except PlanPatchError as e:
    print(f"error: {e}", file=sys.stderr)
    return 1
  if args.stdout:
    sys.stdout.write(new)
  else:
    Path(args.file).write_text(new)
  return 0


if __name__ == "__main__":
  sys.exit(main())
