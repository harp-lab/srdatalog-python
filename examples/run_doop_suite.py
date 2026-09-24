#!/usr/bin/env python3
"""Run selected prepared DOOP datasets to complete fixedpoints and export exact results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from doop_suite.catalog import load_catalog, select_datasets
from doop_suite.suite import run_matrix


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  selection = parser.add_mutually_exclusive_group(required=True)
  selection.add_argument('--all', action='store_true')
  selection.add_argument('--dataset', nargs='+', metavar='NAME')
  selection.add_argument('--tier', choices=list(load_catalog()['tiers']))
  parser.add_argument(
    '--root', type=Path, required=True, help='Data root used by doop_benchmark.py prepare'
  )
  parser.add_argument('--output', type=Path, required=True, help='New external result directory')
  parser.add_argument('--backend', choices=['cpu', 'gpu'], required=True)
  parser.add_argument('--plan', choices=['baseline', 'bitmap'], default='baseline')
  parser.add_argument('--threads', type=int, default=12, help='CPU execution threads')
  parser.add_argument('--jobs', type=int, default=2, help='GPU compile jobs')
  parser.add_argument(
    '--timeout', type=int, default=900, help='Per build/execution process timeout in seconds'
  )
  parser.add_argument('--warmups', type=int, default=1)
  parser.add_argument('--repeats', type=int, default=3)
  parser.add_argument(
    '--reference',
    type=Path,
    help='Reference suite.json; require every selected relation set to match',
  )
  args = parser.parse_args(argv)
  try:
    report = run_matrix(
      select_datasets(args.dataset, args.tier),
      args.root,
      args.output,
      backend=args.backend,
      plan=args.plan,
      threads=args.threads,
      jobs=args.jobs,
      timeout=args.timeout,
      warmups=args.warmups,
      repeats=args.repeats,
      reference=args.reference,
    )
  except (OSError, ValueError, RuntimeError) as error:
    print(f'[error] {error}', file=sys.stderr)
    return 1
  print(
    json.dumps({'status': report['status'], 'report': str(args.output.resolve() / 'suite.json')})
  )
  return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
  raise SystemExit(main())
