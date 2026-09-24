#!/usr/bin/env python3
"""Prepare and inspect the pinned real-application DOOP benchmark suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from doop_suite.catalog import (
  extract_archive,
  fetch_archive,
  load_catalog,
  select_datasets,
  verify_prepared,
  verify_raw,
)


def selection(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
  group = parser.add_mutually_exclusive_group(required=required)
  group.add_argument('--all', action='store_true', help='Select all catalog datasets')
  group.add_argument('--dataset', nargs='+', metavar='NAME', help='Select named datasets')
  group.add_argument('--tier', choices=list(load_catalog()['tiers']))


def prepare_dataset(dataset: dict, root: Path, archive_cache: Path | None = None) -> dict:
  from doop_suite.prepare import prepare

  output = root / 'prepared' / dataset['name']
  if output.exists():
    return verify_prepared(output, dataset)
  archive = fetch_archive(dataset, root, archive_cache)
  raw = root / 'raw' / dataset['name']
  if raw.exists():
    verify_raw(archive, dataset, raw)
  else:
    raw = extract_archive(archive, dataset, root)
  provenance = dict(dataset)
  return prepare(raw, output, provenance=provenance)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest='command', required=True)
  listing = commands.add_parser('list', help='Show tiers, names and canonical reference sizes')
  selection(listing, required=False)
  listing.add_argument('--json', action='store_true')
  for command in ('fetch', 'prepare'):
    action = commands.add_parser(
      command,
      help='Download pinned archives'
      if command == 'fetch'
      else 'Download and normalize complete input relation sets',
    )
    selection(action)
    action.add_argument('--root', type=Path, required=True, help='External data/artifact directory')
    action.add_argument('--archive-cache', type=Path, help='Optional read-only existing zip cache')
  args = parser.parse_args(argv)
  try:
    datasets = select_datasets(args.dataset, args.tier)
    if args.command == 'list':
      if args.json:
        print(json.dumps(datasets, indent=2))
      else:
        print('Local tiers use measured canonical VarPointsTo rows, NOT input or archive size.')
        print(f'{"DATASET":12} {"TIER":8} {"REFERENCE VPT":>15} {"ARCHIVE MB":>12}')
        for item in datasets:
          print(
            f'{item["name"]:12} {item["tier"]:8} {item["reference_vpt_rows"]:>15,}'
            f' {item["archive_bytes"] / 1_000_000:>12.1f}'
          )
      return 0
    root = args.root.resolve()
    for dataset in datasets:
      print(f'[{args.command}] {dataset["name"]}', flush=True)
      if args.command == 'fetch':
        archive = fetch_archive(dataset, root, args.archive_cache)
        print(
          json.dumps({'dataset': dataset['name'], 'archive': str(archive), 'verified': True}),
          flush=True,
        )
      else:
        manifest = prepare_dataset(dataset, root, args.archive_cache)
        print(
          json.dumps(
            {
              'dataset': dataset['name'],
              'status': manifest['status'],
              'prepared_input_rows': sum(r['rows'] for r in manifest['relations'].values()),
              'prepared_input_bytes': sum(r['bytes'] for r in manifest['relations'].values()),
              'directory': str(root / 'prepared' / dataset['name']),
            }
          ),
          flush=True,
        )
    return 0
  except (OSError, ValueError, RuntimeError) as error:
    print(f'[error] {error}', file=sys.stderr)
    return 1


if __name__ == '__main__':
  raise SystemExit(main())
