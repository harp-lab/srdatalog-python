"""Compare complete logical tuple sets, independent of engine export row order."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


def compare_results(left: dict, right: dict, output: Path) -> dict:
  if left['status'] != 'passed' or right['status'] != 'passed':
    raise ValueError('Only successful complete fixedpoints can be compared')
  for key in ('dataset', 'source_sha256', 'metadata_sha256', 'input_manifest_sha256'):
    if not left.get(key) or left[key] != right.get(key):
      raise ValueError(f'Cannot compare different {key}')
  expected = set(left['expected_relations'])
  inputs = set(left['input_relations'])
  outputs = expected - inputs
  if (
    not outputs
    or expected != set(right['expected_relations'])
    or inputs != set(right['input_relations'])
  ):
    raise ValueError('Reference and candidate have different relation schemas')
  for result in (left, right):
    if set(result['relation_counts']) != expected or set(result['outputs']) != outputs:
      raise ValueError('Result does not cover every relation in the query')
  output.parent.mkdir(parents=True, exist_ok=True)
  if output.exists():
    raise FileExistsError(f'Refusing to overwrite comparison: {output}')
  checks = []
  with tempfile.TemporaryDirectory(prefix='.compare-', dir=output.parent) as temporary:
    scratch = Path(temporary)
    for name in sorted(expected):
      check = {
        'relation': name,
        'left_rows': left['relation_counts'][name],
        'right_rows': right['relation_counts'][name],
      }
      check['passed'] = check['left_rows'] == check['right_rows']
      if name in outputs:
        for label, result in (('left', left), ('right', right)):
          subprocess.run(
            [
              'sort',
              '-u',
              '-T',
              str(scratch),
              '-o',
              str(scratch / label),
              str(Path(result['outputs'][name]).resolve(strict=True)),
            ],
            env=dict(os.environ, LC_ALL='C'),
            check=True,
          )
        equal = True
        rows = [0, 0]
        with (scratch / 'left').open('rb') as lhs, (scratch / 'right').open('rb') as rhs:
          while True:
            a, b = lhs.read(8 * 1024 * 1024), rhs.read(8 * 1024 * 1024)
            if not a and not b:
              break
            equal = equal and a == b
            rows[0] += a.count(b'\n')
            rows[1] += b.count(b'\n')
        check['export_rows'] = rows
        check['equal_tuple_sets'] = equal
        check['passed'] = (
          check['passed'] and equal and rows == [check['left_rows'], check['right_rows']]
        )
      checks.append(check)
  result = {
    'passed': all(row['passed'] for row in checks),
    'dataset': left['dataset'],
    'method': 'Complete lexicographically sorted integer TSV tuple sets; no sampling',
    'input_manifest_sha256': left['input_manifest_sha256'],
    'relations': checks,
  }
  with output.open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
  return result
