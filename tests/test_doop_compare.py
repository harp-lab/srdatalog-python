"""Equal cardinalities do not establish relation equality; row order is immaterial."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from doop_suite.compare import compare_results


def report(path):
  return {
    'status': 'passed',
    'dataset': 'sample',
    'source_sha256': 'same-source',
    'metadata_sha256': 'same-meta',
    'input_manifest_sha256': 'same-inputs',
    'expected_relations': ['Input', 'Output'],
    'input_relations': ['Input'],
    'relation_counts': {'Input': 1, 'Output': 2},
    'outputs': {'Output': str(path)},
  }


def test_comparison_checks_tuples_not_cardinality_or_export_order(tmp_path):
  left, right = tmp_path / 'left.tsv', tmp_path / 'right.tsv'
  left.write_text('1\t2\n3\t4\n')
  right.write_text('3\t4\n1\t2\n')
  assert compare_results(report(left), report(right), tmp_path / 'equal.json')['passed']
  right.write_text('3\t4\n1\t9\n')
  assert not compare_results(report(left), report(right), tmp_path / 'different.json')['passed']


def test_comparison_rejects_missing_exports_and_different_inputs(tmp_path):
  path = tmp_path / 'out.tsv'
  path.write_text('1\t2\n3\t4\n')
  incomplete = report(path)
  incomplete['outputs'] = {}
  with pytest.raises(ValueError, match='every relation'):
    compare_results(report(path), incomplete, tmp_path / 'missing.json')
  changed = report(path)
  changed['input_manifest_sha256'] = 'different-input'
  with pytest.raises(ValueError, match='input_manifest_sha256'):
    compare_results(report(path), changed, tmp_path / 'changed.json')
