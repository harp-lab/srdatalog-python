"""Sequential, failure-accounted execution of explicitly selected DOOP datasets."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from .catalog import digest, load_catalog, verify_prepared
from .compare import compare_results

REPO = Path(__file__).resolve().parents[2]


def logical_program(metadata: Path):
  sys.path.insert(0, str(REPO / 'src'))
  source = REPO / 'examples' / 'doop.py'
  spec = importlib.util.spec_from_file_location('doop_suite_model', source)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.build_doopdb_program(json.loads(metadata.read_text()))


def run_matrix(
  datasets: list[dict],
  root: Path,
  output: Path,
  *,
  backend: str,
  plan: str = 'baseline',
  threads: int = 12,
  jobs: int = 2,
  timeout: int = 900,
  warmups: int = 1,
  repeats: int = 3,
  reference: Path | None = None,
) -> dict:
  if backend not in ('cpu', 'gpu'):
    raise ValueError(f'Unknown backend: {backend}')
  if plan not in ('baseline', 'bitmap') or (backend == 'cpu' and plan != 'baseline'):
    raise ValueError('CPU executes logical baseline; bitmap is an opt-in GPU plan')
  if min(threads, jobs, timeout, repeats) < 1 or warmups < 0:
    raise ValueError('Positive threads/jobs/timeout/repeats and nonnegative warmups required')
  root, output = root.resolve(), output.resolve()
  references = {}
  if reference is not None:
    document = json.loads(reference.read_text())
    references = {item['dataset']: item for item in document['results']}
    if any(item['name'] not in references for item in datasets):
      raise ValueError('Reference does not cover every selected dataset')
  output.mkdir(parents=True, exist_ok=False)
  report = {
    'schema_version': 1,
    'suite': load_catalog()['suite'],
    'status': 'running',
    'backend': backend,
    'plan': plan,
    'selected_datasets': [d['name'] for d in datasets],
    'tier_metric': load_catalog()['tier_metric'],
    'results': [],
  }

  def save():
    (output / 'suite.json').write_text(json.dumps(report, indent=2) + '\n')

  save()
  for dataset in datasets:
    name = dataset['name']
    facts = root / 'prepared' / name
    print(f'[run] {name} backend={backend} plan={plan}', flush=True)
    try:
      manifest = verify_prepared(facts, dataset)
      if not manifest['entrypoints']['selected_main_methods']:
        raise ValueError('Refusing an empty-root DOOP benchmark: no main method was selected')
      program = logical_program(facts / 'meta.json')
      expected = {relation.name for relation in program.relations}
      inputs = {relation.name for relation in program.relations if relation.input_file}
      from .prepare import SCHEMA

      if set(manifest['relations']) != set(SCHEMA) or not inputs <= SCHEMA.keys():
        raise ValueError('Prepared relation contract differs from the declared inputs')
      if backend == 'cpu':
        from .cpu import run_cpu

        result = run_cpu(
          facts, output / name, threads=threads, timeout=timeout, warmups=warmups, repeats=repeats
        )
      else:
        from .gpu import run_gpu

        result = run_gpu(
          facts,
          output / name,
          plan=plan,
          jobs=jobs,
          timeout=timeout,
          warmups=warmups,
          repeats=repeats,
        )
      if result['status'] != 'passed':
        raise RuntimeError(f'Backend did not complete: {result["status"]}')
      if set(result['relation_counts']) != expected or set(result['outputs']) != expected - inputs:
        raise ValueError('Backend did not report/export every query relation')
      if result['source_sha256'] != digest(REPO / 'examples' / 'doop.py') or result[
        'metadata_sha256'
      ] != digest(facts / 'meta.json'):
        raise ValueError('Backend used a different query or dataset metadata')
      result.update(
        tier=dataset['tier'],
        reference_vpt_rows=dataset['reference_vpt_rows'],
        expected_relations=sorted(expected),
        input_relations=sorted(inputs),
        input_manifest_sha256=digest(facts / 'manifest.json'),
        input_rows=sum(manifest['relations'][name]['rows'] for name in inputs),
        input_bytes=sum(manifest['relations'][name]['bytes'] for name in inputs),
        prepared_input_rows=sum(r['rows'] for r in manifest['relations'].values()),
        prepared_input_bytes=sum(r['bytes'] for r in manifest['relations'].values()),
        correctness='not_compared',
      )
      if reference is not None:
        try:
          comparison = compare_results(references[name], result, output / name / 'comparison.json')
        except Exception as error:
          result['status'] = 'failed'
          result['error'] = f'Reference comparison failed: {error!r}'
        else:
          result['correctness'] = 'exact_match' if comparison['passed'] else 'mismatch'
          if not comparison['passed']:
            result['status'] = 'failed'
      report['results'].append(result)
      print(
        json.dumps(
          {
            'dataset': name,
            'status': result['status'],
            'correctness': result['correctness'],
            'timings_seconds': result['timings_seconds'],
          }
        ),
        flush=True,
      )
    except Exception as error:
      report['results'].append(
        {
          'dataset': name,
          'tier': dataset['tier'],
          'backend': backend,
          'plan': plan,
          'status': 'failed',
          'error': repr(error),
        }
      )
      print(f'[failed] {name}: {error}', file=sys.stderr, flush=True)
    save()
  report['status'] = (
    'passed' if all(r['status'] == 'passed' for r in report['results']) else 'failed'
  )
  save()
  return report
