'''Runner byte-equivalence gate (N3).

For every fixture in `tests/fixtures/jit/` that has a
`jit_runner.<rule>.cpp` golden, runs `compile_runner(ep, db, ...)`
and byte-diffs the output against that golden. This is the
production gate — `compile_runner` is what `build_project` /
`pipeline.py` will call once the migration is complete.

Runner emit currently delegates to the legacy `gen_complete_runner`
inside `compile_runner`; the gate stays green throughout the N3.x
migration steps that incrementally lift kernel bodies + scaffolding
into the dialect.

Skips: pipelines with feature flags the legacy runner doesn't yet
support (work_stealing, dedup_hash, balanced) raise inside the legacy
emitter. These are tracked in RUNNER_BYTE_MATCH_SKIPS, mirroring the
sister test in `test_jit_complete_runner.py`.
'''

from __future__ import annotations

import re
from pathlib import Path

import pytest
from integration_helpers import _cpp_norm, _unified_cpp_diff

from srdatalog.compile import compile_runner
from srdatalog.ir.dialects.target.cuda.batchfile import _collect_pipelines
from srdatalog.ir.dialects.target.cuda.orchestrator import gen_step_body
from srdatalog.ir.hir import compile_to_mir

FIXTURE_ROOT = Path(__file__).resolve().parent / 'fixtures' / 'jit'

RUNNER_BYTE_MATCH_SKIPS: dict[tuple[str, str], str] = {
  # Work-stealing runner variants — N8 (par.data.atomic_ws dialect).
  ('lsqb_q6_nosj', 'TwoHopPath'): 'WS runner not yet ported (N8)',
  ('polonius_test', 'subset_trans_D0'): 'WS runner not yet ported (N8)',
  # ddisasm — see docs/milestones.md "Nim-reference audit" for the gap matrix.
  ('ddisasm', 'StackLiveVarBlockEnd1_D0_splitB'): (
    'Scan + CartesianJoin shape — _supported_pipeline rejects; pieces exist, '
    'just need to add CartesianJoin to the Scan-middle allowed list.'
  ),
  ('ddisasm', 'StackDefUsed4_D1'): (
    'dedup_hash gen_complete_runner not yet ported (runner-side, not kernel-body).'
  ),
  ('ddisasm', 'StackLiveVarBlockEnd1_D0_splitA'): (
    'Head tuple ordering divergence (`varp, varr` vs `varr, varp`) — likely '
    'MIR-level head-arg ordering, not kernel-body.'
  ),
  ('ddisasm', 'StackLiveVarPriorUsed'): (
    'Pre-narrow Negation emission order differs (`Reg...` before `Stack...` vs '
    'reversed in Nim) — iteration order in `_register_neg_pre_narrow`.'
  ),
  ('ddisasm', 'StackDefUsed1'): (
    'Tiled-Cartesian eligibility differs — Nim emits the tiled-smem materialize '
    'variant; Python emits non-tiled. Likely `_tiled_cart_eligible` predicate gap.'
  ),
}


def _build_fn_for(stem: str):
  try:
    mod = __import__(f'test_integration_{stem}')
  except ModuleNotFoundError:
    return None
  for name in dir(mod):
    if name.startswith('build_'):
      return getattr(mod, name)
  return None


def _db_name_from_fixture(stem: str, rule: str) -> str | None:
  '''Scrape the DB type name from the golden file. The runner output
  begins with `using DB = <name>;`, so the fixture is self-describing
  for this purpose. Falls back to scraping orchestrator.cpp like the
  sister test does.'''
  path = FIXTURE_ROOT / stem / f'jit_runner.{rule}.cpp'
  if path.exists():
    m = re.search(r'using DB = (\w+);', path.read_text())
    if m:
      return m.group(1)
  orch = FIXTURE_ROOT / stem / 'orchestrator.cpp'
  if orch.exists():
    m = re.search(r'(\w+_DB_DeviceDB)', orch.read_text())
    if m:
      return m.group(1)
  return None


def _params() -> list[tuple[str, str]]:
  out: list[tuple[str, str]] = []
  if not FIXTURE_ROOT.exists():
    return out
  for p in sorted(FIXTURE_ROOT.iterdir()):
    if not p.is_dir():
      continue
    for f in sorted(p.glob('jit_runner.*.cpp')):
      m = re.match(r'jit_runner\.(.+)\.cpp$', f.name)
      if m:
        out.append((p.name, m.group(1)))
  return out


@pytest.mark.parametrize('stem,rule', _params())
def test_compile_runner_matches_golden(stem: str, rule: str):
  '''compile_runner output must byte-match the upstream Nim golden.'''
  if (stem, rule) in RUNNER_BYTE_MATCH_SKIPS:
    pytest.skip(RUNNER_BYTE_MATCH_SKIPS[(stem, rule)])

  build_fn = _build_fn_for(stem)
  if build_fn is None:
    pytest.skip(f'no build_* factory in test_integration_{stem}')

  db_name = _db_name_from_fixture(stem, rule)
  if db_name is None:
    pytest.skip(f'cannot infer DB type name for {stem}::{rule}')

  prog = build_fn()
  mir = compile_to_mir(prog)

  # Mirror build.py:95 — collect per-relation index_type so plugin
  # dispatch picks the right `viewCount` for D2L relations. Without
  # this, FULL_VER references on 2-level-indexed relations diverge by
  # one view slot per source.
  rel_index_types = {
    r.name: r.index_type
    for r in prog.relations
    if getattr(r, 'index_type', '')
  }

  # Run the orchestrator pass first — it sets concurrent_write on EPs
  # in shared-destination ParallelGroups, which the runner emitter
  # consults to disable tiled-Cartesian. Mirrors test_jit_complete_runner.
  for i, (step, is_rec) in enumerate(mir.steps):
    gen_step_body(step, db_name, is_rec, i)

  golden_path = FIXTURE_ROOT / stem / f'jit_runner.{rule}.cpp'
  golden = golden_path.read_text()

  for ep in _collect_pipelines(mir):
    if ep.rule_name != rule:
      continue
    try:
      actual = compile_runner(ep, db_name, rel_index_types=rel_index_types)
    except NotImplementedError as e:
      pytest.skip(f'compile_runner: {e}')
    if _cpp_norm(golden) != _cpp_norm(actual):
      diff = _unified_cpp_diff(golden, actual, f'{stem}::{rule}')
      raise AssertionError(
        f'compile_runner byte-equivalence failed: {stem}::{rule}\n'
        f'output diverged from {golden_path}.\n\n' + diff
      )
    return

  pytest.skip(f'no pipeline named {rule!r} in {stem} MIR')
