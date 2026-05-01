'''Byte-equivalence gate for Stage 2.

Walks every fixture in tests/fixtures/jit/, compiles each program's
ExecutePipelines via both:

  - `gen_jit_file_content_from_execute_pipeline(ep)` — legacy emitter
  - `compile_pipeline(ep)`                          — new entry point

and asserts the C++ output is byte-equivalent under `_cpp_norm`.

For M0 (test harness setup), both paths are the same call (compile
delegates to the legacy emitter), so this passes trivially. Its
purpose is to:

  1. Establish the regression baseline. Every fixture currently
     usable by the legacy path is enumerated here.
  2. Catch any future divergence between the legacy emitter and the
     dialect-based path the moment a milestone introduces it.
  3. Provide a parameterized harness that subsequent milestones (M1+)
     don't have to re-engineer — they just register their MIR ops to
     the dialect, and any fixture that exercises that path stays green
     or fails loudly here.

Skips: fixtures already known to diverge from upstream Nim goldens
have entries in BATCH_BYTE_MATCH_SKIPS / ORCH_BYTE_MATCH_SKIPS in
test_jit_codegen_all_fixtures.py. We don't reuse those — this test
checks legacy-vs-new equivalence, which is a different axis: a
fixture that diverges from Nim is still expected to be self-consistent
between legacy and new emitters.
'''

from __future__ import annotations

import re
from pathlib import Path

import pytest
from integration_helpers import _cpp_norm, _unified_cpp_diff

from srdatalog.codegen.batchfile import _collect_pipelines
from srdatalog.codegen.jit.file import gen_jit_file_content_from_execute_pipeline
from srdatalog.compile import compile_pipeline
from srdatalog.hir import compile_to_mir

FIXTURE_ROOT = Path(__file__).resolve().parent / 'fixtures' / 'jit'


def _build_fn_for(stem: str):
  '''Look up the build_<thing>() factory in test_integration_<stem>.py.

  Mirrors the discovery used by test_jit_codegen_all_fixtures.py: each
  fixture stem has a sibling test module exposing one or more
  `build_*` factories that produce the Program.
  '''
  try:
    mod = __import__(f'test_integration_{stem}')
  except ModuleNotFoundError:
    return None
  for name in dir(mod):
    if name.startswith('build_'):
      return getattr(mod, name)
  return None


def _fixture_params() -> list[tuple[str, str]]:
  '''Discover every (stem, rule) pair where the legacy emitter has a
  golden. The set defines the byte-equivalence baseline: each pair
  is a fixture run through both paths.'''
  out: list[tuple[str, str]] = []
  if not FIXTURE_ROOT.exists():
    return out
  for p in sorted(FIXTURE_ROOT.iterdir()):
    if not p.is_dir():
      continue
    for f in sorted(p.glob('jit_batch.*.cpp')):
      m = re.match(r'jit_batch\.(.+)\.cpp$', f.name)
      if m:
        out.append((p.name, m.group(1)))
  return out


@pytest.mark.parametrize('stem,rule', _fixture_params())
def test_legacy_vs_compile_pipeline_byte_equivalent(stem: str, rule: str):
  '''Stage 2 byte-equivalence gate.

  For each (fixture, rule), both the legacy emitter and the new
  `compile_pipeline` entry point must produce identical C++ source
  after `_cpp_norm` normalization.

  M0: trivially passes — `compile_pipeline` delegates to the legacy
  emitter. As M1+ rewire pieces to the dialect-based path, this test
  catches divergence per-fixture, per-rule.
  '''
  build_fn = _build_fn_for(stem)
  if build_fn is None:
    pytest.skip(f'no build_* factory in test_integration_{stem}')

  prog = build_fn()
  mir = compile_to_mir(prog)

  for ep in _collect_pipelines(mir):
    if ep.rule_name != rule:
      continue
    legacy = gen_jit_file_content_from_execute_pipeline(ep)
    new = compile_pipeline(ep)
    if _cpp_norm(legacy) != _cpp_norm(new):
      diff = _unified_cpp_diff(legacy, new, f'{stem}::{rule}')
      raise AssertionError(
        f'Stage 2 byte-equivalence failed: {stem}::{rule}\n'
        f'legacy emitter and compile_pipeline diverged.\n'
        f'Triage: localize divergence to a feature-flag axis (see '
        f'docs/stage2_emitter_audit.md §3) and the corresponding\n'
        f'lowering rule (§10 of ir_lowering_semantics.md).\n\n'
        + diff
      )
    return

  pytest.skip(f'no pipeline named {rule!r} in {stem} MIR')


def test_compile_pipeline_rejects_unknown_target():
  '''The entry point validates its target argument; unknown targets
  fail loudly rather than silently emitting CUDA.'''
  build_fn = _build_fn_for('triangle')
  if build_fn is None:
    pytest.skip('no triangle build_*')
  mir = compile_to_mir(build_fn())
  eps = _collect_pipelines(mir)
  if not eps:
    pytest.skip('no pipelines in triangle MIR')
  with pytest.raises(ValueError, match='unsupported target'):
    compile_pipeline(eps[0], target='nonexistent')  # type: ignore[arg-type]


def test_fixture_set_is_nonempty():
  '''Sanity check: the fixture discovery actually finds something.
  If this fails the byte-equivalence gate is silently doing nothing.'''
  params = _fixture_params()
  assert len(params) > 0, (
    'No fixtures discovered under tests/fixtures/jit/. '
    'The byte-equivalence gate would silently pass — investigate.'
  )
