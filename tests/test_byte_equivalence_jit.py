'''Byte-equivalence gate for the dialect-based JIT codegen.

For every fixture in `tests/fixtures/jit/`, runs each rule's
`ExecutePipeline` through `compile_pipeline()` and byte-diffs the
output against the checked-in `jit_batch.<rule>.cpp` golden under
`_cpp_norm` normalization.

History: an earlier incarnation of this test compared the dialect
output against `gen_jit_file_content_from_execute_pipeline()` (the
legacy emitter) — useful while Stage 2 milestones were progressively
porting paths into the dialect, but redundant once `compile_pipeline`
routes 100% of supported shapes through the dialect AND the legacy
emitter is on its way out. The gate now anchors directly on the
upstream Nim goldens so it survives deletion of the legacy
inner-body emitters.

Skips: fixtures whose Nim goldens are known to diverge from the
Python emitters (none today; entries take the form
`{('stem', 'rule'): 'reason'}`).
'''

from __future__ import annotations

import re
from pathlib import Path

import pytest
from integration_helpers import _cpp_norm, _unified_cpp_diff

from srdatalog.compile import compile_pipeline
from srdatalog.dialects.target.cuda.batchfile import _collect_pipelines
from srdatalog.hir import compile_to_mir

FIXTURE_ROOT = Path(__file__).resolve().parent / 'fixtures' / 'jit'

# (stem, rule) → reason. Empty today; entries here document a known
# upstream divergence and are excluded from the gate.
GOLDEN_BYTE_MATCH_SKIPS: dict[tuple[str, str], str] = {}


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
  '''Discover every (stem, rule) pair where a `jit_batch.<rule>.cpp`
  golden exists. The set defines the byte-equivalence baseline.'''
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
def test_compile_pipeline_byte_matches_golden(stem: str, rule: str):
  '''Dialect output must byte-match the upstream Nim golden for every
  (fixture, rule) pair, modulo `_cpp_norm` normalization.
  '''
  if (stem, rule) in GOLDEN_BYTE_MATCH_SKIPS:
    pytest.skip(GOLDEN_BYTE_MATCH_SKIPS[(stem, rule)])

  build_fn = _build_fn_for(stem)
  if build_fn is None:
    pytest.skip(f'no build_* factory in test_integration_{stem}')

  prog = build_fn()
  mir = compile_to_mir(prog)

  golden_path = FIXTURE_ROOT / stem / f'jit_batch.{rule}.cpp'
  golden = golden_path.read_text()

  for ep in _collect_pipelines(mir):
    if ep.rule_name != rule:
      continue
    actual = compile_pipeline(ep)
    if _cpp_norm(golden) != _cpp_norm(actual):
      diff = _unified_cpp_diff(golden, actual, f'{stem}::{rule}')
      raise AssertionError(
        f'compile_pipeline byte-equivalence failed: {stem}::{rule}\n'
        f'dialect output diverged from {golden_path}.\n'
        f'Triage: localize the divergence to a feature-flag axis (see '
        f'docs/stage2_emitter_audit.md §3) and the corresponding '
        f'lowering rule (§10 of ir_lowering_semantics.md).\n\n'
        + diff
      )
    return

  pytest.skip(f'no pipeline named {rule!r} in {stem} MIR')


def test_compile_pipeline_rejects_unknown_target():
  '''compile_pipeline raises on unsupported targets.'''
  build_fn = _build_fn_for('debug_triangle')
  assert build_fn is not None, 'debug_triangle fixture is required for this test'
  prog = build_fn()
  mir = compile_to_mir(prog)
  ep = next(iter(_collect_pipelines(mir)))
  with pytest.raises(ValueError, match='unsupported target'):
    compile_pipeline(ep, target='cpp_tbb')  # type: ignore[arg-type]


def test_fixture_set_is_nonempty():
  '''Sanity check: there's at least one fixture to compare against.'''
  params = _fixture_params()
  assert len(params) > 0, (
    f'no fixtures found under {FIXTURE_ROOT}; '
    f'the byte-equivalence gate has nothing to validate'
  )
