'''Count-phase byte-equivalence gate (N1).

For every fixture in `tests/fixtures/jit/`, renders the count-phase
kernel body via:
  - legacy `jit_pipeline(pipeline, source_specs, count_ctx)`
  - dialect `compile_kernel_body(ep, is_counting=True, ...)`

and asserts the strings match under `_cpp_norm`. The legacy emit IS
the spec for count-phase shape — there are no isolated count-phase
goldens (count bodies live inside `jit_runner.<rule>.cpp`), so the
legacy emitter is the reference until N3 lands the runner gate.

Mirror of the `_make_kernel_ctx(..., is_counting=True, output_var_name='output_ctx')`
setup that `complete_runner._gen_kernel_count` uses, plus the
`__skip_counting__` sentinel for multi-head rules' secondary outputs.

Skips: as with the materialize gate, fixtures with known upstream
divergence go in COUNT_PHASE_SKIPS with a reason.
'''

from __future__ import annotations

import re
from pathlib import Path

import pytest
from integration_helpers import _cpp_norm, _unified_cpp_diff

from srdatalog.codegen.batchfile import _collect_pipelines
from srdatalog.codegen.jit.context import new_code_gen_context
from srdatalog.codegen.jit.emit_helpers import assign_handle_positions
from srdatalog.codegen.jit.pipeline import jit_pipeline
from srdatalog.codegen.jit.view_management import compute_view_slot_offsets
from srdatalog.compile import compile_kernel_body
from srdatalog.hir import compile_to_mir

FIXTURE_ROOT = Path(__file__).resolve().parent / 'fixtures' / 'jit'

COUNT_PHASE_SKIPS: dict[tuple[str, str], str] = {}


def _build_fn_for(stem: str):
  try:
    mod = __import__(f'test_integration_{stem}')
  except ModuleNotFoundError:
    return None
  for name in dir(mod):
    if name.startswith('build_'):
      return getattr(mod, name)
  return None


def _fixture_params() -> list[tuple[str, str]]:
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


def _legacy_count_body(ep) -> str:
  '''Reproduce the count-phase body emit from
  `complete_runner._gen_kernel_count` — fresh ctx with is_counting=True,
  output_var_name='output_ctx', view_slot_offsets populated via
  compute_view_slot_offsets (matching `_make_kernel_ctx`), and
  secondary outputs marked `__skip_counting__`. Returns the body string
  (view_decls + kernel body) without the kernel signature/envelope.'''
  pipeline = list(ep.pipeline)
  assign_handle_positions(pipeline)
  ctx = new_code_gen_context()
  ctx.indent = 4
  ctx.is_counting = True
  ctx.is_jit_mode = True
  ctx.output_var_name = 'output_ctx'
  # Mirror _make_kernel_ctx's slot offset population — without it, legacy
  # falls back to handle_idx for multi-occurrence views, diverging from
  # the dialect (which always uses positional first-occurrence slots).
  ctx.view_slot_offsets = compute_view_slot_offsets(ep.source_specs, {})
  if ep.dest_specs:
    ctx.output_vars[ep.dest_specs[0].rel_name] = 'output_ctx'
    for dest in ep.dest_specs[1:]:
      ctx.output_vars[dest.rel_name] = '__skip_counting__'
  return jit_pipeline(pipeline, ep.source_specs, ctx)


def _dialect_count_body(ep) -> str:
  '''Same setup, but routed through the dialect.'''
  output_vars = {}
  if ep.dest_specs:
    output_vars[ep.dest_specs[0].rel_name] = 'output_ctx'
    for dest in ep.dest_specs[1:]:
      output_vars[dest.rel_name] = '__skip_counting__'
  return compile_kernel_body(
    ep,
    is_counting=True,
    output_var_name='output_ctx',
    output_vars=output_vars,
  )


@pytest.mark.parametrize('stem,rule', _fixture_params())
def test_count_phase_dialect_matches_legacy(stem: str, rule: str):
  '''Dialect count-phase body must byte-match the legacy `jit_pipeline`
  count emit for every fixture, modulo `_cpp_norm` normalization.
  '''
  if (stem, rule) in COUNT_PHASE_SKIPS:
    pytest.skip(COUNT_PHASE_SKIPS[(stem, rule)])

  build_fn = _build_fn_for(stem)
  if build_fn is None:
    pytest.skip(f'no build_* factory in test_integration_{stem}')

  prog = build_fn()
  mir = compile_to_mir(prog)

  for ep in _collect_pipelines(mir):
    if ep.rule_name != rule:
      continue
    legacy = _legacy_count_body(ep)
    actual = _dialect_count_body(ep)
    if _cpp_norm(legacy) != _cpp_norm(actual):
      diff = _unified_cpp_diff(legacy, actual, f'{stem}::{rule}')
      raise AssertionError(
        f'count-phase byte-equivalence failed: {stem}::{rule}\n'
        f'dialect count emit diverged from legacy jit_pipeline.\n\n'
        + diff
      )
    return

  pytest.skip(f'no pipeline named {rule!r} in {stem} MIR')
