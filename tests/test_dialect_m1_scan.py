'''M1 byte-equivalence — Scan + InsertInto pipelines via dialect.

For every fixture whose pipeline shape is `[Scan, InsertInto]` (the
M1-supported subset), runs:

  legacy = jit_pipeline(pipeline_ops, [], legacy_ctx)   # inner body
  new    = emit(lower_scan_pipeline(pipeline_ops, ctx), emit_ctx)

and asserts byte-equivalence under `_cpp_norm`.

Scope: only the inner kernel body. The functor envelope, view
declarations, and runner remain produced by the legacy emitter; M1
proves the dialect can produce equivalent inner-body C++ on the
simplest pipelines, without touching `compile_pipeline`'s codepath.

Once M2-M13 cover all op kinds, `compile_pipeline` flips to route
through the dialect, and the M0 byte-equivalence harness becomes the
end-to-end gate.
'''

from __future__ import annotations

from pathlib import Path

import pytest
from integration_helpers import _cpp_norm, _unified_cpp_diff

import srdatalog.mir.types as mir
from srdatalog.codegen.batchfile import _collect_pipelines
from srdatalog.codegen.jit.context import new_code_gen_context
from srdatalog.codegen.jit.emit_helpers import assign_handle_positions
from srdatalog.codegen.jit.pipeline import jit_pipeline
from srdatalog.codegen.jit.view_management import (
  collect_unique_view_specs,
  jit_emit_view_declarations,
)
from srdatalog.dialects.sorted_array.lowerings import (
  LoweringCtx,
  _supported_pipeline,
  lower_scan_pipeline,
)
from srdatalog.dialects.target_cuda.emit import EmitCtx, emit
from srdatalog.hir import compile_to_mir

FIXTURE_ROOT = Path(__file__).resolve().parent / 'fixtures' / 'jit'


def _build_fn_for(stem: str):
  try:
    mod = __import__(f'test_integration_{stem}')
  except ModuleNotFoundError:
    return None
  for name in dir(mod):
    if name.startswith('build_'):
      return getattr(mod, name)
  return None


def _scan_pipeline_params() -> list[tuple[str, str]]:
  '''Find every (stem, rule) where the MIR ExecutePipeline has the
  shape M1 supports.

  We can't fully discover this without building the MIR for every
  fixture. To keep the test cheap, return the full list of fixtures
  and let each test case skip itself when the pipeline doesn't match.
  '''
  out: list[tuple[str, str]] = []
  if not FIXTURE_ROOT.exists():
    return out
  for p in sorted(FIXTURE_ROOT.iterdir()):
    if not p.is_dir():
      continue
    for f in sorted(p.glob('jit_batch.*.cpp')):
      stem = p.name
      rule = f.name[len('jit_batch.'):-len('.cpp')]
      out.append((stem, rule))
  return out


def _legacy_inner_body(pipeline_ops: list[mir.MirNode], ctx) -> str:
  '''Produce the inner-body C++ via the legacy emitter — what
  `jit_pipeline()` would emit inside operator(). This includes view
  declarations + the actual pipeline emission.
  '''
  return jit_pipeline(pipeline_ops, [], ctx)


def _dialect_inner_body(
  pipeline_ops: list[mir.MirNode],
  legacy_ctx,
) -> str:
  '''Lower MIR -> IIR -> C++ via the dialect, prepended with the
  legacy view-declarations prelude (which is shared infrastructure
  M1 doesn't reimplement).'''
  view_specs = collect_unique_view_specs(pipeline_ops)
  # jit_emit_view_declarations seeds legacy_ctx.view_vars and returns
  # the prelude string we prepend to the dialect-emitted body.
  prelude = jit_emit_view_declarations(view_specs, pipeline_ops, [], legacy_ctx)

  lower_ctx = LoweringCtx(
    view_var_names={
      k: v for k, v in legacy_ctx.view_vars.items() if k.isdigit()
    },
    is_counting=legacy_ctx.is_counting,
    inside_cartesian=legacy_ctx.inside_cartesian,
    output_var=legacy_ctx.output_var_name,
    tile_var=legacy_ctx.tile_var,
    debug=legacy_ctx.debug,
    output_var_overrides=dict(legacy_ctx.output_vars),
  )
  iir = lower_scan_pipeline(pipeline_ops, lower_ctx)
  emit_ctx = EmitCtx(indent_level=legacy_ctx.indent, tile_var=legacy_ctx.tile_var)
  return prelude + emit(iir, emit_ctx)


@pytest.mark.parametrize('stem,rule', _scan_pipeline_params())
def test_scan_pipeline_dialect_byte_equivalent(stem: str, rule: str):
  '''M1 gate: dialect-emitted inner body == legacy-emitted inner body
  for every fixture whose pipeline has shape [Scan, InsertInto].

  Fixtures with other shapes are skipped at runtime — M2+ extend the
  dialect to cover them.
  '''
  build_fn = _build_fn_for(stem)
  if build_fn is None:
    pytest.skip(f'no build_* factory in test_integration_{stem}')

  prog = build_fn()
  mir_prog = compile_to_mir(prog)

  for ep in _collect_pipelines(mir_prog):
    if ep.rule_name != rule:
      continue
    pipeline_ops = list(ep.pipeline)
    if not _supported_pipeline(pipeline_ops):
      pytest.skip(
        f'{stem}::{rule} pipeline shape is '
        f'{[type(o).__name__ for o in pipeline_ops]}, not M1-supported'
      )

    # Two independent contexts so legacy and dialect paths each get
    # a fresh name counter starting at 0.
    legacy_ctx = new_code_gen_context()
    legacy_ctx.indent = 4  # operator() body indent
    assign_handle_positions(pipeline_ops)
    legacy_body = _legacy_inner_body(pipeline_ops, legacy_ctx)

    # Build a parallel legacy_ctx for the dialect path so view_vars
    # gets populated and counter starts fresh — both paths must see
    # identical input state.
    dialect_legacy_ctx = new_code_gen_context()
    dialect_legacy_ctx.indent = 4
    assign_handle_positions(pipeline_ops)
    # Run view_management to populate view_vars; jit_pipeline call
    # inside _legacy_inner_body would also do this, but we want a
    # clean ctx for the dialect path.
    view_specs = collect_unique_view_specs(pipeline_ops)
    jit_emit_view_declarations(
      view_specs, pipeline_ops, [], dialect_legacy_ctx
    )
    dialect_body = _dialect_inner_body(pipeline_ops, dialect_legacy_ctx)

    if _cpp_norm(legacy_body) != _cpp_norm(dialect_body):
      diff = _unified_cpp_diff(legacy_body, dialect_body, f'{stem}::{rule}')
      raise AssertionError(
        f'M1 dialect byte-equivalence failed: {stem}::{rule}\n'
        f'legacy jit_pipeline() and dialect emit() diverged.\n\n'
        f'--- LEGACY ---\n{legacy_body}\n'
        f'--- DIALECT ---\n{dialect_body}\n'
        f'--- DIFF ---\n{diff}'
      )
    return

  pytest.skip(f'no pipeline named {rule!r} in {stem} MIR')


def test_lower_rejects_unsupported_pipeline_shape():
  '''The lowering raises ValueError for non-[Scan, InsertInto] shapes.
  This is a structural sanity check — M1 must not silently accept
  unknown shapes and produce garbage.'''
  scan = mir.Scan(
    vars=['x'], rel_name='R', version=mir.Version.FULL, index=[0]
  )
  ctx = LoweringCtx(view_var_names={'-1': 'view_R'})

  with pytest.raises(ValueError, match='supports only'):
    lower_scan_pipeline([scan], ctx)
  with pytest.raises(ValueError, match='supports only'):
    lower_scan_pipeline([scan, scan], ctx)
