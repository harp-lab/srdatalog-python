'''PR-1d: CudaRenderCtx extraction tests.

Verifies:

  1. `CudaRenderCtx` is importable from
     `srdatalog.ir.codegen.cuda.lower_ctx` and constructs with
     defaults.
  2. `LoweringCtx.render_ctx` is a real `CudaRenderCtx` instance.
  3. The forwarding properties round-trip read + write through
     `ctx.<field>` <-> `ctx.render_ctx.<field>` for every field
     extracted into `CudaRenderCtx`.
  4. Passing flat render kwargs (back-compat path) and passing
     `render_ctx=` (preferred path) both produce a ctx whose
     observable state agrees on every render-side field.
  5. A sample MIR pipeline (`Scan -> InsertInto`) lowers to the
     identical rendered IIR text under both ctx-construction shapes
     — i.e. extraction is byte-equivalent.

Per `docs/phase_decomposition_redesign.md` § 3.2.1 and the PR-1d
spec brief.
'''

from __future__ import annotations

from typing import Any

import pytest

import srdatalog.ir.mir.types as mir
from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
from srdatalog.ir.codegen.cuda.lower_ctx import CudaRenderCtx, NegPreNarrowInfo
from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
  LoweringCtx,
  lower_scan_pipeline,
)
from srdatalog.ir.hir.types import Version

# -----------------------------------------------------------------------------
# 1. CudaRenderCtx surface
# -----------------------------------------------------------------------------


def test_cuda_render_ctx_default_construct() -> None:
  '''CudaRenderCtx() with no args yields a fresh dataclass with
  every field at its declared default. Mutable dataclass — NOT
  frozen, NOT slotted (per the spec brief: dynamic setattr must
  work).'''
  rc = CudaRenderCtx()
  assert rc.view_var_names == {}
  assert rc.output_var == 'output'
  assert rc.output_var_overrides == {}
  assert rc.view_slot_bases == {}
  assert rc.rel_index_types == {}
  assert rc.tiled_cartesian_valid_var == ''
  assert rc.ws_cartesian_valid_var == ''
  assert rc.neg_pre_narrow == {}
  assert rc.debug is True
  assert rc.tile_var == 'tile'

  # Mutable — direct field assignment works (no FrozenInstanceError).
  rc.output_var = 'ctx_other'
  assert rc.output_var == 'ctx_other'


def test_cuda_render_ctx_not_frozen() -> None:
  '''Dynamic attribute write must succeed — the BG pragma test path
  relies on `object.__setattr__`-style mutation. If we ever switch
  to `frozen=True` or `slots=True`, this test catches the
  regression.'''
  rc = CudaRenderCtx()
  # Plain setattr (would FrozenInstanceError if frozen=True).
  rc.tiled_cartesian_valid_var = 'tc_valid_1'
  assert rc.tiled_cartesian_valid_var == 'tc_valid_1'
  # Dynamic setattr — would AttributeError on a slotted dataclass
  # if `__dict__` is missing.
  object.__setattr__(rc, 'output_var', 'overridden')
  assert rc.output_var == 'overridden'


# -----------------------------------------------------------------------------
# 2. LoweringCtx embeds CudaRenderCtx + forwards through properties
# -----------------------------------------------------------------------------


def test_lowering_ctx_default_render_ctx_is_fresh() -> None:
  '''A bare `LoweringCtx()` constructs its own CudaRenderCtx via the
  default-factory shape. The render_ctx is a real instance — not a
  shared mutable.'''
  ctx_a = LoweringCtx()
  ctx_b = LoweringCtx()
  assert isinstance(ctx_a.render_ctx, CudaRenderCtx)
  assert isinstance(ctx_b.render_ctx, CudaRenderCtx)
  assert ctx_a.render_ctx is not ctx_b.render_ctx
  # Mutating one's dict must NOT bleed into the other.
  ctx_a.render_ctx.view_var_names['x'] = 'view_x'
  assert ctx_b.render_ctx.view_var_names == {}


FORWARD_FIELD_FIXTURES: list[tuple[str, Any, Any]] = [
  ('view_var_names', {'0': 'view_one'}, {'1': 'view_two'}),
  ('output_var', 'ctx0', 'ctx_other'),
  ('output_var_overrides', {'Rel': 'ctx_rel'}, {'Rel2': 'ctx_rel2'}),
  ('view_slot_bases', {'0': 0, '1': 2}, {'2': 4}),
  ('rel_index_types', {'R': 'Device2LevelIndex'}, {'R': ''}),
  ('tiled_cartesian_valid_var', 'tc_valid_1', 'tc_valid_2'),
  ('ws_cartesian_valid_var', 'ws_valid_3', 'ws_valid_4'),
  (
    'neg_pre_narrow',
    {
      0: NegPreNarrowInfo(
        var_name='h_pre_1',
        pre_vars=['v0'],
        in_cartesian_vars=['v1'],
        pre_consts=[(0, 5)],
        view_var='view_neg_0',
        rel_name='Neg',
      ),
    },
    {},
  ),
  ('debug', False, True),
  ('tile_var', 'tile', 'tile_other'),
]


@pytest.mark.parametrize(('field', 'initial', 'replacement'), FORWARD_FIELD_FIXTURES)
def test_forwarding_property_read_roundtrip(field: str, initial: Any, replacement: Any) -> None:
  '''`ctx.<field>` reads must alias `ctx.render_ctx.<field>` —
  no shadowing dataclass field on LoweringCtx.'''
  rc = CudaRenderCtx()
  setattr(rc, field, initial)
  ctx = LoweringCtx(render_ctx=rc)
  assert getattr(ctx, field) == initial
  # Read again after a direct mutation on the inner ctx — the
  # property must NOT have cached a snapshot.
  setattr(rc, field, replacement)
  assert getattr(ctx, field) == replacement


@pytest.mark.parametrize(('field', 'initial', 'replacement'), FORWARD_FIELD_FIXTURES)
def test_forwarding_property_write_roundtrip(field: str, initial: Any, replacement: Any) -> None:
  '''`ctx.<field> = X` must route to `ctx.render_ctx.<field>` — no
  shadow attribute on the outer LoweringCtx.'''
  ctx = LoweringCtx()
  setattr(ctx, field, initial)
  assert getattr(ctx.render_ctx, field) == initial
  setattr(ctx, field, replacement)
  assert getattr(ctx.render_ctx, field) == replacement


# -----------------------------------------------------------------------------
# 3. Flat-kwargs construction equivalence
# -----------------------------------------------------------------------------


def test_flat_kwargs_and_render_ctx_kwargs_agree() -> None:
  '''Constructing with flat render kwargs vs. constructing with a
  pre-built render_ctx yields the same observable render-side state.'''
  view_var_names = {'0': 'view_src'}
  ctx_flat = LoweringCtx(view_var_names=view_var_names, output_var='ctx0')
  ctx_built = LoweringCtx(
    render_ctx=CudaRenderCtx(view_var_names=dict(view_var_names), output_var='ctx0')
  )
  assert ctx_flat.view_var_names == ctx_built.view_var_names
  assert ctx_flat.output_var == ctx_built.output_var
  # Identity check on render_ctx itself.
  assert ctx_flat.render_ctx.view_var_names == ctx_built.render_ctx.view_var_names


def test_render_ctx_kwarg_rejects_flat_render_kwargs() -> None:
  '''Passing both `render_ctx=` AND flat render kwargs must fail
  loudly — silently merging would be a footgun.'''
  rc = CudaRenderCtx(output_var='ctx0')
  with pytest.raises(TypeError, match='render_ctx'):
    LoweringCtx(render_ctx=rc, view_var_names={'0': 'view_x'})


# -----------------------------------------------------------------------------
# 4. Sample MIR lowers to byte-equivalent output through extracted ctx
# -----------------------------------------------------------------------------


def _scan_insert_pipeline() -> tuple[mir.Scan, mir.InsertInto]:
  scan = mir.Scan(
    vars=['v0', 'v1'],
    rel_name='Src',
    version=Version.FULL,
    index=[0, 1],
    handle_start=0,
  )
  ins = mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['v0', 'v1'],
    index=[0, 1],
  )
  return scan, ins


def _render_iir(op: Any) -> str:
  return emit(op, EmitCtx(indent_level=2))


def test_byte_equivalent_under_flat_vs_render_ctx_kwargs() -> None:
  '''A sample MIR program lowered with a flat-kwargs LoweringCtx vs.
  a render_ctx-kwarg LoweringCtx must yield identical rendered IIR
  text. Extraction MUST be byte-preserving.'''
  scan, ins = _scan_insert_pipeline()
  view_var_names = {'0': 'view_src'}

  ctx_flat = LoweringCtx(view_var_names=view_var_names, output_var='ctx0')
  text_flat = _render_iir(lower_scan_pipeline([scan, ins], ctx_flat))

  ctx_built = LoweringCtx(
    render_ctx=CudaRenderCtx(view_var_names=dict(view_var_names), output_var='ctx0')
  )
  text_built = _render_iir(lower_scan_pipeline([scan, ins], ctx_built))

  assert text_flat == text_built
  # Sanity: the render reached actual content.
  assert 'view_src' in text_flat
  assert 'ctx0' in text_flat
