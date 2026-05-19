'''CudaRenderCtx — PR-1 foundation tests.

Per `docs/phase_decomposition_redesign.md` § 3.2.1 (Wave T1).

`CudaRenderCtx` is the CUDA-render-side companion to the framework
`LowerCtx`. PR-1 introduces it as the canonical home for the 11
CUDA-identifier-bearing fields that previously lived flat on the
dialect-level `LoweringCtx`. Future PRs migrate every per-op render
to consume `ctx.render_ctx.<field>`; PR-1 establishes the shape.

These tests pin the construction contract, defaults, and the
threading-into-`LoweringCtx` shape.
'''

from __future__ import annotations

from srdatalog.ir.codegen.cuda.lower_ctx import CudaRenderCtx
from srdatalog.ir.dialects.relation.sorted_array.lowerings import LoweringCtx


def test_cuda_render_ctx_constructible_with_no_args() -> None:
  '''Every field defaults to an empty / off value — back-compat
  with the legacy flat-field `LoweringCtx` construction.'''
  ctx = CudaRenderCtx()
  assert ctx.view_var_names == {}
  assert ctx.output_var == 'output'
  assert ctx.output_var_overrides == {}
  assert ctx.view_slot_bases == {}
  assert ctx.rel_index_types == {}
  assert ctx.tiled_cartesian_valid_var == ''
  assert ctx.ws_cartesian_valid_var == ''
  assert ctx.ws_cartesian_bound_vars == []
  assert ctx.neg_pre_narrow == {}
  assert ctx.dedup_hash_vars == []
  assert ctx.debug is True


def test_cuda_render_ctx_constructible_with_explicit_args() -> None:
  '''Every field takes a positional override.'''
  ctx = CudaRenderCtx(
    view_var_names={'0': 'view_R_0'},
    output_var='ctx0',
    output_var_overrides={'R': 'ctx_R'},
    view_slot_bases={'0': 0, '1': 2},
    rel_index_types={'R': 'Device2LevelIndex'},
    tiled_cartesian_valid_var='_tc_valid_1',
    ws_cartesian_valid_var='_ws_valid_1',
    ws_cartesian_bound_vars=['v0', 'v1'],
    neg_pre_narrow={},
    dedup_hash_vars=['v0'],
    debug=False,
  )
  assert ctx.view_var_names == {'0': 'view_R_0'}
  assert ctx.output_var == 'ctx0'
  assert ctx.debug is False


def test_cuda_render_ctx_mutability() -> None:
  '''The dataclass is mutable (not frozen) — lowering helpers flip
  individual fields during the walk (e.g. neg_pre_narrow gets
  populated by `_register_neg_pre_narrow`).'''
  ctx = CudaRenderCtx()
  ctx.view_var_names['0'] = 'view_X_0'
  assert ctx.view_var_names == {'0': 'view_X_0'}


def test_lowering_ctx_threads_render_ctx_slot() -> None:
  '''The dialect-level `LoweringCtx` holds the `CudaRenderCtx` via
  the `render_ctx` slot. Constructing a `LoweringCtx` with no args
  builds a default `CudaRenderCtx` internally.'''
  ctx = LoweringCtx()
  assert isinstance(ctx.render_ctx, CudaRenderCtx)
  assert ctx.render_ctx.view_var_names == {}
  assert ctx.render_ctx.output_var == 'output'


def test_lowering_ctx_accepts_prebuilt_render_ctx() -> None:
  '''New-style construction: pass a fully-built `CudaRenderCtx`.'''
  render = CudaRenderCtx(view_var_names={'0': 'view_R_0'}, output_var='ctx0')
  ctx = LoweringCtx(render_ctx=render)
  assert ctx.render_ctx is render
  assert ctx.render_ctx.view_var_names == {'0': 'view_R_0'}


def test_lowering_ctx_accepts_legacy_flat_kwargs() -> None:
  '''Back-compat construction: legacy callers pass the flat CUDA
  field names; the constructor builds a `CudaRenderCtx` internally
  populated from those kwargs.'''
  ctx = LoweringCtx(view_var_names={'0': 'view_R_0'}, output_var='ctx0')
  assert ctx.render_ctx.view_var_names == {'0': 'view_R_0'}
  assert ctx.render_ctx.output_var == 'ctx0'
  # The forwarding properties expose the same underlying storage.
  assert ctx.view_var_names == {'0': 'view_R_0'}
  assert ctx.output_var == 'ctx0'


def test_lowering_ctx_legacy_field_reads_go_through_render_ctx() -> None:
  '''Every legacy flat field name still reads correctly from the
  dialect ctx — the properties delegate to render_ctx.'''
  ctx = LoweringCtx(view_var_names={'0': 'a'}, output_var='o')
  # Writes via legacy property surface land on the render_ctx too.
  ctx.view_var_names['1'] = 'b'
  assert ctx.render_ctx.view_var_names == {'0': 'a', '1': 'b'}


def test_lowering_ctx_render_ctx_is_shared_storage() -> None:
  '''Mutating render_ctx directly is visible through the forwarding
  property (and vice-versa). PR-1 keeps the legacy surface live so
  per-op render migrations can switch one site at a time without
  breaking byte-equivalence.'''
  ctx = LoweringCtx()
  ctx.render_ctx.tiled_cartesian_valid_var = '_tc_1'
  assert ctx.tiled_cartesian_valid_var == '_tc_1'
  ctx.tiled_cartesian_valid_var = '_tc_2'
  assert ctx.render_ctx.tiled_cartesian_valid_var == '_tc_2'
