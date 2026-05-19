'''Byte-equivalence test for Wave 2A B-Cart migration.

Per `docs/phase_b_lowering_dispatcher.md` §4.2 (per-PR acceptance
gate): each per-MIR-op migration ships a
`test_lower_mir_<op>_byte_equivalent` test that runs the migrated
path on every relevant fixture and asserts byte equality with the
legacy `if isinstance(head, mir.X):` branch.

For CartesianJoin the relevant fixtures (per
`docs/phase_b_lowering_dispatcher.md` §4 row B-Cart, "hard" —
`_lower_root_cart` / `_lower_nested_cart` interact with multi-source
shapes, prefix narrowing, the C5 `TiledCartesian` typed pragma, and
the `_register_neg_pre_narrow` -> `ctx.neg_pre_narrow` contract
consumed by mid-chain `mir.Negation`):

  - root 2-source Cart with trailing InsertInto (`_lower_root_cart`),
  - root 3-source Cart (CartesianNDecompose path),
  - mid-chain 2-source Cart under a Scan root,
  - mid-chain 3-source Cart under a Scan root,
  - mid-chain Cart with inner Negation (verifies the
    `ctx.neg_pre_narrow` registration in `_lower_nested_cart` still
    flows through `lower_mir_negation_in_chain`),
  - mid-chain Cart under a multi-source ColumnJoin root,
  - tiled-eligible shape (2 sources, 1 var per source, materialize
    phase) — verifies the C5 `TiledCartesian` wrap-op path remains
    intact (the wrap fires BEFORE this Cart entry sees the op,
    because `_should_use_declarative` returns True for
    `mir.CartesianJoin` but the wrap op is a different type;
    `_lower_inner_chain`'s `mir.TiledCartesian` branch still
    catches the wrap form).
  - legacy `ctx.tiled_cartesian=True` short-circuit path: the bare
    `mir.CartesianJoin` reaches `lower_mir_cart_in_chain` ->
    `_lower_nested_cart`, which forwards to `_lower_nested_cart_tiled`
    via its own `_tiled_cart_eligible` check — verifies the dual-write
    transition still emits byte-identical IIR.
  - full pipeline through `compile_pipeline` for both root and nested
    shapes.

The test compares two compilation paths:

  - LEGACY: `USE_DECLARATIVE` patched to NOT contain
    `mir.CartesianJoin`, so `lower_scan_pipeline` /
    `_lower_inner_chain` fall into the legacy imperative branches
    (`_lower_root_cart` / `_lower_nested_cart`).
  - NEW: `USE_DECLARATIVE` left alone (CartesianJoin IS in the set),
    so dispatch routes through `lower_mir_cart_root` /
    `lower_mir_cart_in_chain`.

Byte-equivalence is asserted on the rendered IIR text. The C5
end-to-end test (`tests/test_pragma_tiled_cartesian_end_to_end.py`)
re-running green under this PR is the strongest signal that the C5
path remains intact; this file adds direct-call coverage of the
non-tiled corner cases plus the tiled coexistence anchor.
'''

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

import srdatalog.ir.mir.types as mir
from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
from srdatalog.ir.dialects.iir.cf import Block as IirBlock
from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
  LoweringCtx,
  _should_use_declarative,
  lower_scan_pipeline,
)
from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_cart import (
  lower_mir_cart,
  lower_mir_cart_in_chain,
  lower_mir_cart_root,
)
from srdatalog.ir.hir.types import Version

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _render(op: Any) -> str:
  '''Render an IIR op tree through the CUDA emitter — the same
  surface the byte-equivalence harnesses exercise.'''
  return emit(op, EmitCtx(indent_level=2))


@contextmanager
def _force_legacy_branch():
  '''Temporarily strip `mir.CartesianJoin` from `USE_DECLARATIVE` so
  `lower_scan_pipeline` / `_lower_inner_chain` fall into the legacy
  imperative branches (`_lower_root_cart` / `_lower_nested_cart`).

  Save / restore as a context manager — the dialect module's
  `USE_DECLARATIVE` is a frozenset re-bound name; the dispatch
  helper `_should_use_declarative` re-reads it on each call, so
  swapping the frozenset for the duration of one block is enough to
  force the legacy path.
  '''
  import srdatalog.ir.dialects.relation.sorted_array as sa_dialect

  saved = sa_dialect.USE_DECLARATIVE
  sa_dialect.USE_DECLARATIVE = frozenset(saved - {mir.CartesianJoin})
  try:
    yield
  finally:
    sa_dialect.USE_DECLARATIVE = saved


def _new_ctx(**kwargs: Any) -> LoweringCtx:
  '''Fresh LoweringCtx — counters reset so both paths bump identically.'''
  view_var_names = kwargs.pop('view_var_names', None) or {
    '0': 'v_Src_full',
    '1': 'v_R_full',
    '2': 'v_S_full',
    '3': 'v_T_full',
  }
  return LoweringCtx(output_var='ctx0', view_var_names=view_var_names, **kwargs)


def _column_source(
  rel: str,
  handle_start: int,
  prefix_vars: tuple[str, ...] = (),
  index: tuple[int, ...] = (0,),
) -> mir.ColumnSource:
  return mir.ColumnSource(
    rel_name=rel,
    version=Version.FULL,
    index=list(index),
    prefix_vars=list(prefix_vars),
    handle_start=handle_start,
  )


def _cart(
  *sources: mir.ColumnSource,
  vars_: tuple[str, ...],
  var_from_source: tuple[tuple[str, ...], ...],
  handle_start: int = -1,
) -> mir.CartesianJoin:
  return mir.CartesianJoin(
    vars=list(vars_),
    sources=list(sources),
    var_from_source=[list(vfs) for vfs in var_from_source],
    handle_start=handle_start,
  )


def _insert_into(vars_: tuple[str, ...]) -> mir.InsertInto:
  return mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=list(vars_),
    index=list(range(len(vars_))),
  )


def _scan(vars_: tuple[str, ...] = ('x',), handle_start: int = 0) -> mir.Scan:
  return mir.Scan(
    vars=list(vars_),
    rel_name='Src',
    version=Version.FULL,
    index=list(range(len(vars_))),
    handle_start=handle_start,
  )


# -----------------------------------------------------------------------------
# 1. Root Cart — `lower_scan_pipeline` -> `lower_mir_cart_root`
# -----------------------------------------------------------------------------


def test_root_cart_two_sources_byte_equivalent():
  '''Root 2-source Cart with trailing InsertInto. Emits per-source
  handle binds (SaRoot), combined validity `return`, per-source
  degree, total, total-zero `return`, ParallelFor grid-stride loop
  with per-source decompose + SaGetValAt var binds.'''
  cart = _cart(
    _column_source('R', handle_start=1),
    _column_source('S', handle_start=2),
    vars_=('y', 'z'),
    var_from_source=(('y',), ('z',)),
    handle_start=1,
  )
  ins = _insert_into(('y', 'z'))

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([cart, ins], _new_ctx()))
  new_text = _render(lower_scan_pipeline([cart, ins], _new_ctx()))

  assert legacy_text == new_text
  # Root-Cart scaffold markers: `return` (not `continue`) on validity.
  assert 'return' in legacy_text
  assert 'h_R_1' in legacy_text and 'h_S_2' in legacy_text


def test_root_cart_three_sources_byte_equivalent():
  '''Root 3-source Cart: emits CartesianNDecompose for per-thread idx.'''
  cart = _cart(
    _column_source('R', handle_start=1),
    _column_source('S', handle_start=2),
    _column_source('T', handle_start=3),
    vars_=('y', 'z', 'w'),
    var_from_source=(('y',), ('z',), ('w',)),
    handle_start=1,
  )
  ins = _insert_into(('y', 'z', 'w'))

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([cart, ins], _new_ctx()))
  new_text = _render(lower_scan_pipeline([cart, ins], _new_ctx()))

  assert legacy_text == new_text
  assert 'h_R_1' in legacy_text
  assert 'h_S_2' in legacy_text
  assert 'h_T_3' in legacy_text


def test_root_cart_one_source_byte_equivalent():
  '''Root 1-source Cart (legal per `_lower_root_cart` assertion
  `num_sources >= 1`). Emits the same scaffold without the
  per-source decompose.'''
  cart = _cart(
    _column_source('R', handle_start=1),
    vars_=('y',),
    var_from_source=(('y',),),
    handle_start=1,
  )
  ins = _insert_into(('y',))

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([cart, ins], _new_ctx()))
  new_text = _render(lower_scan_pipeline([cart, ins], _new_ctx()))

  assert legacy_text == new_text


# -----------------------------------------------------------------------------
# 2. Mid-chain Cart under a Scan root — `_lower_inner_chain` ->
# `lower_mir_cart_in_chain`
# -----------------------------------------------------------------------------


def test_nested_cart_two_sources_under_scan_byte_equivalent():
  '''2-source Cart nested under a Scan root. Emits the
  nested-Cart scaffold (lane/group_size, per-source handle, total,
  flat_idx, idx vars).'''
  scan = _scan(('x',))
  cart = _cart(
    _column_source('R', handle_start=1),
    _column_source('S', handle_start=2),
    vars_=('y', 'z'),
    var_from_source=(('y',), ('z',)),
    handle_start=1,
  )
  ins = _insert_into(('x', 'y', 'z'))

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([scan, cart, ins], _new_ctx()))
  new_text = _render(lower_scan_pipeline([scan, cart, ins], _new_ctx()))

  assert legacy_text == new_text
  # Nested-Cart marker: lane / group_size scaffold.
  assert 'lane' in legacy_text
  assert 'group_size' in legacy_text


def test_nested_cart_three_sources_under_scan_byte_equivalent():
  '''3-source nested Cart: uses CartesianNDecompose for idx vars.'''
  scan = _scan(('x',))
  cart = _cart(
    _column_source('R', handle_start=1),
    _column_source('S', handle_start=2),
    _column_source('T', handle_start=3),
    vars_=('y', 'z', 'w'),
    var_from_source=(('y',), ('z',), ('w',)),
    handle_start=1,
  )
  ins = _insert_into(('x', 'y', 'z', 'w'))

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([scan, cart, ins], _new_ctx()))
  new_text = _render(lower_scan_pipeline([scan, cart, ins], _new_ctx()))

  assert legacy_text == new_text


def test_nested_cart_one_source_byte_equivalent():
  '''1-source nested Cart: minimal scaffold, single idx var.'''
  scan = _scan(('x',))
  cart = _cart(
    _column_source('R', handle_start=1),
    vars_=('y',),
    var_from_source=(('y',),),
    handle_start=1,
  )
  ins = _insert_into(('x', 'y'))

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([scan, cart, ins], _new_ctx()))
  new_text = _render(lower_scan_pipeline([scan, cart, ins], _new_ctx()))

  assert legacy_text == new_text


# -----------------------------------------------------------------------------
# 3. Nested Cart with mid-chain Negation — `ctx.neg_pre_narrow` contract
# -----------------------------------------------------------------------------


def test_nested_cart_with_inner_negation_byte_equivalent():
  '''Nested Cart followed by a Negation whose prefix var is bound
  OUTSIDE the Cart (by the Scan). `_lower_nested_cart` registers a
  `neg_pre_narrow` entry for the negation's handle_idx BEFORE
  rendering the body; the body's chain dispatcher then routes the
  Negation through `lower_mir_negation_in_chain` which sees the
  registration.

  Both paths must produce identical text — the `neg_pre_narrow`
  flow is preserved by construction (the B-Cart migration delegates
  to `_lower_nested_cart`, which is the registrar).
  '''
  scan = _scan(('x',))
  cart = _cart(
    _column_source('R', handle_start=1),
    _column_source('S', handle_start=2),
    vars_=('y', 'z'),
    var_from_source=(('y',), ('z',)),
    handle_start=1,
  )
  neg = mir.Negation(
    rel_name='Bad',
    version=Version.FULL,
    prefix_vars=['x'],
    index=[0],
    handle_start=3,
  )
  ins = _insert_into(('x', 'y', 'z'))

  views = {
    '0': 'v_Src_full',
    '1': 'v_R_full',
    '2': 'v_S_full',
    '3': 'v_Bad_full',
  }

  with _force_legacy_branch():
    legacy_text = _render(
      lower_scan_pipeline([scan, cart, neg, ins], _new_ctx(view_var_names=views))
    )
  new_text = _render(lower_scan_pipeline([scan, cart, neg, ins], _new_ctx(view_var_names=views)))

  assert legacy_text == new_text
  # Pre-narrow handle var (registered by _register_neg_pre_narrow)
  # appears in the rendered output.
  assert 'h_Bad_neg_pre' in legacy_text


def test_nested_cart_with_negation_inside_cart_vars_byte_equivalent():
  '''Negation whose prefix vars come from INSIDE the Cart: no
  pre-narrow registration (the helper's `pre_vars` ends up empty
  because the var is in `cartesian_bound_set`). Both paths should
  still match byte-for-byte.'''
  scan = _scan(('x',))
  cart = _cart(
    _column_source('R', handle_start=1),
    _column_source('S', handle_start=2),
    vars_=('y', 'z'),
    var_from_source=(('y',), ('z',)),
    handle_start=1,
  )
  # Prefix var `y` is bound INSIDE the Cart (var_from_source[0]).
  neg = mir.Negation(
    rel_name='Bad',
    version=Version.FULL,
    prefix_vars=['y'],
    index=[0],
    handle_start=3,
  )
  ins = _insert_into(('x', 'y', 'z'))

  views = {
    '0': 'v_Src_full',
    '1': 'v_R_full',
    '2': 'v_S_full',
    '3': 'v_Bad_full',
  }

  with _force_legacy_branch():
    legacy_text = _render(
      lower_scan_pipeline([scan, cart, neg, ins], _new_ctx(view_var_names=views))
    )
  new_text = _render(lower_scan_pipeline([scan, cart, neg, ins], _new_ctx(view_var_names=views)))

  assert legacy_text == new_text


# -----------------------------------------------------------------------------
# 4. Nested Cart under a multi-source ColumnJoin root
# -----------------------------------------------------------------------------


def test_nested_cart_under_cj_root_byte_equivalent():
  '''A multi-source ColumnJoin root with a nested Cart in the middle:
  exercises the Scan-rooted vs CJ-rooted dispatch in
  `_supported_pipeline` — both must accept the shape and both
  byte-equivalence paths must agree.'''
  cj_root = mir.ColumnJoin(
    var_name='x',
    sources=[
      mir.ColumnSource(
        rel_name='A',
        version=Version.FULL,
        index=[0],
        prefix_vars=[],
        handle_start=0,
      ),
      mir.ColumnSource(
        rel_name='B',
        version=Version.FULL,
        index=[0],
        prefix_vars=[],
        handle_start=1,
      ),
    ],
    handle_start=0,
  )
  cart = _cart(
    _column_source('R', handle_start=2),
    _column_source('S', handle_start=3),
    vars_=('y', 'z'),
    var_from_source=(('y',), ('z',)),
    handle_start=2,
  )
  ins = _insert_into(('x', 'y', 'z'))

  views = {
    '0': 'v_A_full',
    '1': 'v_B_full',
    '2': 'v_R_full',
    '3': 'v_S_full',
  }

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([cj_root, cart, ins], _new_ctx(view_var_names=views)))
  new_text = _render(lower_scan_pipeline([cj_root, cart, ins], _new_ctx(view_var_names=views)))

  assert legacy_text == new_text


# -----------------------------------------------------------------------------
# 5. Tiled-eligible shape — C5 dual-write coexistence
# -----------------------------------------------------------------------------


def test_tiled_eligible_cart_legacy_bool_path_byte_equivalent():
  '''Tiled-eligible 2-source / 1-var-per-source Cart with
  `ctx.tiled_cartesian=True` (the legacy runner-driven bool path,
  pre-A3). The bare `mir.CartesianJoin` reaches
  `lower_mir_cart_in_chain` -> `_lower_nested_cart`, which forwards
  to `_lower_nested_cart_tiled` via its own `_tiled_cart_eligible`
  check.

  Both paths produce the tiled IIR (TiledBallotBlock +
  SaTiledCartesian2D scaffold) — the C5 dual-write transition is
  preserved.
  '''
  scan = _scan(('x',))
  cart = _cart(
    _column_source('R', handle_start=1),
    _column_source('S', handle_start=2),
    vars_=('y', 'z'),
    var_from_source=(('y',), ('z',)),
    handle_start=1,
  )
  ins = _insert_into(('x', 'y', 'z'))

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([scan, cart, ins], _new_ctx(tiled_cartesian=True)))
  new_text = _render(lower_scan_pipeline([scan, cart, ins], _new_ctx(tiled_cartesian=True)))

  assert legacy_text == new_text
  # Tiled-mode marker: the tc_valid_<n> name appears in the body.
  assert 'tc_valid' in legacy_text


def test_tiled_eligible_cart_wrap_op_path_byte_equivalent():
  '''Tiled-eligible Cart wrapped in `mir.TiledCartesian` (the C5
  post-A3 / pure-typed-pragma form). The wrap op is a DIFFERENT
  MIR op type — `_lower_inner_chain` dispatches it through
  `lower_tiled_cartesian_in_chain` (C5), NOT through the B-Cart
  entry. This test verifies that B-Cart's `USE_DECLARATIVE`
  membership does NOT perturb the C5 wrap-op dispatch.

  Both paths produce byte-identical IIR via the same
  `_lower_nested_cart_tiled` helper.
  '''
  scan = _scan(('x',))
  bare_cart = _cart(
    _column_source('R', handle_start=1),
    _column_source('S', handle_start=2),
    vars_=('y', 'z'),
    var_from_source=(('y',), ('z',)),
    handle_start=1,
  )
  wrap = mir.TiledCartesian(inner=bare_cart)
  ins = _insert_into(('x', 'y', 'z'))

  with _force_legacy_branch():
    legacy_text = _render(lower_scan_pipeline([scan, wrap, ins], _new_ctx()))
  new_text = _render(lower_scan_pipeline([scan, wrap, ins], _new_ctx()))

  assert legacy_text == new_text
  # Tiled-mode marker present.
  assert 'tc_valid' in legacy_text


def test_tiled_eligible_cart_wrap_op_unaffected_by_use_declarative():
  '''Pin the C5 invariant: the `mir.TiledCartesian` wrap op never
  routes through `_should_use_declarative` (because
  `mir.TiledCartesian` is not in `USE_DECLARATIVE`). The B-Cart
  migration adds `mir.CartesianJoin` to the set, but the wrap op
  type is distinct.'''
  from srdatalog.ir.dialects.relation.sorted_array import USE_DECLARATIVE

  # Pin: the wrap op type is NOT in USE_DECLARATIVE.
  assert mir.TiledCartesian not in USE_DECLARATIVE
  # And `_should_use_declarative` returns False for an instance of it.
  wrap = mir.TiledCartesian(
    inner=mir.CartesianJoin(vars=[], sources=[], var_from_source=[], handle_start=-1)
  )
  assert _should_use_declarative(wrap) is False
  # Bare CartesianJoin IS in the set and gates True.
  bare = mir.CartesianJoin(vars=[], sources=[], var_from_source=[], handle_start=-1)
  assert _should_use_declarative(bare) is True


# -----------------------------------------------------------------------------
# 6. Direct call into the chain-aware variant
# -----------------------------------------------------------------------------


def test_lower_mir_cart_in_chain_returns_block():
  '''Pin the IIR tree shape directly: nested Cart produces a `Block`
  with at least the scaffold (lane bind, group_size bind, per-source
  handle binds, total bind, loop). Same shape as legacy
  `_lower_nested_cart`.'''
  cart = _cart(
    _column_source('R', handle_start=1),
    _column_source('S', handle_start=2),
    vars_=('y', 'z'),
    var_from_source=(('y',), ('z',)),
    handle_start=1,
  )
  ins = _insert_into(('y', 'z'))
  ctx = _new_ctx(debug=False)

  out = lower_mir_cart_in_chain(cart, [ins], ctx)
  assert isinstance(out, IirBlock)
  # The block must contain a lane bind and group_size bind among its
  # opening stmts — the unique signature of `_lower_nested_cart`.
  rendered = _render(out)
  assert 'lane' in rendered
  assert 'group_size' in rendered


def test_lower_mir_cart_root_returns_block():
  '''Pin the IIR tree shape directly: root Cart produces a `Block`
  with the root-Cart scaffold (per-source SaRoot binds, combined
  validity `return`, per-source degree, ParallelFor).'''
  cart = _cart(
    _column_source('R', handle_start=1),
    _column_source('S', handle_start=2),
    vars_=('y', 'z'),
    var_from_source=(('y',), ('z',)),
    handle_start=1,
  )
  ins = _insert_into(('y', 'z'))
  ctx = _new_ctx(debug=False)

  out = lower_mir_cart_root(cart, [ins], ctx)
  assert isinstance(out, IirBlock)
  rendered = _render(out)
  # Root-Cart uses `return` (not `continue`).
  assert 'return' in rendered


# -----------------------------------------------------------------------------
# 7. Registry contract — stub asserts on direct call + ownership
# -----------------------------------------------------------------------------


def test_lower_mir_cart_registry_stub_asserts():
  '''The `@lowering(target=iir.cf, source=mir.CartesianJoin)`
  registry entry is a stub that asserts on direct invocation —
  dispatch flows through `lower_scan_pipeline` (root) or
  `_lower_inner_chain` (mid-chain) to the position-aware variants.
  Mirrors the C5 `lower_tiled_cartesian` split + B-Scan / B-Negation
  splits.'''
  cart = _cart(
    _column_source('R', handle_start=1),
    vars_=('y',),
    var_from_source=(('y',),),
    handle_start=1,
  )
  ctx = _new_ctx()
  with pytest.raises(AssertionError, match=r'lower_mir_cart_(root|in_chain)'):
    lower_mir_cart(cart, ctx)


def test_lower_mir_cart_is_registered_on_sorted_array_dialect():
  '''The `CartesianJoin` `@lowering` is registered on the
  `relation.sorted_array` dialect. Pins dialect ownership per
  `docs/phase_b_lowering_dispatcher.md` §4 (one `@lowering` per
  MIR op, on the dialect that lowers it).'''
  from srdatalog.ir.dialects.relation.sorted_array import DIALECT as SA_DIALECT

  matched = [low for low in SA_DIALECT.lowerings if low.matches is mir.CartesianJoin]
  assert len(matched) == 1
  assert matched[0].consumes == ('mir',)
  assert 'iir.cf' in matched[0].produces


# -----------------------------------------------------------------------------
# 8. USE_DECLARATIVE invariants
# -----------------------------------------------------------------------------


def test_use_declarative_contains_cart():
  '''Pin the ratchet: after this Wave 2A PR, `mir.CartesianJoin`
  must appear in `USE_DECLARATIVE`. The monotonic discipline test
  (when it lands) catches accidental removals.'''
  from srdatalog.ir.dialects.relation.sorted_array import USE_DECLARATIVE

  assert mir.CartesianJoin in USE_DECLARATIVE
  # Sanity: previously-migrated ops still in the set.
  assert mir.Filter in USE_DECLARATIVE
  assert mir.ConstantBind in USE_DECLARATIVE
  assert mir.InsertInto in USE_DECLARATIVE
  assert mir.Scan in USE_DECLARATIVE
  assert mir.Negation in USE_DECLARATIVE
  assert mir.Aggregate in USE_DECLARATIVE
  assert mir.ColumnJoin in USE_DECLARATIVE


# -----------------------------------------------------------------------------
# 9. Smoke test through the full compile_pipeline surface
# -----------------------------------------------------------------------------


def test_root_cart_in_full_pipeline_byte_equivalent():
  '''Smoke test: a root 2-source Cart + InsertInto pipeline rendered
  through `compile_pipeline` (the production surface that the
  byte-equivalence harnesses guard) must produce identical CUDA
  under both `USE_DECLARATIVE` states.'''
  from srdatalog.compile import compile_pipeline

  cart = _cart(
    _column_source('R', handle_start=0),
    _column_source('S', handle_start=1),
    vars_=('y', 'z'),
    var_from_source=(('y',), ('z',)),
    handle_start=0,
  )
  ins = mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['y', 'z'],
    index=[0, 1],
  )
  ep = mir.ExecutePipeline(
    pipeline=[cart, ins],
    source_specs=[*cart.sources],
    dest_specs=[ins],
    rule_name='RootCart',
  )

  with _force_legacy_branch():
    legacy_out = compile_pipeline(ep)
  new_out = compile_pipeline(ep)

  assert legacy_out == new_out


def test_nested_cart_in_full_pipeline_byte_equivalent():
  '''Smoke test: a Scan + 2-source Cart + InsertInto pipeline.'''
  from srdatalog.compile import compile_pipeline

  scan = _scan(('x',))
  cart = _cart(
    _column_source('R', handle_start=1),
    _column_source('S', handle_start=2),
    vars_=('y', 'z'),
    var_from_source=(('y',), ('z',)),
    handle_start=1,
  )
  ins = mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['x', 'y', 'z'],
    index=[0, 1, 2],
  )
  ep = mir.ExecutePipeline(
    pipeline=[scan, cart, ins],
    source_specs=[scan, *cart.sources],
    dest_specs=[ins],
    rule_name='NestedCart',
  )

  with _force_legacy_branch():
    legacy_out = compile_pipeline(ep)
  new_out = compile_pipeline(ep)

  assert legacy_out == new_out


def test_nested_cart_with_filter_in_full_pipeline_byte_equivalent():
  '''Scan + 2-source Cart + Filter + InsertInto: middle-of-chain
  Filter follows the Cart, exercising the Cart's body-render path
  that re-enters `_lower_inner_chain` (which routes Filter through
  the new path under USE_DECLARATIVE).'''
  from srdatalog.compile import compile_pipeline

  scan = _scan(('x',))
  cart = _cart(
    _column_source('R', handle_start=1),
    _column_source('S', handle_start=2),
    vars_=('y', 'z'),
    var_from_source=(('y',), ('z',)),
    handle_start=1,
  )
  flt = mir.Filter(vars=['y', 'z'], code='y < z')
  ins = mir.InsertInto(
    rel_name='Dst',
    version=Version.NEW,
    vars=['x', 'y', 'z'],
    index=[0, 1, 2],
  )
  ep = mir.ExecutePipeline(
    pipeline=[scan, cart, flt, ins],
    source_specs=[scan, *cart.sources],
    dest_specs=[ins],
    rule_name='CartFilter',
  )

  with _force_legacy_branch():
    legacy_out = compile_pipeline(ep)
  new_out = compile_pipeline(ep)

  assert legacy_out == new_out


# -----------------------------------------------------------------------------
# 10. Multi-source Cart still routes correctly (sanity: no shape gate)
# -----------------------------------------------------------------------------


def test_should_use_declarative_no_source_count_gate_for_cart():
  '''Unlike B-CJ-single (which gates on `len(sources) == 1`),
  B-Cart has NO source-count gate in `_should_use_declarative` —
  all source counts route through the new path. Pin the invariant
  so a future B-Cart-multi PR doesn't accidentally re-add the gate.
  '''
  one_src = _cart(
    _column_source('R', handle_start=1),
    vars_=('y',),
    var_from_source=(('y',),),
    handle_start=1,
  )
  two_src = _cart(
    _column_source('R', handle_start=1),
    _column_source('S', handle_start=2),
    vars_=('y', 'z'),
    var_from_source=(('y',), ('z',)),
    handle_start=1,
  )
  three_src = _cart(
    _column_source('R', handle_start=1),
    _column_source('S', handle_start=2),
    _column_source('T', handle_start=3),
    vars_=('y', 'z', 'w'),
    var_from_source=(('y',), ('z',), ('w',)),
    handle_start=1,
  )
  assert _should_use_declarative(one_src) is True
  assert _should_use_declarative(two_src) is True
  assert _should_use_declarative(three_src) is True
