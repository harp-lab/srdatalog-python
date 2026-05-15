'''F3 / Layer 1 foundation tests — `LowerCtx`, `NameGen`,
`ViewLayout`, `Scope`, `EmptyScope`.

Covers the contract from `docs/phase_zero_prerequisites.md` §3.2 +
§3.3 and `docs/phase_b_lowering_dispatcher.md` §3 (field-by-field
LowerCtx design).

Self-contained: synthetic `Op` + `Dialect` fixtures, no production
dialect imports. Mirrors the style of `tests/test_core_pass_kinds.py`.
'''

from __future__ import annotations

import dataclasses
from dataclasses import FrozenInstanceError

import pytest

from srdatalog.ir.core import (
  Compiler,
  Dialect,
  EmptyScope,
  LowerCtx,
  LoweringMissingError,
  LoweringPass,
  NameGen,
  Op,
  Scope,
  ViewLayout,
)
from srdatalog.ir.core.passes import lowering

# -----------------------------------------------------------------------------
# Synthetic op fixtures
# -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _SrcLeaf(Op):
  '''Source-dialect leaf op carrying an integer payload.'''

  value: int


@dataclasses.dataclass(frozen=True, slots=True)
class _SrcWrap(Op):
  '''Source-dialect wrapper holding one inner source op.'''

  inner: object


@dataclasses.dataclass(frozen=True, slots=True)
class _TgtLeaf(Op):
  '''Target-dialect leaf op carrying the lowered payload.'''

  value: int


@dataclasses.dataclass(frozen=True, slots=True)
class _TgtWrap(Op):
  '''Target-dialect wrapper holding one lowered inner op.'''

  inner: object


def _src_dialect() -> Dialect:
  d = Dialect(name='test.src')
  d.ops = [_SrcLeaf, _SrcWrap]
  return d


def _tgt_dialect() -> Dialect:
  d = Dialect(name='test.tgt')
  d.ops = [_TgtLeaf, _TgtWrap]
  return d


# -----------------------------------------------------------------------------
# LowerCtx shape: 5 fields, frozen
# -----------------------------------------------------------------------------


def test_lower_ctx_has_exactly_five_fields():
  '''D10: `LowerCtx` is pinned at 5 fields. Adding a 6th requires
  a doc amendment + owner sign-off (see
  `docs/phase_zero_prerequisites.md` §3.2 and
  `docs/code_discipline.md` D10).'''
  fields = dataclasses.fields(LowerCtx)
  assert len(fields) == 5
  names = {f.name for f in fields}
  assert names == {'compiler', 'name_gen', 'view_layout', 'plugin_registry', 'target'}


def test_lower_ctx_is_frozen():
  '''Field reassignment on a `LowerCtx` instance raises
  `FrozenInstanceError`.'''
  ctx = LowerCtx(
    compiler=Compiler(),
    name_gen=NameGen(),
    view_layout=ViewLayout(),
    plugin_registry=None,
    target='cuda',
  )
  with pytest.raises(FrozenInstanceError):
    ctx.target = 'cpp_tbb'  # type: ignore[misc]


# -----------------------------------------------------------------------------
# LowerCtx.lower dispatch via attached _table
# -----------------------------------------------------------------------------


def test_lower_ctx_dispatches_via_attached_table():
  '''`LoweringPass.apply` constructs a `LowerCtx`, attaches the
  per-call dispatch table via ``object.__setattr__``, and the
  registered ``@lowering`` fires through ``ctx.lower(op)``.'''
  src = _src_dialect()
  tgt = _tgt_dialect()

  observed: list[type] = []

  @lowering(tgt, _SrcLeaf, consumes=('test.src',), produces=('test.tgt',))
  def _l_leaf(op, ctx):
    # Verify the ctx is a real LowerCtx with all five fields populated.
    assert isinstance(ctx, LowerCtx)
    assert ctx.target == 'test.tgt'
    assert isinstance(ctx.name_gen, NameGen)
    assert isinstance(ctx.view_layout, ViewLayout)
    observed.append(type(op))
    return _TgtLeaf(value=op.value * 10)

  c = Compiler()
  c.register_dialect(src)
  c.register_dialect(tgt)

  p = LoweringPass(
    name='src_to_tgt',
    consumes=('test.src',),
    produces=('test.tgt',),
    target_dialect_name='test.tgt',
  )
  out = p.apply(_SrcLeaf(value=4), c)
  assert out == _TgtLeaf(value=40)
  assert observed == [_SrcLeaf]


def test_lower_ctx_lower_raises_for_unregistered_op_type():
  '''`LowerCtx.lower(op)` raises `LoweringMissingError` when no
  ``@lowering`` is registered for ``type(op)`` on the active
  target dialect.'''
  src = _src_dialect()
  tgt = _tgt_dialect()

  c = Compiler()
  c.register_dialect(src)
  c.register_dialect(tgt)

  p = LoweringPass(
    name='src_to_tgt_empty',
    consumes=('test.src',),
    produces=('test.tgt',),
    target_dialect_name='test.tgt',
  )
  with pytest.raises(LoweringMissingError) as ei:
    p.apply(_SrcLeaf(value=1), c)
  assert ei.value.op_type is _SrcLeaf
  assert ei.value.target_dialect_name == 'test.tgt'


def test_lower_ctx_explicit_recursion_through_lower():
  '''A lowering that calls ``ctx.lower(child)`` recursively walks
  the synthetic tree (per R1: explicit recursion, no auto-walk).
  End-to-end verification that `LoweringPass.apply` threads
  `LowerCtx` through nested dispatches.'''
  src = _src_dialect()
  tgt = _tgt_dialect()

  @lowering(tgt, _SrcLeaf)
  def _l_leaf(op, ctx):
    return _TgtLeaf(value=op.value + 100)

  @lowering(tgt, _SrcWrap)
  def _l_wrap(op, ctx):
    # Explicit recursion via ctx.lower — the canonical LowerCtx use.
    return _TgtWrap(inner=ctx.lower(op.inner))

  c = Compiler()
  c.register_dialect(src)
  c.register_dialect(tgt)

  p = LoweringPass(
    name='src_to_tgt_rec',
    consumes=('test.src',),
    produces=('test.tgt',),
    target_dialect_name='test.tgt',
  )
  tree = _SrcWrap(inner=_SrcWrap(inner=_SrcLeaf(value=2)))
  out = p.apply(tree, c)
  assert out == _TgtWrap(inner=_TgtWrap(inner=_TgtLeaf(value=102)))


# -----------------------------------------------------------------------------
# NameGen
# -----------------------------------------------------------------------------


def test_name_gen_fresh_bumps_counter():
  '''Successive `NameGen.fresh` calls produce ``<prefix>_1``,
  ``<prefix>_2``, ... matching legacy `LoweringCtx.fresh`.'''
  ng = NameGen()
  assert ng.fresh('h') == 'h_1'
  assert ng.fresh('h') == 'h_2'
  # Different prefix shares the same counter (legacy parity).
  assert ng.fresh('v') == 'v_3'


def test_name_gen_start_offsets_counter():
  '''`NameGen(start=10).fresh('h')` returns ``h_11``.'''
  ng = NameGen(start=10)
  assert ng.fresh('h') == 'h_11'
  assert ng.fresh('h') == 'h_12'


# -----------------------------------------------------------------------------
# ViewLayout
# -----------------------------------------------------------------------------


def test_view_layout_is_frozen():
  '''`ViewLayout` is frozen+slots — field reassignment raises.'''
  vl = ViewLayout()
  with pytest.raises(FrozenInstanceError):
    vl.view_var_names = {}  # type: ignore[misc]


def test_view_layout_defaults_to_empty():
  '''`ViewLayout()` constructs with empty ``view_var_names`` and
  ``slot_bases`` mappings (F5 will populate from real ``prog``).'''
  vl = ViewLayout()
  assert vl.view_var_names == {}
  assert vl.slot_bases == {}


def test_view_layout_accepts_explicit_mappings():
  '''Direct construction with mappings works — used by F5 once the
  pipeline shim wires real ``prog`` into the layout.'''
  vl = ViewLayout(
    view_var_names={'h0': 'view_a'},
    slot_bases={'h0': 0},
  )
  assert vl.view_var_names == {'h0': 'view_a'}
  assert vl.slot_bases == {'h0': 0}


# -----------------------------------------------------------------------------
# Scope + EmptyScope
# -----------------------------------------------------------------------------


@pytest.mark.parametrize('cls', [Scope, EmptyScope])
def test_scope_classes_are_frozen_and_slotted(cls):
  '''Both the `Scope` marker base and `EmptyScope` use
  ``frozen=True, slots=True`` so concrete subclasses inherit the
  discipline without restating it (D16-style cap on subclass
  fields).'''
  # slots=True → class declares __slots__ (and instances lack __dict__).
  assert hasattr(cls, '__slots__')
  inst = cls()
  assert not hasattr(inst, '__dict__')


def test_scope_subclass_field_reassignment_raises_frozen_error():
  '''A user-defined `Scope` subclass with a field rejects field
  reassignment with `FrozenInstanceError` — proving subclasses
  inherit the frozen contract.'''

  @dataclasses.dataclass(frozen=True, slots=True)
  class _SyntheticScope(Scope):
    bound: int = 0

  s = _SyntheticScope(bound=3)
  with pytest.raises(FrozenInstanceError):
    s.bound = 4  # type: ignore[misc]


def test_empty_scope_is_subclass_of_scope():
  '''`EmptyScope` is the F3 convenience subclass — instances are
  `Scope` instances, so dispatchers can pass `EmptyScope()` wherever
  a `Scope` is expected.'''
  assert isinstance(EmptyScope(), Scope)
  assert issubclass(EmptyScope, Scope)
