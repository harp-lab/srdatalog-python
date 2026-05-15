'''Discipline test: every concrete MIR type is Op-subclassed and
frozen+slots (Phase A1).

Per docs/phase_a_mir_onto_op.md. After Phase A1, MIR types are
walkable by `core.passes._walk` and field reassignment is blocked
(though list contents remain mutable; A2 introduces tuple coercion
+ true immutability).
'''

from __future__ import annotations

import dataclasses
import inspect

import pytest

import srdatalog.ir.mir.types as mir
from srdatalog.ir.core import Op
from srdatalog.ir.hir.types import Version


def _walk(op):
  '''Pre-order Op walker. Inlined here because the production
  framework's `_walk` (in core/passes.py) lands in a downstream PR.
  This minimal version proves MIR types are walkable per Phase A's
  contract: dataclass fields containing Op or list/tuple-of-Op are
  the recursion children.'''
  if not isinstance(op, Op):
    return
  yield op
  for f in dataclasses.fields(op):
    v = getattr(op, f.name)
    yield from _walk_value(v)


def _walk_value(v):
  if isinstance(v, Op):
    yield from _walk(v)
  elif isinstance(v, (list, tuple)):
    for x in v:
      yield from _walk_value(x)


def _concrete_mir_op_classes() -> list[type]:
  '''Every concrete @dataclass class defined in mir.types — i.e. every
  MIR op subclass that callers can construct.'''
  out: list[type] = []
  for _, obj in inspect.getmembers(mir, inspect.isclass):
    if obj is Op or obj is mir.Op:
      continue
    if obj.__module__ != mir.__name__:
      continue
    if not dataclasses.is_dataclass(obj):
      continue
    out.append(obj)
  return out


# -----------------------------------------------------------------------------
# Discipline: every MIR class is an Op subclass, frozen + slots.
# -----------------------------------------------------------------------------


@pytest.mark.parametrize('cls', _concrete_mir_op_classes())
def test_mir_class_is_op_subclass(cls: type) -> None:
  assert issubclass(cls, Op), f'{cls.__name__} must subclass core.Op'


@pytest.mark.parametrize('cls', _concrete_mir_op_classes())
def test_mir_class_is_frozen(cls: type) -> None:
  params = cls.__dataclass_params__  # type: ignore[attr-defined]
  assert params.frozen, f'{cls.__name__} must be @dataclass(frozen=True)'


@pytest.mark.parametrize('cls', _concrete_mir_op_classes())
def test_mir_class_is_slotted(cls: type) -> None:
  assert hasattr(cls, '__slots__'), (
    f'{cls.__name__} must be @dataclass(slots=True) — needs __slots__'
  )


# -----------------------------------------------------------------------------
# Discipline: ExecutePipeline gains the open `pragmas` field.
# -----------------------------------------------------------------------------


def test_execute_pipeline_has_pragmas_field() -> None:
  fields = {f.name: f.type for f in dataclasses.fields(mir.ExecutePipeline)}
  assert 'pragmas' in fields, 'ExecutePipeline must declare a `pragmas` field'


def test_execute_pipeline_pragmas_default_is_empty_tuple() -> None:
  ep = mir.ExecutePipeline(pipeline=[], source_specs=[], dest_specs=[])
  assert ep.pragmas == (), f'default pragmas must be empty tuple, got {ep.pragmas!r}'


def test_execute_pipeline_pragmas_accepts_open_keys() -> None:
  ep = mir.ExecutePipeline(
    pipeline=[],
    source_specs=[],
    dest_specs=[],
    pragmas=(('dedup_hash', True), ('my_external_pragma', {'k': 1})),
  )
  assert dict(ep.pragmas) == {'dedup_hash': True, 'my_external_pragma': {'k': 1}}


# -----------------------------------------------------------------------------
# Discipline: framework `_walk` walks MIR.
# -----------------------------------------------------------------------------


def test_walk_descends_into_mir_program() -> None:
  '''A simple Program → walk yields all descendant Op-typed nodes.'''
  cs = mir.ColumnSource(rel_name='Edge', version=Version.FULL, index=[0, 1])
  scan = mir.Scan(vars=['x', 'y'], rel_name='Edge', version=Version.FULL, index=[0, 1])
  ins = mir.InsertInto(rel_name='Path', version=Version.NEW, vars=['x', 'y'], index=[0, 1])
  ep = mir.ExecutePipeline(pipeline=[scan, ins], source_specs=[cs], dest_specs=[ins])
  prog = mir.Program(steps=[(ep, False)])

  walked = list(_walk(prog))
  assert prog in walked
  assert ep in walked
  assert scan in walked
  assert ins in walked
  # ColumnSource is referenced via source_specs (a list field) — _walk
  # follows list fields too.
  assert cs in walked


# -----------------------------------------------------------------------------
# Discipline: frozen blocks field reassignment.
# -----------------------------------------------------------------------------


def test_field_reassignment_blocked() -> None:
  scan = mir.Scan(vars=['x'], rel_name='R', version=Version.FULL, index=[0])
  with pytest.raises(dataclasses.FrozenInstanceError):
    scan.handle_start = 99


def test_pragmas_field_reassignment_blocked() -> None:
  ep = mir.ExecutePipeline(pipeline=[], source_specs=[], dest_specs=[])
  with pytest.raises(dataclasses.FrozenInstanceError):
    ep.pragmas = (('x', 1),)
