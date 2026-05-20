'''PR-P0 tests — pragma plugin framework primitives.

Covers the seven new primitives delivered by PR-P0
(`docs/phase_decomposition_redesign.md` § 6.0.0):

  1. `PragmaPlugin` construction (positional + decorator factory).
  2. `Services.register / get / try_get` (typed retrieval).
  3. `AttributeDict.__getitem__ / __setitem__ / __contains__ / get`.
  4. `Compiler.register_pragma_plugin` atomicity (build a plugin
     with a conflict; assert no partial state).
  5. `topo_sort_passes` cycle detection.
  6. `_compiler_registration_scope` thread-locality (decorators
     stage into compiler when scope active).
  7. Back-compat: legacy `@pragma_handler` still works (minimal
     pragma + handler + run through `MaterializePragmaPass`).

These tests live OUTSIDE `tests/test_core_pragma.py` because the
existing file pins the back-compat surface; PR-P0 introduces new
primitives without disturbing it.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import pytest

from srdatalog.ir.core import (
  Attribute,
  AttributeDict,
  AttributeNameCollisionError,
  Compiler,
  MaterializePragmaPass,
  MissingRequiredServiceError,
  Op,
  OpNameCollisionError,
  Pass,
  PassCycleError,
  Pragma,
  PragmaPlugin,
  Render,
  RenderDoubleRegistrationError,
  ServiceConflictError,
  ServiceMissingError,
  Services,
  pragma_handler,
  pragma_plugin,
  topo_sort_passes,
)
from srdatalog.ir.core.dialect import _compiler_registration_scope, _get_current_compiler
from srdatalog.ir.core.pragma import PragmaRegistration

# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def isolated_pragma_registry(monkeypatch):
  '''Swap the module-global `_PRAGMA_REGISTRY` for an empty list so
  each test's decorations stay isolated. Mirrors the existing
  `isolated_registry` fixture in `tests/test_core_pragma.py`.
  '''
  fresh: list[PragmaRegistration] = []
  monkeypatch.setattr('srdatalog.ir.core.pragma._PRAGMA_REGISTRY', fresh)
  return fresh


# A toy Pragma + ops + attributes used across multiple tests.


@final
@dataclass(frozen=True, slots=True)
class _ToyPragma(Pragma):
  capacity: int = 10


@final
@dataclass(frozen=True, slots=True)
class _ToyOp(Op):
  name: str = 'toy'


@final
@dataclass(frozen=True, slots=True)
class _OtherToyOp(Op):
  value: int = 0


@final
@dataclass(frozen=True, slots=True)
class _ToyAttr(Attribute):
  threshold: float = 0.5


# A bare service type used to exercise the Services dict.


class _NameGenLike:
  def __init__(self) -> None:
    self.counter = 0

  def fresh(self, prefix: str) -> str:
    self.counter += 1
    return f'{prefix}_{self.counter}'


class _ViewLayoutLike:
  pass


# =============================================================================
# 1. PragmaPlugin construction
# =============================================================================


def test_pragma_plugin_constructs_with_defaults():
  '''Bare `PragmaPlugin(pragma_cls=...)` works — every other field
  defaults to () / empty dict per spec § 3.2.1.3.'''
  plugin = PragmaPlugin(pragma_cls=_ToyPragma)
  assert plugin.pragma_cls is _ToyPragma
  assert plugin.new_ops == ()
  assert plugin.new_attributes == ()
  assert plugin.passes == ()
  assert plugin.lowerings == ()
  assert plugin.renders == {}
  assert plugin.requires_services == ()
  assert plugin.produces_ops == ()
  assert plugin.consumes_ops == ()
  assert plugin.preserves == ()


def test_pragma_plugin_constructs_with_fields_populated():
  '''Every field on `PragmaPlugin` is settable via the constructor
  (the spec § 3.2.1.3 example uses all of them).'''
  plugin = PragmaPlugin(
    pragma_cls=_ToyPragma,
    new_ops=(_ToyOp,),
    new_attributes=(_ToyAttr,),
    requires_services=(_NameGenLike,),
    produces_ops=(_ToyOp,),
    consumes_ops=(_OtherToyOp,),
    preserves=(_NameGenLike,),
  )
  assert plugin.new_ops == (_ToyOp,)
  assert plugin.new_attributes == (_ToyAttr,)
  assert plugin.requires_services == (_NameGenLike,)
  assert plugin.produces_ops == (_ToyOp,)
  assert plugin.consumes_ops == (_OtherToyOp,)
  assert plugin.preserves == (_NameGenLike,)


def test_pragma_plugin_is_frozen():
  '''`PragmaPlugin` is a frozen dataclass; field reassignment fails.'''
  from dataclasses import FrozenInstanceError

  plugin = PragmaPlugin(pragma_cls=_ToyPragma)
  with pytest.raises(FrozenInstanceError):
    plugin.pragma_cls = _ToyPragma  # type: ignore[misc]


def test_pragma_plugin_decorator_factory_builds_a_plugin():
  '''`@pragma_plugin(...)` class-decorator factory returns a
  PragmaPlugin instance bound to the class name.'''

  @pragma_plugin(
    _ToyPragma,
    new_ops=(_ToyOp,),
    produces_ops=(_ToyOp,),
  )
  class _MyToyPlugin:
    """Sugar form: class body discarded, decorator returns the plugin."""

  assert isinstance(_MyToyPlugin, PragmaPlugin)
  assert _MyToyPlugin.pragma_cls is _ToyPragma
  assert _MyToyPlugin.new_ops == (_ToyOp,)
  assert _MyToyPlugin.produces_ops == (_ToyOp,)


def test_pragma_plugin_decorator_renders_default_is_empty_dict():
  '''The decorator factory's `renders=None` default becomes `{}` on
  the resulting plugin (not None — avoids the mutable-default trap
  for the constructor's `default_factory=dict`).'''

  @pragma_plugin(_ToyPragma)
  class _Plugin:
    """No renders contributed."""

  assert isinstance(_Plugin, PragmaPlugin)
  assert _Plugin.renders == {}


# =============================================================================
# 2. Services typed-key dict
# =============================================================================


def test_services_register_and_get():
  '''Round-trip: register an instance, look it up via its type.'''
  services = Services()
  ng = _NameGenLike()
  services.register(_NameGenLike, ng)
  assert services.get(_NameGenLike) is ng


def test_services_get_returns_the_exact_instance():
  '''No deep copy — the same object is returned.'''
  services = Services()
  vl = _ViewLayoutLike()
  services.register(_ViewLayoutLike, vl)
  assert services.get(_ViewLayoutLike) is vl


def test_services_get_raises_when_missing():
  '''Looking up an unregistered type raises ServiceMissingError
  (subclass of KeyError so legacy `except KeyError` keeps catching).'''
  services = Services()
  with pytest.raises(ServiceMissingError) as ei:
    services.get(_NameGenLike)
  assert issubclass(type(ei.value), KeyError)
  assert ei.value.service_type is _NameGenLike


def test_services_try_get_returns_none_when_missing():
  '''try_get is the nullable form; never raises.'''
  services = Services()
  assert services.try_get(_NameGenLike) is None


def test_services_try_get_returns_instance_when_present():
  services = Services()
  ng = _NameGenLike()
  services.register(_NameGenLike, ng)
  assert services.try_get(_NameGenLike) is ng


def test_services_register_rejects_double_registration():
  '''Re-registering the same TYPE raises ServiceConflictError —
  silent overwrites would mismatch downstream captured handles.'''
  services = Services()
  services.register(_NameGenLike, _NameGenLike())
  with pytest.raises(ServiceConflictError) as ei:
    services.register(_NameGenLike, _NameGenLike())
  assert ei.value.service_type is _NameGenLike


def test_services_contains():
  '''`MyService in services` returns True iff registered.'''
  services = Services()
  assert _NameGenLike not in services
  services.register(_NameGenLike, _NameGenLike())
  assert _NameGenLike in services


def test_services_typed_get_infers_concrete_type():
  '''mypy/runtime parity: get(NameGenLike) returns a NameGenLike,
  not Any. The runtime check here pairs with the typing tests in
  the module's stub: `services.get(_NameGenLike).fresh(...)`
  type-checks at mypy and works at runtime.'''
  services = Services()
  services.register(_NameGenLike, _NameGenLike())
  ng = services.get(_NameGenLike)
  # Runtime: the returned value supports the protocol of _NameGenLike
  assert ng.fresh('x') == 'x_1'


# =============================================================================
# 3. AttributeDict typed-key dict
# =============================================================================


def test_attribute_dict_set_and_get_via_indexing():
  attrs = AttributeDict()
  v = _ToyAttr(threshold=0.7)
  attrs[_ToyAttr] = v
  assert attrs[_ToyAttr] is v
  assert attrs[_ToyAttr].threshold == 0.7


def test_attribute_dict_getitem_raises_when_missing():
  attrs = AttributeDict()
  with pytest.raises(KeyError) as ei:
    _ = attrs[_ToyAttr]
  assert '_ToyAttr' in str(ei.value)


def test_attribute_dict_get_returns_none_when_missing():
  attrs = AttributeDict()
  assert attrs.get(_ToyAttr) is None


def test_attribute_dict_get_returns_value_when_present():
  attrs = AttributeDict()
  v = _ToyAttr()
  attrs[_ToyAttr] = v
  assert attrs.get(_ToyAttr) is v


def test_attribute_dict_contains():
  attrs = AttributeDict()
  assert _ToyAttr not in attrs
  attrs[_ToyAttr] = _ToyAttr()
  assert _ToyAttr in attrs


def test_attribute_dict_setitem_validates_value_type():
  '''Writing a mismatched value type fails at write time, not at
  the eventual read crash.'''

  @final
  @dataclass(frozen=True, slots=True)
  class OtherAttr(Attribute):
    pass

  attrs = AttributeDict()
  with pytest.raises(TypeError, match=r'requires an instance of'):
    attrs[_ToyAttr] = OtherAttr()  # type: ignore[assignment]


def test_attribute_dict_setitem_overwrites_silently():
  '''Repeat writes of the SAME type silently overwrite — per the
  class docstring, conflict detection is the producer's responsibility.'''
  attrs = AttributeDict()
  v1 = _ToyAttr(threshold=0.1)
  v2 = _ToyAttr(threshold=0.9)
  attrs[_ToyAttr] = v1
  attrs[_ToyAttr] = v2
  assert attrs[_ToyAttr] is v2


def test_attribute_dict_len_and_iter():
  '''Mirror dict semantics: len() and iter() over keys.'''
  attrs = AttributeDict()
  assert len(attrs) == 0
  attrs[_ToyAttr] = _ToyAttr()
  assert len(attrs) == 1
  assert list(attrs) == [_ToyAttr]


def test_attribute_dict_repr():
  attrs = AttributeDict()
  assert repr(attrs) == 'AttributeDict()'
  attrs[_ToyAttr] = _ToyAttr(threshold=0.3)
  assert '_ToyAttr' in repr(attrs)


def test_op_default_attributes_is_a_fresh_dict_per_instance():
  '''Two ops with default-factory `attributes` get independent dicts
  (the `field(default_factory=...)` is per-instance).'''

  @dataclass(frozen=True, slots=True)
  class MyOp(Op):
    name: str = 'a'

  a, b = MyOp(), MyOp()
  assert a.attributes is not b.attributes
  a.attributes[_ToyAttr] = _ToyAttr()
  assert _ToyAttr not in b.attributes


def test_op_attributes_does_not_affect_equality():
  '''Two ops with the same fields but different attribute states
  are still equal — `compare=False` on the parent's `attributes`
  field excludes it from structural identity.'''

  @dataclass(frozen=True, slots=True)
  class MyOp(Op):
    name: str = 'a'

  a, b = MyOp(), MyOp()
  a.attributes[_ToyAttr] = _ToyAttr()
  assert a == b
  assert hash(a) == hash(b)


def test_op_attributes_excluded_from_repr():
  '''`repr=False` on the parent's `attributes` field keeps repr
  byte-equivalent with the pre-PR-P0 shape.'''

  @dataclass(frozen=True, slots=True)
  class MyOp(Op):
    name: str = 'a'

  r = repr(MyOp(name='foo'))
  # The repr should NOT include the attributes= kwarg or its
  # AttributeDict marker.
  assert 'attributes=' not in r
  assert 'AttributeDict' not in r


# =============================================================================
# 4. Compiler.register_pragma_plugin atomicity
# =============================================================================


def _make_simple_plugin(pragma_cls: type[Pragma] = _ToyPragma) -> PragmaPlugin:
  '''A baseline plugin contributing one new op + one new attribute.'''
  return PragmaPlugin(
    pragma_cls=pragma_cls,
    new_ops=(_ToyOp,),
    new_attributes=(_ToyAttr,),
  )


def test_register_pragma_plugin_succeeds_on_clean_compiler():
  c = Compiler()
  plugin = _make_simple_plugin()
  c.register_pragma_plugin(plugin)
  assert '_ToyOp' in c.ops
  assert c.ops['_ToyOp'] is _ToyOp
  assert '_ToyAttr' in c.attributes
  assert plugin in c.pragma_plugins


def test_register_pragma_plugin_op_name_collision_is_atomic():
  '''Op-name collision aborts the whole registration; the second
  plugin's other contributions are NOT applied.'''

  # First plugin
  c = Compiler()
  c.register_pragma_plugin(PragmaPlugin(pragma_cls=_ToyPragma, new_ops=(_ToyOp,)))

  # Second plugin with a different Op class but same __name__.
  # Build a synthetic class with the same name.
  @dataclass(frozen=True, slots=True)
  class _Shadow(Op):
    pass

  _Shadow.__name__ = '_ToyOp'

  attrs_before = dict(c.attributes)
  ops_before = dict(c.ops)
  passes_before = list(c.passes)
  with pytest.raises(OpNameCollisionError):
    c.register_pragma_plugin(
      PragmaPlugin(
        pragma_cls=_ToyPragma,
        new_ops=(_Shadow,),
        new_attributes=(_ToyAttr,),
      )
    )
  # No state changed: attributes / passes untouched
  assert c.ops == ops_before
  assert c.attributes == attrs_before
  assert c.passes == passes_before


def test_register_pragma_plugin_attribute_name_collision_is_atomic():
  c = Compiler()
  c.register_pragma_plugin(PragmaPlugin(pragma_cls=_ToyPragma, new_attributes=(_ToyAttr,)))

  @dataclass(frozen=True, slots=True)
  class _ShadowAttr(Attribute):
    pass

  _ShadowAttr.__name__ = '_ToyAttr'

  ops_before = dict(c.ops)
  with pytest.raises(AttributeNameCollisionError):
    c.register_pragma_plugin(
      PragmaPlugin(
        pragma_cls=_ToyPragma,
        new_ops=(_ToyOp,),
        new_attributes=(_ShadowAttr,),
      )
    )
  # The op contribution from the failing plugin was NOT applied
  assert '_ToyOp' not in ops_before
  assert '_ToyOp' not in c.ops


def test_register_pragma_plugin_render_double_registration_is_atomic():
  '''Two plugins registering a render for the same (op_type, target)
  pair: second registration fails atomically.'''
  c = Compiler()

  def _render_a(op, ctx):
    return 'a'

  def _render_b(op, ctx):
    return 'b'

  c.register_pragma_plugin(
    PragmaPlugin(
      pragma_cls=_ToyPragma,
      renders={'cuda': (Render(op_type=_ToyOp, fn=_render_a),)},
    )
  )

  # Second plugin tries to render the same (op_type, target) pair.
  ops_before = dict(c.ops)
  with pytest.raises(RenderDoubleRegistrationError):
    c.register_pragma_plugin(
      PragmaPlugin(
        pragma_cls=_ToyPragma,
        new_ops=(_OtherToyOp,),
        renders={'cuda': (Render(op_type=_ToyOp, fn=_render_b),)},
      )
    )
  # `_OtherToyOp` contribution was rolled back
  assert '_OtherToyOp' not in c.ops
  assert dict(c.ops) == ops_before


def test_register_pragma_plugin_missing_required_service_is_atomic():
  c = Compiler()
  ops_before = dict(c.ops)
  with pytest.raises(MissingRequiredServiceError) as ei:
    c.register_pragma_plugin(
      PragmaPlugin(
        pragma_cls=_ToyPragma,
        new_ops=(_ToyOp,),
        requires_services=(_NameGenLike,),
      )
    )
  assert '_NameGenLike' in str(ei.value)
  # No op contribution from the failing plugin
  assert c.ops == ops_before


def test_register_pragma_plugin_satisfied_service_succeeds():
  c = Compiler()
  c.services.register(_NameGenLike, _NameGenLike())
  c.register_pragma_plugin(
    PragmaPlugin(
      pragma_cls=_ToyPragma,
      new_ops=(_ToyOp,),
      requires_services=(_NameGenLike,),
    )
  )
  assert '_ToyOp' in c.ops


def test_register_pragma_plugin_pass_cycle_is_atomic():
  '''Two passes whose produces/consumes form a cycle: registration
  fails, neither pass is added to `compiler.passes`.'''

  @dataclass(frozen=True)
  class _PA(Pass):
    name: str = 'pa'
    produces_ops: tuple[type, ...] = (_ToyOp,)
    consumes_ops: tuple[type, ...] = (_OtherToyOp,)

    def apply(self, prog, compiler):
      return prog

  @dataclass(frozen=True)
  class _PB(Pass):
    name: str = 'pb'
    produces_ops: tuple[type, ...] = (_OtherToyOp,)
    consumes_ops: tuple[type, ...] = (_ToyOp,)

    def apply(self, prog, compiler):
      return prog

  c = Compiler()
  passes_before = list(c.passes)
  with pytest.raises(PassCycleError):
    c.register_pragma_plugin(
      PragmaPlugin(
        pragma_cls=_ToyPragma,
        passes=(_PA(), _PB()),
      )
    )
  # No partial pass state
  assert c.passes == passes_before


# =============================================================================
# 5. topo_sort_passes cycle detection
# =============================================================================


def test_topo_sort_passes_returns_input_when_no_deps():
  '''An empty list of passes returns an empty list. Independent
  passes preserve a stable order keyed on name.'''

  @dataclass(frozen=True)
  class _P(Pass):
    name: str = 'p'

    def apply(self, prog, compiler):
      return prog

  assert topo_sort_passes([]) == []
  out = topo_sort_passes([_P(name='b'), _P(name='a')])
  assert [p.name for p in out] == ['a', 'b']


def test_topo_sort_passes_respects_produces_consumes():
  '''A pass that produces an op must run before any pass that
  consumes it.'''

  @dataclass(frozen=True)
  class _Producer(Pass):
    name: str = 'producer'
    produces_ops: tuple[type, ...] = (_ToyOp,)

    def apply(self, prog, compiler):
      return prog

  @dataclass(frozen=True)
  class _Consumer(Pass):
    name: str = 'consumer'
    consumes_ops: tuple[type, ...] = (_ToyOp,)

    def apply(self, prog, compiler):
      return prog

  out = topo_sort_passes([_Consumer(), _Producer()])
  assert [p.name for p in out] == ['producer', 'consumer']


def test_topo_sort_passes_self_edge_tolerated():
  '''A pass that both produces and consumes the same op type is
  allowed (the materialization passes do this); no cycle is raised.'''

  @dataclass(frozen=True)
  class _Materializer(Pass):
    name: str = 'mat'
    produces_ops: tuple[type, ...] = (_ToyOp,)
    consumes_ops: tuple[type, ...] = (_ToyOp,)

    def apply(self, prog, compiler):
      return prog

  out = topo_sort_passes([_Materializer()])
  assert len(out) == 1


def test_topo_sort_passes_raises_on_cycle():
  '''An A->B->A cycle raises PassCycleError naming both passes.'''

  @dataclass(frozen=True)
  class _A(Pass):
    name: str = 'a'
    produces_ops: tuple[type, ...] = (_ToyOp,)
    consumes_ops: tuple[type, ...] = (_OtherToyOp,)

    def apply(self, prog, compiler):
      return prog

  @dataclass(frozen=True)
  class _B(Pass):
    name: str = 'b'
    produces_ops: tuple[type, ...] = (_OtherToyOp,)
    consumes_ops: tuple[type, ...] = (_ToyOp,)

    def apply(self, prog, compiler):
      return prog

  with pytest.raises(PassCycleError) as ei:
    topo_sort_passes([_A(), _B()])
  assert set(ei.value.cycle) == {'a', 'b'}


# =============================================================================
# 6. _compiler_registration_scope thread-locality
# =============================================================================


def test_compiler_registration_scope_sets_thread_local():
  c = Compiler()
  assert _get_current_compiler() is None
  with _compiler_registration_scope(c):
    assert _get_current_compiler() is c
  assert _get_current_compiler() is None


def test_compiler_registration_scope_nests_and_restores():
  c1 = Compiler()
  c2 = Compiler()
  with _compiler_registration_scope(c1):
    assert _get_current_compiler() is c1
    with _compiler_registration_scope(c2):
      assert _get_current_compiler() is c2
    assert _get_current_compiler() is c1
  assert _get_current_compiler() is None


def test_compiler_registration_scope_restores_on_exception():
  c = Compiler()
  with pytest.raises(RuntimeError), _compiler_registration_scope(c):
    assert _get_current_compiler() is c
    raise RuntimeError('boom')
  assert _get_current_compiler() is None


def test_pragma_handler_stages_into_compiler_when_scope_active(isolated_pragma_registry):
  '''When a `_compiler_registration_scope(c)` is active, a
  `@pragma_handler` decoration stages its registration into the
  compiler's per-instance handler list — NOT the module-global.'''

  c = Compiler()

  with _compiler_registration_scope(c):

    @pragma_handler(_ToyPragma, on=_ToyOp)
    def _handler(op, pragma, ctx):
      return op

  # Module-global registry stayed empty
  assert isolated_pragma_registry == []
  # Per-Compiler list got the registration
  staged: list[PragmaRegistration] = getattr(c, '_pragma_handlers', [])
  assert len(staged) == 1
  assert staged[0].pragma_cls is _ToyPragma


def test_pragma_handler_falls_back_to_module_global_without_scope(isolated_pragma_registry):
  '''Outside a registration scope, `@pragma_handler` keeps writing
  to the module-global (back-compat with pre-PR-P0 code paths).'''

  @pragma_handler(_ToyPragma, on=_ToyOp)
  def _handler(op, pragma, ctx):
    return op

  assert len(isolated_pragma_registry) == 1
  assert isolated_pragma_registry[0].pragma_cls is _ToyPragma


# =============================================================================
# 7. Back-compat: legacy @pragma_handler + MaterializePragmaPass + Compiler.run
# =============================================================================


def test_materialize_pragma_pass_runs_handler_via_compiler_run(isolated_pragma_registry):
  '''End-to-end back-compat: a `@pragma_handler` registered inside a
  compiler scope, then a `MaterializePragmaPass(_ToyPragma)` Pass
  inserted into `compiler.passes`, then `compiler.run(prog)` invokes
  the handler against matching ops in `prog`.

  This validates spec § 6.0.0 row 8 (the legacy `@pragma_handler`
  back-compat shim).
  '''

  # An op carrying pragmas (mimics mir.ExecutePipeline shape).
  @final
  @dataclass(frozen=True, slots=True)
  class _Carrier(Op):
    name: str = ''
    pragmas: tuple[Pragma, ...] = ()

  # Handler returns a NEW _Carrier with `name='materialized'`.
  call_log: list[str] = []

  c = Compiler()
  with _compiler_registration_scope(c):

    @pragma_handler(_ToyPragma, on=_Carrier)
    def _materialize(op, pragma, ctx):
      call_log.append(f'fired:{op.name}')
      return _Carrier(name='materialized', pragmas=())

  # Build the pipeline using the MaterializePragmaPass shim.
  c.passes.append(MaterializePragmaPass(name='mat', pragma_cls=_ToyPragma))

  # Run against a Carrier op carrying the toy pragma.
  prog = _Carrier(name='before', pragmas=(_ToyPragma(),))
  result = c.run(prog)

  assert call_log == ['fired:before']
  assert isinstance(result, _Carrier)
  assert result.name == 'materialized'


def test_materialize_pragma_pass_is_noop_when_no_pragma_matches(isolated_pragma_registry):
  '''Carrier op without the matching pragma → handler never fires;
  op returned unchanged.'''

  @final
  @dataclass(frozen=True, slots=True)
  class _Carrier(Op):
    pragmas: tuple[Pragma, ...] = ()

  call_log: list[str] = []

  c = Compiler()
  with _compiler_registration_scope(c):

    @pragma_handler(_ToyPragma, on=_Carrier)
    def _materialize(op, pragma, ctx):
      call_log.append('fired')
      return op

  c.passes.append(MaterializePragmaPass(name='mat', pragma_cls=_ToyPragma))
  prog = _Carrier(pragmas=())
  result = c.run(prog)
  assert call_log == []
  assert result is prog


def test_compiler_run_uses_self_passes_when_no_pipeline_given():
  '''PR-P0 (spec § 6 row 6): `compiler.run(prog)` without
  `pipeline=...` consults `self.passes` directly. Existing
  `compiler.run(prog, pipeline=[...])` continues to work
  (back-compat).
  '''

  @dataclass(frozen=True)
  class _Identity(Pass):
    name: str = 'identity'

    def apply(self, prog, compiler):
      return prog

  c = Compiler()
  c.passes.append(_Identity())

  out = c.run('hello')
  assert out == 'hello'

  out2 = c.run('explicit', pipeline=[_Identity()])
  assert out2 == 'explicit'


# =============================================================================
# 8. Per-Compiler registries are independent across instances
# =============================================================================


def test_per_compiler_registries_do_not_leak_across_instances():
  '''Risk-4 fix: registries are per-Compiler. A plugin registered on
  c1 does NOT show up in c2.'''
  c1 = Compiler()
  c2 = Compiler()
  c1.register_pragma_plugin(PragmaPlugin(pragma_cls=_ToyPragma, new_ops=(_ToyOp,)))
  assert '_ToyOp' in c1.ops
  assert '_ToyOp' not in c2.ops


# =============================================================================
# 9. Sanity: PragmaPlugin __all__ surface
# =============================================================================


def test_pragma_plugin_imports_are_public():
  '''Every NEW PR-P0 primitive must be importable from the public
  package surface (`srdatalog.ir.core.__all__`).'''
  from srdatalog.ir.core import __all__ as core_all

  expected = {
    'Attribute',
    'AttributeDict',
    'MaterializePragmaPass',
    'PassCycleError',
    'PragmaPlugin',
    'PragmaPluginConflictError',
    'Render',
    'Services',
    'pragma_plugin',
    'topo_sort_passes',
  }
  missing = expected - set(core_all)
  assert not missing, f'Missing from srdatalog.ir.core.__all__: {missing}'
