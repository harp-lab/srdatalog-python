'''PragmaPlugin schema — the atomic pragma-as-plugin contract.

PR-P0 / spec § 3.2.1.3: a pragma is packaged as one `PragmaPlugin`
that bundles every artifact the pragma needs — its `Pragma` subclass,
any new IR ops, any new typed attributes, the passes that materialize
+ optimize it, the lowerings + per-target renders. The whole bundle
registers atomically with one `compiler.register_pragma_plugin(...)`
call; partial state never leaks.

This is the ACID-test fix (spec § 1): "adding a new pragma = new
plugin module, ZERO edits to existing source files." Today's
`@pragma_handler` cannot ship new ops, cannot bundle lowerings, and
cannot declare dependencies — see the five "what
`@pragma_handler` CANNOT do" bullets in spec § 3.2.1.3.

Schema sketch (spec § 3.2.1.3):

    @final
    @dataclass(frozen=True, slots=True)
    class BlockGroupPlugin(PragmaPlugin):
      pragma_cls:        type[Pragma]                = BlockGroupPragma
      new_ops:           tuple[type[Op], ...]        = (BgRoot, ...)
      new_attributes:    tuple[type[Attribute], ...] = (BgPlacementAttr,)
      passes:            tuple[Pass, ...]            = (MaterializeBlockGroup(),)
      lowerings:         tuple[Lowering, ...]        = (LowerBgRoot(), ...)
      renders:           dict[str, tuple[Render, ...]] = field(default_factory=lambda: {
        'cuda': (RenderBgRootCuda(), ...),
      })
      requires_services: tuple[type, ...]            = (NameGen, ViewLayout)
      produces_ops:      tuple[type[Op], ...]        = (BgRoot, ...)
      consumes_ops:      tuple[type[Op], ...]        = (mir.ExecutePipeline,)
      preserves:         tuple[type, ...]            = (semi_naive_safety, pragma_compat)

The `register_pragma_plugin(plugin)` API on `Compiler` (spec § 6.0.0)
atomically unpacks all of these into the per-Compiler registries;
conflict detection on op-name, attribute-name, render-double-
registration, pass-cycle, and missing required service is part of
that path. See `dialect.py` for the unpack logic.

Module discipline: per D6, this file lives in `core/` and imports no
concrete dialect; the plugin schema is dialect-agnostic. The
schema's fields are typed-shells (`type[Op]`, `type[Attribute]`,
`type[Pragma]`) — every concrete subclass lives in its owning plugin
module.
'''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, final

from srdatalog.ir.core.attribute import Attribute
from srdatalog.ir.core.ops import Op
from srdatalog.ir.core.passes import Lowering, Pass
from srdatalog.ir.core.pragma import Pragma


@dataclass(frozen=True)
class Render:
  '''A per-target render registration packaged inside a `PragmaPlugin`.

  A render produces target-specific output text (CUDA C++ today; TBB
  / SYCL / ... post-Phase B2) for one Op type. Bundled into a plugin
  so the plugin owns ALL of its renders — adding a new render does
  not edit any framework file.

  Fields:
    op_type — Op subclass this render handles (one render per
              (op_type, target) pair; conflict detection in
              `register_pragma_plugin` enforces uniqueness).
    fn      — render callable. Signature is target-specific (CUDA
              today uses `(op, EmitCtx) -> str`; per-target details
              live in the target's own contract). Framework-side
              this is `Callable[..., Any]`.
    mode    — render dispatch mode. CUDA distinguishes 'stmt' vs
              'expr'; other targets may ignore. Defaults to 'stmt'
              for back-compat with the existing CUDA registry.

  Per spec § 3.2.1.3, the `renders` field on `PragmaPlugin` is a
  `dict[str, tuple[Render, ...]]` keyed on target name; one
  `Render` is one (op_type, mode) registration for that target.
  '''

  op_type: type
  fn: Callable[..., Any]
  mode: str = 'stmt'


@dataclass(frozen=True)
class PragmaPlugin:
  '''A pragma packaged as a compiler plugin.

  Bundles every artifact a pragma needs into one atomic registration.
  See spec § 3.2.1.3 for the full contract and rationale.

  Construct directly:

      plugin = PragmaPlugin(
        pragma_cls=MyPragma,
        new_ops=(MyRoot, MyScope),
        passes=(MaterializeMy(),),
        lowerings=(LowerMyRoot(),),
        renders={'cuda': (Render(MyRoot, render_my_root_cuda),)},
        produces_ops=(MyRoot, MyScope),
        consumes_ops=(MirExecutePipeline,),
      )
      compiler.register_pragma_plugin(plugin)

  Or build subclasses with `@final @dataclass(frozen=True, slots=True)`
  (matching the same discipline as `Pragma` / `Op` — pure data).
  Subclasses should use `field(default=...)` or `field(default_factory=...)`
  for any pre-populated field, exactly mirroring the spec § 3.2.1.3
  example.

  Atomicity: `Compiler.register_pragma_plugin(plugin)` is all-or-nothing.
  Conflicts detected up front (op-name collision, attribute-name
  collision, render double-registration, pass-cycle, missing required
  service) abort the registration with NO partial state in the per-
  Compiler registries. See `dialect.py:register_pragma_plugin`.

  Discipline:
    - `slots=True` is INTENTIONALLY omitted on `PragmaPlugin` itself
      (not on its subclasses, which should still use it). The base
      uses field defaults heavily; slots+inheritance with mixed
      defaults pushes Python's dataclass machinery into edge cases
      that don't pay back in this code path (the schema is constructed
      at plugin-load time, not in a hot loop).
    - frozen=True so a registered plugin object is immutable after
      handoff; passes that capture a plugin reference cannot mutate
      its bundle.
  '''

  pragma_cls: type[Pragma]
  new_ops: tuple[type[Op], ...] = ()
  new_attributes: tuple[type[Attribute], ...] = ()
  passes: tuple[Pass, ...] = ()
  lowerings: tuple[Lowering, ...] = ()
  renders: dict[str, tuple[Render, ...]] = field(default_factory=dict)
  requires_services: tuple[type, ...] = ()
  produces_ops: tuple[type[Op], ...] = ()
  consumes_ops: tuple[type[Op], ...] = ()
  preserves: tuple[type, ...] = ()


def pragma_plugin(
  pragma_cls: type[Pragma],
  *,
  new_ops: tuple[type[Op], ...] = (),
  new_attributes: tuple[type[Attribute], ...] = (),
  passes: tuple[Pass, ...] = (),
  lowerings: tuple[Lowering, ...] = (),
  renders: dict[str, tuple[Render, ...]] | None = None,
  requires_services: tuple[type, ...] = (),
  produces_ops: tuple[type[Op], ...] = (),
  consumes_ops: tuple[type[Op], ...] = (),
  preserves: tuple[type, ...] = (),
) -> Callable[[type], PragmaPlugin]:
  '''Class-decorator factory: build a `PragmaPlugin` from a class body.

  Sugar over the bare `PragmaPlugin(pragma_cls=..., ...)` constructor.
  Symmetric with `@pragma_handler` / `@lowering` / `@register_render`
  — the decorator IS the registration shape (Triton-style; see
  `feedback_decorator_registries.md`).

  Usage:

      @pragma_plugin(
        BlockGroupPragma,
        new_ops=(BgRoot, BgScope, BgScheduler),
        passes=(MaterializeBlockGroup(),),
        produces_ops=(BgRoot, BgScope, BgScheduler),
        consumes_ops=(MirExecutePipeline,),
      )
      class BlockGroupPlugin:
        """Doc body lives here. The decorator throws the class away
        and returns a PragmaPlugin instance bound to its name."""

  The decorated object's class body is discarded — only its name is
  used (for diagnostics). For richer subclassing (where the plugin
  has methods or pre-populated typed fields), construct via
  `@final @dataclass(frozen=True, slots=True)` subclassing of
  `PragmaPlugin` directly, per the spec § 3.2.1.3 example.

  The bare-constructor form (`PragmaPlugin(pragma_cls=...)`) MUST
  also work; this decorator is purely optional sugar.
  '''

  def _wrap(_cls: type) -> PragmaPlugin:
    return PragmaPlugin(
      pragma_cls=pragma_cls,
      new_ops=new_ops,
      new_attributes=new_attributes,
      passes=passes,
      lowerings=lowerings,
      renders=renders if renders is not None else {},
      requires_services=requires_services,
      produces_ops=produces_ops,
      consumes_ops=consumes_ops,
      preserves=preserves,
    )

  return _wrap


class PragmaPluginConflictError(ValueError):
  '''Raised by `Compiler.register_pragma_plugin(plugin)` when the
  plugin's contributions conflict with already-registered state.

  Subclasses of this error name the specific conflict kind for
  programmatic recovery; the base class is what callers normally
  `except` against.

  Per spec § 3.2.1.3: registration is atomic. On conflict, the
  per-Compiler registries are left UNCHANGED — no half-loaded plugin
  state. See `dialect.py:register_pragma_plugin` for the
  try-everything-then-commit transaction shape.
  '''


@final
class OpNameCollisionError(PragmaPluginConflictError):
  '''A new op in `plugin.new_ops` has the same `__name__` as an op
  already registered with the Compiler.'''


@final
class AttributeNameCollisionError(PragmaPluginConflictError):
  '''A new attribute in `plugin.new_attributes` has the same
  `__name__` as an attribute already registered with the Compiler.'''


@final
class RenderDoubleRegistrationError(PragmaPluginConflictError):
  '''A `(op_type, target)` pair in `plugin.renders` already has a
  render registered with the Compiler.'''


@final
class MissingRequiredServiceError(PragmaPluginConflictError):
  '''A type in `plugin.requires_services` is not registered with the
  Compiler's `Services` handle.'''


__all__ = [
  'AttributeNameCollisionError',
  'MissingRequiredServiceError',
  'OpNameCollisionError',
  'PragmaPlugin',
  'PragmaPluginConflictError',
  'Render',
  'RenderDoubleRegistrationError',
  'pragma_plugin',
]
