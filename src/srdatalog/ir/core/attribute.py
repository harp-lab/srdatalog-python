'''Typed Op attributes — Risks 2 + 5 fix from spec § 3.2.1.2.

Replaces "per-feature named fields on Op classes" with an MLIR-style
typed attribute dict. Adding a per-feature carrier (e.g. `ViewBinding`,
`OutputBinding`, `BgPlacementAttr`) is no longer a field-add on a
shared Op class — it's a new `Attribute` subclass packaged in a
`PragmaPlugin.new_attributes` tuple, looked up at the call site as
`op.attributes[ViewBinding]`.

Per spec § 3.2.1.2:

    Risk 2: IR op classes grow per-feature with named binding fields
            (`KernelDef.view_bindings`, `KernelDef.output_binding`,
            ...). Each feature edits `KernelDef`. Bool-field anti-
            pattern at the IR layer.
    Fix:    MLIR-style typed attribute dict on every Op:
            `op.attributes[ViewBinding]`. Pragma plugins register
            their own attribute types; Op classes don't grow.

    Risk 5: Op classes themselves carry fixed schemas (...) IS a
            fixed-schema attribute carrier at the IR layer.
    Fix:    Ops are typed shells declaring only name + region
            structure + verification. Per-feature data lives in
            `op.attributes`. (Risk 2 fix generalized.)

The `Attribute` base class is the pragma-side parallel of `Op` /
`Pragma` / `Type` (also pure data, frozen, slotted). The
`AttributeDict` storage is typed-key — each attribute kind is its own
TYPE, mirroring the `Services` shape (Risk 1 fix).

This module is part of `core/`. Per D6 it knows no concrete dialects.

See spec § 11.2 for the direct MLIR precedent.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

# Bounded TypeVar so `AttributeDict[T]` only accepts Attribute
# subclasses. PEP 695 (`class AttributeDict: def __getitem__[T: ...]`)
# is Python 3.12+; the project targets 3.10, so we use the classic
# TypeVar form with a `bound` argument. The public API still types
# correctly under mypy --strict.


@dataclass(frozen=True, slots=True)
class Attribute:
  '''Base class for typed attributes on Ops.

  Subclasses are typically `@final @dataclass(frozen=True, slots=True)`,
  per the same discipline as `Pragma` (spec § 3.2.1.3) and `Op` (D1 /
  D2 / D3 / D4 in `docs/code_discipline.md`). Each subclass declares
  the per-feature data it carries; new features ship new Attribute
  subclasses in a `PragmaPlugin.new_attributes` tuple. Op classes do
  NOT grow per-feature fields.

  Usage (from a pragma plugin):

      @final
      @dataclass(frozen=True, slots=True)
      class ViewBinding(Attribute):
        handle_idx: int
        var_name: str

      # In the producing pass:
      op.attributes[ViewBinding] = ViewBinding(handle_idx=0, var_name='v0')

      # In the consuming render:
      vb = op.attributes[ViewBinding]
      render(vb.var_name, vb.handle_idx)

  See spec § 3.2.1.2 (Risks 2 + 5).
  '''


_T = TypeVar('_T', bound=Attribute)


class AttributeDict:
  '''Typed-key dict on every Op. `attributes[T] -> T instance`.

  Storage is `dict[type[Attribute], Attribute]` internally. The
  method-level type variables surface mypy-correct retrieval at the
  call site: `op.attributes[ViewBinding]` infers `ViewBinding`, not
  `Any`.

  Mutation semantics:
    - `__setitem__` overwrites silently. Attribute dicts are typically
      populated by one producer pass; if two passes try to write the
      same Attribute type, that is a coordination bug and the second
      writer should be aware. We do NOT enforce conflict detection on
      attribute writes — that's the producer's responsibility (each
      producer pass should write its own Attribute types). See spec §
      3.2.1.3 for the plugin-contract conflict-detection that DOES
      enforce ownership at registration time.
    - `__delitem__` is intentionally not provided; per spec § 3.2.1.2
      Bucket 1, attributes ARE the cross-pass IR carrier, so deletion
      is conceptually deleting IR data. The framework doesn't need it
      yet; YAGNI.

  Discipline:
    - Adding a new attribute kind = new `Attribute` subclass, registered
      via a `PragmaPlugin.new_attributes` tuple. The `Op` class never
      grows per-feature fields — all per-feature data lives here.
    - Two pragma plugins MUST NOT register the same Attribute subclass.
      `PragmaPlugin` registration detects the conflict atomically
      (spec § 3.2.1.3); this dict trusts the registration layer.

  This object is mutable — it has to be, because passes populate it
  incrementally. The owning `Op` is still frozen via `compare=False,
  repr=False` on its `attributes` field (see `op.py`); the dict's
  identity is excluded from op equality / hashing so two ops with the
  same fields but different attribute states stay equal at the IR
  level. Cross-pass data semantics live in the attributes; equality
  semantics live in the op's frozen fields.
  '''

  __slots__ = ('_by_type',)

  def __init__(self) -> None:
    self._by_type: dict[type[Attribute], Attribute] = {}

  def __getitem__(self, attr_type: type[_T]) -> _T:
    '''Look up the Attribute instance keyed by `attr_type`.

    Raises `KeyError` (with the Attribute subclass name) if missing.
    Callers that prefer a None fallback use `.get(...)` instead.
    '''
    try:
      return self._by_type[attr_type]  # type: ignore[return-value]
    except KeyError:
      raise KeyError(
        f'no Attribute of type {attr_type.__name__!r} on this op; '
        f'call sites that may legitimately have no instance should use '
        f'`op.attributes.get({attr_type.__name__})` instead.'
      ) from None

  def __setitem__(self, attr_type: type[_T], value: _T) -> None:
    '''Store `value` under the key `attr_type`. Silent overwrite —
    see the class docstring for the conflict-discipline rationale.

    Validates that `value` is an instance of `attr_type` (cheap
    guardrail against the `op.attributes[A] = b_instance` typo —
    catches it at write time, not at the eventual read crash).
    '''
    if not isinstance(value, attr_type):
      raise TypeError(
        f'AttributeDict[{attr_type.__name__}] requires an instance of '
        f'{attr_type.__name__}; got {type(value).__name__!r}.'
      )
    self._by_type[attr_type] = value

  def __contains__(self, attr_type: object) -> bool:
    '''Membership test: `ViewBinding in op.attributes`.

    The argument is typed `object` so callers can write the natural
    `if X in op.attributes:` without mypy complaining about generic-
    parameter inference. Production lookups go through `__getitem__`
    / `get(...)`; this is the cheap "do you have one?" path.
    '''
    return attr_type in self._by_type

  def get(self, attr_type: type[_T]) -> _T | None:
    '''Look up the Attribute instance keyed by `attr_type`, or None.

    Symmetric with `dict.get(key, None)`. Never raises.
    '''
    val = self._by_type.get(attr_type)
    if val is None:
      return None
    # Defensive: a misregistered attribute under the wrong key would
    # otherwise mistype this return; surface it loudly.
    if not isinstance(val, attr_type):  # pragma: no cover (registration-time guard)
      raise TypeError(
        f'AttributeDict.get({attr_type.__name__}) found a {type(val).__name__!r} '
        f'under the {attr_type.__name__!r} key. Registration discipline broken; '
        f'see spec § 3.2.1.3 for the plugin-contract conflict-detection rules.'
      )
    return val

  def __len__(self) -> int:
    return len(self._by_type)

  def __iter__(self) -> Any:
    '''Iterate over registered Attribute TYPES. Mirrors `dict.__iter__`
    which yields keys.'''
    return iter(self._by_type)

  def __repr__(self) -> str:
    if not self._by_type:
      return 'AttributeDict()'
    inner = ', '.join(f'{k.__name__}={v!r}' for k, v in self._by_type.items())
    return f'AttributeDict({inner})'


__all__ = [
  'Attribute',
  'AttributeDict',
]
