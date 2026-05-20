'''Typed-key Services dict — Risk 1 fix from spec § 3.2.1.2.

`Services` is a per-`Compiler.run` typed-key dict of injected services.
Adding a service = new TYPE, never a new attribute on this class.

The shape is the MLIR/LLVM AnalysisManager / React ContextProvider
pattern (spec § 11.2 / § 11.4): services are injected, not accumulated
into a fixed-schema bag. A fresh-name generator, a view-layout
provider, a per-`Compiler` plugin registry — each is registered under
its own TYPE key. Type the public methods so `services.get(NameGen)`
infers `NameGen` at the call site.

Per spec § 3.2.1.2:

    Risk 1: `Services` becomes a named-field carrier.
            services.name_gen, services.compiler, ... — each new
            service edits the `Services` class. Same fixed-schema
            anti-pattern.
    Fix:    `Services` is a typed-key dict: `services.get[T](T) → T`.
            Adding a service = new type, no edit.

Usage:

    from srdatalog.ir.core.lower_ctx import NameGen
    services = Services()
    services.register(NameGen, NameGen())

    ng = services.get(NameGen)  # mypy infers NameGen
    maybe_ng = services.try_get(NameGen)  # mypy infers NameGen | None

This module is part of `core/`; per D6 it imports no concrete dialect
and no concrete service type. The Services class is the typed shell;
each service is its own type, owned by whoever ships it.
'''

from __future__ import annotations

from typing import Any, TypeVar

# TypeVar over service classes. PEP 695 syntax (`def get[T](...)`) is
# Python 3.12+; this project's `requires-python = ">=3.10"` mandates
# the classic TypeVar form for now. The public surface still types
# correctly under mypy --strict for callers.
_T = TypeVar('_T')


class ServiceMissingError(KeyError):
  '''Raised by `Services.get(T)` when no instance of T is registered.

  Subclass of `KeyError` so existing `except KeyError` clauses keep
  catching it; carries the requested type for diagnostics.
  '''

  def __init__(self, service_type: type) -> None:
    self.service_type = service_type
    super().__init__(
      f'no service of type {service_type.__name__!r} registered; '
      f'register one via `services.register({service_type.__name__}, '
      f'<instance>)` before this call. See spec § 3.2.1.2 Risk 1.'
    )


class ServiceConflictError(ValueError):
  '''Raised by `Services.register(T, instance)` when T is already
  registered.

  Re-registering would silently overwrite the prior instance — a
  surprise for downstream passes that already captured a handle. The
  conflict is loud at registration time; if a re-register is
  intentional, the caller must explicitly `services.replace(T, ...)`
  (NOT yet implemented; YAGNI until a real caller needs it).
  '''

  def __init__(self, service_type: type) -> None:
    self.service_type = service_type
    super().__init__(
      f'service of type {service_type.__name__!r} is already registered; '
      f'register() does not allow silent overwrite. See spec § 3.2.1.2 Risk 1.'
    )


class Services:
  '''Per-`Compiler.run` typed-key dict of injected services.

  No named-field attributes. Internally a `dict[type, Any]`. The
  public methods are typed so mypy infers `services.get(NameGen)` ->
  `NameGen` at the call site, NOT `Any`.

  Spec: docs/phase_decomposition_redesign.md § 3.2.1.2 (Risk 1 fix)
  and § 3.2.1.3 (consumed by `PragmaPlugin.requires_services` for
  conflict detection at plugin-registration time).

  Discipline:
    - NEVER add a typed attribute to this class. The whole point is
      that the API surface is fixed; new services live in their own
      types (Risk 1).
    - The internal storage is `_by_type: dict[type, Any]`. The
      method-level type vars keep callers type-safe without leaking
      the storage shape.
  '''

  __slots__ = ('_by_type',)

  def __init__(self) -> None:
    self._by_type: dict[type, Any] = {}

  def register(self, service_type: type[_T], instance: _T) -> None:
    '''Register `instance` under the key `service_type`.

    Raises `ServiceConflictError` if `service_type` is already
    registered. Re-registration is loud by design — see the
    `ServiceConflictError` docstring.
    '''
    if service_type in self._by_type:
      raise ServiceConflictError(service_type)
    self._by_type[service_type] = instance

  def get(self, service_type: type[_T]) -> _T:
    '''Look up the instance registered under `service_type`.

    Raises `ServiceMissingError` if no instance is registered.
    Callers that prefer a None fallback use `try_get(...)` instead.
    '''
    try:
      return self._by_type[service_type]  # type: ignore[no-any-return]
    except KeyError:
      raise ServiceMissingError(service_type) from None

  def try_get(self, service_type: type[_T]) -> _T | None:
    '''Look up the instance registered under `service_type`, or None.

    Symmetric with `dict.get(key, None)`; never raises.
    '''
    return self._by_type.get(service_type)

  def __contains__(self, service_type: object) -> bool:
    '''Membership test: `MyService in services`. The argument is
    typed as `object` (not `type[_T]`) so callers can write
    `if MyService in services:` without mypy complaining about
    generic-parameter inference.
    '''
    return service_type in self._by_type


__all__ = [
  'ServiceConflictError',
  'ServiceMissingError',
  'Services',
]
