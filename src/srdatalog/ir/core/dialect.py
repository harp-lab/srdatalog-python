'''Dialect base + Compiler registry.

A `Dialect` is a coherent set of types, ops, lowerings, rewrites, and
a verifier. Dialects register with a `Compiler` at init time, which
indexes them by name and exposes lookups for the pass driver.

The registry has no central enum of dialects — Property P1. Adding a
new dialect = constructing a `Dialect` and calling `register_dialect`.
No edits to this module.

See docs/ir_lowering_semantics.md, section 19.
'''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Dialect:
  '''A registered dialect.

  Fields:
    name      — dialect identifier (e.g. "relation.sorted_array").
    types     — Type subclasses owned by this dialect.
    ops       — Op subclasses owned by this dialect.
    lowerings — Lowering rules emitted *out* of this dialect.
    rewrites  — Rewrite rules *within* this dialect.
    verifier  — optional callable that validates the dialect's IR
                shape; returns a list of VerificationError on failure.
  '''

  name: str
  types: list[type] = field(default_factory=list)
  ops: list[type] = field(default_factory=list)
  lowerings: list[Any] = field(default_factory=list)
  rewrites: list[Any] = field(default_factory=list)
  verifier: Callable[[Any], list[Any]] | None = None


class Compiler:
  '''Holds the registered dialects.

  The registry is the single source of truth for what dialects exist.
  Lookups happen by name; cross-dialect lowerings are resolved by
  matching the source op kind against each dialect's `lowerings` list.
  '''

  def __init__(self) -> None:
    self._dialects: dict[str, Dialect] = {}

  def register_dialect(self, d: Dialect) -> None:
    '''Register a dialect. Raises ValueError if `d.name` is already taken.'''
    if d.name in self._dialects:
      raise ValueError(f'Dialect {d.name!r} already registered')
    self._dialects[d.name] = d

  def get_dialect(self, name: str) -> Dialect:
    '''Look up a registered dialect by name. Raises KeyError if missing.'''
    return self._dialects[name]

  @property
  def dialects(self) -> list[Dialect]:
    '''All registered dialects, in registration order.'''
    return list(self._dialects.values())

  def run(self, prog: Any, *, pipeline: list[Any]) -> Any:
    '''Drive a list of `Pass` instances over `prog`. Returns the
    (possibly transformed) prog.

    Pre-flight: validate pipeline ordering. For each `Pass`, check
    that all `consumes` dialects are either registered with this
    Compiler OR produced by an earlier Pass in the list. Raises
    `PassOrderingError` on mismatch — at construction time, before
    any pass executes.

    Per `docs/compiler_redesign.md` §4 and the R4 research report:
    pipelines are data; ordering errors are caught up-front.
    '''
    # Local import to avoid a circular at module-load time
    # (passes.py imports Compiler from this module).
    from srdatalog.ir.core.passes import PassOrderingError

    available: set[str] = {d.name for d in self.dialects}
    for i, p in enumerate(pipeline):
      for needed in p.consumes:
        if needed not in available:
          raise PassOrderingError(p.name, needed, i)
      available |= set(p.produces)

    for p in pipeline:
      prog = p.apply(prog, self)
    return prog
