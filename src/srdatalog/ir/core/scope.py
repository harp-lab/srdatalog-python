'''Scope — per-op-family lexical context, passed explicitly to
lowerings that need it.

Per `docs/phase_zero_prerequisites.md` §3.3 and decision #2: each
concrete `Scope` subclass has at most 8 fields. A discipline test
enforces this cap (the D16-style rule). Without the cap, `Scope`
becomes the new god-object — defeating the whole point of shrinking
`LowerCtx`.

The base class is empty (a marker). Subclasses live alongside the ops
they serve — typically per dialect family. Phase B introduces
concrete ones (`JoinScope`, `FilterScope`, etc.); F3 just lays the
base.
'''

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Scope:
  '''Marker base for per-family lexical scope.

  Subclasses MUST be ``@final + @dataclass(frozen=True, slots=True)``
  and have at most 8 fields (the D16-style cap). Concrete subclasses
  land in the dialects/lowerings that need them.
  '''


@dataclass(frozen=True, slots=True)
class EmptyScope(Scope):
  '''Convenience: explicit "no scope information" — used as the
  initial argument from the `LoweringPass` dispatcher root call.
  '''


__all__ = [
  'EmptyScope',
  'Scope',
]
