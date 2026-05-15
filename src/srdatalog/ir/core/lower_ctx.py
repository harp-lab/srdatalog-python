'''LowerCtx — per-pass state for LoweringPass dispatchers.

Spec: `docs/phase_zero_prerequisites.md` §3.2 and
`docs/phase_b_lowering_dispatcher.md` §3.

DISCIPLINE D10: this dataclass has EXACTLY 5 fields. Adding a 6th
requires a doc amendment + owner sign-off. Anything that wants to
live on the ctx but doesn't fit one of the 5 slots:

  - lexical scope         → use a `Scope` subclass passed explicitly
                             (`docs/code_discipline.md` D16-style cap).
  - pragma flag           → materialize as MIR op insertion (Phase C).
  - per-Op-family state   → put on the Op or thread via a `Scope`
                             subclass.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from srdatalog.ir.core.dialect import Compiler


class NameGen:
  '''Counter-backed unique-name generator.

  Mutable wrapper so the enclosing `LowerCtx` stays frozen. Production
  parity: bump order matches the legacy `LoweringCtx.fresh()` so
  byte-equivalence holds. Fresh names are ``<prefix>_<n>``, with ``n``
  starting from ``start + 1``.

  Per `docs/phase_b_lowering_dispatcher.md` §3.2.
  '''

  def __init__(self, start: int = 0) -> None:
    self._counter = start

  def fresh(self, prefix: str) -> str:
    '''Bump the counter and return ``<prefix>_<n>``.'''
    self._counter += 1
    return f'{prefix}_{self._counter}'


@dataclass(frozen=True, slots=True)
class ViewLayout:
  '''Per-relation view variable names + slot bases.

  Computed once at the start of each `LoweringPass.apply` from
  ``prog``'s declared views and the registered index plugins (so D2L
  ``FULL_VER`` takes 2 slots etc.). Read-only during lowering.

  F3 introduces this as an empty default; `F5` will populate it from
  the real ``prog``.

  Per `docs/phase_b_lowering_dispatcher.md` §3.3.
  '''

  view_var_names: dict[str, str] = field(default_factory=dict)
  slot_bases: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class LowerCtx:
  '''Per-pass state. EXACTLY 5 fields. Frozen.

  Per `docs/phase_zero_prerequisites.md` §3.2 and
  `docs/phase_b_lowering_dispatcher.md` §3.

  The dispatch table is attached by `LoweringPass.apply` via
  ``object.__setattr__`` (decision #1 in
  `docs/phase_zero_prerequisites.md` §1). This is the one permitted
  use of ``object.__setattr__`` on `LowerCtx`; it is framework infra,
  NOT a transition shim, and is not counted in the D18 ratchet.

  Fields:
    compiler        — running `Compiler`, for cross-dialect dispatch
                       and registry lookups.
    name_gen        — mutable `NameGen` wrapper for fresh names.
    view_layout     — per-relation `ViewLayout` (read-only during
                       lowering).
    plugin_registry — per-`Compiler` plugin registry (typed ``Any``
                       to avoid an F4 import cycle until that PR
                       lands).
    target          — active render target name (e.g. ``'cuda'``).
  '''

  compiler: Compiler
  name_gen: NameGen
  view_layout: ViewLayout
  plugin_registry: Any
  target: str

  def lower(self, op: Any) -> Any:
    '''Dispatch entry.

    Looks up the registered ``@lowering`` for ``type(op)`` on the
    active `LoweringPass`'s ``target_dialect_name`` and applies it.
    Lowerings call this method for child ops they want to recurse
    into (per the R1 design report: explicit recursion, no auto-walk).

    The dispatch table is attached by `LoweringPass.apply` as the
    private attribute ``_table`` via ``object.__setattr__``.
    '''
    table = self._table  # type: ignore[attr-defined]
    low = table.get(type(op))
    if low is None:
      from srdatalog.ir.core.passes import LoweringMissingError

      raise LoweringMissingError(type(op), self.target)
    return low.apply(op, self)


__all__ = [
  'LowerCtx',
  'NameGen',
  'ViewLayout',
]
