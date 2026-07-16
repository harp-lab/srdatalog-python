'''Base classes for IR ops and types.

`Op` and `Type` are pure-data marker bases. Subclasses are dataclasses
with `frozen=True, slots=True`. The bases carry no methods and no
state.

Discipline rules (see docs/design_principles.md):

  - Subclasses are pure data. No methods. No virtual dispatch.
  - Frozen: IR is immutable.
  - Slots: forbid attribute injection at runtime.
  - Dispatch via `match`, not method override.

These are enforced by tests/test_ir_core_discipline.py.
'''

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Type:
  '''Base class for IR value and state types.

  Each dialect defines its own subclasses. For example, the
  IIR-sorted-array dialect provides `SaState`, `SaView`, `SaHandle`.
  Subclasses must use `@dataclass(frozen=True, slots=True)` and may
  add fields; they must not add methods.
  '''


@dataclass(frozen=True, slots=True)
class Op:
  '''Base class for IR operations.

  Each dialect defines its own subclasses, typically as dataclasses.
  Op instances form an IR program; the pass driver walks them via
  pattern matching, never via virtual dispatch on methods.
  '''
