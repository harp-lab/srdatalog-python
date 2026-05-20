'''Base classes for IR ops and types.

`Op` and `Type` are pure-data marker bases. Subclasses are dataclasses
with `frozen=True, slots=True`. The bases carry no methods and no
state — *with the single exception of `Op.attributes`*, the typed
attribute dict introduced in PR-P0 (spec § 3.2.1.2 Risks 2 + 5).

Discipline rules (see docs/design_principles.md):

  - Subclasses are pure data. No methods. No virtual dispatch.
  - Frozen: IR is immutable.
  - Slots: forbid attribute injection at runtime.
  - Dispatch via `match`, not method override.

These are enforced by tests/test_ir_core_discipline.py.

About `Op.attributes` (PR-P0, spec § 3.2.1.2 / § 3.2.1.3):

  - Default factory: a fresh empty `AttributeDict` per instance.
  - `kw_only=True`: keeps positional construction stable for every
    existing subclass that already declares its own positional
    fields. Without this, adding any defaulted parent field would
    break every subclass with non-defaulted fields (Python's
    "non-default arg follows default arg" rule on the generated
    `__init__`).
  - `compare=False, hash=False`: attribute state is per-pass metadata,
    NOT part of structural op identity. Two ops with the same frozen
    fields are equal regardless of attribute state — equality lives
    in the IR shape, cross-pass data lives in attributes (spec §
    3.2.1.3, MLIR precedent § 11.2).
  - `repr=False`: keeps the repr clean and byte-equivalent with the
    pre-PR-P0 shape; byte-equivalence goldens that print ops are
    unaffected.
'''

from __future__ import annotations

from dataclasses import dataclass, field

from srdatalog.ir.core.attribute import AttributeDict


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

  `attributes` is the MLIR-style typed attribute dict (PR-P0, spec §
  3.2.1.2 Risks 2 + 5). Per-feature data — view bindings, output
  bindings, index bindings, block-group placement, etc. — lives there,
  registered by the owning pragma plugin via
  `PragmaPlugin.new_attributes`. Op subclasses do NOT grow per-feature
  fields. See `attribute.py` for the storage type.
  '''

  attributes: AttributeDict = field(
    default_factory=AttributeDict,
    kw_only=True,
    compare=False,
    hash=False,
    repr=False,
  )
