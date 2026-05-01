'''Base classes for IR ops and types.

Each dialect subclasses `Op` for its operations and `Type` for its
value/state types. Subclasses are dataclasses with whatever fields
are needed; the bases are intentionally empty marker classes — they
exist so the pattern matcher and verifier can identify "this is an IR
op" vs "this is some other Python object."

See docs/ir_lowering_semantics.md, section 19.
'''

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Type:
  '''Base class for IR value and state types.

  Each dialect defines its own subclasses. For example, the
  IIR-sorted-array dialect provides `SaState`, `SaView`, `SaHandle`.
  '''


@dataclass
class Op:
  '''Base class for IR operations.

  Each dialect defines its own subclasses, typically as dataclasses.
  Op instances form an IR program; the pass driver walks them.
  '''
