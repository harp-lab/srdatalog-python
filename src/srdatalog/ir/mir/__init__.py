'''Mid-level IR (MIR): types, passes, emit.

Submodules:
  - ``types``  — the MIR ADT (Scan, ColumnJoin, CartesianJoin, etc.)
  - ``passes`` — MIR optimization passes (clause reordering, etc.)
  - ``emit``   — S-expression printer for golden-diff tests
'''

from __future__ import annotations

# No re-exports here by design — submodule-qualified imports read
# cleanly and avoid circular-import hazards (types ↔ passes cross-ref).
