'''Mid-level IR (MIR): commands, types, passes, emit, and runner.

Subpackage formed from the former top-level ``mir_*.py`` files — keeping
them flat was fine when there were two or three of them, but with six
the prefix-namespace got unwieldy. Old imports like
``from srdatalog.mir_types import MirNode`` are now
``from srdatalog.mir.types import MirNode``.

No public-API symbols moved, only their module paths. Everything the
outside world needs is still accessible via ``srdatalog.mir.<submodule>``.
'''

from __future__ import annotations

# No re-exports here by design — submodule-qualified imports read
# cleanly and avoid circular-import hazards that a big umbrella
# re-export would introduce (types ↔ commands ↔ passes all cross-ref).
