'''Pattern matching helpers for IR walkers.

The bulk of pattern matching uses Python 3.10+ structural `match`
statements directly inside Lowering/Rewrite `apply` functions. This
module hosts the helpers that don't fit cleanly there:

  - Multi-node patterns (subtree matchers).
  - Tree traversal utilities (find-all, walk-with-context).
  - Cross-scope predicates (e.g. "is this op inside an IterURV scope")
    used by sa.hint's verifier.

Stage 1: stub. Real walkers are added when dialects need them.

See docs/ir_lowering_semantics.md, section 20.
'''

from __future__ import annotations
