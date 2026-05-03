'''Index-aware IR cross-cutting dialects.

These dialects supply structural ops shared across data-structure
dialects: control flow, memory operations, etc. They are not
data-structure-specific themselves; instead they're the compositional
glue that data-structure ops compose with.

Currently registered:
  - cf: control flow — Block, Bind, GridStrideLoop, ParallelFor, Phase,
    LaneZeroGuard, WriteOutput, AddCount, IndentBlock, BlankLine,
    Comment, RawString, IfReturnIfNot, VarRef.

Planned:
  - mem: arena allocation, hugepage hints (spec §18).
'''
