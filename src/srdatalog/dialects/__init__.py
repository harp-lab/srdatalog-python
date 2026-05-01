'''Registered dialects.

Each subpackage is one dialect, providing types, ops, lowerings,
rewrites, and a verifier. Adding a new data structure (e.g. LSM<K>,
union-find, bitmap) means adding a sibling subpackage here — no
edits to existing code.

Stage 1: empty. The IIR-sorted-array dialect arrives in Stage 3.

See docs/ir_lowering_semantics.md, Part IV for sketches of planned
dialects and Part VII for the implementation roadmap.
'''
