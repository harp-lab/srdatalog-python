# ir_core — Dialect-Agnostic IR Framework

## Spec

Full specification in [`docs/ir_lowering_semantics.md`](../../../docs/ir_lowering_semantics.md), Part VI (sections 19–22).

## What goes here

Framework code only — no specific dialects. Dialects live in
[`../dialects/`](../dialects/).

## Design invariants

- **No imports from `dialects/` or specific data-structure code.** This
  module must compile with zero registered dialects.
- **No central enum of dialect kinds.** Adding a dialect is purely
  additive — it must not require edits to any file in this directory.
- **Pass driver consults the registry, not a hardcoded list.** The
  driver knows about `Dialect`, `Lowering`, `Rewrite` — never about
  specific dialect contents.

## Stage 1 status

- Skeleton in place: `Compiler`, `Dialect`, `Op`/`Type` bases,
  `Lowering`/`Rewrite` types, `PassDriver` shell, `VerificationError`.
- Smoke test: [`tests/test_ir_core_scaffold.py`](../../../tests/test_ir_core_scaffold.py).
- Real implementations land as dialects come online (Stage 3+ in the
  roadmap, section 25 of the spec).

## Adding a dialect

See [`docs/ir_lowering_semantics.md`](../../../docs/ir_lowering_semantics.md), Part IV
(sections 13–15) for sketches of LSM⟨K⟩, union-find, bitmap. Each
follows the pattern:

1. Subpackage in `../dialects/<name>/`.
2. Subclass `Op`/`Type` for the dialect's vocabulary.
3. Construct a `Dialect(...)` with lists of types, ops, lowerings,
   rewrites, and an optional verifier.
4. Register at compiler init: `compiler.register_dialect(my_dialect)`.

No edits to anything in this directory.
