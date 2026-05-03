---
orphan: true
---

# Design Principles: IR + Dialect Architecture

This document is the pinned reference for design discipline in the
multi-dialect IR. It is short on purpose. The goal is to lock in
research-validated practices so we don't recreate the engineering
mistakes of LLVM/MLIR while building the same conceptual stack.

The companion document is [`ir_lowering_semantics.md`](./ir_lowering_semantics.md),
which gives the formal semantics and the per-MIR-op lowering rules.
This document gives the *how-we-write-it* rules.

## 0. Heritage

We are not inventing PL theory. The architecture draws from:

- **Term rewriting**: Baader & Nipkow, *Term Rewriting and All That*; Visser,
  *Stratego/XT*.
- **Pattern matching**: Maranget, "Compiling pattern matching to good
  decision trees", ML 2008.
- **ADTs and sums**: ML, Standard ML, Haskell, Rust enums.
- **Free functions over methods**: Scheme, Standard ML; the "expression
  problem" answers (Wadler 1998; Carette-Kiselyov-Shan, "Finally tagless").
- **Generic traversal**: Lämmel & Peyton Jones, "Scrap Your Boilerplate", TLDI 2003.
- **Locally nameless / de Bruijn for binding**: Aydemir et al., POPL 2008.
- **Algebraic effects**: Plotkin & Pretnar, "Handling algebraic effects", LMCS 2013.
- **MLIR concepts** (multi-dialect IR, lowerings) — but **not its engineering**.

Whenever the rules below have a research source, it is named in the
"Source" column.

## 1. Discipline rules (D-rules)

Every contributor follows these. They are checked by
[`tests/test_ir_core_discipline.py`](../tests/test_ir_core_discipline.py)
where automatable.

### D1 — `Op` and `Type` subclasses are pure data

No methods. No virtual dispatch. No `op.lower()`. Algorithms live in
external functions, dispatched via `match`.

**Source**: Standard ML, Haskell. The "expression problem" makes
methods on data the wrong default for compilers.

**Enforced by**: `test_op_subclasses_have_no_user_methods` and the
matching test for `Type`.

### D2 — IR is immutable

`@dataclass(frozen=True)` on every Op/Type subclass. Mutation forces
shared-state reasoning we don't want.

**Source**: every functional language; pragmatically reduces a class
of bugs to nothing.

**Enforced by**: `test_op_subclasses_are_frozen` + runtime check.

### D3 — Slots required

`@dataclass(slots=True)` forbids attribute injection at runtime.
Catches the "let me stick a flag on this op for one pass" anti-pattern.

**Enforced by**: `test_op_subclasses_use_slots` + runtime check.

### D4 — All op/type kinds are dataclasses

Needed for: generic field iteration in strategy combinators, sexpr
round-trip, structural equality, pattern matching `case` arms.

**Enforced by**: `test_op_subclasses_are_dataclasses`.

### D5 — Dispatch via `match` over closed unions

Never `isinstance` chains, never virtual methods, never visitor objects.
Each dialect declares a closed union: `IIRSortedArrayOp = SaRoot | SaPref | …`,
and dispatchers exhaust it via `match`.

```python
def lower(op: IIRSortedArrayOp) -> list[Op]:
  match op:
    case SaRoot(view=v):     return _lower_root(v)
    case SaPref(handle=h, key=k): return _lower_pref(h, k)
    ...
    case _:                   assert_never(op)
```

**Source**: Standard ML, Haskell GADTs, Rust enums. Enables compile-time
exhaustiveness checking.

**Enforced by**: mypy `--strict` exhaustiveness checking on union types
plus `assert_never` in the catch-all branch.

### D6 — Rewrites are pure functions

A rewrite is `Op → list[Op]` (or `Op → Op | None` for strategies).
No state. No hooks. No `PatternRewriter`-style rewriter object.
Composition via the strategy combinators in
[`ir/core/strategy.py`](../src/srdatalog/ir/core/strategy.py).

**Source**: Stratego (Visser); functional rewriting (Baader-Nipkow).

### D7 — Strategies are first-class values

`top_down(rule)`, `bottom_up(rule)`, `repeat(rule)`, `seq(s1, s2)`,
`choice(s1, s2)`, `try_(s)`, `all_(s)`, `one(s)`, `some(s)`. Build
walks by composing combinators; do not write ad-hoc tree walkers per
pass.

**Source**: Stratego/XT.

### D8 — Lexical binding only

Variables and handles bound in scopes use lexical `Let(v, e, body)`.
No string-keyed dicts that look up bindings by name across scopes.
Hoisting and scope manipulation become structural rewrites.

**Source**: λ-calculus, Scheme. The current GPU emitter's
`handle_vars` dict is the anti-pattern this rule prevents.

### D9 — Smart constructors for cross-node invariants

When a node's correctness depends on its surrounding scope (e.g.,
`sa.hint` requires being inside `IterURV`), construct it via a
builder that requires the scope token. Make ill-formed IR
unconstructible rather than relying on after-the-fact verifier passes.

**Source**: Standard ML refinement types in spirit; phantom-type
techniques in Haskell.

### D10 — Verification at every level

Each dialect ships a verifier that runs after every pass on its IR
shapes. Verifiers catch the predicates D9 didn't make
unconstructible. See [`ir/core/verifier.py`](../src/srdatalog/ir/core/verifier.py).

### D11 — `@final` on every concrete op subclass

Closed sums require closed hierarchies. `typing.@final` tells
mypy/pyright: this class has no further subclasses. Catches the
"add a wrapping subclass for one pass's purpose" anti-pattern.

### D12 — Static type checking in CI

`mypy --strict` (or `pyright` strict) on `ir/core/`, `dialects/`,
and any new dialect-aware module. Without this, D5's exhaustiveness
guarantees disappear and the closed-union story collapses.

## 2. Anti-rules (A-rules)

Things we explicitly do not do, with the corresponding mistake they
prevent.

### A1 — No SSA / Region / Block hierarchy

LLVM/MLIR carry SSA + Region + Block infrastructure that is overkill
for a relational IR. Lexical `Let` binding plus tree-shaped Op nesting
is enough.

### A2 — No TableGen-equivalent DSL

Python dataclasses + decorators are sufficient. If declarative dialect
specs become useful, generate them from a small sexpr DSL we control,
not a custom language.

### A3 — No interface / mixin / trait soup on ops

Behavior is functions, not interfaces. If an op needs to participate
in two different algorithms, both algorithms are external functions
that pattern-match it. No `OpInterface`, no traits.

### A4 — No reflective dispatch

No `getattr`-based dispatch. No name-string-driven branching. Either
pattern-match or use the registry.

### A5 — No mutable rewriter state

Rewrites compose via strategy combinators. Mutable rewriters (MLIR's
`PatternRewriter` with insertion points and listeners) are out.

### A6 — No global mutable state in the framework

`Compiler` is a value. `PassDriver` is a value. Constructing them
twice gives two independent compilers. No singletons.

### A7 — No metaclasses, no import-time side effects

Imports are predictable. Dialects register *explicitly* by calling
`compiler.register_dialect(...)`, not by side effects of `import`.

### A8 — No `dyn_cast` / `isa` / runtime type queries beyond `match`

The C++ trap in disguise. Python's `isinstance` is fine inside `match`
patterns; standalone `isinstance` chains are an A8 violation.

## 3. Concrete enforcement

| Rule | Enforced by |
|---|---|
| D1, D2, D3, D4 | [`tests/test_ir_core_discipline.py`](../tests/test_ir_core_discipline.py) |
| D5 | mypy strict + `assert_never` in match defaults |
| D6, D7 | Convention + code review + strategy module shape |
| D8 | Convention; will be checked by the lowering pass for IIR (see ir_lowering_semantics.md §10.4) |
| D9 | Per-dialect smart constructors; D10 catches bypass attempts |
| D10 | Per-dialect verifier passes |
| D11 | mypy strict on `@final` |
| D12 | CI |
| A1–A8 | Code review against this document |

## 4. When to deviate

Add a new D-rule or A-rule to this document. Don't deviate silently.
If a rule turns out to be too strict in practice, raise it as an
explicit decision and document the relaxation.

## 5. Background reading

For contributors new to the PL/compilers space:

- *Compiling Pattern Matching to Good Decision Trees* — Maranget. The
  why-match-is-the-right-tool reference.
- *Stratego/XT Tutorial* (Visser et al.). The strategy-combinators
  reference; this codebase's `ir/core/strategy.py` is a Python port.
- *Term Rewriting and All That* — Baader & Nipkow. Background on
  confluence, termination, normal forms.
- *Scrap Your Boilerplate* — Lämmel & Peyton Jones. Justifies generic
  traversal over per-op visitor methods.
- The MLIR documentation for *multi-dialect compilation*, *lowering*,
  *progressive lowering* — read for the conceptual model. **Do not**
  use it as an engineering reference; the implementation choices are
  what we are deliberately not repeating.
