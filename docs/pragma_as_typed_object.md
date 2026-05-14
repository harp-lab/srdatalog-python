---
orphan: true
---

# Pragmas as Typed Compile-Time Objects

This doc supersedes the string-keyed pragma model from
[`phase_c_pragma_materialization.md`](phase_c_pragma_materialization.md)
§2 and [`compiler_redesign.md`](compiler_redesign.md) §6. **Pragmas
are typed compile-time objects, not string keys.** The old
string-keyed sketch was a half-measure; this doc replaces it.

The change is architectural enough to deserve its own spec — it
shapes the DSL surface, the MIR field, the `MirPragmaPass`
algorithm, and the plugin extension model.

## 1. Why string keys aren't enough

The pre-redesign sketch had:

```python
@pragma(name="dedup_hash", on=mir.ExecutePipeline, value_type=bool)
def materialize_dedup_hash(op, ctx): ...

Rule(...).with_pragma("dedup_hash", True)
mir.ExecutePipeline(..., pragmas=(("dedup_hash", True),))
```

Problems:

- **Stringly-typed config in a typed compiler.** `Rule.with_pragma("didup_hash", True)` is a silent typo; the error fires deep in `MirPragmaPass`'s "unconsumed pragma" check, not at the user's keystroke.
- **No autocomplete.** The DSL surface has no anchor for IDE tooling.
- **Flat bool values everywhere.** Every pragma defaults to `bool` because the value type is opaque; structured config (`BlockGroup(threads_per_warp=128, ws_steal_threshold=64)`) is awkward and untyped.
- **Cross-pragma references via strings.** `before=("count_as_product",)` — also stringly-typed; topo-sort errors fire at runtime instead of import time.
- **External-plugin name collisions.** Two plugins both registering `name="dedup_hash"` need a `replaces=()` escape hatch (per R3 design); with typed objects, two `class DedupHash(Pragma):` in different packages are distinct types — collisions are structurally impossible.

The fix: pragmas are **first-class compile-time values of first-class compile-time types**.

## 2. The `Pragma` base class

```python
# core/pragma.py (new file in Layer 1):

@dataclass(frozen=True, slots=True)
class Pragma:
    """Base class for compile-time pragma objects.

    A pragma is a typed, structured fact about a rule that the
    compiler partially-evaluates by inserting wrap ops into MIR.
    Pragmas:

      - Are constructed at DSL-time (eager Python type checks via
        the dataclass __init__).
      - Carry structured config (not flat bool/Any).
      - Are consumed once by MirPragmaPass — they don't survive
        past it. Their effect is the wrap-op insertion; their
        evidence downstream is the typed op tree, not the pragma
        itself.
      - Dispatch by type, not string name.

    Discipline (parallel to Op subclass rules):

      - Subclass with @final + @dataclass(frozen=True, slots=True).
      - Pure data: no methods. Behavior lives in a separately-
        registered @pragma_handler — symmetric with @lowering /
        @rewrite (every transformation is a registered decoration,
        separate from the data it operates on).
      - Use __post_init__ for cross-field validation if needed.
    """
```

Subclassing pattern (built-in `dedup_hash`):

```python
# pragmas/dedup_hash.py:

from srdatalog.core.pragma import Pragma

@final
@dataclass(frozen=True, slots=True)
class DedupHash(Pragma):
    """Atomically dedup emitted tuples via a GPU hash table.

    Materialization wraps each InsertInto in a DedupGate MIR op,
    which Phase B's MIR→IIR lowering turns into the existing
    `sorted_array.DedupTryInsert` COMPOUND op.
    """
    hash_size: int = 1024 * 1024  # GPU hash table capacity (default 1 Mi entries)
```

Subclassing pattern (richer pragma — `block_group`):

```python
# pragmas/block_group.py:

@final
@dataclass(frozen=True, slots=True)
class BlockGroup(Pragma):
    """Block-group work-balanced root iteration for a rule.

    Materialization wraps the root multi-source ColumnJoin in a
    `BlockGroupRoot` MIR op (per `phase_c §4` evidence,
    `lowerings.py:617` shows the legacy `ctx.bg_enabled` branch
    only fires on the root multi-source CJ).
    """
    threads_per_warp: int = 32
    ws_steal_threshold: int = 64

    def __post_init__(self):
        if self.threads_per_warp not in (16, 32):
            raise PragmaConfigError(
                f"BlockGroup.threads_per_warp must be 16 or 32, "
                f"got {self.threads_per_warp}"
            )
```

The `__post_init__` validation fires at `BlockGroup(threads_per_warp=64)` construction time — at the user's DSL keystroke, not in MirPragmaPass.

## 3. The `@pragma_handler` decorator

Replaces the old `@pragma(name=..., value_type=...)`. Type-based instead of string-based.

```python
# core/pragma.py (continued):

def pragma_handler(
    pragma_cls: type[Pragma],
    *,
    on: type[Op],                              # MIR op type that carries the pragma
    before: tuple[type[Pragma], ...] = (),     # other pragma types this must run BEFORE
    after: tuple[type[Pragma], ...] = (),      # ... AFTER
) -> Callable:
    """Register a materialization handler for a pragma class.

    The decorated function fires during MirPragmaPass on every op of
    type `on` whose pragmas tuple contains a `pragma_cls` instance.

    Signature:
        def handler(op: Op, pragma: pragma_cls, ctx: PragmaCtx) -> Op | None:

    Returns the transformed op (with the pragma instance removed
    from op.pragmas), OR None to skip (no-op for this pragma).

    `before` / `after` reference pragma TYPES, not strings. A typo
    (`before=(WorkStealng,)`) is a Python NameError at decoration —
    immediate, traceable.
    """
```

Usage:

```python
@pragma_handler(DedupHash, on=mir.ExecutePipeline,
                before=(CountAsProduct,))
def materialize_dedup_hash(op, pragma: DedupHash, ctx) -> Op | None:
    """`pragma` is the typed DedupHash instance — full struct, not bool."""
    new_pipeline = tuple(
        mir.DedupGate(inner=child, hash_size=pragma.hash_size)
        if isinstance(child, mir.InsertInto)
        else child
        for child in op.pipeline
    )
    new_pragmas = tuple(p for p in op.pragmas if not isinstance(p, DedupHash))
    return op.replace(pipeline=new_pipeline, pragmas=new_pragmas)
```

Key changes vs the string-keyed shape:
- The handler receives `pragma: DedupHash` directly — no `.get("dedup_hash")` lookup, no `value_type=` validation needed (Python type system enforces it).
- Removal: `tuple(p for p in op.pragmas if not isinstance(p, DedupHash))` — type-discriminated, not string-keyed.
- `before` / `after` reference `CountAsProduct` (a class), not `"count_as_product"` (a string).

## 4. MIR carries pragma instances

`mir.ExecutePipeline.pragmas` is now `tuple[Pragma, ...]` — an ordered collection of typed pragma instances.

```python
@dataclass(frozen=True, slots=True)
class ExecutePipeline(Op):
    pipeline: tuple[Op, ...]
    source_specs: tuple[Op, ...]
    dest_specs: tuple[InsertInto, ...]
    rule_name: str = ''
    clause_order: tuple[int, ...] = ()
    pragmas: tuple[Pragma, ...] = ()  # tuple of typed pragma instances
```

Construction (DSL → HIR → MIR carries the instances through):

```python
mir.ExecutePipeline(
    pipeline=(...),
    source_specs=(...),
    dest_specs=(...),
    pragmas=(DedupHash(hash_size=2**20), BlockGroup(threads_per_warp=32)),
)
```

Lookup helpers (free functions on the pragma module, not methods on `Op`):

```python
def get_pragma(op: Op, pragma_cls: type[Pragma]) -> Pragma | None:
    """Return the first pragma of type pragma_cls on op, or None."""
    return next((p for p in op.pragmas if isinstance(p, pragma_cls)), None)

def has_pragma(op: Op, pragma_cls: type[Pragma]) -> bool:
    return any(isinstance(p, pragma_cls) for p in op.pragmas)
```

In handlers / debugging code:

```python
if (dh := get_pragma(op, DedupHash)) is not None:
    print(f"dedup with hash size {dh.hash_size}")
```

## 5. `MirPragmaPass` — type-based dispatch

```python
class MirPragmaPass(RewritePass):
    """Walks the registered @pragma_handler set, applies each in
    topo-sorted order. After this pass, op.pragmas is empty for
    every op in the tree.
    """
    dialect = MIR_DIALECT
    until_fixpoint = False  # one-shot

    def apply(self, prog, compiler):
        regs = compiler.pragma_handler_registry()  # list of (cls, on, before, after, fn)

        # Topo-sort by pragma TYPE (not string).
        graph = {r.pragma_cls: set() for r in regs}
        for r in regs:
            for nxt in r.before: graph[r.pragma_cls].add(nxt)
            for prv in r.after:
                graph.setdefault(prv, set()).add(r.pragma_cls)
        try:
            order = _kahn(graph)  # list[type[Pragma]]
        except CycleError as e:
            raise PragmaOrderingError(
                f"pragma cycle: {' -> '.join(c.__name__ for c in e.cycle)}"
            )

        # Apply in order.
        by_cls: dict[type[Pragma], PragmaReg] = {r.pragma_cls: r for r in regs}
        for cls in order:
            if cls not in by_cls: continue   # forward-decl OK
            reg = by_cls[cls]
            prog = self._apply_one(prog, reg)

        # Discipline: every pragma instance must have been claimed.
        leaks = []
        for op in walk(prog):
            if hasattr(op, 'pragmas') and op.pragmas:
                leaks.append((op, op.pragmas))
        if leaks:
            op, ps = leaks[0]
            cls_names = sorted(c.__name__ for c in by_cls)
            raise UnconsumedPragmaError(
                f"pragma {ps[0]!r} on {type(op).__name__} survived "
                f"MirPragmaPass; no @pragma_handler claims its type. "
                f"Did you mean one of: {cls_names}?"
            )
        return prog

    def _apply_one(self, prog, reg):
        # For every op of type reg.on whose pragmas contain a reg.pragma_cls instance:
        # find the instance, call reg.fn(op, instance, ctx), use returned op.
        ...
```

Key shifts vs the string-keyed sketch:
- Topo-sort keyed by `type[Pragma]`, not `str`.
- Lookup: `isinstance(p, reg.pragma_cls)` per pragma instance, not `op.pragmas.get(name)`.
- Cycle / leak errors reference class names — clearer than string names; map directly to imports.

## 6. DSL surface

```python
# Today (string-based):
Rule(...)
    .with_pragma("dedup_hash", True)
    .with_pragma("block_group", True)
    .with_pragma("work_stealing", True)

# After this redesign (typed-object):
from srdatalog.pragmas import DedupHash, BlockGroup, WorkStealing

Rule(...)
    .with_pragma(DedupHash())
    .with_pragma(BlockGroup(threads_per_warp=32))
    .with_pragma(WorkStealing())

# Or in a single call:
Rule(...).with_pragmas(
    DedupHash(hash_size=2**20),
    BlockGroup(threads_per_warp=32, ws_steal_threshold=64),
    WorkStealing(),
)
```

DSL `with_pragma(p: Pragma)` validates the argument is a `Pragma` subclass and that the type has a registered handler in the active Compiler. If not, raises a typed `UnregisteredPragmaError` at the keystroke — pointing at the missing import or plugin.

Back-compat: the `(string, value)` form is **deprecated** but kept during the migration period. It synthesizes a one-off `Pragma` subclass per call (or maps to a registered legacy adapter). Removed in Layer 3 cleanup.

## 7. External plugin worked example

A third-party `my-srdatalog-jaccard` package on PyPI:

```python
# my_srdatalog_jaccard/pragma.py:
from srdatalog.core.pragma import Pragma, pragma_handler
from srdatalog.dialects import mir
from .ops import JaccardGate

@final
@dataclass(frozen=True, slots=True)
class Jaccard(Pragma):
    """Compute pairwise Jaccard similarity between rule outputs."""
    threshold: float = 0.5
    hash_func: str = 'xxhash3'

    def __post_init__(self):
        if not 0.0 <= self.threshold <= 1.0:
            raise PragmaConfigError(f"Jaccard.threshold must be in [0,1]; got {self.threshold}")

@pragma_handler(Jaccard, on=mir.ExecutePipeline)
def materialize_jaccard(op, pragma: Jaccard, ctx):
    new_pipeline = tuple(
        JaccardGate(inner=child, threshold=pragma.threshold, hash_func=pragma.hash_func)
        if isinstance(child, mir.InsertInto)
        else child
        for child in op.pipeline
    )
    new_pragmas = tuple(p for p in op.pragmas if not isinstance(p, Jaccard))
    return op.replace(pipeline=new_pipeline, pragmas=new_pragmas)
```

End-user code:

```python
from srdatalog import Compiler, Program, Rule, Var
from my_srdatalog_jaccard import Jaccard         # ← typed import

X, Y = Var('x'), Var('y')
similar_pairs = (
    Rule(...)
        .with_pragma(Jaccard(threshold=0.7, hash_func='xxhash3'))
)

compiler = Compiler.with_default_plugins()       # auto-loads my_srdatalog_jaccard
result = compiler.run(Program(rules=[similar_pairs]))
```

Compare to the string-keyed alternative:

```python
# Discouraged — what we DON'T want:
Rule(...).with_pragma("jaccard", {"threshold": 0.7, "hash_func": "xxhash3"})
# Issues: typo "jaccrd" silent until MirPragmaPass; "threshold" key string;
# no IDE knowledge that this is a Jaccard pragma; nested dict instead of
# named fields; can't validate threshold range at construction time.
```

The typed form gives all of: imports as evidence, autocomplete, structured config, eager validation, refactor-safe references.

## 8. Discipline implications

These rules supersede the corresponding string-based ones in
[`code_discipline.md`](code_discipline.md):

| ID | Rule (revised) | Test |
|---|---|---|
| **D8 (revised)** | Hardcoded pragma class references in core. | `test_core_has_no_pragma_class_imports` — AST scan: `from srdatalog.pragmas import *` etc. forbidden in `core/`. (Pragma classes are plugin concerns; core knows only the `Pragma` base.) |
| **R5 (revised)** | Every concrete `Pragma` subclass the DSL can produce has a `@pragma_handler` registration. | `test_pragma_handler_registry_completeness` — for every `Pragma` subclass discoverable via plugins, assert handler registered. |
| **NEW** | Every `Pragma` subclass is `@dataclass(frozen=True, slots=True)` + `@final`. | `test_pragma_subclasses_are_frozen_final` — parametrized over all known subclasses. |
| **NEW** | `op.pragmas` is empty after `MirPragmaPass`. | `test_pragmas_empty_after_materialization` (already in phase_c §7). |
| **NEW** | DSL `with_pragma(arg)` rejects non-Pragma args at the call site. | `test_with_pragma_rejects_non_pragma` — `Rule(...).with_pragma("foo")` raises `TypeError` (after migration window). |

## 9. Migration impact summary

| Doc / module | Change required |
|---|---|
| `compiler_redesign.md` §6 | Replace string-based `materialize_dedup_hash` example with typed-object form. Add reference to this doc. |
| `phase_c_pragma_materialization.md` | Replace the `@pragma(name=...)` decorator design with `@pragma_handler(PragmaCls, ...)`. Update worked examples. Update DSL surface section. |
| `phase_e_plugin_extensibility.md` | Update worked external-plugin example to use typed Pragma. |
| `code_discipline.md` | Update D8 / R5 per §8 above. Add three new discipline rules. |
| `phase_a_mir_onto_op.md` | Note: A1's `pragmas: tuple[tuple[str, Any], ...]` field becomes `pragmas: tuple[Pragma, ...]` once `core/pragma.py` lands. A1's `tuple-of-tuples` form is the **transition shape**; A2 or a follow-up converts to `tuple-of-Pragma`. |
| `phase_b_lowering_dispatcher.md` | No structural change (lowerings consume MIR ops; MIR by Phase C has typed pragmas). |
| `phase_d_hir_passes.md` | No structural change (HIR carries `pragmas: tuple[Pragma, ...]` instead of dict). |

## 10. Why this isn't OO

A natural alternative: put the materialize logic on the `Pragma` class as a method.

```python
# Rejected alternative (Option B from the design discussion):
class DedupHash(Pragma):
    def materialize(self, op, ctx) -> Op | None: ...
```

Rejected because:

1. **Inconsistent with the rest of the framework.** `@lowering`, `@rewrite`, `@register_render` are all separate-decoration patterns. Pragma should be too.
2. **Harder to override / patch externally.** A third party that wants to swap a built-in pragma's materialization (e.g., a `dedup_hash` benchmark variant) would need to subclass `DedupHash` and re-register. With separate `@pragma_handler`, they just register a new handler.
3. **Pragma is data, not behavior.** "Compile-time fact" framing — the pragma IS the structured config; the materialization IS the act of consuming it. They're separable concerns.
4. **D1 discipline parallel.** `Op` subclasses are pure data, no methods (per `core/CLAUDE.md`). `Pragma` follows the same rule for the same reasons.

## 11. Sign-off

- [ ] `Pragma` base class lives in `core/pragma.py` (not `core/op.py` — distinct concept).
- [ ] `@pragma_handler(PragmaCls, on=MirOpCls, before=, after=)` is the only registration shape.
- [ ] `mir.ExecutePipeline.pragmas: tuple[Pragma, ...]` is the wire format.
- [ ] DSL `Rule.with_pragma(p: Pragma)` accepts only Pragma instances (string form deprecated).
- [ ] Topo-sort + cycle detection + unconsumed-pragma errors all reference class names, not strings.
- [ ] Discipline rules updated per §8.
- [ ] All affected phase docs updated per §9.

Reviewer initials and date below.
