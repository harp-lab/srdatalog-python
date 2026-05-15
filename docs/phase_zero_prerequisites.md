---
orphan: true
---

# Phase Zero Prerequisites — Cross-Cutting Decisions

After Layer 0 design docs landed, four parallel research subagents
investigated the unclear cross-cutting concerns (R1–R4 reports). Their
findings surfaced ~16 design tradeoffs that needed resolution **before
parallel implementation agents could ship code without rework**.

This doc consolidates those decisions plus the typed-pragma decision
(`pragma_as_typed_object.md`). Every Layer 1+ implementation agent
operates against this contract; ambiguities here block parallel work.

Companion to:
- [`compiler_redesign.md`](compiler_redesign.md) — architectural spine.
- [`code_discipline.md`](code_discipline.md) — enforcement.
- [`pragma_as_typed_object.md`](pragma_as_typed_object.md) — typed pragma model.
- Phase docs A/B/C/D/E.

## 1. The 16 cross-cutting decisions

| # | Concern | Resolution | Rationale |
|---|---|---|---|
| 1 | `LoweringPass._table` attachment on frozen `LowerCtx` | **`object.__setattr__`** in `LoweringPass.apply()`; one line; the convention pattern already used elsewhere | Accepted minor grime over a 6th field or wrapper class; preserves §5 ceiling at 5 fields |
| 2 | `Scope` size growth risk (becoming new god-object) | **Per-family Scope subclasses with hard cap of 8 fields per subclass.** Specifically: `Scope` is an abstract base; concrete instances are `JoinScope`, `FilterScope`, `AggregateScope`, etc. Each is a frozen dataclass with ≤8 fields. Discipline test enforces. | Without this, we just rename `LoweringCtx` to `Scope`. R1's loud warning. |
| 3 | Recursion model in lowerings | **Explicit `ctx.lower(child)`.** No auto-walk via `bottom_up`. Each lowering decides which children to recurse into; some children are values not control. | Mirrors MLIR ConversionPattern. Auto-walk breaks when source-dialect children have no target-dialect lowering. |
| 4 | NameGen counter back-flow during partial migration | **`NameGen` is mutable wrapper; bridge writes `name_gen._counter` back onto legacy `LoweringCtx.name_counter` after dispatched call returns.** Discipline-tested by per-PR byte-equiv suite. Removed in Layer 3 cleanup. | Byte-equiv requires identical bump order across legacy + new paths during migration. Not negotiable. |
| 5 | Multi-target priority for ambiguous `@lowering` | **Hard-fail on duplicate `(source_op_type, target)` registration.** No PatternBenefit-style priority. | Simplest correct option; defer until a real plugin needs richer dispatch. R1 recommendation. |
| 6 | `block_group` precondition failure mode | **Loud `PragmaApplicationError`.** Pragma asserts a fact; if IR contradicts, that's a bug. | Partial-eval framing: pragmas are facts about the rule, not best-effort hints. R2 recommendation. |
| 7 | `dedup_hash` + `work_stealing` interaction | **Document mutual exclusion + assert at MirPragmaPass.** No legacy combine; defer real combine until use case appears. | Today's legacy code never combined them on disjoint code paths. No byte-equiv reference for combination. |
| 8 | `count` pragma fit | **Defer to separate `PhasePass` mechanism.** `count` is a phase flag (count-vs-materialize), not a wrap-op pragma. Don't force-fit into `@pragma_handler`. | R2 + R4 both flagged. Forcing it into the wrap-op shape is the wrong abstraction. |
| 9 | Per-Compiler pragma overrides for tests | **Module-global `_PRAGMA_REGISTRY` for now; per-Compiler overlay deferred to Phase E.** Tests use `monkeypatch` on the module-global registry. | Acceptable for Phase A-D; flagged for E. |
| 10 | `provides`/`requires` placement on plugin register callable | **Function attribute** (`register.provides = (...)`, `register.requires = (...)`). Loaded via `ep.load()` then read. | Ergonomic; defer separate-EP-group optimization until a real perf concern. |
| 11 | Auto-load env var (`SRDATALOG_AUTOLOAD=1`) | **No.** `Compiler()` is empty by default; `Compiler.with_default_plugins()` is opt-in. No env-var override. | Convenient for notebooks but dangerous for tests. Add when a concrete user asks. |
| 12 | Pragma namespace policy | **Flat namespace + `replaces=()` escape hatch for built-in collisions.** External plugins use Python-import-path identity; structural collisions impossible. | R3 recommendation. Per `pragma_as_typed_object.md`, typed pragmas make string-name collisions structurally impossible — but the `replaces=()` escape hatch stays for plugin-vs-plugin handler conflicts on the same pragma class on the same op type. |
| 13 | `dependency_graph` granularity | **Dialect-name strings.** Op-class fingerprints are precise but require new contract. Tighten if needed. | Defer until granularity matters. |
| 14 | Fixpoint pass termination | **Per-pass `max_iters` field with default cap of 100** (matches `core/strategy.py:repeat`'s existing default of 1024 but tightened for safety). | Per-pass tunable; default catches divergent rewrites. |
| 15 | `compile_to_hir` sub-pipeline filtering | **Filter `DEFAULT_PIPELINE` by `isinstance(p, ProgramPass) and 'hir' in p.consumes`.** Add `Pass.phase: Literal['hir','mir','iir','render'] | None = None` only if filtering grows complex. | Start simple; add the phase tag when fragility shows. |
| 16 | Per-Compiler dialect registration vs process-global | **Process-global decorator state for now (matches existing `@lowering`/`@rewrite`); per-Compiler overlay deferred to Phase E.** | Same call as #9; consistent policy across pragma + lowering registries. |

## 2. The typed-pragma decision (#17)

Beyond the 16 from R1–R4: **pragmas are typed Python objects, not
strings.** `Pragma` base class + `@pragma_handler(PragmaCls, on=...)`.
Full spec: [`pragma_as_typed_object.md`](pragma_as_typed_object.md).

Implications threaded into `code_discipline.md` (D8/D13/D14/D15/R5/R5b)
and the affected phase docs.

## 2.1 The transitional-state ratchet decision (#18)

Beyond #17: **transitional mutable state must ratchet down.** Any
`object.__setattr__` shim on frozen Op, `# DEPRECATED:` field pending
removal, or module-global mutable registry with a documented
per-Compiler migration target requires:

1. An inline `# TODO(phase-X): <action>` comment naming the phase
   that removes it.
2. Inclusion in a discipline ratchet test (`test_discipline_transitional_state_ratchet`)
   that fails CI if the count goes UP.

Same shape as D5 (`test_iir_no_raw_string_growth`) and D12
(`USE_DECLARATIVE` monotonic-add).

**Why this matters:** the redesign accepts transitional debt because
byte-equiv migration requires it. PR #28's A1 mutation shims (4
sites, all marked TODO) are the first inventory entries. Without the
ratchet, transitional shims become permanent — exactly the failure
mode this whole redesign reverses (the "framework with shims that
never get removed" anti-pattern).

The first ratchet inventory snapshot lands when the test ships
(probably with F1 or earlier in Layer 1). Subsequent PRs that remove
shims update the cap atomically in the same commit.

## 3. Locked design contracts (Layer 1+ baseline)

Implementation agents writing Layer 1 / Layer 2 PRs operate against
these. Deviations require an amendment via the `discipline-amendment/<id>`
flow per `code_discipline.md` §9.

### 3.1 Pass framework (F1)

```python
class Pass(ABC):
    name: str
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    def apply(self, prog, compiler) -> Any: ...

class LoweringPass(Pass):
    target_dialect_name: str
    # apply() builds dispatch table per-call, dispatches via type(op),
    # uses _LoweringDispatcher (F1) — full LowerCtx is F3.

class RewritePass(Pass):
    dialect_name: str
    until_fixpoint: bool = True
    # Wraps existing apply_rewrites_to_fixpoint.

class ProgramPass(Pass):
    fn: Callable[[Any, Any], Any]
    # apply() = fn(prog, compiler). Used for HIR planning passes.
```

`Compiler.run(prog, *, pipeline: list[Pass])`:
- Pre-flight: walk `pipeline` left-to-right; maintain `available_dialects` from `Compiler.dialects` ∪ `produces` of earlier passes. Raise `PassOrderingError` on `consumes` mismatch.
- Apply each Pass in order; thread `prog` through.

### 3.2 LowerCtx (F3)

```python
@dataclass(frozen=True)
class LowerCtx:
    compiler: Compiler           # cross-dialect dispatch + registries
    name_gen: NameGen            # mutable wrapper, fresh(prefix) -> str
    view_layout: ViewLayout      # per-relation view_var_names + slot bases
    plugin_registry: PluginRegistry
    target: str

    def lower(self, op):
        # Routes via _table attached by LoweringPass.apply via
        # object.__setattr__ (decision #1).
```

Discipline-pinned at 5 fields (D10).

### 3.3 Scope (F3 follow-up)

```python
class Scope(ABC):
    """Lexical-context base. Concrete subclasses per op family.
    Discipline: each subclass is frozen+slots, ≤ 8 fields."""

@dataclass(frozen=True, slots=True)
class JoinScope(Scope):
    bound_vars: tuple[str, ...]
    cartesian_bound_vars: tuple[str, ...]
    handle_vars: dict[str, str]
    is_counting: bool
    inside_cartesian: bool
    # ≤ 8 fields total

@dataclass(frozen=True, slots=True)
class FilterScope(Scope):
    bound_vars: tuple[str, ...]
    # ≤ 8 fields total

# etc.
```

Discipline test `test_scope_subclass_field_count_capped` parametrized
over all `Scope` subclasses.

### 3.4 Pragma (typed)

Per [`pragma_as_typed_object.md`](pragma_as_typed_object.md). Quick
recap:

```python
@dataclass(frozen=True, slots=True)
class Pragma: ...   # base, no fields, no methods

@pragma_handler(PragmaCls, on=MirOpCls,
                before=(OtherPragmaCls,), after=())
def materialize(op, pragma: PragmaCls, ctx: PragmaCtx) -> Op | None: ...

# DSL:
Rule(...).with_pragma(DedupHash(hash_size=2**20))

# MIR:
ExecutePipeline(..., pragmas=(DedupHash(...), BlockGroup(...)))
```

### 3.5 Plugin discovery (F4)

```python
@classmethod
def Compiler.with_default_plugins(cls, *, group="srdatalog.plugins") -> "Compiler":
    c = cls()
    eps = _topo_sort(_discover(group))   # by provides/requires attrs
    for ep in eps:
        c.register_plugin(ep)
    return c

# Conflict: loud failure (PluginConflictError) with `replaces=()`
# escape hatch on the colliding plugin's register fn.
```

### 3.6 Pragma materialization (Phase C — per typed model)

```python
class MirPragmaPass(RewritePass):
    dialect_name = "mir"
    until_fixpoint = False  # one-shot

    def apply(self, prog, compiler):
        regs = get_pragma_registrations()
        order = _kahn_topo_sort(regs)         # by class identity
        for cls in order:
            prog = self._apply_one(prog, by_cls[cls])
        # Discipline: every pragma instance has been removed from op.pragmas
        # else raise UnconsumedPragmaError with did-you-mean class names.
        return prog
```

## 4. New discipline rules consolidated

Beyond the existing D1–D12 / R1–R7 in `code_discipline.md`:

| ID | Rule | Test |
|---|---|---|
| **D13** | `Pragma` subclass without `@final + @dataclass(frozen=True, slots=True)` | `test_pragma_subclasses_are_frozen_final` |
| **D14** | Method on `Pragma` subclass other than `__post_init__` | `test_pragma_subclasses_are_pure_data` |
| **D15** | DSL `Rule.with_pragma(...)` accepts non-`Pragma` arg (post-migration) | `test_with_pragma_rejects_non_pragma` |
| **D16** | `Scope` subclass with > 8 fields | `test_scope_subclass_field_count_capped` |
| **D17** | `LoweringPass.apply` skipping the table-build step (dispatch via isinstance directly) | `test_lowering_pass_uses_table_dispatch` (AST scan in `core/passes.py`) |
| **D18** | **Transitional mutable state without ratchet.** `object.__setattr__` shims on frozen Op + `# DEPRECATED:` fields + module-global mutable registries that have a documented per-Compiler migration target. Every occurrence MUST carry a `# TODO(phase-X): <action>` comment AND be inventoried by a per-category cap that monotonically decreases. | `test_discipline_transitional_state_ratchet` — counts each category against a pinned cap; CI fails if count goes UP. Same shape as D5 (`test_iir_no_raw_string_growth`) and D12 (`USE_DECLARATIVE` monotonic). The redesign accepts transitional debt because byte-equiv migration requires it (PR #28's A1 mutation shims are the first example), but the debt MUST visibly decrease. |
| **R5b** | `op.pragmas` empty after `MirPragmaPass` | `test_pragmas_empty_after_materialization` |
| **R8** | Every Pass in `DEFAULT_PIPELINE` has `consumes` satisfied at its position | enforced at runtime by `Compiler.run`'s pre-flight; discipline test pins `DEFAULT_PIPELINE` is well-ordered |

## 5. Open questions still requiring user decision (none load-bearing)

These do NOT block Layer 1 / 2 work but should be resolved before
relevant phases:

| Phase | Question | Acceptable defaults until decided |
|---|---|---|
| A2 | Should `concurrent_write` move to a MIR pass (computed once) or stay as transition shim? | Stay as shim; fix in A2 |
| C | Should `count` use `PhasePass` or stay imperative for now? | Stay imperative; revisit when adding 2nd target |
| E | Should plugin discovery support an env var override (`SRDATALOG_AUTOLOAD`)? | No; revisit on user request |
| E | Should `replaces=()` escape hatch be limited to opted-in plugins or open to any? | Open; loud error on collision otherwise |

## 6. Phase-by-phase impact summary

| Phase | What's locked by this doc |
|---|---|
| **A2** | Mutation site cleanup uses (1) `dataclasses.replace` where parallel references aren't an issue, (2) explicit threading where they are. The `name_gen` back-flow pattern from #4 is the model. |
| **A3** | Removes deprecated named pragma fields; `pragmas: tuple[Pragma, ...]` is the final shape. |
| **F1** | Pass kinds per §3.1. `Compiler.run` per §3.1. |
| **F3** | LowerCtx 5 fields per §3.2. Scope per-family subclasses per §3.3. |
| **F4** | Plugin discovery per §3.5. |
| **F5** | Declarative pipeline shim — `DEFAULT_PIPELINE` is `list[Pass]`; `compile_kernel_body` reduces to `Compiler.run(prog, pipeline=...)`. |
| **B (Wave 2A)** | Each per-MIR-op `@lowering` uses explicit `ctx.lower(child)` recursion (decision #3). USE_DECLARATIVE ratchet per phase_b §5. |
| **C (Wave 2C)** | Each pragma is a typed `Pragma` subclass per §3.4. `block_group` precondition is loud (#6). `count` deferred to PhasePass (#8). |
| **D (Wave 2B)** | Each HIR pass becomes a `ProgramPass` per §3.1. |
| **E (Wave 2D)** | Plugin discovery per §3.5. Built-ins re-shipped as plugins to validate. |

## 7. Sign-off

This doc is the contract between the design phase and the
implementation phase. Layer 1+ agents reference this; deviations
require a documented amendment.

- [ ] All 17 cross-cutting decisions (16 from R1–R4 + typed pragma)
  have a stated resolution.
- [ ] The locked contracts in §3 are concrete enough that an
  implementation agent can ship without asking clarifying questions.
- [ ] New discipline rules in §4 are integrated into the test inventory.
- [ ] Open questions in §5 don't block Layer 1 work.

Reviewer initials and date below.
