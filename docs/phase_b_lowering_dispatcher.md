---
orphan: true
---

# Phase B — `LoweringPass` dispatcher + per-MIR-op migration

The phase that actually deletes the imperative monolith. Every MIR op
gets one `@lowering` registration in its own file; the per-op feature
flag (`USE_DECLARATIVE`) ratchets up monotonically; the legacy
`lower_scan_pipeline` shrinks until it disappears.

Companion to [`compiler_redesign.md`](compiler_redesign.md) §8
(migration policy) and [`code_discipline.md`](code_discipline.md) §6
(migration-period rules).

## 1. Goal

After Phase B:

- Every concrete MIR op has an `@lowering(target=IIR, source=mir.X)`
  registered in its own file under
  `dialects/relation/sorted_array/lowerings/lower_mir_<op>.py`.
- `LoweringPass(source=MIR, target=IIR)` is a real, dispatching pass:
  walks the MIR tree, looks up registrations by op type, applies them.
- `lower_scan_pipeline` and `_legacy_imperative_lower` are deleted.
- `LoweringCtx` (the legacy 25-field god-object) is replaced by the
  small `LowerCtx` (5 fields).
- `compile_kernel_body` is reduced to `compiler.run(prog,
  pipeline=DEFAULT_PIPELINE)`.

Phase B does NOT yet:

- Materialize pragmas as op insertions (Phase C).
- Touch HIR pass framework migration (Phase D).
- Touch plugin extensibility validation (post-Layer-2).

## 2. `LoweringPass` algorithm

```python
class LoweringPass(Pass):
    """Cross-dialect, source-driven, per-op dispatch.

    For every Op in the input tree:
      1. Look up registered @lowering(target=self.target_dialect,
         source=type(op)).
      2. If found, apply it. The lowering function returns either:
         (a) An Op in the target dialect (the lowered form).
         (b) An Op that itself contains source-dialect children;
             the dispatcher recurses on those children.
      3. If NOT found, raise LoweringMissingError(op_type, target).
         Discipline test R4 (test_every_mir_op_has_lowering)
         catches this at registration time, but the runtime check
         is the safety net.

    LoweringCtx provides:
      - ctx.lower(op) → Op  (recursive dispatch entry; lowerings
        call this for child ops)
      - the small fixed LowerCtx state (5 fields per
        compiler_redesign.md §5).
    """
    target_dialect: Dialect

    def apply(self, prog, compiler):
        ctx = LowerCtx(
            compiler=compiler,
            name_gen=NameGen(),
            view_layout=ViewLayout.from_program(prog, compiler),
            plugin_registry=compiler.plugin_registry_for(self.target_dialect),
            target=self.target_dialect.name,
        )
        return ctx.lower(prog)
```

Key properties:

- **Source-driven dispatch.** The walker is parameterized over the
  source op's type, not the target. One pass = one source-to-target
  direction.
- **Recursive entry.** `ctx.lower(op)` is the only way to dispatch.
  Lowerings call it for their own children; no manual isinstance
  trees inside lowering bodies.
- **Stateless across ops.** `ctx` is frozen; `name_gen.fresh(...)`
  and `view_layout` are the only mutation-via-replacement points.
- **No fallback path.** Either an op has a lowering or compilation
  fails loudly.

## 3. `LowerCtx` field-by-field

```python
@dataclass(frozen=True)
class LowerCtx:
    """Per-pass state. Frozen — replace via `dataclasses.replace`.

    DISCIPLINE: This dataclass has exactly 5 fields. Adding a 6th
    requires a doc amendment per code_discipline.md §9.
    """
    compiler: Compiler
    name_gen: NameGen
    view_layout: ViewLayout
    plugin_registry: PluginRegistry
    target: str

    def lower(self, op: Op) -> Op:
        """Dispatch entry; looks up @lowering(target=self.target,
        source=type(op)) and applies it."""
        ...
```

### 3.1 `compiler: Compiler`

The running Compiler instance. Lets a lowering call back into other
registries (e.g. resolve a sub-dialect's renderer) when needed. The
compiler reference is the back-pointer that unifies cross-pass state.

### 3.2 `name_gen: NameGen`

```python
class NameGen:
    """Counter-backed unique name generator.

    Mutating: name_gen.fresh('h_X') bumps internal counter.

    Production parity: bump order matches legacy `LoweringCtx.fresh()`
    so byte-equivalence holds. Specifically, the per-rule counter
    starts at 0 for each kernel body and bumps in the same order the
    legacy emitter did.
    """
    def fresh(self, prefix: str) -> str: ...
```

The one mutable state in `LowerCtx`. Wrapped so the dataclass itself
stays frozen.

### 3.3 `view_layout: ViewLayout`

```python
@dataclass(frozen=True)
class ViewLayout:
    """Per-relation view variable names + slot bases.

    Computed once at the start of each LoweringPass.apply from
    `prog`'s declared views and the registered index plugins
    (so D2L FULL_VER takes 2 slots etc.). Read-only during lowering.
    """
    view_var_names: dict[str, str]   # handle_idx → view var name
    slot_bases: dict[str, int]       # handle_idx → base slot
```

Replaces the legacy `LoweringCtx.view_var_names` +
`LoweringCtx.view_slot_bases` pair.

### 3.4 `plugin_registry: PluginRegistry`

The per-Compiler `PluginRegistry` (encapsulated in PR #26). Lowerings
call `ctx.plugin_registry.gen_root_handle(view_var, index_type)` etc.
instead of the module-level `plugin_gen_root_handle(...)` shims.

### 3.5 `target: str`

Identifies the active render target (`'cuda'`, future `'cpp_tbb'`,
`'metal'`, ...). Used by lowerings that need target-conditional
behavior — though the goal is that lowerings are target-agnostic and
target-specifics live in render or in target-specific plugins.

### 3.6 What is NOT on `LowerCtx`

| Removed field | Lives now as |
|---|---|
| `is_counting`, `inside_cartesian` | Lexical scope, passed via `Scope` parameter to lowerings that need it. Or op-level (`Phase(C, body)` vs `Phase(M, body)`). |
| `dedup_hash`, `tiled_cartesian`, `bg_enabled`, `ws_enabled` | Op insertions (Phase C); consumed by Phase B's lowering for the wrap op. |
| `output_var`, `output_var_overrides` | Property of `WriteOutput`/`AddCount` op or its containing scope. |
| `bound_vars`, `cartesian_bound_vars`, `handle_vars`, `pre_narrow_infos` | Lexical scope; explicit `Scope` parameter. |
| `debug` | Compiler-level option; on `compiler` if needed. |
| `tile_var` | Op-level (e.g., `LaneZeroGuard.tile_var: str` or compile-time constant on the kernel envelope). |

The `Scope` parameter idea:

```python
@dataclass(frozen=True)
class Scope:
    """Lexical scope, passed explicitly to lowerings that need it."""
    bound_vars: tuple[str, ...]
    cartesian_bound_vars: tuple[str, ...]
    handle_vars: dict[str, str]
    is_counting: bool
    inside_cartesian: bool

@lowering(target=IIR, source=mir.Filter)
def lower_filter(op: mir.Filter, ctx: LowerCtx, scope: Scope) -> Op:
    body = ctx.lower(child_op, scope.entering_filter())
    ...
```

The dispatcher passes `Scope.empty()` at the top level; each lowering
function decides how to update Scope when recursing.

## 4. Per-MIR-op work-unit table (Wave 2A)

Each PR migrates exactly one MIR op type. New file
`dialects/relation/sorted_array/lowerings/lower_mir_<op>.py`. Single
`@lowering` decoration. Single addition to `USE_DECLARATIVE`.

| Wave 2A PR | Branch | MIR op | Difficulty | Migration order |
|---|---|---|---|---|
| **B-Filter** | `feat/lower-mir-filter` | `mir.Filter` | easy | 1 |
| **B-ConstantBind** | `feat/lower-mir-constant-bind` | `mir.ConstantBind` | easy | 2 |
| **B-Scan** | `feat/lower-mir-scan` | `mir.Scan` | medium | 3 |
| **B-InsertInto** | `feat/lower-mir-insert-into` | `mir.InsertInto` | medium | 4 |
| **B-CJ-single** | `feat/lower-mir-cj-single` | `mir.ColumnJoin` (single-source) | medium | 5 |
| **B-CJ-multi** | `feat/lower-mir-cj-multi` | `mir.ColumnJoin` (multi-source) | hard | 6 |
| **B-Cart** | `feat/lower-mir-cart` | `mir.CartesianJoin` | hard | 7 |
| **B-Aggregate** | `feat/lower-mir-aggregate` | `mir.Aggregate` | hard | 8 |
| **B-Negation** | `feat/lower-mir-negation` | `mir.Negation` | hard | 9 |
| **B-ExecutePipeline** | `feat/lower-mir-execute-pipeline` | `mir.ExecutePipeline` | medium | 10 |

Migration order is a recommendation, not a hard sequence — Wave 2A
PRs can land in any order as long as they're file-disjoint and the
foundation (Phase A + F4) is in.

### 4.1 Per-PR template

```python
# File: dialects/relation/sorted_array/lowerings/lower_mir_scan.py

from srdatalog.core import lowering
from srdatalog.dialects.iir.cf import IIR_CF
from srdatalog.dialects import mir
from srdatalog.dialects.relation.sorted_array.types import SaHandle
# ... other imports

@lowering(target=IIR_CF, source=mir.Scan)
def lower_mir_scan(op: mir.Scan, ctx: LowerCtx, scope: Scope) -> Op:
    """Lower mir.Scan to iir.cf — root scan over a relation.

    Per docs/phase_b_lowering_dispatcher.md §4 — one lowering per
    file; production parity preserved against the legacy
    `_lower_root_scan` branch.
    """
    # ... pure function, returns one Op ...
```

```python
# Single-line addition to dialects/relation/sorted_array/__init__.py:

USE_DECLARATIVE: frozenset[type] = frozenset({
    mir.Filter,
    mir.ConstantBind,
    mir.Scan,           # ← added by this PR
})
```

### 4.2 Per-PR acceptance gate

Beyond the global per-PR Done definition (`code_discipline.md` §4):

- The migrated op produces the same IIR tree as the legacy branch on
  every fixture that exercises it.
- The CUDA emit (post-IIR canonicalization + render) is byte-identical
  to legacy.
- A new test `test_lower_mir_<op>_byte_equivalent` runs the migrated
  path on every relevant fixture and asserts byte equality.

If byte-equivalence fails:

1. First — debug. Likely a missed scope-parameter, a name-counter
   bump-order mismatch, or a missing structured op (Stage 4 vocabulary
   gap).
2. If irreducible — document the divergence, update the affected
   golden(s), get owner sign-off, ship.

## 5. Migration ratchet (`USE_DECLARATIVE`)

```python
# In dialects/relation/sorted_array/__init__.py

# DISCIPLINE: This set is monotonically growing during Phase B.
# Removing an entry requires owner sign-off (see code_discipline.md
# D12).
USE_DECLARATIVE: frozenset[type] = frozenset({
    # filled in by Wave 2A PRs, one entry per PR
})
```

In `compile_kernel_body` (during migration):

```python
def lower_via_dispatcher(op, ctx_legacy):
    if type(op) in USE_DECLARATIVE:
        # New framework path
        new_ctx = LowerCtx.from_legacy(ctx_legacy)
        return new_ctx.lower(op)
    # Legacy fallback — disappears in Layer 3 cleanup
    return _legacy_imperative_lower(op, ctx_legacy)
```

Discipline test `test_use_declarative_is_monotonic`:

- Reads the `USE_DECLARATIVE` set from the current commit.
- Compares against the previous commit (or a pinned baseline).
- Fails CI if any op was removed.

## 6. Cleanup PR (Layer 3)

After all 60 MIR op types are in `USE_DECLARATIVE`:

```diff
-USE_DECLARATIVE: frozenset[type] = frozenset({...all 60 types...})
-
-def lower_via_dispatcher(op, ctx_legacy):
-    if type(op) in USE_DECLARATIVE:
-        new_ctx = LowerCtx.from_legacy(ctx_legacy)
-        return new_ctx.lower(op)
-    return _legacy_imperative_lower(op, ctx_legacy)
-
-def _legacy_imperative_lower(op, ctx):
-    ...  # 2500 LOC deleted
```

Plus:

- Delete `LoweringCtx` (replaced by `LowerCtx`).
- Delete `lower_scan_pipeline` (replaced by the `LoweringPass`
  dispatcher).
- `compile_kernel_body` reduces to one call:
  `compiler.run(prog, pipeline=DEFAULT_PIPELINE)`.

The Layer 3 PR should not introduce NEW logic — it's pure deletion +
the `compile_kernel_body` simplification.

## 7. Risks + mitigations

| Risk | Mitigation |
|---|---|
| The `Scope` parameter design doesn't fit some legacy lowerings (e.g., the negation pre-narrow tracking) | F3 (LowerCtx skeleton) PR establishes Scope; migrate the Scope-easiest ops first (Filter, ConstantBind) to validate before tackling Negation |
| Name-counter bump order differs between legacy and new path | Per-PR byte-equivalence test catches it; if irreducible, the new ordering can be documented and goldens updated (owner sign-off) |
| Some MIR ops have such complex lowering that one file becomes large again | Split into helpers within the same file. The discipline rule is "one `@lowering` per file", not "the file is small". Helpers are fine. |
| Two parallel agents pick the same MIR op | Wave 2A's branch-naming convention (`feat/lower-mir-<op>`) makes conflicts visible; single-line `USE_DECLARATIVE` additions create predictable merge conflicts that signal the duplication |

## 8. Sign-off

Phase B is complete iff:

- [ ] All 10 Wave 2A PRs (one per MIR op family) merged.
- [ ] Layer 3 cleanup PR merged: `lower_scan_pipeline` deleted,
  `LoweringCtx` deleted, `USE_DECLARATIVE` flag deleted.
- [ ] `test_every_mir_op_has_lowering` (R4) passes (no longer xfail).
- [ ] `compile_kernel_body` is < 50 LOC and contains no `if
  isinstance(op, ...)` chains.
- [ ] Byte-equivalence preserved across the entire fixture set (or
  documented divergences with owner sign-off).
- [ ] `LowerCtx` still has exactly 5 fields (D10).

After Phase B sign-off, Phase C (pragma materialization) and Phase D
(HIR passes onto framework) unblock.
