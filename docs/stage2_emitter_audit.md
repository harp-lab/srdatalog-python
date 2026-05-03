---
orphan: true
---

# Stage 2 Spec Extraction: Existing Emitter Audit

**Status:** Stage 2 entry artifact. No code is ported until the audit
locates every emission path the byte-equivalence gate must reproduce
and maps it onto the IIR-sorted-array dialect from §8 of
[ir_lowering_semantics.md](./ir_lowering_semantics.md).

**Branch:** `feat/iir-sorted-array-cuda` (off `feat/ir-dialect-infra`).

**Companion docs:**
- [ir_lowering_semantics.md](./ir_lowering_semantics.md) — formal spec, IIR node set §8
- [design_principles.md](./design_principles.md) — discipline rules

## 1. Why this audit exists

Stage 2 ports the existing Python emitter to an IIR-sorted-array
dialect + target.cuda lowering, validated by **byte-equivalence**:
every existing test fixture must produce identical C++ from the new
path, modulo deterministic name renaming.

The risk: if the IR design (§8 of the spec) doesn't admit some
emission path the current emitter takes, byte-equivalence silently
fails on whichever fixture exercises that path. The audit's job is
to find those gaps **before any code is moved**, not after.

This document inventories:

- All emission template files and their public surface.
- All feature-flag axes and their interactions.
- The CodeGenContext state machine.
- The plugin-dispatched index expressions.
- Each MIR op's emission paths.
- A gap analysis vs §8 of the spec.
- The implementation order with risk gates.

## 2. Source coverage

Existing emitter footprint (~10 K LOC total, ~4 K LOC pure emission):

| File | LOC | Role | Audit status |
|---|--:|---|---|
| `codegen/jit/pipeline.py` | 466 | Top-level walker; nested-op dispatch | covered |
| `codegen/jit/root.py` | 967 | Root-level emitters (Scan, ColumnJoin, CartesianJoin, BlockGroup) | covered |
| `codegen/jit/instructions.py` | 1092 | Nested ColumnJoin, CartesianJoin, PositionedExtract | covered |
| `codegen/jit/kernel_functor.py` | 328 | Functor envelope; DedupTable struct | covered |
| `codegen/jit/scan_negation.py` | 284 | Scan, Negation, Aggregate emitters | covered |
| `codegen/jit/emit_helpers.py` | 332 | Filter, ConstantBind, InsertInto | covered |
| `codegen/jit/view_management.py` | 324 | View-decl block at kernel start | covered |
| `codegen/jit/context.py` | 544 | CodeGenContext state machine | covered |

Other JIT directory files (`compiler.py`, `compiler_ninja.py`,
`loader.py`, `cache.py`, `materialized.py`, `complete_runner.py`,
`orchestrator_jit.py`, `main_file.py`, `file.py`, `plugin.py`,
`indexes/two_level.py`) are build orchestration / plugin
infrastructure, not emission templates. Stage 2 keeps these
unchanged; the new compile entry point delegates to them after
emission.

## 3. Feature-flag axes (the 12 dispatch dimensions)

Every emitter in §2 checks some subset of these. The byte-equivalence
port must reproduce each branch.

| # | Flag | What it controls | Currently active? |
|--:|---|---|---|
| 1 | `ctx.is_counting` | Counting phase (count outputs) vs materialize phase (write) | yes |
| 2 | `ctx.scalar_mode` | Thread-per-row scalar kernel vs warp cooperative | yes |
| 3 | `ctx.inside_cartesian` | Seq-prefix (per-thread) vs cooperative-prefix (tile) | yes |
| 4 | `ctx.dedup_hash_enabled` | FNV+atomicCAS try_insert / winner-guard | yes |
| 5 | `ctx.tiled_cartesian_enabled` + valid_var | 2-source Cartesian ballot-coalesced writes | yes |
| 6 | `ctx.bg_enabled` (root CJ) | Block-group with binary-search + warp-row redistribution | yes |
| 7 | `plugin_view_count(version, index_type) > 1` | Multi-view source (e.g. Device2LevelIndex HEAD+FULL) | yes |
| 8 | `ctx.ws_enabled` | Work-stealing per-warp atomic counter | **no — raises NotImplementedError** |
| 9 | `ctx.bg_histogram_mode` | Block-group histogram-mode counting | **no — raises NotImplementedError** |
| 10 | `ctx.is_fan_out_explore` | Heterogeneous fanout mode | **no — raises NotImplementedError** |
| 11 | `ctx.is_leaf_level` | Tile dispatch at leaf level (no further ColumnJoins ahead) | yes |
| 12 | `ctx.neg_pre_narrow[src_idx]` | Pre-narrowed negation handle hoisted before Cartesian | yes |

**Stage 2 byte-equivalence scope** = flags 1–7 + 11–12. Flags 8–10
already raise; the new path must also raise on those, no semantic
work needed.

**Critical interactions** (combinations the emitter handles
specifically):

- `inside_cartesian + dedup_hash`: dedup goes through atomicAdd write
  position even inside Cartesian.
- `tiled_cartesian + multiple InsertIntos`: ballot computed once at
  first InsertInto (memoized via `tiled_cartesian_ballot_done`), reused
  for subsequent ones in the same body.
- `is_counting + tiled_cartesian + Cartesian-as-product`: short-circuits
  flat loop with `add_count(degree_product)` (only when body is
  pure InsertInto — see `_rest_is_safe_for_cartesian_product`).
- `is_counting + unused_var`: skip `auto v = get_val(...)` when `v`
  is not referenced in the body.
- `multi_view + Cartesian` (BG + non-first source): segment loops nest
  inside warp-row narrowing.

## 4. MIR op coverage

For each MIR op, the emission paths the emitter takes.

### 4.1 `Scan` (`scan_negation.py:jit_scan`)

Emission paths:
- **Default**: warp-strided grid-stride loop, prefix-narrow handle,
  bind vars per remaining column.
- Counting-mode optimization: skip `get_val` for vars not used in body.

State touched: `ctx.bound_vars`, `ctx.indent`. No `inside_cartesian`
interaction (Scan only fires at root or inside non-Cartesian scope).

### 4.2 `ColumnJoin` (`root.py:jit_root_column_join`, `instructions.py:jit_nested_column_join`)

Emission paths split by **position** (root vs nested) and **source count** (single vs multi):

| Position | Sources | Path |
|---|---|---|
| Root | 1 | Grid-stride over root handle's degree; bind var; emit child handle |
| Root | ≥ 2 | `IterUniqueRootValues`: iterate `root_unique_values[]`; first source uses `Hint(lo, hi, 0).prefix(root_val)`; remaining sources use `prefix(root_val)`; emit `IntersectHandles` |
| Root + BG | ≥ 2 | Block-group: binary-search `bg_cumulative_work[]`, warp-row redistribution, narrow first source via `HandleType(begin+lo, begin+hi, depth)`, then segment loops for multi-view non-first sources |
| Nested | 1 | Grid-stride over narrowed parent handle; emit child handle |
| Nested | ≥ 2 | `intersect_handles({h1, h2, …})`; segment loops for multi-view "fresh" sources |

**Multi-view variant**: when any source has `plugin_view_count > 1`,
wrap the whole body in `for (_seg = 0; _seg < view_count; ++_seg)`.
For the multi-source path, segment loops nest only around fresh
(prefix-empty) sources; prefixed sources stay inside the parent's
segment.

### 4.3 `CartesianJoin` (`root.py:jit_root_cartesian_join`, `instructions.py:jit_nested_cartesian_join`)

Emission paths:

- **Standard Cartesian**: degree decomposition (1-src identity, 2-src div/mod, N-src countdown), grid-stride over flat product, bind per-source vars.
- **Tiled Cartesian** (root + 2-source + 1-var-per-source + not counting): emit dual body — tiled (ballot) and fallback (direct) — inside a `if (total > 32) { tiled } else { fallback }` dispatch.
- **Cartesian-as-product** (counting + body is pure InsertInto): replace flat loop with `add_count(degree_product)`.
- **Negation pre-narrow**: any `Negation` immediately following Cartesian gets its prefix split into pre-Cartesian (constant) and in-Cartesian (per-thread) parts. Pre-Cartesian narrowing emits before the Cartesian loop body.

State touched: `ctx.inside_cartesian = True`, `ctx.cartesian_bound_vars`, `ctx.cartesian_as_product`, `ctx.tiled_cartesian_valid_var`, `ctx.tiled_cartesian_ballot_done`.

### 4.4 `Filter` (`emit_helpers.py:jit_filter`)

Emission paths:

- Inside WS Cartesian batch: `valid &= (cond);`
- Inside tiled Cartesian: `valid &= (cond);` (folded into ballot)
- Default: `if (cond) { body }`

### 4.5 `ConstantBind` (`emit_helpers.py:jit_constant_bind`)

Single path: `auto var = code; <body>`.

### 4.6 `Negation` (`scan_negation.py:jit_negation`)

Emission paths:

- **Pre-narrowed** (`ctx.neg_pre_narrow[src_idx]` set): use pre-computed handle, optionally apply in-Cartesian additional prefix calls.
- **Standard narrowing**:
  - Apply const_args first (HIR indexCols order).
  - Then variable prefixes:
    - `inside_cartesian` → `prefix_seq(...)` (sequential)
    - else → `prefix(..., tile, ...)` (cooperative)
- **Body emission**:
  - WS batch loop → fold into `valid &= !handle.valid()`.
  - Tiled Cartesian → fold into `valid &= !handle.valid()`.
  - Default → `if (!valid) { body }`.

### 4.7 `Aggregate` (`scan_negation.py:jit_aggregate`)

Single path: `auto result_var = aggregate<Func>(handle, view); <body>`.
Specialization for COUNT/SUM/MIN/MAX done in the C++ runtime, not at
emission.

### 4.8 `InsertInto` (`emit_helpers.py:jit_insert_into`) — most complex

Decision tree:

```
InsertInto (out_var resolved from ctx.output_vars[rel_name] or default)
├─ inside_cartesian == False
│  └─ wrap in lane-0 guard: if (tile.thread_rank() == 0) { body }
├─ dedup_hash_enabled
│  └─ open: { bool _p = dedup_table.try_insert(...); if (_p) { body } }
├─ Counting phase (is_counting)
│  ├─ skip-counting marker (out_var == "__skip_counting__")
│  ├─ ws_enabled       → out_var++
│  └─ default          → out_var.emit_direct()
└─ Materialize phase
   ├─ ws_cartesian_valid_var  → emit_warp_coalesced(tile, valid, vars...)
   ├─ tiled_cartesian_valid_var
   │  ├─ ballot_done == False (first InsertInto in body)
   │  │  └─ ballot, popcount, write at output_data[col*stride + pos]
   │  │     set ballot_done = True
   │  └─ ballot_done == True (subsequent InsertIntos)
   │     └─ reuse ballot, write at incremented offset
   ├─ dedup_hash_enabled
   │  └─ atomicAdd(atomic_write_pos, 1u); column writes
   └─ default
      └─ out_var.emit_direct(vars...)
```

### 4.9 `PositionedExtract` (`instructions.py:jit_positioned_extract`)

Used after BalancedScan at root. Currently only nested-position is
supported (root BalancedScan raises NotImplementedError). Emits a
point lookup with optional bind_vars wrapped in extra for-loop.

### 4.10 `BalancedScan` (root)

**Raises NotImplementedError** — out of Stage 2 scope.

## 5. CodeGenContext state machine

The context is a mutable bag carried through the recursive emit walk.
Mutation is delicate; some fields are checkpoint/restore (manual), some
are scope-local appended/popped, some are global counters.

### 5.1 Fields by purpose

**Scope tracking**:
- `bound_vars: list[str]` — vars in current emit scope (Scan, Aggregate, ConstantBind results); appended/popped.
- `cartesian_bound_vars: list[str]` — vars bound by current Cartesian; pushed at root, restored on exit.
- `ws_cartesian_bound_vars: list[str]` — same, for WS path.

**Handle / view registries**:
- `handle_vars: dict[str, str]` — `str(handle_start)` → handle var name, plus semantic state keys.
- `view_vars: dict[str, str]` — spec-key + `str(handle_start)` → view var name (multiple aliases).
- `view_slot_offsets: dict[int, int]` — `handle_start` → base view slot.
- `rel_index_types: dict[str, str]` — relation name → custom index type ("" = DSAI default).

**Indent / unique names**:
- `indent: int` — indentation level (2-space units); mutated in place by `inc/dec_indent`.
- `name_counter: int` — global per-context unique-name counter; `gen_unique_name(prefix)` bumps and returns `prefix_<n>`.

**Phase / mode flags**:
- `is_counting: bool` — counting vs materialize.
- `scalar_mode: bool` — thread-per-row leaf kernel.
- `is_leaf_level: bool` — set by ColumnJoin emitter when no more CJs ahead.
- `is_jit_mode: bool` — JIT vs AOT.
- `debug: bool` — emit debug comments.

**Cartesian / negation state**:
- `inside_cartesian: bool` — current op is inside Cartesian; toggles seq vs coop prefix.
- `cartesian_as_product: bool` — count-as-product short-circuit eligible.
- `neg_pre_narrow: dict[int, NegPreNarrowInfo]` — pre-narrowed negation handles.

**Tiled-Cartesian state**:
- `tiled_cartesian_enabled: bool` — feature flag.
- `tiled_cartesian_valid_var: str` — valid-flag var name (e.g., `_tc_valid_3`).
- `tiled_cartesian_ballot_done: bool` — ballot computed for current body, reused.

**Dedup-hash state**:
- `dedup_hash_enabled: bool`.
- `dedup_hash_vars: list[str]` — column vars used as hash key.

**Block-group state**:
- `bg_enabled: bool` — root-level BG enabled.
- `bg_histogram_mode: bool` — raises NotImplementedError.
- `bg_warp_begin_var, bg_warp_end_var, bg_cumulative_var, bg_done_var: str`.
- `cartesian_as_product: bool` — also used by BG histogram.

**Work-stealing state** (raises NotImplementedError):
- `ws_enabled, ws_level, ws_queue_var, ws_range_board_var, ws_live_handles, ws_has_cartesian, ws_cartesian_valid_var`.

**Fan-out explore**: `is_fan_out_explore` (raises NotImplementedError).

**Output**:
- `output_var_name: str` — single-output legacy default ("output").
- `output_vars: dict[str, str]` — relation name → output object var.

**Tile / group**:
- `group_size: int` — usually 32.
- `tile_var, parent_tile_var: str`.

**Hooks**: `hooks: CodeGenHooks` — per-feature override modules (BG, WS, dedup) supply custom emit functions.

### 5.2 Save/restore patterns

The emitter uses three patterns for scope management:

1. **Direct mutation + manual restore**: `inside_cartesian`, `cartesian_bound_vars`, `cartesian_as_product`, `bg_enabled`, `tiled_cartesian_valid_var`, `tiled_cartesian_ballot_done`. Caller saves the previous value, sets new, calls recursive emit, restores in `finally`.
2. **Counter checkpoint**: `name_counter` saved/restored across body emission so root's subsequent unique-names don't see the body's bumps. Critical for byte-match.
3. **Append/pop**: `bound_vars` items appended on entry, popped on exit. `handle_vars` entries inserted under multiple keys (numeric + semantic), removed on exit.

The pattern is consistent enough that the IIR can replace it with
**lexical Let-bindings** (D8 in design_principles.md). This is one of
the simplifications Stage 2 introduces: no more string-keyed handle
dicts.

## 6. Plugin-dispatched expression helpers

These are the index-type-polymorphic emission functions; each one
checks `ctx.rel_index_types` to dispatch to the right plugin.
Wrappers in `context.py`:

| Wrapper | Plugin signature | Default (DSAI) emission |
|---|---|---|
| `gen_root_handle(view, index_type)` | `plugin_gen_root_handle` | `HandleType(0, view.num_rows_, 0)` |
| `gen_degree(handle, index_type)` | `plugin_gen_degree` | `(handle.end() - handle.begin())` |
| `gen_valid(handle, index_type)` | `plugin_gen_valid` | `handle.is_valid()` |
| `gen_get_value_at(handle, view, idx, index_type)` | `plugin_gen_get_value_at` | `view.get_value_at(handle.begin(), idx)` |
| `gen_get_value(view, col, pos, index_type)` | `plugin_gen_get_value` | `view.col_data[col][pos]` |
| `gen_child(handle, idx, index_type)` | `plugin_gen_child` | `handle.child(idx)` |
| `gen_child_range(handle, pos, key, tile, view, index_type)` | `plugin_gen_child_range` | `handle.child_range(...)` |
| `gen_iterators(handle, view, index_type)` | `plugin_gen_iterators` | iterator pair |
| `gen_chained_prefix_calls(parent, prefix_vars, view, cart_vars, scalar, index_type)` | `plugin_chained_prefix_calls` | `.prefix(v1).prefix(v2)…` (coop or seq depending on flags) |
| `gen_chained_prefix_with_last_lower_bound(...)` | `plugin_chained_prefix_with_last_lower_bound` | `.prefix(...).prefix_lower_bound(last)` |

**Implication for IIR design**: each handle op (`Pref`, `Child`,
`Hint`, `Root`, etc.) carries the `index_type` implicitly via the
view it operates on. Lowering to target.cuda dispatches per
`index_type` — this is **already** how the existing emitter works.
Property P2 (index polymorphism via view params, not via base class)
is preserved.

## 7. Mapping to IIR nodes (gap analysis)

Cross-referencing the emission paths in §4 against the IIR node set
in §8 of [ir_lowering_semantics.md](./ir_lowering_semantics.md).

### 7.1 Covered by §8 as-is

| Emission path | IIR node(s) |
|---|---|
| Root Scan | `ParallelFor(par.warp_strided, ScanGS(j, deg, body))` |
| Root single-source CJ | Same as Scan + `Bind(child=sa.child(H, j))` |
| Root multi-source CJ | `IterURV(driver, IntersectHandles({sa.pref(...)}, body))` |
| Root multi-source CJ + hint | `IterURV(driver, IntersectHandles({sa.pref(sa.hint(...), root_val), …}, body))` |
| Nested CJ | `ScanGS(j, sa.degree(H_p), body)` |
| Standard Cartesian | `Cartesian(vars, sources, IndexDecompose, Bind, body)` |
| Filter (default) | `If(p, body)` |
| ConstantBind | `Let(v, e, body)` |
| Negation (default) | `Let(H = sa.pref(...), If(¬sa.exists(H), body))` |
| Negation pre-narrow | Same with `Let(H = ...)` hoisted out of Cartesian — natural under D8 lexical scope |
| Aggregate | `Let(H = sa.pref(...), Let(v = Fold(f, sa.values(H)), body))` |
| InsertInto (default counting) | `Phase(C, AddCount(1))` |
| InsertInto (default materialize) | `Phase(M, WriteOutput(rel, vals, sr))` |
| Multi-view source | `IterSeg(view_count, body)` |
| Counting phase | `Phase(C, body)` |
| Materialize phase | `Phase(M, body)` |
| Dedup-hash scope | `WithDedup(table, body)` |

### 7.2 Gaps — IIR design needs new nodes / fields

These emission paths in the existing emitter don't have a direct
node in §8 of the spec. They need to be added.

#### G1 — Tiled Cartesian (ballot-based coalesced writes)

The existing emitter has a specialized path for 2-source Cartesians
with single-var-per-source: emit a **dual body** (tiled + fallback)
inside `if (total > 32) { tiled } else { fallback }`. The tiled body
uses `tile.ballot()` for offset computation; the InsertInto inside
shares the ballot across multiple writes.

**Proposed IIR node**:

```
TiledCartesian(vars, sources, tiled_body, fallback_body)
```

with verifier-enforced constraints: `len(sources) == 2`, single var
per source, `tiled_body` and `fallback_body` denote the same multiset.

Lowering rule: the same IIR program is emitted twice — once with a
"ballot context" active (which `WriteOutput` consumes), once without.
The dual body is selected at runtime by the `if (total > 32)` guard
(emitted directly by the target.cuda lowering of `TiledCartesian`).

#### G2 — Cartesian-as-product short-circuit

`Phase(C, Cart(vars, sources, [II]))` → `AddCount(∏ degrees)`. This
is rule **R1** in §11 of the spec, already specified as a rewrite. No
new node needed; the rewrite engine fires it before emission.

**Status**: covered by R1.

#### G3 — Block-group root ColumnJoin partitioning

The existing block-group path does:

1. Block-level work assignment via `bg_work_per_block`.
2. Binary search on `bg_cumulative_work[]` to find starting key.
3. Warp-row redistribution within the chosen key.
4. Narrow the first-source handle to the warp's row range
   (`HandleType(narrow_begin, narrow_end, depth)` constructor).
5. Segment loops for 2-level non-first sources nested inside.

**Proposed IIR shape**: `par.data.block_group` is already in §16 as a
parallelism dialect. The block-group lowering produces:

```
ParallelFor(par.data.block_group(cumulative_work), 
   IterKeyRange(...,
     WarpRowDistribute(deg, work_range, narrow,
       Bind(root_val = ...,
         IntersectHandles({sa.pref(sa.narrow(H_first, narrow_range), root_val), …}, 
            body)))))
```

The `WarpRowDistribute` and `NarrowHandle` ops are already mentioned
in §8. **What's missing**: an explicit `IterKeyRange` node distinct
from `IterURV` — the BG variant iterates a key range with cumulative
work tracking, not just unique-root-values one-by-one.

**Proposed addition**: `IterKeyRange(driver, cumulative_work, body)`.

#### G4 — Lane-0 guard

Outside Cartesian, the emitter wraps single-thread writes in
`if (tile.thread_rank() == 0) { ... }` to avoid 32x overcount. This
is target-cuda-specific; the IIR doesn't need a node. Add to
**target.cuda lowering rules** as a wrapper applied to
`WriteOutput` when not under a `Cartesian` ancestor.

**Status**: target lowering concern; no IR node.

#### G5 — Cooperative vs sequential prefix

`prefix(k, tile, view)` (cooperative) vs `prefix_seq(k, view)`
(sequential). Dispatch by `inside_cartesian` flag. **Already
expressible**: `sa.pref(H, k)` lowers per surrounding scope — the
target.cuda lowering inspects whether the emit point is inside a
Cartesian scope and picks the right C++ form.

**Status**: target lowering concern; no IR node.

#### G6 — `prefix_lower_bound` (chained_prefix_with_last_lower_bound)

Used in some balanced-scan paths and aggregate range scans. Variant
of `sa.pref` where the last key uses `prefix_lower_bound` instead
of `prefix`.

**Proposed addition**: `sa.pref_lb(H, k)` op in the sorted_array
dialect — "narrow by lower bound on key". Used in tail-position of
prefix chains.

#### G7 — Multi-aliased view variables

`view_vars` registers each view under both a numeric handle key and
multiple semantic keys (rel_VER_cols, etc.). This is a string-keyed
side table.

**Status**: replaced by D8 lexical Let-bindings of view variables.
The IR doesn't need it.

#### G8 — Counter checkpoint for unique names

The existing emitter saves/restores `name_counter` across body
emission so root's subsequent unique-names start from the original
counter. This is a *byte-equivalence* concern — the C++ string output
must use specific names like `y_idx_1` not `y_idx_9`.

**Proposed approach**: the IIR emission walker carries a
deterministic name allocator per scope. The target.cuda lowering
serializes IIR ops in a fixed order; name generation follows the
same checkpoint/restore pattern but as a structured property of the
emission walker, not state in the IIR.

**Status**: target lowering concern. The IR is unique-name-agnostic;
target.cuda's emitter checkpoints the counter.

#### G9 — Multi-view segment loop nesting

The existing emitter nests segment loops only around "fresh"
(prefix-empty) sources in multi-source CJ. Prefixed sources stay
inside the parent's segment.

**Proposed IIR shape**: `IterSeg(view_count, body)` already in §8.
The lowering rule for multi-source CJ chooses where to insert the
`IterSeg` based on each source's prefix structure. Need to specify
this in the lowering rule explicitly:

> When a ColumnJoin has multiple sources of which some are multi-view
> and "fresh" (no prefix vars), wrap the segment loop only around the
> fresh source's narrowing inside the body. Prefixed multi-view
> sources have the segment loop hoisted to the outer scope.

**Status**: lowering-rule clarification; no new IR node.

#### G10 — Tiled-Cartesian ballot reuse memoization

`ctx.tiled_cartesian_ballot_done` flag: first InsertInto in the
tiled body computes ballot+popcount; subsequent ones reuse the
result. Emit-level sharing of `_tc_off`.

**Proposed IIR shape**: emission state inside `TiledCartesian`'s
target.cuda lowering. The IR carries the dual body; the target
emitter manages "first vs subsequent InsertInto in this body". No
IR node needed.

**Status**: target lowering concern.

#### G11 — `WS Cartesian` (NotImplementedError)

Out of Stage 2 scope. The IIR design should leave room (e.g., a
`par.task.ws` strategy slot) but emission for this path raises
NotImplementedError, matching current behavior.

#### G12 — `bg_histogram_mode` and `is_fan_out_explore` (NotImplementedError)

Out of Stage 2 scope. Same as G11.

### 7.3 Additions to §8 of ir_lowering_semantics.md

Summary of IIR additions needed:

| New node | Purpose | Source gap |
|---|---|---|
| `TiledCartesian(vars, sources, tiled_body, fallback_body)` | 2-source ballot-coalesced Cartesian path | G1 |
| `IterKeyRange(driver, cumulative_work, body)` | BG-style key-range iteration | G3 |
| `sa.pref_lb(H, k)` | Lower-bound prefix variant | G6 |

These three additions extend the dialect by 3 ops. No removals from
§8. The §8 spec needs to be patched (post-Stage-2 commit) to
include them.

## 8. Stage 2 implementation order

The audit dictates a specific port order to make the byte-equivalence
gate fall out incrementally rather than in one big-bang verification.

### 8.1 Per-rule milestone gates

Each milestone introduces one or more lowering rules and adds the
corresponding fixtures to the byte-equivalence test set. **A
milestone is not done until its byte-equivalence tests all pass.**

| Milestone | Rules added | Byte-equiv fixtures |
|---|---|---|
| M1: Scan | §10.1 lowering | tests with single-Scan rules |
| M2: Filter, ConstantBind | §10.6, §10.7 lowerings | tests with simple filtering |
| M3: Single-source CJ root + nested | §10.2, §10.4 | TC base case |
| M4: Multi-source CJ + IterURV + Hint | §10.3 | TC recursive case |
| M5: Negation | §10.8 (default + pre-narrow) | tests with `not edge(...)` |
| M6: Aggregate | §10.9 | tests with `count {…}` |
| M7: Standard Cartesian | §10.5 | tests with cross-products |
| M8: Tiled Cartesian | new G1 node | tiled-cartesian-eligible fixtures |
| M9: Block-group | new G3 node | BG-enabled fixtures |
| M10: Multi-view (segment loops) | G9 lowering rule | Device2LevelIndex fixtures |
| M11: Dedup-hash | `WithDedup` lowering | dedup-hash fixtures |
| M12: PositionedExtract | nested-only | (small, may roll into M3) |
| M13: NotImplementedError parity | G11/G12 stubs | tests asserting raises |

### 8.2 Per-milestone artifacts

Each milestone produces:

1. **Lowering function** in `dialects/sorted_array/lowerings.py` (or `target_cuda/lowerings.py` for target-side rules).
2. **Tests**: byte-equivalence fixture run for every existing test that exercises this path.
3. **mypy + ruff clean**.
4. **Discipline tests still pass**.

### 8.3 Branch / commit cadence

- Each milestone is one or more commits on `feat/iir-sorted-array-cuda`.
- Milestones merge to the branch in order; the byte-equivalence test
  set grows monotonically.
- After M13, the branch is rebased / squashed for review and merged
  to `feat/ir-dialect-infra`, then to `main`.

## 9. Byte-equivalence test strategy

### 9.1 Test shape

For every existing emitter fixture, emit C++ via the **old path**
(`jit_kernel_full(node)`) and the **new path**
(`compiler.compile_pipeline(node)`). Diff the strings.

```python
def test_byte_equivalence(fixture):
    mir_node = build_fixture(fixture)
    old_cpp = jit_kernel_full(mir_node)
    new_cpp = compiler.compile_pipeline(mir_node)
    assert _normalize(old_cpp) == _normalize(new_cpp), \
        f"byte-equivalence failed on {fixture.name}"
```

### 9.2 Normalization

The old emitter's name allocator is global (per-context). The new
emitter has scope-local name allocation. Some divergence in unique
names is unavoidable. Normalization rewrites both emissions to
canonical names (e.g., `auto x_idx_<n>` → `auto x_idx`) before diff.

This is the **only** allowed normalization. Anything else means the
new emission is materially different — IR design bug.

### 9.3 Fixture coverage

Every test in `tests/` that calls `jit_kernel_full` or
`jit_kernel_definition` becomes a byte-equivalence fixture. List
will be enumerated in the M0 commit (test harness setup).

### 9.4 Failure mode

When a fixture fails byte-equivalence, the failure points to a
specific milestone's lowering rule. Triage:

1. Diff localizes the divergence to a specific C++ snippet.
2. Snippet maps to a feature flag / op kind via §3 / §4 of this
   audit.
3. The corresponding IIR lowering rule is reviewed and corrected.
4. **Do not relax the gate.** If the IR design genuinely cannot
   express the emission, that's a §8 spec bug; patch the spec first,
   then update the lowering.

## 10. Risks identified

### 10.1 Scope creep

Stage 2 explicitly excludes WS, BG histogram, fan-out, BalancedScan
root. If a new fixture appears that exercises any of these,
**defer**, do not in-scope it.

### 10.2 Index-plugin churn

The plugin-dispatched expression helpers (§6) live in
`codegen/jit/plugin.py`. The new dialect calls these via the same
plugin registry — preserving Property P2 — but if the plugin
interface changes during the port, byte-equivalence breaks.
**Mitigation**: freeze plugin interface for Stage 2; don't change it
until M13.

### 10.3 Ballot reuse in tiled Cartesian (G10)

The first-vs-subsequent `tiled_cartesian_ballot_done` memoization is
target-emission state, not IR state. Risk: the new target.cuda
lowering must replicate this exactly, including the placement of
the `_tc_off` variable. **Mitigation**: M8 has a dedicated
byte-equivalence test for fixtures with multiple InsertIntos in a
single tiled body.

### 10.4 BG `HandleType(begin+lo, begin+hi, depth)` direct construction

The block-group path constructs handles directly via the
`HandleType(lo, hi, depth)` constructor, bypassing the standard
`Root().prefix()` API. This is sorted-array-specific. If we lift
this to a generic IR op (`NarrowHandle`), it must be implementable
by every `CNodeHandle`-conforming index. **Mitigation**: keep
`NarrowHandle` in the sorted_array dialect only; other dialects
can choose not to expose it.

### 10.5 Counter checkpoint divergence

If the new emitter's name allocator drifts from the old one's
checkpoint/restore pattern, byte-equivalence fails uniformly across
many fixtures. **Mitigation**: M1 establishes the canonical
name-allocation pattern and the normalization hook (§9.2). All
subsequent milestones use the same approach.

## 11. Open questions

Q1. **PositionedExtract** is currently emitted only in nested
position. Is this a complete path or a partial Nim port? The
existing emitter raises NotImplementedError for root BalancedScan —
PositionedExtract may transitively be unreachable in current
fixtures. Audit M0: enumerate fixtures touching PositionedExtract;
if zero, defer to post-Stage-2.

Q2. **`_skip_counting__` marker**. The InsertInto emitter has a
secondary-output skip-counting branch (line 201 of emit_helpers.py).
Is this exercised by any fixture? If not, defer; if yes, model as
a `WriteOutput` flag in the IR.

Q3. **HIR / MIR migration**. Stage 2 does not migrate HIR/MIR to
dialects. After Stage 4 demonstrates pluggability, Stage 5+ may
migrate them. The audit pre-commits to keeping HIR/MIR external for
Stage 2.

Q4. **Symbol table population**. Section 2 of `Compiler cooperation
discussion` implies a `SymbolTable` mapping `relation_name → dialect`.
For Stage 2 (sorted-array-only), the table is trivially populated
from `ctx.rel_index_types`. M0 includes wiring the symbol table
construction inside `compile_pipeline()`.

## 12. Next action

**M0 — Test harness setup.** Before any dialect code lands:

1. Enumerate every existing test fixture that calls `jit_kernel_full`
   or `jit_kernel_definition`; mark each by which feature flags it
   exercises (cross-reference §3 of this audit).
2. Build the byte-equivalence diff harness with the normalization hook.
3. Make sure all current fixtures still pass through the old path
   (regression baseline).
4. The new compile entry point `compile_pipeline(node, target='cuda')`
   exists and currently delegates to `jit_kernel_full` (back-compat
   shim). Until M1 lands, the new path is the old path.

M0 is one commit. Then M1 (Scan lowering) starts the byte-equivalence
gate.
