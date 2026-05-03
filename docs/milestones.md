# Milestone tracker

Live status for the Stage 2 dialect migration + follow-on cleanup. Companion
to [`stage2_emitter_audit.md`](./stage2_emitter_audit.md) (the original
audit) and [`ir_lowering_semantics.md`](./ir_lowering_semantics.md) (the
formal lowering spec).

**Goal of Stage 2:** retire the legacy `codegen/jit/pipeline.py` emit chain
in favor of a dialect-based IR pyramid (HIR → MIR → IIR → CUDA C++).

**Verification gate:** every kernel body in every test fixture must lower
through the dialect and produce byte-identical C++ to the upstream Nim
reference, modulo `_cpp_norm` whitespace/comment normalization.

## Status snapshot

| Layer | Done | Outstanding |
|---|---|---|
| Stage 1 (dialect skeleton, M1–M7.x) | ✅ | — |
| Stage 2 (legacy-emit retirement) | ✅ N1, N2, N3.x, N4.x, N5.0–N5.2, N6, N7, N8 | ⬜ N5.3, N5.4 |
| Legacy elimination | ✅ all `codegen/jit/{pipeline,instructions,root,scan_negation,kernel_functor,file}.py` deleted | — |
| Layout reorg | ✅ Phase A (`codegen/jit/` → `ir/dialects/target/cuda/`), Phase B (`ir/` namespace) | — |
| Docs / test rename sync | ✅ README + 10 `test_jit_*.py` → `test_cuda_*.py` | — |
| Open work | — | WS full runner, CPU/WASM target, HIR/MIR as proper dialects, `complete_runner.py` templating |

**Test gates currently passing:**
- 249/249 runner byte-equivalence (`tests/test_runner_byte_equivalence.py`) — 2 skipped (WS runner unported)
- 253/253 jit-batch byte-equivalence (`tests/test_byte_equivalence_jit.py`)
- 1033 / 1033 total in suite (2 unrelated skips)

---

## Stage 1 — dialect skeleton (pre-Stage 2)

Built the IR core, MIR types, and progressively-extended dialect lowering
of the simplest pipeline shapes against byte-equivalence harness.

| ID | Status | Description |
|---|---|---|
| M0 | ✅ | byte-equivalence gate + dialect skeleton |
| M1 | ✅ | `[Scan, InsertInto]` |
| M2 | ✅ | `[Scan, (Filter \| ConstantBind)*, InsertInto]` |
| M3 | ✅ | multi-source ColumnJoin (root + nested) |
| M5 | ✅ | Negation (baseline, no pre-narrow) |
| M5.x | ✅ | Negation pre-narrow + 3+-source CartesianJoin |
| M5.y | ✅ | complete byte-equivalence (252/252 jit_batch goldens) |
| M6 | ✅ | multi-InsertInto pipelines |
| M7 | ✅ | nested CartesianJoin |
| M7.x | ✅ | root CartesianJoin |

Stage 1 ended with `compile_pipeline` switched to route through the dialect
for all supported shapes; legacy pipeline.py remained as a fallback.

---

## Stage 2 — retire the legacy emit chain

Each milestone moves one shape from "still goes through `jit_pipeline`" to
"natively lowered in the dialect". Verified after each milestone via the
runner byte-equivalence suite.

| ID | Status | Commit | Description |
|---|---|---|---|
| N1 | ✅ | `8f399c7` | Phase-aware count emission in dialect |
| N2 | ✅ | `cc356df` | Fused kernel emit via dialect (single-pass materialize) |
| N3.0 | ✅ | (folded into N3) | Drop `_dialect_safe_kernel` legacy fallback for count/materialize |
| N3 | ✅ | `c2662c2` | Runner gate + dialect kernel-body emit for non-D2L runners |
| N3.2 | ✅ | `d03e4de` | Lift runner namespace + leaf phase emitters into dialect |
| N4.0 | ✅ | `c107529` | Establish `par.data.block_group` dialect, lift BG histogram emit |
| N4.1 | ✅ | `6253ed0` | BG root multi-source CJ via dialect (`BgRootCjMulti` op + emit) |
| N5.0 | ✅ | `0c7955b` | Establish `relation.d2l` dialect, plug view-count slot accounting |
| N5.1 | ✅ | `a343938` | `D2lSegmentLoop` op + multi-source nested CJ over D2L FULL_VER |
| N5.2 | ✅ | `9fa764a` | D2L segment loop in root CJ multi non-first source |
| **N5.3** | ⬜ | — | **Single-source nested CJ over D2L FULL_VER** — `_lower_nested_cj_single` doesn't exist yet; the legacy `_nested_column_join_single` had its own segment-loop wrap. **No fixtures exercise this today;** falls through `_supported_pipeline` as "unsupported". |
| **N5.4** | ⬜ | — | **Scan / Negation / Aggregate rooted at D2L FULL_VER** — those root ops over a multi-view source need their own segment-loop wrap. **No fixtures exercise this today;** Scan-rooted with multi-view source is rejected by `_supported_pipeline`. |
| N6 | ✅ | `3cc879d` | Dedup-hash WriteOutput variant via dialect |
| N7 | ✅ | `68df9f4` | Tiled-Cartesian dispatch + ballot-coalesced writes via dialect |
| N8 | ✅ | `f586d25` | Kernel-functor work-stealing emit variants via dialect (`<out>++` count, `emit_warp_coalesced` materialize, Filter/Negation valid-flag folding) |

### What still routes through legacy after N1–N8 + cleanup

After commit `3fe98c4` (Cleanup: delete legacy codegen/jit emit chain),
**none** of the legacy `codegen/jit/pipeline.py` chain is reachable from
production code. Every kernel body in every fixture lowers through the
dialect.

The `_dialect_safe_kernel` predicate that previously gated which kernels
would fall back to legacy was removed — the dialect now handles everything
in the test set. N5.3 / N5.4 are the only IR shapes the dialect explicitly
doesn't lower; if a workload exercises one, `_supported_pipeline` raises
loudly rather than silently producing wrong C++.

---

## Cleanup commits (post-Stage 2)

| ID | Commit | Description |
|---|---|---|
| Cleanup-A | `34b3161` | Drop unreachable legacy fallback paths in `complete_runner` (`_dialect_safe_kernel`, `_make_kernel_ctx`, fallback else-branches) |
| Cleanup-B | `3fe98c4` | Delete legacy `codegen/jit/{pipeline,instructions,root,scan_negation,kernel_functor,file}.py` + 10 unit-test files exclusively testing them |
| Phase A | `3409cdf` | Kill `codegen/jit/` namespace — move CUDA target backend into `ir/dialects/target/cuda/` (with `build/` and `plugins/` sub-packages); D2L plugin lives at `ir/dialects/relation/d2l/cuda.py` (auto-registers on D2L dialect import) |
| Phase B | `ace889d` | Group IR layers under `ir/` (`ir_core/` → `ir/core/`, `hir/` → `ir/hir/`, `mir/` → `ir/mir/`, `dialects/` → `ir/dialects/`) |
| Docs sync | `2e1d032` | README architecture refresh + `test_jit_*` → `test_cuda_*` rename |

---

## N5.3 / N5.4 — what's needed if/when they land

### N5.3 — single-source nested CJ over D2L FULL_VER

**MIR shape:** `[..., ColumnJoin(sources=[X], var_name=v), ..., InsertInto]` where `X` is a D2L FULL_VER source with no prefix vars.

**Legacy emit shape** (from the deleted `_nested_column_join_single`):

```cpp
for (int _seg = 0; _seg < 2; _seg++) {            // segment loop wrap
  auto view_X = views[base_slot + _seg];           // shadowed view per segment
  auto h_X = HandleType(0, view_X.num_rows_, 0);   // root handle from new view
  if (!h_X.valid()) continue;
  uint32_t degree_X = h_X.degree();

  for (uint32_t idx_X = 0; idx_X < degree_X; ++idx_X) {
    auto v = view_X.get_value_at(h_X, idx_X);
    auto ch_X = h_X.child(idx_X);
    <body>
  }
}
```

**Dialect work:**
- Add a `_lower_nested_cj_single(cj_op, rest, ctx)` to `relation/sorted_array/lowerings.py`
  (currently `_lower_inner_chain` only dispatches to `_lower_nested_cj_multi` for
  `len(sources) >= 2`).
- When the source is multi-view fresh, wrap with `D2lSegmentLoop(declare=True)`
  (the shadowing variant — see `D2lSegmentLoop`'s docstring for the two-shape
  toggle).
- Without segment loop, the single-source nested CJ degenerates to a
  GridStrideLoop-like body — straightforward dialect composition.

### N5.4 — Scan / Negation / Aggregate over D2L FULL_VER

**MIR shapes:**
- `[Scan(rel=X, ver=FULL), ..., InsertInto]` with X declared `Device2LevelIndex`
- `[..., Negation(rel=X, ver=FULL), ..., InsertInto]` with X D2L FULL
- `[..., Aggregate(rel=X, ver=FULL), ..., InsertInto]` with X D2L FULL

**Legacy emit shape:** outer segment loop wrapping the existing scan / negation /
aggregate emit. Each segment uses a fresh per-segment view.

**Dialect work:**
- Detect multi-view source in `_lower_root_scan` / `_lower_negation` / aggregate
  (no aggregate lowering exists yet — would need that too).
- Wrap with `D2lSegmentLoop(declare=True)` similar to N5.3.
- Aggregate is unported; would also need an `Aggregate` IR op + lowering.

**Why these aren't urgent:** zero fixtures exercise them. The legacy
`_nested_column_join_single` and root-scan/negation segment-loop paths
exist in the upstream Nim reference and would need to be ported only if a
workload starts producing those MIR shapes.

---

## Out-of-scope work on the radar

Not part of any milestone series; flagged for awareness.

| Topic | Effort | Trade-off |
|---|---|---|
| **Promote HIR / MIR to real dialects** | medium | They predate the dialect framework. Would unlock pattern matchers / strategies on HIR/MIR ops. Mostly rename + `Op` mixin. |
| **`complete_runner.py` templating** | large | ~700 lines of f-string-heavy CUDA emission. Either lift more shapes into structured ops, or introduce a templating layer (jinja-style or `quote`-style). Long-term direction question. |
| **WS full runner** | LARGE | Net-new — legacy never finished it. Would design the WCOJTask queue, cross-warp stealing, `par.data.atomic_ws` dialect. Currently blocks 2 skipped fixtures. |
| **Second target (CPU / WASM)** | very large | Would actually exercise the IR layering — same IIR, different emit. Validates the "target.cuda" prefix in the layout. |
| **Mypy cleanup in `ir/dialects/target/cuda/`** | medium | ~6 pre-existing errors (unreachable, missing type args). Inherited from legacy moves; not blocking. |
| **`ir/core/` strategy / pattern infra** | medium | Stratego-style combinators in `strategy.py` are barely used. Either commit to using them in lowering / passes, or trim. |
| **Provenance / SR semiring path** | very large | Touched in `provenance.py` but not woven through dialect lowering. Would be a sizable IR project if provenance becomes first-class. |
| **HIR-level passes (`stratify`, `split`, `semi_naive`, `plan`)** | medium | Real passes but not under `ir/core` pass framework. Aligning would unify pass infrastructure. |
| **`ir/dialects/target/cuda/context.py` simplification** | small | After legacy deletion, only `materialized.py` / `view_slots.py` / `block_group.py` (lazy) and unit tests still need it. Mostly inlinable. |

---

## How to update this doc

When landing a milestone, edit the row above and append a one-line entry
under the cleanup list with the commit short-hash. Don't repeat what the
commit message says — link to it.

When the next batch of work starts (new dialect, new target, new IR
layer), add a new "Stage 3" / "Stage 4" section above instead of stuffing
it into the existing tables.
