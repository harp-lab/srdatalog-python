---
orphan: true
---

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
| Stage 2 (legacy-emit retirement) | ✅ N1, N2, N3.x, N4.x, N5.0–N5.2, N6, N7, N8 | see "Refactor PR" below — landed |
| Legacy elimination | ✅ all `codegen/jit/{pipeline,instructions,root,scan_negation,kernel_functor,file}.py` deleted | — |
| Layout reorg | ✅ Phase A (`codegen/jit/` → `ir/dialects/target/cuda/`), Phase B (`ir/` namespace) | ⬜ Phase C (R1–R6 below) |
| Docs / test rename sync | ✅ README + 10 `test_jit_*.py` → `test_cuda_*.py` | — |
| Open work | — | WS full runner, CPU/WASM target, HIR/MIR as proper dialects, `complete_runner.py` templating |

**Test gates currently passing (after Refactor PR R1–R9):**
- 272/272 runner byte-equivalence (`tests/test_runner_byte_equivalence.py`) — 6 skipped: 2 WS runner + 4 ddisasm F1–F3 gaps (no compile-errors)
- 253/253 jit-batch byte-equivalence (`tests/test_byte_equivalence_jit.py`)
- 4/4 N5.3/N5.4 guard tests (`tests/test_n5_3_n5_4_guards.py`)
- **24/27 ddisasm runner rules byte-equal** to upstream Nim, **0 compile-errors**, 3 documented divergences (F1 ×2 + F3 ×1; F2 landed)
- 1005 / 1005 total in suite (6 documented skips)

---

## Refactor PR — close out the structural reorg + ddisasm compile

**Branch:** `feat/iir-sorted-array-cuda` (current). **Merge target:** `main`.

**Goal:** ship a self-contained PR that (a) finishes the namespace reorg
started by Phase A/B, (b) retires the legacy `SRDatalogProgram` MIR chain
that duplicates `Version` / `FactIndex` / `ColumnSource` symbols, and
(c) makes ddisasm compile end-to-end through the dialect (no
`NotImplementedError` / `ValueError`). Byte-equivalence for the 3
ddisasm rules that currently produce divergent-but-valid C++ is
explicitly out of scope — tracked as F1–F3 follow-ups.

After this PR, "the dialect handles every shape we test against" will
be true, and the package layout will be coherent (no top-level files
that should be in IR, no parallel-but-misnamed legacy MIR).

### Acceptance gate

Each item below must be green before merge:

- All structural moves (R1–R6) leave existing imports working via
  thin re-exports OR all internal call-sites updated. No `pip install
  -e .` regressions.
- Full suite: `pytest tests/` passes (modulo the documented 7 skips).
- ddisasm fixture: `tests/test_integration_ddisasm.py` passes; the
  runner sweep checks 24+/27 rules byte-equal (vs. 22/27 today).
- `RUNNER_BYTE_MATCH_SKIPS` reduced by 2 entries (R8, R9 land); the
  remaining 5 (2 WS + 3 ddisasm-divergence) carry concise reasons.
- [`docs/milestones.md`](./milestones.md) updated to reflect what landed.

### Structural moves (R1–R6) — finish the namespace ✅

| ID | Move | Status | Notes |
|---|---|---|---|
| **R1** | `rule_rewrite.py` → `ir/hir/rule_rewrite.py` | ✅ landed | HIR-level passes alongside `index.py`, `plan.py`, etc. |
| **R2** | `provenance.py` → `ir/hir/provenance.py` | ✅ landed | DSL imports `USER_PROVENANCE` sentinel via the new path. |
| **R3** | `compile.py` → `ir/dialects/target/cuda/api.py` | ✅ landed | Thin re-export shim kept at `srdatalog.compile` for downstream users. |
| **R4** | `pipeline.py` → `ir/pipeline.py` | ✅ landed | `build.py` + `viz/bundle.py` updated. |
| **R5** | Retire legacy `SRDatalogProgram` chain (~1000 LOC) | ✅ landed | Deleted: `srdatalog_program.py`, `ir/mir/{commands,runner,schema}.py`, 3 test files. cffi/FFI wrapper stays. |
| **R6** | Delete `example_program.py` | ✅ landed | Folded into R5 — was the only non-test consumer of the legacy chain, with hardcoded `/home/miakerchen/...` paths and no `__main__` guard. |

### Correctness moves (R7–R9) — make ddisasm + N5.4-Scan honest ✅

| ID | Move | Status | Notes |
|---|---|---|---|
| **R7** | Revert N5.4-Scan over-implementation | ✅ landed | Now matches Nim's `jitRootScan` (no segment-loop wrap). Test flipped to assert *no* `_seg_` in emit. |
| **R8** | Implement `Scan + CartesianJoin` shape | ✅ landed | `_supported_pipeline` accepts `CartesianJoin` in Scan-middle. `_lower_nested_cart` learned the "fresh-root + chained `.prefix()`" pattern. `_lower_root_scan` count-phase var elision now uses substring-on-rendered-body (matching Nim's `varName notin body`). |
| **R9** | Port `dedup_hash` path in `gen_complete_runner` | ✅ landed | DedupTable struct emit, LaunchParams fields, kernel signatures, setup hash-table alloc, execute clear-table flow. Cartesian count-as-product short-circuit now disabled for dedup_hash (each tuple needs the in-kernel try_insert test). |

### Out of scope (follow-up PRs)

| ID | Topic | Why deferred |
|---|---|---|
| **F1** | ddisasm head-tuple ordering (`varr,varp` vs `varp,varr` in `StackLiveVarBlockEnd1_D0_splitA`) | MIR-level head-arg ordering investigation. Not structural; produces valid (just different) C++. |
| **F2** | ddisasm pre-narrow Negation iteration order (`StackLiveVarPriorUsed`) | ✅ landed. Reproduces Nim's `Table[int, ...]` hash-bucket iteration via a hashWangYi1 port + linear-probe slot resolution. |
| **F3** | ddisasm `_tiled_cart_eligible` predicate gap (`StackDefUsed1`) | Tiled-Cartesian eligibility predicate disagrees with Nim. Investigation, not refactor. |
| **F4** | N5.3 (single-source nested CJ over D2L FULL_VER) | No live workload. Pinned by guard test. |
| **F5** | N5.4-Negation / N5.4-Aggregate over D2L FULL_VER | Both Nim and dialect broken; defer until upstream fixes. |

### Suggested commit boundaries

To keep review tractable, land R1–R9 as one PR but split internally
into reviewable commits:

1. R1 (rule_rewrite move) — small, mechanical
2. R2 (provenance move) — small, mechanical
3. R3 (compile move) — small, mechanical
4. R4 (pipeline move) — small, mechanical
5. R5 + R6 (legacy MIR delete + example_program delete) — bigger but pure deletion
6. R7 (N5.4-Scan revert + test flip) — small, with reasoning in commit msg
7. R8 (Scan+Cart shape) — small
8. R9 (dedup_hash port) — biggest of the bunch
9. Docs + milestone update

Each commit should keep tests green so bisect stays useful.

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
| **N5.3** | ⬜ | — | **Single-source nested CJ over D2L FULL_VER** — `_lower_nested_cj_single` doesn't exist yet; `_supported_pipeline` rejects the shape with a clear `ValueError("unsupported pipeline shape ...")`. Pinned by `tests/test_n5_3_n5_4_guards.py::test_n5_3_*`. |
| **N5.4** (Scan) | ⚠ R7 in Refactor PR | `a90062d` | Over-implementation. We segment-wrap; Nim doesn't. R7 reverts. See "Refactor PR" above. |
| **N5.4** (Negation, std-path) | ⬜ Nim-broken-too (F5) | — | Standard-path Negation over D2L FULL_VER raises `NotImplementedError`. Audit found Nim's `jitNegation` ALSO has no segment-loop wrap. Both ends broken. Defer. |
| **N5.4** (Aggregate) | ⬜ Nim-broken-too (F5) | — | Aggregate over D2L FULL_VER. No `Aggregate` IR op exists today. Audit found Nim's `jitAggregate` ALSO has no segment-loop wrap. Both ends broken. Defer. |
| **Scan + Cart shape** | ⬜ R8 in Refactor PR | — | `[Scan, CartesianJoin, InsertInto]`. Pieces exist; only need to add `CartesianJoin` to the Scan-middle allowed list. Hit by ddisasm `StackLiveVarBlockEnd1_D0_splitB`. |
| **dedup_hash runner** | ⬜ R9 in Refactor PR | — | `gen_complete_runner` raises `dedup_hash not yet ported`. Runner-side. Hit by ddisasm `StackDefUsed4_D1`. |
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

## Nim-reference audit (2026-05-03)

Real workloads (specifically `examples/ddisasm.py`) trip patterns the
dialect raises on. Before implementing anything, audited the Nim
reference at `~/workspace/SRDatalog/src/srdatalog/codegen/target_jit/`
to find which gaps are dialect-only (Nim has a working emit we can
port) vs both-broken (Nim also doesn't handle the shape).

Outcome: not every "missing" pattern is worth implementing — for some,
Nim itself silently emits wrong code. Byte-equivalence with Nim
*requires matching its bugs* on those paths.

**Status after Refactor PR (R1–R9 landed):**

- 23 / 27 ddisasm rules byte-equal to Nim through the dialect.
- 0 / 27 fail kernel-body compile.
- 3 / 27 compile but disagree with Nim — F1 head ordering (×2), F3
  tiled-Cartesian eligibility (×1). Tracked in
  [`tests/test_runner_byte_equivalence.py`](../tests/test_runner_byte_equivalence.py)
  and [`tests/test_cuda_complete_runner.py`](../tests/test_cuda_complete_runner.py)
  `RUNNER_BYTE_MATCH_SKIPS` with concise reasons. F2 (neg-prenarrow
  ordering) landed via Nim Table-iteration port.

| Pattern | Dialect | Nim | Hit by ddisasm? | Verdict |
|---|---|---|---|---|
| `Scan + CartesianJoin + InsertInto` | ✅ landed (R8) | ✅ emits | ✅ `StackLiveVarBlockEnd1_D0_splitB` | Compiles. Byte-divergence remains due to F1 (head ordering). |
| `dedup_hash` in `gen_complete_runner` | ✅ landed (R9) | ✅ emits | ✅ `StackDefUsed4_D1` | Compiles + byte-equal to Nim. |
| **N5.3** — single-source nested CJ over D2L FULL_VER | ❌ `_supported_pipeline` rejects | ✅ emits seg-loop wrap ([jit_instructions.nim:42-143](https://github.com/.../jit_instructions.nim)) | ❌ no | Real gap, but no live workload. Defer; pinned by `tests/test_n5_3_n5_4_guards.py`. |
| **N5.4-Scan** — root Scan over D2L FULL_VER | ✅ wraps in `D2lSegmentLoop(declare=True)` | ❌ **NO** seg-loop wrap ([jit_root.nim:61-126](https://github.com/.../jit_root.nim)) | ❌ no | **We over-implemented**. Diverges from Nim. Revert pending — see "Pending revert" below. |
| **N5.4-Negation** — std-path over D2L FULL_VER | ❌ raises `NotImplementedError` | ❌ **NO** seg-loop wrap ([jit_scan_negation.nim:142-187](https://github.com/.../jit_scan_negation.nim)) | ❌ no | **Both broken.** Nim silently emits single-view `valid()` check; semantically wrong on FULL_VER (HEAD/FULL split). Defer until upstream fixes. |
| **N5.4-Aggregate** — over D2L FULL_VER | ❌ no `Aggregate` IR op | ❌ **NO** seg-loop wrap ([jit_scan_negation.nim:212-276](https://github.com/.../jit_scan_negation.nim)) | ❌ no | **Both broken.** Nim emits a single `aggregate<>(...)` call against one view. Defer. |
| ddisasm: head tuple ordering (`varr,varp` vs `varp,varr`) | ❌ wrong order | ✅ correct | ✅ `StackLiveVarBlockEnd1_D0_splitA` | MIR-level head ordering bug — not kernel-body. Investigate HIR→MIR pass for head-arg ordering. |
| ddisasm: pre-narrow Negation emission order | ✅ landed (F2) | ✅ correct | ✅ `StackLiveVarPriorUsed` | Nim iterates `Table[int, ...]` in hash-bucket order (slot = `hashWangYi1(handle_idx) & 63`). Ported as `_nim_table_iter_order` helper. |
| ddisasm: tiled-Cartesian eligibility | ❌ skips tiling | ✅ tiles | ✅ `StackDefUsed1` | `_tiled_cart_eligible` predicate disagrees with Nim — likely relevant when Cartesian source has prefix vars but the binding is single-source/single-var. |

**Note on N5.4-Scan over-implementation:** commit `a90062d` added a
`D2lSegmentLoop(declare=True)` wrap to `_lower_root_scan` for D2L
FULL_VER on the theory that the existing single-view scan was a "silent
miscompile." The audit disproved that theory: `jitRootScan` ([jit_root.nim:61-126](file:///home/stargazermiao/workspace/SRDatalog/src/srdatalog/codegen/target_jit/jit_root.nim#L61))
emits a single-view scan with no segment-loop. **R7 in the Refactor PR
above** reverts this; no fixture currently catches the divergence
because zero integration fixtures exercise root Scan over D2L FULL_VER.

### N5.3 / N5.4 — what's needed if/when they land

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
