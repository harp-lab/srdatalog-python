---
orphan: true
---

# Phase A3 — Remove deprecated bool fields on `mir.ExecutePipeline`

The Layer 3 cleanup wave. Phase A1/A2 added `pragmas: tuple[Pragma, ...]`
as the open, typed pragma surface; Phase C wired one `@pragma_handler`
per built-in pragma; Phase A3 is the symmetric ending — the deprecated
named bool fields disappear and the typed Pragma becomes the **sole**
source of truth.

Companion to `docs/phase_a_mir_onto_op.md` (A1 + A2 template),
`docs/phase_c_pragma_materialization.md` §4-5 (per-pragma wrap-op
design + dual-write contract), `docs/pragma_as_typed_object.md` §3
(`@pragma_handler` contract), and `docs/code_discipline.md` D19
(dead-code marker ratchet that A3 drives to zero).

## 1. Goal

End-state after Phase A3:

- `mir.ExecutePipeline` has NO deprecated bool fields. The six
  fields removed: `dedup_hash`, `work_stealing`, `block_group`,
  `use_fan_out`, `count`, `tiled_cartesian`. (`concurrent_write`
  stays — it is a computed marker, not a user-set pragma.)
- The DSL dual-write helpers `_BUILTIN_BOOL_SHADOW_PRAGMAS`,
  `_BUILTIN_RULE_BOOL_SHADOW_PRAGMAS`, `_legacy_bool_kwargs_for`
  and `_legacy_rule_bool_kwargs_for` (`src/srdatalog/dsl.py`)
  are deleted. `Rule.with_pragma(...)` only writes the typed
  `Pragma` instance into the `pragmas` tuple.
- Every `@pragma_handler` materializer drops its
  `if op.<bool>: return` short-circuit and always emits the
  wrap op. The handlers in:
    - `src/srdatalog/ir/dialects/relation/sorted_array/pragmas/dedup_hash.py`
    - `src/srdatalog/ir/dialects/parallel/atomic_ws/pragmas/work_stealing.py`
    - `src/srdatalog/ir/dialects/parallel/block_group/pragmas/block_group.py`
    - `src/srdatalog/ir/dialects/relation/sorted_array/pragmas/fanout.py`
    - `src/srdatalog/ir/dialects/iir/cf/pragmas/count.py`
    - `src/srdatalog/ir/dialects/relation/sorted_array/pragmas/tiled_cartesian.py`
- The `if ctx.<pragma>:` consumers in
  `src/srdatalog/ir/dialects/relation/sorted_array/lowerings/__init__.py`
  either (a) consume the new typed wrap op via the
  registered `@lowering`, or (b) get deleted as dead. Each
  removal retires one `DEAD CODE NOTE` (D19) marker.
- Runner-side consumers in `src/srdatalog/ir/codegen/cuda/runner.py`
  + `src/srdatalog/ir/codegen/cuda/envelope.py` +
  `src/srdatalog/ir/codegen/cuda/complete_runner.py` read from a
  typed-pragma helper (`ep_has_pragma(ep, BlockGroup)` etc.)
  rather than the bool field. The runner is the migration
  surface that the original C-wave PRs declared "out of scope";
  A3 brings it in.
- `LoweringCtx` (`relation/sorted_array/lowerings/__init__.py`
  lines 122 / 129 / 141 / 155) keeps `dedup_hash`, `tiled_cartesian`,
  `ws_enabled`, `bg_enabled` only as **lowering-internal scratch**
  flipped True for the duration of a wrap-op lowering by the
  pragma module's `lower_*` helper. They are no longer read from
  the EP — they are written by the `@lowering` and read by the
  legacy emit helpers that the lowering delegates to.

Phase A3 does NOT:

- Delete `lower_scan_pipeline` (Layer 3 proper, post-A3).
- Migrate the runner-side scheduling for `FanOut` to the typed
  IIR (A3 reads the pragma instead of the bool; the runtime
  `jit_fanout_executor.h` integration remains in
  `complete_runner.py`).
- Remove `LoweringCtx.dedup_hash` etc. — those are scratch and
  removed by Phase B (per-MIR-op `@lowering` lands inline body
  that doesn't need scratch state).

## 2. Inventory

The six bool fields on `mir.ExecutePipeline`
(`src/srdatalog/ir/mir/types.py` lines 510-515) and one on
`Rule` (`src/srdatalog/dsl.py` via `Rule.count`).

### 2.1 `dedup_hash` (PR A3-1)

- Field: `mir.ExecutePipeline.dedup_hash: bool`
  (`src/srdatalog/ir/mir/types.py` line 513).
- Pragma: `DedupHash` (registered at
  `src/srdatalog/ir/dialects/relation/sorted_array/pragmas/dedup_hash.py`).
- Materialization handler:
  `materialize_dedup_hash` at
  `src/srdatalog/ir/dialects/relation/sorted_array/pragmas/dedup_hash.py:144`.
- Wrap op: `mir.DedupGate` (`src/srdatalog/ir/mir/types.py:235-257`).
- Consumers (the bool field surface that A3-1 retires):
    - DSL dual-write: `_BUILTIN_BOOL_SHADOW_PRAGMAS` entry at
      `src/srdatalog/dsl.py:749-752`.
    - Handler short-circuit:
      `src/srdatalog/ir/dialects/relation/sorted_array/pragmas/dedup_hash.py:147`
      (`if op.dedup_hash: return dataclasses.replace(op, pragmas=new_pragmas)`).
    - Production lowering: `use_dedup = ctx.dedup_hash and bool(vars_list)`
      at `relation/sorted_array/lowerings/__init__.py:2625`;
      gating the `try_insert + if (_p) { write }` block at
      lines 2627-2630 (D19 marker at line 2551).
    - MIR predicate:
      `src/srdatalog/ir/mir/passes.py:389`
      (`not ep.work_stealing and not ep.dedup_hash` —
      see also A3-2 for the `work_stealing` half).
    - Runner: `if ep.dedup_hash:` at
      `src/srdatalog/ir/codegen/cuda/runner.py:180` (kernel
      args for atomic write pos + dedup table) and
      `src/srdatalog/ir/codegen/cuda/envelope.py:525, 530`
      (functor header dedup_table struct).
    - Pipeline plumbing: `dedup_hash=state.ep.dedup_hash` at
      `src/srdatalog/ir/default_pipelines.py:293`.
    - HIR plumbing: `dedup_hash` kwarg threaded through
      `src/srdatalog/ir/hir/lower.py:605, 633` (and
      `dedup_hash=variant.dedup_hash` at all the
      `wrap_in_execute_pipeline(...)` callsites).
- New typed path: handler always wraps. Lowering for
  `mir.DedupGate` already registered (delegates to
  `_lower_insert_into` with `ctx.dedup_hash` flipped True for
  the duration; see
  `relation/sorted_array/pragmas/dedup_hash.py:lower_dedup_gate`
  line 158). Runner consumes via new
  `ep_has_pragma(ep, DedupHash)` helper.

### 2.2 `work_stealing` (PR A3-2)

- Field: `mir.ExecutePipeline.work_stealing: bool`
  (`src/srdatalog/ir/mir/types.py:511`).
- Pragma: `WorkStealing` at
  `src/srdatalog/ir/dialects/parallel/atomic_ws/pragmas/work_stealing.py`.
- Materialization handler: `materialize_work_stealing` (same
  module).
- Wrap op: `mir.WSScope` (`src/srdatalog/ir/mir/types.py:260-288`).
- Consumers:
    - DSL dual-write: `_BUILTIN_BOOL_SHADOW_PRAGMAS` entry at
      `src/srdatalog/dsl.py:753-756`.
    - Handler short-circuit: in
      `parallel/atomic_ws/pragmas/work_stealing.py` (the
      `if op.work_stealing:` clause; module-level docstring
      lines 125-130 describe the contract).
    - Production lowering: `if ctx.ws_enabled:` at
      `relation/sorted_array/lowerings/__init__.py:2633`
      (count-phase `<out>++` instead of
      `<out>.emit_direct()`); D19 marker at line 2565.
    - MIR predicate: `not ep.work_stealing` at
      `src/srdatalog/ir/mir/passes.py:389`.
    - HIR plumbing: `work_stealing` kwarg in
      `src/srdatalog/ir/hir/lower.py:602, 630` and at each
      variant lowering site (~4 sites).
- New typed path: handler always wraps `InsertInto` in
  `WSScope`; registered `@lowering(target=iir.cf, source=WSScope)`
  delegates to `_lower_insert_into` with `ctx.ws_enabled` flipped
  True. Runner does not consume `work_stealing` directly today
  (runner-side WS scaffolding is the unfinished
  `WCOJTask` queue — out of A3 scope, matches the C4 PR
  constraint).

### 2.3 `block_group` (PR A3-3)

- Field: `mir.ExecutePipeline.block_group: bool`
  (`src/srdatalog/ir/mir/types.py:512`).
- Pragma: `BlockGroup` at
  `src/srdatalog/ir/dialects/parallel/block_group/pragmas/block_group.py`.
- Materialization handler: `materialize_block_group` (same
  module, around line 141).
- Wrap op: `mir.BlockGroupRoot` (`src/srdatalog/ir/mir/types.py:355-383`).
- Consumers:
    - DSL dual-write: `_BUILTIN_BOOL_SHADOW_PRAGMAS` entry at
      `src/srdatalog/dsl.py:771-774`.
    - Handler short-circuit: in
      `parallel/block_group/pragmas/block_group.py:146` area
      (`if op.block_group: return ...`).
    - Production lowering: `if ctx.bg_enabled:` at
      `relation/sorted_array/lowerings/__init__.py:717`
      (dispatches to `_lower_root_cj_bg`); D19 marker at line 703.
    - Save/restore scratch: lines 320-325, 997-1002 inside
      `relation/sorted_array/lowerings/__init__.py` flip
      `ctx.bg_enabled` for body / nested-CJ recursion — these
      stay (they are lowering-internal, not EP-bool-driven).
    - Runner: `if ep.block_group:` at
      `src/srdatalog/ir/codegen/cuda/runner.py:204, 240, 654`
      (3 sites: launch_materialize, launch_fused,
      execute_fused — emit BG histogram / dispatch kernel
      variant). The C3 PR declared these "out of scope"; A3-3
      brings them in.
    - HIR plumbing: `block_group` kwarg in
      `src/srdatalog/ir/hir/lower.py:603, 631` + variant
      lowering callsites.
- New typed path: handler always wraps the root op;
  `@lowering(target=iir.cf, source=mir.BlockGroupRoot)` already
  registered. Runner consumes via `ep_has_pragma(ep, BlockGroup)`.

### 2.4 `use_fan_out` (PR A3-4 — the hardest)

- Field: `mir.ExecutePipeline.use_fan_out: bool`
  (`src/srdatalog/ir/mir/types.py:510`).
- Pragma: `FanOut` at
  `src/srdatalog/ir/dialects/relation/sorted_array/pragmas/fanout.py`.
- Materialization handler: `materialize_fanout` (same module,
  line 91 area; short-circuit at line 136).
- Wrap op: `mir.FanOut` (`src/srdatalog/ir/mir/types.py:326-352`).
- Consumers:
    - DSL dual-write: `_BUILTIN_BOOL_SHADOW_PRAGMAS` entry at
      `src/srdatalog/dsl.py:764-767`.
    - Handler short-circuit:
      `relation/sorted_array/pragmas/fanout.py:136`
      (`if op.use_fan_out: return dataclasses.replace(op, pragmas=new_pragmas)`).
    - Production lowering: there is no `if ctx.use_fan_out:`
      branch in
      `relation/sorted_array/lowerings/__init__.py` — the
      `use_fan_out` consumer historically lived in the
      runner-side `complete_runner.py` (today, no `use_fan_out`
      grep hit there — the field is THREADED but never read by
      the runner; D19 marker at
      `relation/sorted_array/lowerings/__init__.py:2598`
      describes the orphan).
    - HIR plumbing: `use_fan_out` kwarg in
      `src/srdatalog/ir/hir/lower.py:601, 629` and at all
      `wrap_in_execute_pipeline(...)` callsites (lines 736,
      751, 857, 872).
    - Runtime header reference: the C++ runtime in
      `src/srdatalog/runtime/generalized_datalog/gpu/runtime/jit/jit_fanout_executor.h`
      defines `FanOutTaskQueue` — but no Python codegen
      site currently dispatches to it under the bool. The
      runner-side integration is unwired in this repo
      snapshot.
- Why this is the hardest: even though there are no live
  Python consumers, the field is plumbed through ~6 sites in
  HIR, MIR construction, and a default-value-only path through
  the runner. A3-4 must (a) delete the plumbing (every
  `use_fan_out=` kwarg), (b) confirm zero runtime divergence
  (the FanOutTaskQueue runtime header is never invoked from
  the current Python codegen — A3-4 documents the disconnect
  and ratchets the next migration as a follow-up rather than
  reintroducing the bool).
- New typed path: handler always wraps the trailing
  `InsertInto` in `mir.FanOut`. The `@lowering(target=iir.cf,
  source=mir.FanOut)` rule already registered (delegates to
  `_lower_insert_into` unchanged — kernel body is identical
  to non-fanout shape; the scheduling difference was
  runner-side and currently inert in the Python emit).

### 2.5 `count` (PR A3-5)

- Field: `mir.ExecutePipeline.count: bool`
  (`src/srdatalog/ir/mir/types.py:515`).
- Also: `Rule.count: bool` (DSL) and `HirRuleVariant.count`
  (`src/srdatalog/ir/hir/types.py:104`).
- Pragma: `Count` at
  `src/srdatalog/ir/dialects/iir/cf/pragmas/count.py`.
- Materialization handler: `materialize_count` (same module,
  line 88; short-circuit at line 127).
- Wrap op: `mir.CountPhase` (`src/srdatalog/ir/mir/types.py:291-323`).
- Consumers:
    - DSL dual-write: `_BUILTIN_RULE_BOOL_SHADOW_PRAGMAS`
      entry at `src/srdatalog/dsl.py:785-788`.
    - Handler short-circuit:
      `iir/cf/pragmas/count.py:127`
      (`if op.count: return dataclasses.replace(op, pragmas=new_pragmas)`).
    - Production lowering: the `if ctx.is_counting:` branches
      throughout
      `relation/sorted_array/lowerings/__init__.py` (lines 444,
      451, 650, 2135, 2619, 2632, plus ~30 other
      `is_counting`-keyed sites scattered through the
      monolith). D19 marker at line 2581.
    - HIR plumbing: `count=variant.count` at
      `src/srdatalog/ir/hir/lower.py:739, 754, 860, 875`.
    - HIR variant field: `HirRuleVariant.count`
      (`src/srdatalog/ir/hir/types.py:104`).
    - Rule field: `Rule.count` (via `dataclasses.fields(Rule)`
      in `dsl.py`).
    - HIR plan plumbing: `v.count = rule.count` at
      `src/srdatalog/ir/hir/plan.py:484`.
    - Viz: `rule.count` at `src/srdatalog/viz/bundle.py:79`
      (cosmetic — update to `has_count_pragma(rule)`).
    - Runner: the count phase is dispatched by
      `complete_runner.py:258, 365, 432, 517, 587, 650`
      which pass `is_counting=True/False` to
      `compile_kernel_body(...)` — these stay (they thread
      the phase flag through the legacy emit). The A3-5 cut
      is: `Rule(count=True)` no longer sets a bool — it
      attaches a typed `Count` pragma, and
      `complete_runner.py` reads `ep_has_pragma(ep, Count)`
      to decide whether the kernel is count-only.
- New typed path: handler always wraps the EP's `pipeline`
  in a single `CountPhase(Block(...))`. The
  `@lowering(target=iir.cf, source=CountPhase)` rule emits
  `iir.cf.Phase(mode='C', body=...)`. The legacy
  `is_counting` ctx-scratch flag stays as the in-lowering
  signal that count phase is active — it is set by the
  count-phase pragma's lowering, no longer threaded from the
  EP bool.

### 2.6 `tiled_cartesian` (PR A3-6)

- Field: `mir.ExecutePipeline.tiled_cartesian: bool`
  (`src/srdatalog/ir/mir/types.py:514`).
- Pragma: `TiledCartesian` at
  `src/srdatalog/ir/dialects/relation/sorted_array/pragmas/tiled_cartesian.py`.
- Materialization handler: `materialize_tiled_cartesian`
  (same module, line 144; short-circuit area lines 170-184).
- Wrap op: `mir.TiledCartesian`
  (`src/srdatalog/ir/mir/types.py:386-411`).
- Consumers:
    - DSL dual-write: NONE — `tiled_cartesian` is NOT a
      user-set pragma. It is computed by
      `has_tiled_cartesian_eligible(pipeline)` in
      `src/srdatalog/ir/codegen/cuda/pipeline_utils.py:41`
      and threaded into `compile_kernel_body(tiled_cartesian=
      ...)` by `complete_runner.py:369`. The dual-write
      contract for this pragma is asymmetric: the runner-side
      eligibility detection writes the bool, the typed
      pragma is currently unreachable (no DSL surface).
    - Handler short-circuit:
      `relation/sorted_array/pragmas/tiled_cartesian.py`
      around line 170-184 (handler reads `ep.tiled_cartesian`
      and short-circuits when True).
    - Production lowering:
      `if not ctx.tiled_cartesian or ctx.is_counting:` at
      `relation/sorted_array/lowerings/__init__.py:1289`
      (gates eligibility for the
      `_lower_nested_cart_tiled` dispatch); the dispatch
      call at line 1783; D19 markers at lines 1277 and 1769.
    - Pipeline plumbing:
      `tiled_cartesian=state.tiled_cartesian` at
      `src/srdatalog/ir/default_pipelines.py:294` and
      `tiled_cartesian=tiled_cartesian_eligible` at
      `complete_runner.py:369`.
    - State field: `default_pipelines.py:78` carries
      `tiled_cartesian: bool` on the pipeline state shim
      passed to `LowerScanPipelineShim`.
- New typed path: the runner-side eligibility detection
  (`has_tiled_cartesian_eligible(pipeline)`) becomes a
  **MIR pass** that runs BEFORE `MirPragmaPass` and attaches
  a `TiledCartesian()` pragma to eligible EPs. After A3-6
  the source of truth for "is this pipeline tiled-eligible"
  is `ep_has_pragma(ep, TiledCartesian)`. The
  `@lowering(target=iir.cf, source=mir.TiledCartesian)`
  rule already registered.

## 3. Per-PR partition

A3 is six sequential PRs. Each removes one bool field's full
lifecycle: DSL dual-write helper entry, handler short-circuit,
production consumers, field declaration. Atomic so that
byte-equivalence is preserved per-PR.

Strict ordering rationale: a later PR cannot reference any
helper the earlier PR removed. Order chosen by independence
(A3-1, A3-2, A3-3 are independent within their lowering
slices) and by "easiest first" (A3-4 last because it touches
runner-side and has the field-with-no-consumer surprise).

| PR | Field | Branch | Roughly LOC |
|---|---|---|---|
| A3-1 | `dedup_hash` | `feat/a3-1-remove-dedup-hash-bool` | ~120 |
| A3-2 | `work_stealing` | `feat/a3-2-remove-work-stealing-bool` | ~90 |
| A3-3 | `block_group` | `feat/a3-3-remove-block-group-bool` | ~180 |
| A3-4 | `use_fan_out` | `feat/a3-4-remove-use-fan-out-bool` | ~100 |
| A3-5 | `count` | `feat/a3-5-remove-count-bool` | ~200 |
| A3-6 | `tiled_cartesian` | `feat/a3-6-remove-tiled-cartesian-bool` | ~140 |

### 3.1 A3-1: `dedup_hash`

Steps (in commit order):

1. Add an `ep_has_pragma(ep, PragmaCls) -> bool` helper to
   `src/srdatalog/ir/mir/pragma_pass.py` (or co-located with
   the pragma class). Used by all six A3 PRs.
2. Switch consumer reads to typed:
   - `runner.py:180`, `envelope.py:525, 530` → use
     `ep_has_pragma(ep, DedupHash)`.
   - `mir/passes.py:389` → `not ep_has_pragma(ep, WorkStealing) and not ep_has_pragma(ep, DedupHash)`.
   - `default_pipelines.py:293` → read the pragma off
     `state.ep.pragmas`.
3. Delete handler short-circuit:
   `pragmas/dedup_hash.py:147-148` becomes unconditional wrap.
4. Delete DSL dual-write entry:
   `dsl.py:749-752` (and update `_legacy_bool_kwargs_for`
   if it becomes a no-op for this pragma's family).
5. Delete HIR plumbing: `hir/lower.py:605, 633` and all
   `dedup_hash=variant.dedup_hash` callsites; HIR variant
   field; `Rule.dedup_hash` if present.
6. Delete the field from `mir/types.py:513` and remove
   the back-compat note from the class docstring.
7. Retire D19 marker at
   `relation/sorted_array/lowerings/__init__.py:2551`
   (the `if ctx.dedup_hash` branch in
   `_lower_insert_into` stays — it is now reached ONLY
   via the wrap op's `lower_dedup_gate` flipping the
   ctx scratch flag).

### 3.2 A3-2: `work_stealing`

Mirror A3-1's shape. The runner has NO `work_stealing`
direct consumer today, so step 2 is just the
`mir/passes.py:389` predicate. Retire D19 marker at
`relation/sorted_array/lowerings/__init__.py:2565`.

### 3.3 A3-3: `block_group`

Mirror A3-1. The runner has THREE consumer sites (lines
204, 240, 654 in `runner.py`) — A3-3 swaps each to
`ep_has_pragma(ep, BlockGroup)`. Retire D19 marker at line
703 of the sorted_array lowerings module.

### 3.4 A3-4: `use_fan_out`

The straightforward part is identical to A3-2 (no live
consumer in the Python codegen, only HIR plumbing and the
DSL helper). The investigation part: explicitly document in
the PR description that the `FanOutTaskQueue` runtime
header is unwired from the current Python emit, and that
A3-4 does NOT restore the connection — that is a separate
future work item. Retire D19 marker at line 2598 of the
sorted_array lowerings module.

### 3.5 A3-5: `count`

The largest blast radius — `Rule.count` is read by viz, by
the HIR planning pass, and by every variant lowering. The
PR:

1. Replace `Rule.count` with a derived
   `Rule.has_count_pragma() -> bool` helper that scans
   `Rule.plans[*].pragmas` for `isinstance(p, Count)`.
2. Update `hir/plan.py:484` and `hir/lower.py:739, 754,
   860, 875` to thread `has_count_pragma(rule)` instead of
   the bool.
3. Update `complete_runner.py` to read
   `ep_has_pragma(ep, Count)` when deciding count-only
   pipelines (the `kernel_materialize discarded` comment
   at line 23 of `complete_runner.py` documents the
   behavior).
4. Delete `_BUILTIN_RULE_BOOL_SHADOW_PRAGMAS` +
   `_legacy_rule_bool_kwargs_for` from `dsl.py`.
5. Delete `HirRuleVariant.count`, `mir.ExecutePipeline.count`,
   `Rule.count`.
6. Retire D19 marker at line 2581.

The `is_counting` scratch flag on `LoweringCtx` (line 103
of the sorted_array lowerings module) stays — it is no
longer threaded from `ep.count` but set by
`lower_count_phase` (the `CountPhase` lowering) for the
duration of the body lowering.

### 3.6 A3-6: `tiled_cartesian`

The asymmetric case. The bool is runner-detected, not
user-set. The PR:

1. Add a `MirTiledEligibilityPass` (one-shot `RewritePass`)
   to `src/srdatalog/ir/mir/passes.py` that walks every
   `ExecutePipeline`, runs the existing
   `has_tiled_cartesian_eligible(pipeline)` predicate, and
   attaches a `TiledCartesian()` pragma where eligible.
   Runs BEFORE `MirPragmaPass` (which then materializes
   the wrap op).
2. Delete the handler short-circuit at
   `pragmas/tiled_cartesian.py:170-184`.
3. Delete `tiled_cartesian` plumbing from
   `default_pipelines.py:78, 294` and
   `complete_runner.py:369` — eligibility is now an MIR
   property, not a state-shim field.
4. Update the gate at
   `relation/sorted_array/lowerings/__init__.py:1289` to
   read off the wrap op presence in `_lower_inner_chain`
   (already in place; the bool gate becomes redundant).
5. Delete the field from `mir/types.py:514`.
6. Retire D19 markers at lines 1277 and 1769.

## 4. Acceptance gate per PR

Beyond the global per-PR Done definition (`docs/code_discipline.md`):

- The pragma's existing e2e test continues to pass:
    - A3-1: `tests/test_pragma_dedup_hash_end_to_end.py`
    - A3-2: `tests/test_pragma_work_stealing_end_to_end.py`
    - A3-3: `tests/test_pragma_block_group_end_to_end.py`
    - A3-4: `tests/test_pragma_fanout_end_to_end.py`
    - A3-5: `tests/test_pragma_count_end_to_end.py`
    - A3-6: `tests/test_pragma_tiled_cartesian_end_to_end.py`

  These tests already exercise the typed-pragma path; passing
  them under A3 proves the typed path covers what the bool did.
- Byte-equivalence harness green:
    - `tests/test_runner_byte_equivalence.py` (272 fixtures)
    - `tests/test_jit_batch_byte_equivalence.py` (253 fixtures)
    - `tests/test_dedup_hash_byte_equivalence.py` (A3-1 only)
    - The lowering-byte-equivalence tests
      (`test_lower_mir_*_byte_equivalent.py`) for the dialects
      the PR touches.

  Byte-equivalence is the production regression anchor — any
  divergence blocks the PR.
- D19 dead-code marker count strictly decreases. Each A3 PR
  removes one `DEAD CODE NOTE` comment from the sorted_array
  lowerings module (per the per-PR table in §2).
- `grep -n "ep\.<bool_field>" src/srdatalog/` returns 0
  production hits post-PR. (Doc-string mentions in
  `pragmas/*.py` get cleaned up in the same PR as the bool
  removal.)
- `grep -n "<bool_field>:" src/srdatalog/ir/mir/types.py`
  returns 0 hits.
- `grep -rn "ep\.<bool_field>\b" tests/` — test fixtures
  updated to construct `pragmas=(Pragma(),)` directly. Per
  the A3 PR, `with_pragma(...)` is the only legal construction.

## 5. What A3 unblocks

After A3 is done:

- The DSL `with_pragma(X())` no longer mutates
  `Rule.<bool>` or sets a back-compat kwarg on `PlanEntry`.
  The DSL surface for pragmas is exactly:
  `Rule.with_pragma(X())` → `PlanEntry(pragmas=(X(),))` →
  `mir.ExecutePipeline(pragmas=(X(),))`.
- C-pragma `materialize_<pragma>` handlers no longer
  short-circuit on the bool. Every pragma uniformly
  produces the wrap op; every wrap op has a registered
  `@lowering`. The dual-write contract documented in
  `docs/phase_c_pragma_materialization.md` §5.1 collapses
  to a single-write contract.
- D19 dead-code marker count drops to approximately 0 for
  the bool-field family of `DEAD CODE NOTE (C2)`,
  `(C3)`, `(C4)`, `(C5)`, `(C6 - count)`,
  `(C6 - fanout)`. Other D19 markers (unrelated to
  the bool-field migration) survive.
- The `_legacy_imperative_lower` monolith
  (`relation/sorted_array/lowerings/__init__.py`) shrinks
  by ~6 conditional branches, each ~10-40 LOC. After A3
  the monolith's remaining branches are the legacy
  `is_counting` + `tiled_cartesian_valid_var` ctx-scratch
  reads (set by pragma lowerings, not by EP fields) — the
  remaining tech debt for Phase B / Layer 3 to clean up.
- Layer 3 cleanup of the legacy monolith proceeds in
  earnest. Per `docs/compiler_redesign.md` §10, A3
  sign-off is the gate for declaring Phase A "done end-to-end".

## 6. Risks + mitigations

| Risk | Mitigation |
|---|---|
| A consumer site missed during inventory pass → silent byte-equivalence regression | The byte-equivalence harness (272 runner + 253 jit-batch fixtures) catches divergence on every PR; the gate is hard-fail. |
| Pragma topology breaks because some pragma expected the bool to be set BEFORE `MirPragmaPass` runs (e.g. `materialize_tiled_cartesian` reads `ep.tiled_cartesian` as a runner-detection signal) | A3-6 is sequenced last and explicitly introduces `MirTiledEligibilityPass` to attach the pragma BEFORE `MirPragmaPass`. The pragma-topology discipline test (R5 in `docs/phase_c_pragma_materialization.md` §7) checks every pragma is registered AND every pragma name surfaces a wrap op. |
| External code that already sets `ep.dedup_hash = True` directly on a frozen MIR instance → AttributeError | MIR is `frozen=True` since A1; external code cannot mutate. The only legitimate construction is via DSL `with_pragma(...)`. Discipline test `test_no_mir_mutation_outside_replace` (A2) anchors this. |
| Runner-side `block_group` consumer in `runner.py` reads the pragma but the pragma was filtered out by `MirPragmaPass` (which empties `op.pragmas` after materialization) | A3-3 reads the pragma BEFORE materialization, OR routes the runner read through a stable property on the `BlockGroupRoot` wrap op (presence of wrap op in `ep.pipeline`). Decision in A3-3 design review. |
| `use_fan_out` PR (A3-4) deletes a field whose runner-side `FanOutTaskQueue` integration is currently unwired → loss of a documented intent | A3-4 PR description explicitly calls out the disconnect and links the runtime header as the migration anchor for a follow-up. The deletion is safe because no Python codegen site dispatches under the bool today. |
| `count` PR (A3-5) is the largest blast radius (Rule-level field + viz + HIR + every variant lowering) → high regression risk | A3-5 is sequenced 5th (not last) so subsequent A3-6 work is small. The new `Rule.has_count_pragma()` helper is added in a separate first commit, validated by tests, before any field is removed. |
| Sphinx-myst build fails because the new doc cross-links to a doc without `orphan: true` | The new doc itself has `orphan: true` (this doc). The compiler_redesign.md link uses the markdown form, which works because that target also has `orphan: true`. |

## 7. Sign-off

Phase A3 is complete iff:

- [ ] All six A3 PRs (A3-1 … A3-6) merged.
- [ ] `src/srdatalog/ir/mir/types.py:ExecutePipeline` has no
  `dedup_hash`, `work_stealing`, `block_group`, `use_fan_out`,
  `count`, `tiled_cartesian` fields. Only `pragmas: tuple[
  Pragma, ...]` carries pragma data. `concurrent_write`
  remains (computed-only marker).
- [ ] `src/srdatalog/dsl.py:_BUILTIN_BOOL_SHADOW_PRAGMAS`,
  `_BUILTIN_RULE_BOOL_SHADOW_PRAGMAS`,
  `_legacy_bool_kwargs_for`, `_legacy_rule_bool_kwargs_for`
  are deleted.
- [ ] Every `@pragma_handler` (`materialize_*`) drops its
  `if op.<bool>:` short-circuit. Discipline test
  `test_pragma_handlers_unconditional` (added in A3-1) asserts
  no handler reads `op.<deleted-bool-name>`.
- [ ] D19 dead-code marker count for the bool-field family
  is 0.
- [ ] Byte-equivalence preserved across the full fixture
  set on every A3 PR.
- [ ] `grep -rn "ep\.\(dedup_hash\|work_stealing\|block_group\|use_fan_out\|count\|tiled_cartesian\)" src/srdatalog/`
  returns 0 production hits.

After A3 sign-off, Layer 3 cleanup of the legacy lowering
monolith (`relation/sorted_array/lowerings/__init__.py`,
`_lower_insert_into` etc.) proceeds. Per
`docs/compiler_redesign.md` §10, Phase A is then complete
end-to-end.
