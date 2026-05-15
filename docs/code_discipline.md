---
orphan: true
---

# Code Discipline — Forbidden, Required, Enforced

This is the contract that prevents the redesign from regressing into
the "framework on top of imperative monolith" anti-pattern that
[`compiler_redesign.md`](compiler_redesign.md) §1 diagnoses.

**Sign-off rule:** every PR after Layer 0 (this redesign package) must
satisfy every requirement here. CI failures on discipline tests are
**hard blocks** — only an explicit override comment from the project
owner unblocks them, and the override must include reasoning recorded
in the PR.

## 1. Target code structure

Every file's location is determined by its role. Adding a file to the
wrong directory fails CI.

```
src/srdatalog/
  core/                          # Framework ONLY. No dialect knowledge.
    op.py                        # Op, Type bases
    dialect.py                   # Dialect, Compiler, registration
    passes.py                    # LoweringPass, RewritePass, ProgramPass
    lower_ctx.py                 # Small LowerCtx (5 fields, frozen)
    strategy.py                  # bottom_up/top_down/repeat combinators
    plugin.py                    # Entry-point discovery, register_plugin

  dialects/                      # Every dialect. Built-in AND plugin-installed.
    hir/
      types.py                   # Planning records (NOT Op-subclassed)
      passes/                    # ONE FILE PER PASS
        stratify.py
        split.py
        semi_naive.py
        plan.py
        index.py
        temprel.py
        temp_index.py
    mir/
      types.py                   # Generic MIR ops (Op-subclassed, frozen)
      pragma_ops/                # ONE FILE PER WRAP OP
        dedup_gate.py
        block_group_root.py
        ws_scope.py
    iir/
      cf/{ops,print,__init__}.py
      expr/{ops,print,__init__}.py
    relation/
      sorted_array/
        ops.py                   # Op definitions ONLY
        types.py                 # Type definitions ONLY
        print.py                 # Print form ONLY
        lowerings/               # ONE FILE PER MIR OP
          lower_mir_scan.py
          lower_mir_filter.py
          lower_mir_insert_into.py
          ...
        rewrites.py              # @rewrite registrations only
      d2l/                       # Same shape as sorted_array
    parallel/
      data/                      # ParallelFor, GridStrideLoop
      block_group/               # NEW sub-dialect (Phase 2C)
      atomic_ws/                 # NEW sub-dialect (Phase 2C)

  codegen/
    cuda/
      render/                    # ONE FILE PER DIALECT
        iir_cf.py
        iir_expr.py
        sorted_array.py
        d2l.py
        parallel_data.py
        parallel_block_group.py  # NEW
        parallel_atomic_ws.py    # NEW
      pass.py                    # CudaRenderPass

  pragmas/                       # Each pragma is a plugin module. INCLUDING built-ins.
    dedup_hash.py                # @pragma(name="dedup_hash", on=mir.ExecutePipeline)
    block_group.py
    work_stealing.py
    tiled_cartesian.py

  pipeline.py                    # ONLY DEFAULT_PIPELINE definition + entry
  dsl.py                         # Default Python DSL frontend (also a plugin)
  __init__.py                    # Compiler.with_default_plugins()
```

**Three location rules:**

1. **`core/` knows nothing about specific dialects, ops, pragmas, or
   targets.** Adding a `from srdatalog.ir.dialects` import to any file
   in `core/` fails CI.
2. **One concept per file.** `lowerings/lower_mir_scan.py` lowers
   `mir.Scan` and nothing else. No 2500-LOC files.
3. **Built-in pragmas live in `pragmas/`, not buried in dialect
   lowerings.** Each pragma is a plugin module. The DSL
   `Rule(pragmas={"dedup_hash": True})` works because
   `pragmas/dedup_hash.py` is auto-loaded.

## 2. Forbidden code shapes (CI-enforced)

Each row gets a discipline test in `tests/test_discipline_*.py`.
**A test failure here is a hard PR block.**

| ID | Forbidden pattern | Why | Discipline test |
|---|---|---|---|
| **D1** | `if ctx.<pragma>:` (or any pragma name) outside `pragmas/<pragma>.py` | Pragma flags are partial-evaluation triggers, not branch keys. They live in ONE place. | `test_no_pragma_flags_outside_pragma_modules` — greps `ctx\.<pragma_name>\b` across `src/`, allowlist = `pragmas/<pragma>.py` |
| **D2** | Direct call to `lower_scan_pipeline(...)` or any monolith lowering function | Production must go through `Compiler.run(pipeline=...)`. No bypass. | `test_no_direct_monolith_lowering_calls` — after Layer 3, `lower_scan_pipeline` doesn't exist; during migration, `USE_DECLARATIVE` is ratchet-only |
| **D3** | `isinstance(op, mir.X)` dispatch chain | Cross-dialect dispatch goes through `@lowering` registry, not isinstance. | `test_no_dispatch_isinstance_chains` — AST scan; allowlist = inside one `@lowering`-decorated function body or inside `core/` |
| **D4** | Module-import side effects (registering, mutating globals at import time) | Per S3A.8 / A7. | `test_ir_no_import_side_effects` (already exists, extended) |
| **D5** | New `RawString(...)` call site anywhere in `src/srdatalog/ir/` | Per Stage 4 contract. | `test_iir_no_raw_string_growth` (already exists, whole-tree) |
| **D6** | `from srdatalog.core import` referencing dialect symbols | Core is dialect-agnostic. | `test_core_has_no_dialect_imports` — AST scan of `core/*.py` |
| **D7** | New `if`-branch in an existing lowering function (where a `@lowering` registration is the right shape) | This is the "moving code around" anti-pattern. | `test_lowering_files_have_one_lowering_each` — files matching `lower_mir_*.py` must contain exactly ONE `@lowering` decoration |
| **D8** | Imports of concrete `Pragma` subclasses (e.g. `from srdatalog.pragmas import DedupHash`) anywhere in `core/`. Hardcoded pragma string names also forbidden. | Pragmas are plugin-discovered typed objects. Core knows only the `Pragma` base class — never specific subclasses or names. (Per [`pragma_as_typed_object.md`](pragma_as_typed_object.md) §8.) | `test_core_has_no_pragma_imports` — AST scan for `from srdatalog.pragmas` / `from <plugin>.pragmas` style imports under `core/` |
| **D9** | `@lowering` / `@rewrite` / `@pragma_handler` / `@register_render` registration with no caller-exercised path | The infrastructure-without-consumer anti-pattern (the original sin). | `test_every_registration_is_exercised` — every registration must be hit by at least one production-path test (coverage assertion) |
| **D10** | New field added to `LowerCtx` | `LowerCtx` is fixed at 5 fields after F3. Extending it requires a doc amendment. | `test_lower_ctx_field_count_pinned_at_five` — counts dataclass fields, must be exactly 5 |
| **D11** | New file in `core/` after Layer 1 | Core is frozen post-foundation. | `test_core_module_set_pinned` — pins the file list under `core/` |
| **D12** | Removal of an op from `USE_DECLARATIVE` set during migration | The migration ratchet is monotonic. | `test_use_declarative_is_monotonic` — git-history check (or test fixture pin) |
| **D13** | `Pragma` subclass without `@final + @dataclass(frozen=True, slots=True)` | Pragmas are pure compile-time data; same discipline as `Op` subclasses. (Per [`pragma_as_typed_object.md`](pragma_as_typed_object.md) §8.) | `test_pragma_subclasses_are_frozen_final` — parametrized over all Pragma subclasses |
| **D14** | Method on a `Pragma` subclass other than `__post_init__` (or pure-data dataclass mechanics) | Behavior lives in `@pragma_handler`, not on the class. Symmetric with D1 (no methods on Op subclasses). | `test_pragma_subclasses_are_pure_data` — AST scan |
| **D15** | DSL `Rule.with_pragma(...)` accepts non-`Pragma` arg (after migration window) | Pragmas are typed objects, not strings. The `(name, value)` form is deprecated; final cleanup removes it. | `test_with_pragma_rejects_non_pragma` — `Rule(...).with_pragma("foo", True)` raises `TypeError` |
| **D18** | **Transitional mutable state** that doesn't ratchet down. Three sub-cases: (a) `object.__setattr__(...)` calls on frozen `Op` subclasses (the migration shim from A1 onward); (b) `# DEPRECATED:` fields on dataclasses pending removal in a named phase; (c) module-global mutable registries with a documented per-`Compiler` migration target. Each occurrence MUST: (i) carry an inline `# TODO(phase-X): <action>` comment naming the phase that removes it, AND (ii) be inventoried in a ratchet test that fails CI if the count goes UP. **Permanent framework infra exclusions** (NOT transition shims) must be listed in the ratchet test's `_EXCLUDED` set with an inline rationale at the call site: currently `core/passes.py` (LoweringPass.apply attaches the per-call dispatch table to LowerCtx — keeps LowerCtx pinned at 5 fields per D10). | The redesign accepts transitional tech-debt because byte-equiv migration requires it (see PR #28's mutation shims) — but the debt MUST visibly decrease. Without a ratchet, transitional shims become permanent (the original sin this redesign reverses). Same pattern as D5 (`test_iir_no_raw_string_growth`) and D12 (`USE_DECLARATIVE` monotonic). | `test_transitional_mutable_state_ratchet` — counts `object.__setattr__(` calls on inferable-frozen-Op instances + counts `# DEPRECATED:` fields + counts known module-global mutable registries; per-category cap stored in the test; CI fails if count > cap. Each PR that removes a shim updates the cap atomically (same commit). |

The load-bearing test is **D9**. A registration that no production test
triggers is dead infra. CI fails until either the registration is
removed or a test exercises it via `Compiler.run`. This is the
mechanism that prevents the "build infra, never use it" cycle from
repeating.

## 3. Required code shapes (positive contract)

| ID | Required | Reason | Discipline test |
|---|---|---|---|
| **R1** | Every `Pass` instance referenced in `DEFAULT_PIPELINE` is registered with the `Compiler` | No orphan passes | `test_default_pipeline_passes_all_registered` |
| **R2** | Every dialect ships a `@verifier` (even if no-op) | Per S3A.7 | already enforced, kept |
| **R3** | Every concrete IIR op has either `@register_render(target=cuda)` OR a `@rewrite` registered | Per `ir_dialect_contract.md` §1 (LEAF-or-COMPOUND contract) | `verify_renderability` (already wired in production via `_apply_dialect_rewrites`) |
| **R4** | Every concrete MIR op has a registered `@lowering(target=IIR, source=mir.X)` | Cross-IR completeness | `test_every_mir_op_has_lowering` |
| **R5** | Every concrete `Pragma` subclass the DSL can produce has a registered `@pragma_handler(PragmaCls, on=...)`. (Per [`pragma_as_typed_object.md`](pragma_as_typed_object.md) §8.) | Closed-form pragma surface; missing handler = `UnconsumedPragmaError` at compile, with did-you-mean class-name list. | `test_pragma_handler_registry_completeness` — parametrized over all known `Pragma` subclasses |
| **R5b** | `op.pragmas` is empty after `MirPragmaPass`. | Materialization is the only consumer; surviving pragma instances are bugs. (Per [`pragma_as_typed_object.md`](pragma_as_typed_object.md) §8.) | `test_pragmas_empty_after_materialization` |
| **R6** | Every PR adds NEW files only; existing-file edits are limited to additions in designated registry sets (`DEFAULT_PIPELINE`, `USE_DECLARATIVE`, plugin entry points, `__all__`) | Anti-monolith | per-PR review checklist (§4) |
| **R7** | Every NEW class / function has at least one production-path call site | Anti-dead-infra | overlaps with D9 |

## 4. Per-PR Definition of Done

A PR is **not done** until ALL of these are satisfied. Reviewer
explicitly checks each by ticking the box in the PR template.

```markdown
[ ] Discipline CI passes — all tests in tests/test_discipline_*.py
[ ] No new `if`-branch added to an existing lowering — new file under
    lowerings/ or pragmas/ instead
[ ] Every new @lowering / @rewrite / @pragma / @register_render
    registration has a test that exercises it via the production
    Compiler.run path (NOT a synthetic in-test compiler)
[ ] Byte-equivalence preserved — full byte-equivalence suite green
    OR documented divergence with golden update + reviewer sign-off
[ ] No new file in core/ (core is frozen post-Layer-1)
[ ] No new pragma name appears in core/ — all pragma logic in pragmas/
[ ] Diff size: ONLY new files under the assigned scope + single-line
    additions to designated registry sets
[ ] LowerCtx not extended (LowerCtx is fixed at 5 fields)
[ ] PR description references the design doc section that authorizes
    this change
[ ] Reviewer comment: "Verified no shortcut" — explicit ack required
```

The PR template (`.github/pull_request_template.md`) enforces this
checklist as the body. Reviewer must check every box before merge.

## 5. Enforcement architecture

Three layers of enforcement, defense-in-depth.

```
┌──────────────────────────────────────────────────────┐
│ Layer A: CI discipline tests (mechanical)            │
│   tests/test_discipline_*.py                         │
│   Fail PR on any forbidden / required violation.     │
│   ~12 tests, each enforcing one rule from §2/§3.     │
│   Hard block; only owner-override unblocks.          │
└──────────────────────────────────────────────────────┘
                       │
┌──────────────────────────────────────────────────────┐
│ Layer B: Per-PR template (procedural)                │
│   .github/pull_request_template.md                   │
│   Reviewer must tick every box (§4).                 │
│   Any unchecked box blocks merge.                    │
└──────────────────────────────────────────────────────┘
                       │
┌──────────────────────────────────────────────────────┐
│ Layer C: Design-doc reference (architectural)        │
│   PR body MUST reference docs/<design>.md§X that     │
│   authorizes the change. If no design auth → PR      │
│   rejected.                                          │
│                                                      │
│   Discipline rules themselves are NOT edited inside  │
│   feature PRs — they're amended in their own         │
│   design-update PR with explicit owner sign-off.     │
└──────────────────────────────────────────────────────┘
```

Layer A catches mechanical shortcuts. Layer B catches procedural
shortcuts. Layer C catches scope creep.

## 6. Migration-period special discipline

During Layer 2 (parallel waves), the legacy monolith still exists as
a fallback. Two extra rules apply for the duration:

- **`USE_DECLARATIVE` is monotonically growing.** A PR can only ADD to
  it. Removing requires a documented reason and explicit owner
  sign-off. CI enforces (D12).
- **Every `USE_DECLARATIVE` addition has a paired `@lowering`
  registration in the same commit.** The new file under
  `lowerings/lower_mir_<op>.py` defines the registration; the
  single-line addition to `USE_DECLARATIVE` flips production over to
  it. CI checks the cross-reference: every member of `USE_DECLARATIVE`
  must have a corresponding `@lowering(source=mir.X)` registered.

After Layer 3 cleanup, both rules are obsolete (legacy monolith
deleted; `USE_DECLARATIVE` flag deleted).

## 7. The "no useless complicated pseudo refactor" test

Past PRs in this repo have been the failure mode this redesign
forbids: ship infrastructure (a Lowering registration, a decorator, a
new module) without making it the production path. The framework
existed alongside the imperative monolith; the monolith continued to
do the work; the framework was decoration.

**A PR satisfies this discipline iff it changes the production path,
not just the metadata around it.** Concretely:

- Adding a `@lowering` for `mir.Scan` is **not done** until removing
  the corresponding `if isinstance(head, mir.Scan):` branch from
  `lower_scan_pipeline` (or, during migration, until `mir.Scan ∈
  USE_DECLARATIVE`).
- Adding a `@pragma` for `dedup_hash` is **not done** until the
  corresponding `if ctx.dedup_hash:` branch in the lowering is
  unreachable (after `MirPragmaPass` clears the flag).
- Adding a sub-dialect for `parallel.block_group` is **not done**
  until `if ctx.bg_enabled:` is unreachable in production paths.

Discipline test D9 (`test_every_registration_is_exercised`) is the
mechanical floor. The reviewer's "Verified no shortcut" comment in §4
is the human ceiling. Both must hold.

## 8. CI test inventory

The discipline test suite (`tests/test_discipline_*.py`):

| Test file | Enforces |
|---|---|
| `test_discipline_no_pragma_branches.py` | D1 |
| `test_discipline_no_monolith_calls.py` | D2 |
| `test_discipline_no_isinstance_chains.py` | D3 |
| `test_discipline_no_import_side_effects.py` | D4 (extends existing test) |
| `test_iir_no_raw_string_growth.py` | D5 (already exists, whole-tree) |
| `test_discipline_core_isolation.py` | D6, D8, D11 |
| `test_discipline_one_lowering_per_file.py` | D7 |
| `test_discipline_registrations_exercised.py` | D9 |
| `test_discipline_lower_ctx_pinned.py` | D10 |
| `test_discipline_use_declarative_monotonic.py` | D12 |
| `test_discipline_pragma_typed_objects.py` | D13, D14, D15 (per [`pragma_as_typed_object.md`](pragma_as_typed_object.md)) |
| `test_discipline_transitional_state_ratchet.py` | D18 (mutation shims, deprecated fields, module-global registries — see [`pragma_as_typed_object.md`](pragma_as_typed_object.md) and PR #28's A1 shims for current inventory) |
| `test_discipline_pipeline_completeness.py` | R1, R4, R5, R5b |
| `test_codegen_completeness.py` | R3 (existing — verify_renderability) |

Most ship in Layer 1 (foundation). All must be green for any Layer 2+
PR to merge. Each test is < 100 LOC; the entire discipline suite
should run in < 1 second.

## 9. Amendment process

Discipline rules are not negotiable inside feature PRs. To amend a
rule:

1. Open a separate `discipline-amendment/<id>` branch.
2. The PR body must explain WHY the rule is wrong (with concrete
   evidence — e.g., a real refactor it forbids that should be
   allowed).
3. Owner approval required.
4. The amendment lands as a docs-only PR before any code PR that
   relies on the new rule.

This is the "stop the parallel work, revise the discipline doc, then
resume" process from `compiler_redesign.md` §10.

## 10. Sign-off

This document represents the contract every contributor (human or
agent) operates under. Sign-off includes:

- [ ] CI discipline tests are hard PR blocks (no warning-only mode).
- [ ] Layer A + Layer B + Layer C enforcement architecture is in place
  before Layer 2 parallel work starts.
- [ ] The "no useless complicated pseudo refactor" test in §7 is the
  ceiling-and-floor for every PR.
- [ ] Discipline amendments follow §9; never inside a feature PR.

Reviewer initials and date below.
