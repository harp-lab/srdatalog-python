---
orphan: true
---

# Phase A — MIR onto `Op`

Foundation for the framework-driven compiler. Promote every MIR type
to a `frozen=True, slots=True` `Op` subclass so that
`apply_rewrites_to_fixpoint` and the `LoweringPass` dispatcher can
walk MIR.

Companion to [`compiler_redesign.md`](compiler_redesign.md) §10
(phase order) and [`code_discipline.md`](code_discipline.md)
(forbidden / required patterns).

## 1. Goal

After Phase A:

- Every MIR type is a `@dataclass(frozen=True, slots=True)` `Op`
  subclass.
- The `mir.Program` tree is a frozen Op tree, walkable by
  `core.passes._walk(prog)`.
- `mir.ExecutePipeline` carries `pragmas: tuple[tuple[str, Any], ...]`
  (transition shape — open key/value list) instead of named bool
  fields (`dedup_hash`, `work_stealing`, ...). **This is the Phase A
  shape only.** Once `core/pragma.py` lands (Layer 1), the field
  becomes `pragmas: tuple[Pragma, ...]` per
  [`pragma_as_typed_object.md`](pragma_as_typed_object.md). A2 or a
  follow-up converts the wire format to the typed-object form.
- All sites that mutate MIR in-place (`pipeline.append(...)`,
  `node.handle_start = ...`) have been replaced with
  `dataclasses.replace(...)` or proper builders.
- A new dialect `mir` is registered with the `Compiler` at startup.
- Discipline tests pin: every MIR type is Op-subclassed, no in-place
  mutation, every MIR op has a registered `@lowering`.

Phase A does NOT yet:

- Replace `lower_scan_pipeline` (Phase B).
- Materialize pragmas as op insertions (Phase C).
- Register `@lowering` per MIR op (Phase B).

It only changes the shape of the MIR data so subsequent phases can
operate on it.

## 2. MIR type inventory

60 classes in `src/srdatalog/ir/mir/types.py`. Categorized:

### 2.1 Generic ops (kept)

| Class | Field signature | Notes |
|---|---|---|
| `Program` | `steps: tuple[ProgramStep, ...]` | Top-level container. |
| `ProgramStep` | `node: Op, is_recursive: bool` | Tuple becomes Op subclass too. |
| `ExecutePipeline` | `pipeline: tuple[Op, ...], source_specs: tuple[ColumnSource, ...], dest_specs: tuple[InsertInto, ...], rule_name: str, pragmas: tuple[tuple[str, Any], ...]` | **`pragmas` field replaces `dedup_hash`, `work_stealing`, `block_group`, `fanout`, `count`, `tiled_cartesian`, ...** |
| `InjectCppHook` | `code: str, before_or_after: str` | |
| `ColumnSource` | `rel_name: str, version: Version, index: tuple[int, ...], prefix_vars: tuple[str, ...], handle_start: int, clause_idx: int` | Lists → tuples. |
| `Scan` | `vars: tuple[str, ...], rel_name: str, version: Version, index: tuple[int, ...], prefix_vars: tuple[str, ...], handle_start: int` | Lists → tuples. |
| `ColumnJoin` | `var_name: str, sources: tuple[ColumnSource, ...], handle_start: int` | |
| `CartesianJoin` | `vars: tuple[str, ...], sources: tuple[ColumnSource, ...], var_from_source: tuple[tuple[str, ...], ...], handle_start: int` | Nested lists → nested tuples. |
| `Filter` | `vars: tuple[str, ...], code: str` | |
| `ConstantBind` | `var_name: str, code: str, deps: tuple[str, ...]` | |
| `Aggregate` | `result_var: str, func: str, source: ColumnSource, prefix_vars: tuple[str, ...]` | |
| `Negation` | `source: ColumnSource, prefix_vars: tuple[str, ...]` | |
| `InsertInto` | `rel_name: str, version: Version, vars: tuple[str, ...], index: tuple[int, ...]` | |
| `CreateFlatView` | `rel_name: str, version: Version, index: tuple[int, ...]` | |
| `InnerPipeline` | `pipeline: tuple[Op, ...]` | |
| `ProbeJoin` | (per current types.py) | |
| ... (remaining ~40 types) | | |

### 2.2 Pragma fields removed from `ExecutePipeline`

Currently: 6 named bool fields. After Phase A: removed entirely,
replaced by the open `pragmas` dict.

| Removed field | Lives now as | Materialized by |
|---|---|---|
| `dedup_hash: bool` | `pragmas[("dedup_hash", True)]` | Phase C: `pragmas/dedup_hash.py` |
| `work_stealing: bool` | `pragmas[("work_stealing", True)]` | Phase C: `pragmas/work_stealing.py` |
| `block_group: bool` | `pragmas[("block_group", True)]` | Phase C: `pragmas/block_group.py` |
| `fanout: bool` | `pragmas[("fanout", True)]` | Phase C: `pragmas/fanout.py` (or stay imperative if no consumer) |
| `count: bool` | `pragmas[("count", True)]` | Phase C: `pragmas/count.py` |
| `tiled_cartesian: bool` | (computed, not user-set — kept as field OR passed through pragmas; decision in Phase C) | |

The DSL's `Rule(dedup_hash=True)` becomes sugar over
`Rule.with_pragma("dedup_hash", True)`. The DSL change is part of
Phase A (so the pipeline plumbing produces the right MIR shape),
but the pragma materialization to wrap ops is Phase C.

## 3. Mutation site inventory

Every site that mutates a MIR instance must change to
`dataclasses.replace(...)` or an explicit builder. Inventory (to be
verified by exhaustive grep in the F2 PR):

### 3.1 Known mutation sites

| Site | Mutation | Replacement |
|---|---|---|
| `hir/lower.py:lower_hir_to_mir_steps` | `pipeline.append(...)` while building MIR steps | Build a `list[Op]` locally, then `mir.Program(steps=tuple(local_list))` at the end |
| `mir/passes.py:apply_all_mir_passes` | Mutates `step.pipeline` in place during reorder passes | Each pass returns a new `ProgramStep` via `dataclasses.replace` |
| `dialects/relation/sorted_array/lowerings.py` | Reads `node.handle_start` after mutation (assigned by `assign_handle_positions`) | `assign_handle_positions` returns a new pipeline with `handle_start` set, doesn't mutate |
| `codegen/cuda/api.py:assign_handle_positions` | Sets `.handle_start` on `Scan` / `ColumnJoin` / `CartesianJoin` / `ColumnSource` | Returns a new pipeline with replacements; existing pipeline unchanged |
| `codegen/cuda/complete_runner.py` | Reads (does not mutate) | No change |

### 3.2 Verification approach

A new discipline test `test_no_mir_mutation_outside_replace`:
- AST-scans every `*.py` file under `src/`
- For every assignment `obj.<attr> = ...` where `obj` is statically
  inferable as an `Op` subclass instance — fails CI.
- Allowlist: code inside `core/` (the framework owns `Op` itself).

Implementation: a small AST visitor; ~50 LOC test. Not perfect
static analysis but catches the common patterns.

## 4. PR breakdown

Phase A is 3 PRs, sequential.

### PR A1: `feat/mir-onto-op-types` (~600 LOC diff)

Just the type changes. Production still uses legacy `LoweringCtx`,
legacy `lower_scan_pipeline`. Byte-equivalence preserved.

- Convert all 60 MIR types in `mir/types.py` to
  `@dataclass(frozen=True, slots=True)` + `Op` subclass.
- Convert every `list[X]` field to `tuple[X, ...]`.
- Replace `dedup_hash` / `work_stealing` / `block_group` /
  `fanout` / `count` / `tiled_cartesian` named fields on
  `ExecutePipeline` with `pragmas: tuple[tuple[str, Any], ...]`.
- Add `mir.Program.replace(...)` helpers if needed for ergonomics.
- New `mir/__init__.py` registers `mir` Dialect with all op types.
- Tests: every MIR type round-trips through `dataclasses.replace`,
  is hashable, is walkable by `core.passes._walk`.
- Discipline tests added: `test_mir_ops_are_frozen_op_subclasses`,
  `test_mir_program_walkable`.

### PR A2: `feat/mir-fix-mutation-sites` (~400 LOC diff)

The mechanical change. With frozen MIR, every mutation site fails;
this PR fixes them all.

- `hir/lower.py:lower_hir_to_mir_steps` — build via tuple, never
  append-mutate.
- `mir/passes.py` — each pass returns new instances via
  `dataclasses.replace`.
- `codegen/cuda/api.py:assign_handle_positions` — return new
  pipeline.
- DSL `Rule.with_plan(dedup_hash=True, ...)` — produces
  `pragmas=(("dedup_hash", True),)` instead of named fields.
- HIR `HirRuleVariant.pragmas: dict[str, Any]` field added (existing
  bool fields kept during migration, deprecated).
- Discipline test added: `test_no_mir_mutation_outside_replace`.
- Byte-equivalence: full suite green.

### PR A3: `feat/mir-pragmas-field-only` (~200 LOC diff)

The cleanup. With mutation sites fixed and the new shape in place,
remove the deprecated named bool fields entirely.

- Remove `ExecutePipeline.dedup_hash` etc. — only `pragmas` remains.
- Remove `HirRuleVariant.dedup_hash` etc. — only `pragmas` remains.
- Update `compile_kernel_body` to read `ep.pragmas.get("dedup_hash")`
  during the migration period (Phase B/C will eliminate this read
  entirely).
- Update DSL `PlanEntry` to drop named bool fields.
- Update tests to construct `pragmas=(("dedup_hash", True),)` directly.

After A3: MIR has the final Phase A shape. Phase B can begin.

## 5. Byte-equivalence strategy

Phase A doesn't change the lowering, so the emitted CUDA must be
byte-identical. The risk is that the field re-shape (lists → tuples,
named pragma fields → dict) breaks something subtle.

Per-PR gate:

- A1: full suite green. Type tests added.
- A2: full byte-equivalence suite green (272 runner + 253 jit-batch).
  Mutation-site fixes are mechanical; if a fix changes order or
  identity, byte-equivalence catches it.
- A3: full byte-equivalence suite green. The named-field removal
  changes only the DSL surface; `compile_kernel_body` reads pragmas
  via `ep.pragmas.get(...)` instead of `ep.dedup_hash`.

If byte-equivalence breaks at any step, the PR is blocked until the
divergence is either (a) traced to a Phase A bug and fixed, or (b)
documented as an explicit acceptable change with golden update +
owner sign-off.

## 6. Discipline tests added in Phase A

| Test | Enforces | PR |
|---|---|---|
| `test_mir_ops_are_frozen_op_subclasses` | Every type in `mir/types.py` subclasses `Op` and is `frozen=True, slots=True` | A1 |
| `test_mir_program_walkable` | `mir.Program(...)` is walkable by `core.passes._walk` (yields all descendant ops) | A1 |
| `test_no_mir_mutation_outside_replace` | AST scan: no `obj.<attr> = ...` assignments to statically-inferred `Op` instances outside `core/` | A2 |
| `test_every_mir_op_has_lowering` | Every concrete MIR op type has at least one `@lowering(target=IIR, source=mir.X)` registration. **Stub passes initially** (raises `pytest.xfail` until Phase B); enforces by Phase B end | A1 (stub) |
| `test_pragmas_field_is_open_dict` | `ExecutePipeline.pragmas` is `tuple[tuple[str, Any], ...]`; no named pragma bool fields exist | A3 |

## 7. Risks + mitigations

| Risk | Mitigation |
|---|---|
| `frozen=True` breaks mutation in code we missed | A2's discipline test catches it; CI fails on any regression. |
| `slots=True` breaks `dataclasses.replace` for some types | Tested in A1 round-trip tests; if any type fails, fall back to `frozen=True` only for that type with a documented reason. |
| Tuple-of-tuple fields for nested structures (`var_from_source`) become awkward | Provide a `Mir.cartesian_join(vars=..., sources=..., var_from_source=...)` factory that accepts list inputs and converts. |
| Hashability breaks for types containing dicts (e.g., `Aggregate.func: str` if some site stores a dict) | Audit during A1; convert any dict fields to tuple-of-tuples. |
| HIR↔MIR boundary: HIR is mutable, MIR is frozen — risk of "almost-Op" data crossing | The boundary is `lower_hir_to_mir_steps`. It builds MIR from scratch; no reference shared. |

## 8. Sign-off

Phase A is complete iff:

- [ ] All three PRs (A1, A2, A3) merged.
- [ ] Discipline tests in §6 all green.
- [ ] Byte-equivalence preserved through every PR (or documented
  divergence with sign-off).
- [ ] No `mir.X` reference in `src/` constructs the named pragma
  bool fields (`dedup_hash=`, etc.) — only `pragmas=`.
- [ ] `mir.Program(...)` is walkable by `_walk` end-to-end (test
  fixture covers a real compile output).

After Phase A sign-off, Phase B (per-MIR-op `@lowering` migration)
unblocks.
