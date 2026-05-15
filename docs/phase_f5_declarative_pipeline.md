# Phase F5 — Declarative Pipeline Shim

**Status:** Spec.
**Prerequisites:** F1 (#30), F3 (#34), F4 (#32) — all merged.
**Layer:** Foundation (Layer 1, last piece).

## 1. Goal

End-state per `docs/compiler_redesign.md` §4:

```python
result = Compiler.with_default_plugins().run(prog, pipeline=DEFAULT_PIPELINE)
```

Today `compile_to_mir` (HIR→MIR) and `compile_kernel_body` (per-EP
MIR→IIR→render) are imperative orchestration functions whose control
flow embeds the pipeline shape. F5 reshapes them into thin
`Compiler.run` callers that consume a declared `list[Pass]`.

The pipeline becomes data. Tests + downstream callers can swap,
reorder, or insert custom passes without forking the orchestration
function.

## 2. Scope: SHIM, not native rewrites

F5 is a **shim** — it wraps existing imperative entry points as
`ProgramPass` instances without changing their internal bodies.
Native per-pass implementations land in later phases:

| Phase | Replaces shim with |
|---|---|
| Phase B (Wave 2A) | `MirToIirLowering` as a real `LoweringPass` (per-MIR-op `@lowering`) |
| Phase C (Wave 2C) | `MirPragmaPass` as a real `RewritePass` over typed pragmas |
| Phase D (Wave 2B) | HIR planning passes (`StratifyPass`, `SplitPass`, …) as real `ProgramPass` instances |
| Phase E (Wave 2D) | Built-in dialects re-shipped as plugins via entry points |

F5 ships the framework + the wiring; the bodies stay imperative
until each phase claims them. The byte-equivalence harness anchors
correctness throughout.

## 3. Two pipelines, two entry points

The codebase has two orchestration boundaries; both reduce to
`Compiler.run` in F5:

### 3.1 Program pipeline — `compile_to_mir(program) → mir.Program`

Today (`src/srdatalog/ir/hir/__init__.py:96-124`):

```python
def compile_to_mir(program, *, hir=None, ...):
    if hir is None:
        hir = compile_to_hir(program)
    steps = lower_hir_to_mir_steps(hir)
    if apply_mir_passes:
        steps = apply_all_mir_passes(steps)
    return mir.Program(steps=[...])
```

F5 shape:

```python
DEFAULT_PROGRAM_PIPELINE: list[Pass] = [
    HirPlanningShim(),   # Program → HirProgram     (wraps compile_to_hir)
    HirToMirShim(),      # HirProgram → list[step] (wraps lower_hir_to_mir_steps)
    MirOptShim(),        # list[step] → list[step] (wraps apply_all_mir_passes)
    MirProgramAssembly(),# list[step] → mir.Program
]

def compile_to_mir(program, *, hir=None, apply_mir_passes=True) -> mir.Program:
    pipeline = DEFAULT_PROGRAM_PIPELINE
    if not apply_mir_passes:
        pipeline = [p for p in pipeline if not isinstance(p, MirOptShim)]
    return Compiler.with_default_plugins().run(
        InitialProg(program=program, hir=hir),
        pipeline=pipeline,
    )
```

`InitialProg` is a small frozen dataclass carrying the through-state
(`program`, optional pre-computed `hir`, intermediate `steps`). Each
shim reads what it needs and returns the next state.

### 3.2 Kernel pipeline — `compile_kernel_body(ep, ...) → str`

Today (`src/srdatalog/ir/codegen/cuda/api.py:98-185`): builds
`view_specs`, `view_decls`, `LoweringCtx`, then `lower_scan_pipeline`
+ `emit`.

F5 shape:

```python
DEFAULT_KERNEL_PIPELINE: list[Pass] = [
    AssignHandlesShim(),      # ep → ep with handle_starts
    CollectViewSpecsShim(),   # ep → KernelCtx with view_specs
    EmitViewDeclsShim(),      # KernelCtx → KernelCtx with view_decls + var maps
    LowerScanPipelineShim(),  # KernelCtx → KernelCtx with iir
    CudaRenderShim(),         # KernelCtx → str (final body text)
]

def compile_kernel_body(ep, *, is_counting, ...) -> str:
    return Compiler.with_default_plugins().run(
        KernelCtx(ep=ep, is_counting=is_counting, ...),
        pipeline=DEFAULT_KERNEL_PIPELINE,
    )
```

`KernelCtx` is the through-state carrying the original kwargs
(`is_counting`, `output_var_name`, etc.) plus the progressively
populated derived values (`view_specs`, `view_decls`, `iir`). Frozen;
each shim returns a new instance via `dataclasses.replace`.

## 4. The `ProgramPass` shim pattern

Per `core/passes.py`:

```python
@dataclass(frozen=True)
class ProgramPass(Pass):
    fn: Callable[[Any, Compiler], Any]

    def apply(self, prog, compiler):
        return self.fn(prog, compiler)
```

Each F5 shim is a `ProgramPass` with a one-line `fn` that calls into
the existing imperative function. Example:

```python
class HirToMirShim(ProgramPass):
    name = "hir_to_mir"
    consumes = ()
    produces = ("mir_steps",)

    @staticmethod
    def fn(state, _compiler):
        from srdatalog.ir.hir.lower import lower_hir_to_mir_steps
        steps = lower_hir_to_mir_steps(state.hir)
        return dataclasses.replace(state, steps=steps)
```

Why subclass `ProgramPass` for each shim instead of using bare
`ProgramPass(name=..., fn=lambda ...)` instances?

- **Per-shim `consumes` / `produces` declarations** make the
  pre-flight ordering check (`Compiler.run` in `core/dialect.py:69`)
  meaningful even at the shim stage. A test that swaps `MirOptShim`
  for `LegacyMirOptShim` should fail at construction time if the
  consumes/produces don't line up, not silently re-order.
- **Discoverable in the codebase via `grep "class.*Shim"`** — the
  shim is a known transitional construct, easy to inventory for
  removal later.
- **Phase B/C/D/E replaces shims by deleting the subclass + adding
  the native Pass next to it**, atomic per pass. The `name=` field
  of the new Pass is the same as the old, so the `DEFAULT_PIPELINE`
  list doesn't need to change.

## 5. State threading: `KernelCtx` and `InitialProg`

Two through-state types, one per pipeline. Both are
`@dataclass(frozen=True, slots=True)`. Each shim reads the fields
it needs and returns a new instance with the next field populated.

```python
@dataclass(frozen=True, slots=True)
class InitialProg:
    program: Program
    hir: HirProgram | None = None
    steps: list[tuple[mir.MirNode, bool]] | None = None
    mir_program: mir.Program | None = None

@dataclass(frozen=True, slots=True)
class KernelCtx:
    ep: m.ExecutePipeline
    is_counting: bool
    output_var_name: str = "output"
    output_vars: dict[str, str] | None = None
    slot_mode: str = "positional"
    rel_index_types: dict[str, str] | None = None
    tiled_cartesian: bool = False
    bg_enabled: bool = False
    # populated by shims:
    view_specs: tuple[Any, ...] | None = None
    view_decls: str | None = None
    name_map: dict[str, str] | None = None
    base_map: dict[str, int] | None = None
    iir: Any = None
    body_text: str | None = None
```

The state carries the entire kwarg surface so that pipeline
swapping (test scenarios, alternate targets) is just a matter of
swapping passes — the orchestration function never branches on
kwargs.

## 6. Discovery + plugin wiring (interaction with F4)

`Compiler.with_default_plugins()` already discovers entry-point
plugins per F4. F5 adds:

- `srdatalog` ships a default plugin (entry point
  `srdatalog.builtin.cuda`) that registers no dialects but DOES
  register the F5 shim Passes by name — letting pipeline-swap
  tests reach them via `compiler.get_pass("hir_to_mir")` (a future
  helper) without importing concrete shim classes.

That helper is out of F5's scope; F5 just declares the shims as
module-level singletons that callers import directly:

```python
# src/srdatalog/ir/pipeline.py
DEFAULT_PROGRAM_PIPELINE: list[Pass] = [
    HirPlanningShim(),
    HirToMirShim(),
    MirOptShim(),
    MirProgramAssembly(),
]
```

## 7. Discipline impact

- **D9** ("registered passes must be triggered by a production
  test"): both `DEFAULT_PROGRAM_PIPELINE` and
  `DEFAULT_KERNEL_PIPELINE` are exercised by every byte-equivalence
  test, so D9 stays satisfied.
- **R1** ("every Pass instance referenced in `DEFAULT_PIPELINE` is
  registered with the Compiler"): F5 introduces the wiring; the
  ratchet stays at 0 violations from day one.
- **D6** ("no imports from `dialects/` in core"): the shims live in
  `ir/pipeline.py` (a NEW module under `ir/`, NOT under `ir/core/`).
  `core/` stays dialect-agnostic. The shims are allowed to import
  from `hir/`, `mir/`, `codegen/cuda/` because the shim module
  itself is the integration layer.

## 8. Acceptance test

A single end-to-end test that exercises both pipelines via the
declarative shape:

```python
def test_default_pipelines_match_imperative_baseline():
    program = Program.parse(SAMPLE_DSL)
    # Declarative path:
    declarative_mir = Compiler.with_default_plugins().run(
        InitialProg(program=program),
        pipeline=DEFAULT_PROGRAM_PIPELINE,
    ).mir_program
    # Imperative baseline:
    imperative_mir = compile_to_mir(program)
    assert _normalize(declarative_mir) == _normalize(imperative_mir)
```

Plus the existing 280+ byte-equivalence tests — all must pass
unchanged after `compile_to_mir` + `compile_kernel_body` reduce to
`Compiler.run`.

## 9. Implementation milestones

| Step | What | Lines | Test |
|---|---|---|---|
| **F5.1** | New module `src/srdatalog/ir/pipeline.py` defining `InitialProg`, `KernelCtx`, the 9 shim subclasses, `DEFAULT_PROGRAM_PIPELINE`, `DEFAULT_KERNEL_PIPELINE` | ~250 | `test_pipeline_shims_smoke.py` — each shim runs in isolation |
| **F5.2** | Reduce `compile_to_mir` to a 3-line `Compiler.run` call (preserves `apply_mir_passes=False` knob via pipeline filter) | ~10 changed | full byte-equivalence suite |
| **F5.3** | Reduce `compile_kernel_body` to a 3-line `Compiler.run` call | ~80 changed (helpers already in shims) | full byte-equivalence suite |
| **F5.4** | Add `test_default_pipelines_match_imperative_baseline` (golden anchor for §8) | new test | one-shot |

Each milestone is a separate PR; F5.1 is pure addition (no
behavior change), F5.2 + F5.3 are the byte-equivalence-gated cuts.

## 10. What this unblocks

- **Phase B** can introduce a new `MirToIirLowering` Pass + delete
  `LowerScanPipelineShim` in the same PR — the `DEFAULT_KERNEL_PIPELINE`
  list barely changes.
- **Phase C** can introduce `MirPragmaPass` between `HirToMirShim`
  and `MirOptShim` without touching `compile_to_mir`.
- **Phase D** can replace the monolithic `compile_to_hir` body with
  individual `StratifyPass`, `SplitPass`, etc., one PR each, by
  expanding `HirPlanningShim` into its constituents.
- **Phase E** plugins can register custom passes that user code
  injects via `Compiler.run(prog, pipeline=DEFAULT + [MyPass()])`.
