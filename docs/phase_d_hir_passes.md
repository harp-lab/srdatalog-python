---
orphan: true
---

# Phase D — HIR passes onto `ProgramPass`

The HIR planning passes (stratify, semi-naive, plan, index, ...) are
currently invoked imperatively from `compile_to_hir`. Phase D wraps
each as a `ProgramPass` instance and inserts them into
`DEFAULT_PIPELINE` so they're first-class members of the data-driven
compile.

Companion to [`compiler_redesign.md`](compiler_redesign.md) §4 (Pass
kinds; `ProgramPass` justification) and
[`ir_derivation_topology.md`](ir_derivation_topology.md) §3.1 (HIR
pass table).

## 1. Goal

After Phase D:

- Each HIR pass is a `ProgramPass` instance, in its own file under
  `dialects/hir/passes/<pass>.py`.
- `compile_to_hir` no longer hardcodes the pass sequence; `Compiler.run`
  drives them via the pipeline data.
- HIR types stay as planning records (mutable, not Op-subclassed) —
  the Stage 3B "wrong abstraction" call from earlier still stands.
  ProgramPass is the framework escape hatch for whole-program
  transformations.
- External plugins can ship new HIR passes (e.g., a custom
  optimization or a debugging dump) by registering a `ProgramPass`
  and inserting it into the pipeline at a specified position.

Phase D does NOT:

- Convert HIR types to `Op` subclasses (out of scope; per the
  redesign, HIR stays as records).
- Break HIR's existing API (planning records are still mutable;
  passes still mutate them in-place, just inside `apply()`).

## 2. `ProgramPass` contract

```python
class ProgramPass(Pass):
    """Whole-program transformation. Used for HIR planning passes
    that operate on HirProgram as a unit.

    Unlike LoweringPass / RewritePass, ProgramPass does NOT walk an
    Op tree. It calls `self.fn(prog, compiler)` and returns the
    result. The function may mutate `prog` in place (HIR's
    convention) or return a new HirProgram.

    Usage:
        @program_pass(name="stratify",
                      consumes=("hir",),
                      produces=("hir",))
        def stratify_pass(prog, compiler):
            ...   # imperative HIR transformation
            return prog
    """
    name: str
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    fn: Callable[[Any, Compiler], Any]

    def apply(self, prog, compiler):
        return self.fn(prog, compiler)
```

The `@program_pass` decorator wraps a function as a `ProgramPass`
instance and registers it with the running Compiler.

## 3. Per-HIR-pass migration table (Wave 2B)

| Wave 2B PR | Branch | HIR pass | Existing function (in `hir/`) | Notes |
|---|---|---|---|---|
| **D-Stratify** | `feat/hir-stratify-pass` | `StratifyPass` | `stratify.py:stratify(prog)` | SCC analysis on the rule dep graph |
| **D-Split** | `feat/hir-split-pass` | `SplitPass` | `split.py:split_multihead(prog)` | Splits multi-head rules |
| **D-SemiNaive** | `feat/hir-semi-naive-pass` | `SemiNaivePass` | `semi_naive.py:gen_variants(prog)` | Delta variants for recursive rules |
| **D-Plan** | `feat/hir-plan-pass` | `PlanPass` | `plan.py:plan(prog)` | var_order / clause_order per variant |
| **D-Index** | `feat/hir-index-pass` | `IndexSelectionPass` | `index.py:select_indices(prog)` | Required indices per relation |
| **D-TempRel** | `feat/hir-temprel-synthesis-pass` | `TempRelSynthesisPass` | `temprel.py:synthesize(prog)` | Temp relations (semi-join helpers) |
| **D-TempIndex** | `feat/hir-temp-index-pass` | `TempIndexRegistrationPass` | `temp_index.py:register(prog)` | Temp relation indices |

Each PR is small (~50–100 LOC): wrap an existing imperative function
as a `ProgramPass`, move it to `dialects/hir/passes/<pass>.py`,
register it.

### 3.1 Per-PR template

```python
# File: src/srdatalog/dialects/hir/passes/stratify.py

from srdatalog.core import program_pass

@program_pass(name="stratify", consumes=("hir",), produces=("hir",))
def stratify_pass(prog, compiler):
    """SCC analysis on the rule dependency graph; populates
    prog.strata with the topo-sorted strata.

    Migrated from hir/stratify.py:stratify (Phase D-Stratify).
    Behavior unchanged; this is purely a wrapping move so the pass
    becomes part of DEFAULT_PIPELINE.
    """
    from srdatalog.ir.hir.stratify import stratify as _legacy_stratify
    _legacy_stratify(prog)  # mutates prog in place
    return prog
```

### 3.2 Per-PR acceptance gate

- The migrated pass produces the same `HirProgram` as the legacy
  invocation on every fixture.
- A new test `test_hir_<pass>_pass_via_pipeline` runs the pass via
  `Compiler.run` (with a pipeline containing only this pass) and
  asserts the output matches the legacy direct call.
- The legacy function is **not deleted** in this PR — it stays in
  place as the pass's implementation. Deletion happens in the cleanup
  PR after all 7 passes have been migrated.

## 4. `compile_to_hir` becomes data-driven

### 4.1 Before

```python
def compile_to_hir(program, verbose=False):
    return default_pipeline(verbose=verbose).compile_to_hir(program)

def default_pipeline(verbose=False):
    p = HirCompilerPipeline(verbose=verbose)
    p.add_hir_transform(StratifyPass())
    p.add_hir_transform(SplitPass())
    p.add_hir_transform(SemiNaivePass())
    p.add_hir_transform(PlanPass())
    p.add_hir_transform(TempRelSynthesisPass())
    p.add_hir_transform(IndexSelectionPass())
    p.add_hir_transform(TempIndexRegistrationPass())
    return p
```

### 4.2 After

```python
def compile_to_hir(program, *, compiler=None):
    """Run the HIR sub-pipeline only. Returns HirProgram.

    Convenience for callers that want HIR but not MIR/IIR/codegen.
    For full compile, use Compiler.run(program, pipeline=DEFAULT_PIPELINE).
    """
    compiler = compiler or Compiler.with_default_plugins()
    hir_pipeline = [p for p in DEFAULT_PIPELINE
                    if isinstance(p, ProgramPass)
                    and p.consumes == ("hir",)]
    hir_prog = HirProgram.from_user_program(program)
    return compiler.run(hir_prog, pipeline=hir_pipeline)
```

`DEFAULT_PIPELINE` (in `pipeline.py`) lists all 7 HIR passes by
instance, in order. Inserting a custom pass between two HIR passes
is a one-line change to a user-supplied pipeline list.

## 5. Sign-off

Phase D is complete iff:

- [ ] All 7 Wave 2B PRs merged.
- [ ] `compile_to_hir` no longer constructs a hardcoded
  `HirCompilerPipeline`; it pulls the HIR sub-pipeline from
  `DEFAULT_PIPELINE`.
- [ ] `dialects/hir/passes/` contains exactly 7 modules, each with
  exactly one `@program_pass`-decorated function.
- [ ] Byte-equivalence preserved (HIR / MIR goldens unchanged).
- [ ] A new test asserts inserting a custom user `ProgramPass`
  between two built-in passes works end-to-end (validates the
  extension model from the user side).

After Phase D, every IR layer (HIR / MIR / IIR) is driven by passes
in `DEFAULT_PIPELINE`. No layer has imperative dispatch.
