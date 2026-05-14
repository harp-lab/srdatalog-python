---
orphan: true
---

# Phase E — Plugin extensibility (entry-point discovery + worked examples)

The phase that proves the redesign delivers on its central promise:
**every language feature is a plugin, including built-ins, and an
external package can add new pragmas / dialects / targets without
touching core**.

Companion to [`compiler_redesign.md`](compiler_redesign.md) §3 (the
language is a configuration of the compiler).

## 1. Goal

After Phase E:

- `pip install some-srdatalog-extension` automatically extends the
  compiler. No core changes. No DSL surface changes. No editing
  `DEFAULT_PIPELINE`.
- A worked example external plugin lives in
  `examples/plugin_aggregation_jaccard/` (or as a separately-shipped
  PyPI package) and demonstrates: new pragma, new MIR wrap op, new
  sub-dialect, new IIR ops, new lowerings, new render registrations.
- Built-in pragmas (`pragmas/dedup_hash.py` etc.) are discovered the
  same way as external pragmas — no special-casing.
- Discipline test asserts: removing all pragma modules from
  `pragmas/` reduces the compiler to a no-op (the core knows nothing
  about pragmas; everything is plugin-supplied).

## 2. Plugin discovery — entry points

`pyproject.toml` (in any extension package, including this repo's
own built-ins):

```toml
[project.entry-points."srdatalog.plugins"]
default_lang = "srdatalog.plugins.default_lang:register"
cuda_target = "srdatalog.plugins.cuda_target:register"

# An external package would add:
# my_optimization = "my_pkg.srdatalog_plugin:register"
```

Each entry point points to a `register(compiler)` function:

```python
# srdatalog/plugins/default_lang/__init__.py
def register(compiler):
    """Register all built-in HIR/MIR/IIR dialects, MIR pragma_ops,
    and pragma rewrite rules with the given Compiler."""
    from srdatalog.dialects.hir import DIALECT as HIR_DIALECT
    from srdatalog.dialects.mir import DIALECT as MIR_DIALECT
    # ... all built-in dialects
    for d in (HIR_DIALECT, MIR_DIALECT, ...):
        compiler.register_dialect(d)
    # Pragma modules self-register on import (their @pragma decorators
    # populate the registry on the running compiler instance).
    from srdatalog.pragmas import (dedup_hash, block_group,
                                   work_stealing, tiled_cartesian,
                                   count, fanout)  # noqa: F401
```

`Compiler.with_default_plugins()`:

```python
@classmethod
def with_default_plugins(cls) -> 'Compiler':
    """Construct a Compiler with all entry-point plugins loaded.

    Walks importlib.metadata.entry_points(group='srdatalog.plugins'),
    calls each register(self).

    For tests / custom configs, use Compiler() and call
    register_plugin manually.
    """
    compiler = cls()
    for ep in importlib.metadata.entry_points(group='srdatalog.plugins'):
        register_fn = ep.load()
        register_fn(compiler)
    return compiler
```

## 3. Explicit registration (for tests / custom configs)

```python
compiler = Compiler()
compiler.register_plugin('default_lang')          # by entry-point name
compiler.register_plugin(my_module.register)      # by callable
compiler.register_dialect(my_dialect)             # by Dialect instance
```

All three are the same call shape internally — `register_plugin`
resolves the argument to a `register(compiler)` callable.

## 4. Worked example: ship a custom pragma as an external package

Project layout (notional `my-srdatalog-jaccard` package on PyPI):

```
my-srdatalog-jaccard/
  pyproject.toml
  src/my_srdatalog_jaccard/
    __init__.py
    register.py                  # entry-point target
    pragma.py                    # @pragma(name="jaccard")
    dialect_jaccard/             # new IIR sub-dialect
      ops.py                     # JaccardSimilarity, JaccardThreshold
      print.py
      __init__.py                # DIALECT
    mir_ops/
      jaccard_gate.py            # MIR wrap op
    lowerings/
      lower_mir_jaccard_gate.py  # @lowering for the wrap op
    render/
      cuda.py                    # @register_render(target='cuda') for IIR ops
```

`pyproject.toml`:

```toml
[project]
name = "my-srdatalog-jaccard"
dependencies = ["srdatalog>=2.0"]

[project.entry-points."srdatalog.plugins"]
jaccard = "my_srdatalog_jaccard.register:register"
```

`src/my_srdatalog_jaccard/register.py`:

```python
def register(compiler):
    from . import pragma            # noqa — @pragma decorator fires
    from .dialect_jaccard import DIALECT as JACCARD_DIALECT
    from .mir_ops import jaccard_gate  # noqa — Op class loaded
    from .lowerings import lower_mir_jaccard_gate  # noqa — @lowering fires
    from .render import cuda  # noqa — @register_render fires

    compiler.register_dialect(JACCARD_DIALECT)
```

End-user code (no changes to core, no changes to `srdatalog`'s DSL):

```python
from srdatalog import Compiler, Program, Rule, Var

# Plugin auto-loaded via entry point.
compiler = Compiler.with_default_plugins()

X, Y = Var('x'), Var('y')
similar_pairs = (
    Rule(...)
        .with_pragma('jaccard', {'threshold': 0.7})
)

prog = Program(rules=[similar_pairs])
result = compiler.run(prog)  # uses DEFAULT_PIPELINE; jaccard pragma fires
```

What happens internally:

1. `Compiler.with_default_plugins()` loads `default_lang`,
   `cuda_target`, AND `jaccard` (from the user's installed
   `my-srdatalog-jaccard` package) — all three call `register(compiler)`.
2. The DSL accepts `with_pragma('jaccard', {'threshold': 0.7})`
   because the `@pragma(name='jaccard', value_type=dict)` registered.
3. HIR carries the pragma to MIR.
4. `MirPragmaPass` calls `materialize_jaccard(...)`, which inserts a
   `JaccardGate` MIR wrap op.
5. `MirToIirLowering` finds `@lowering(target=JACCARD_DIALECT,
   source=JaccardGate)` and lowers to IIR using the plugin's IIR ops.
6. `IirCanonicalizePass` decomposes any COMPOUND ops the plugin
   supplied.
7. `verify_renderability` checks that every op type has a CUDA
   renderer; plugin's `register/cuda.py` provided them.
8. `CudaRenderPass` emits the C++.

**Core never knew about jaccard.** The DSL never knew about jaccard.
The default plugin set never knew about jaccard. The plugin itself
provided every step from pragma trigger to rendered text.

## 5. Validation: built-ins ARE plugins (Wave 2D)

To prove the redesign actually delivers extensibility (not just claims
to), Wave 2D PRs re-ship one or two existing built-in features as
plugin packages — i.e., move them out of `srdatalog/` proper into
their own plugin module loaded via entry point.

| PR | Branch | Re-ships | Validates |
|---|---|---|---|
| **E1** | `feat/plugin-aggregation-as-extension` | One aggregation kind (e.g. AggCount) as a plugin | Aggregation framework is plugin-extensible |
| **E2** | `feat/plugin-semiring-as-extension` | One semiring (e.g. BooleanSR) as a plugin | Semiring framework is plugin-extensible |

These PRs are validation; their goal is to confirm the redesign's
extension story works end-to-end. If either fails, the failure is
diagnostic — the redesign needs revision before declaring Phase E
complete.

## 6. Discipline tests added in Phase E

| Test | Enforces | PR |
|---|---|---|
| `test_compiler_with_default_plugins_loads_built_ins` | `Compiler.with_default_plugins()` loads at least the default_lang, cuda_target plugins | E (foundation) |
| `test_external_plugin_can_add_pragma` | A test fixture shipping a fake plugin module + entry point can compile a program using its pragma | E (foundation) |
| `test_core_module_set_pinned` (D11) | The set of files under `core/` is exactly the pinned list — no domain knowledge leaks into core | F1 (foundation), reaffirmed in Phase E |
| `test_default_lang_is_a_plugin` | `srdatalog/plugins/default_lang/__init__.py` exists; built-in pragmas are discoverable through the same entry-point mechanism as external | E (foundation) |
| `test_no_pragma_hardcoded_in_core` (D8) | `core/` has zero string literals matching known pragma names | F1 (foundation), reaffirmed in Phase E |

## 7. Sign-off

Phase E is complete iff:

- [ ] `Compiler.with_default_plugins()` exists and loads via entry
  points.
- [ ] Built-in pragmas live in `pragmas/` and are discovered through
  the same entry-point mechanism as external plugins.
- [ ] At least one external-plugin worked example exists (in
  `examples/`) demonstrating new pragma + new sub-dialect + new
  lowering + new render — all in a separate package, no edits to
  `srdatalog/` proper.
- [ ] Wave 2D validation PRs (E1, E2) merged: at least one
  aggregation and one semiring re-shipped as plugin packages, proving
  the extension story works.
- [ ] Discipline tests in §6 all pass.
- [ ] Documentation in `phase_e_plugin_extensibility.md` (this doc)
  has a step-by-step "How to ship an external plugin" section that a
  user can follow without reading any other doc.

After Phase E sign-off, the compiler is genuinely framework-driven
and genuinely extensible. The redesign delivers what
`compiler_redesign.md` §3 promised.
