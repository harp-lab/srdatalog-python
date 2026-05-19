# srdatalog-jaccard-demo

External-plugin demo for [srdatalog](../../). Proves the central
extensibility claim: a separate Python package, installed via `pip
install`, can extend the srdatalog compiler with a new dialect +
typed Pragma + lowering rule WITHOUT touching any file in
`srdatalog/` core.

## What it ships

A minimal `relation.jaccard` dialect:

- `Jaccard(threshold: float)` — typed `Pragma` subclass; users attach
  it to a rule via `Rule(...).with_pragma(Jaccard(threshold=0.7))`.
- `JaccardIndex(inner: InsertInto, threshold: float)` — MIR wrap op
  the pragma materializes into.
- `@pragma_handler(Jaccard, on=ExecutePipeline)` — wraps each
  trailing `InsertInto` in `JaccardIndex(...)` during
  `MirPragmaPass`.
- `@lowering(target=DIALECT, source=JaccardIndex)` — emits IIR for
  the wrap op (delegates to the sorted_array dialect's
  `_lower_insert_into` under `ctx.dedup_hash=True`, plus a marker
  comment carrying the threshold).

## How discovery works

`pyproject.toml` declares the entry point:

```toml
[project.entry-points."srdatalog.plugins"]
jaccard = "srdatalog_jaccard:register"
```

Once installed, `Compiler.with_default_plugins()` walks the
`srdatalog.plugins` entry-point group, calls
`srdatalog_jaccard.register(compiler)`, and the new dialect lands on
the running compiler. No edits to `src/srdatalog/` and no edits to
the main package's `pyproject.toml` are required.

## Install + run the tests

From the repo root:

```bash
pip install -e examples/srdatalog_jaccard_demo --no-deps
PYTHONPATH=src python -m pytest examples/srdatalog_jaccard_demo/tests/ -v
```

The tests verify:

1. `Compiler.with_default_plugins()` discovers and loads the
   `jaccard` entry point.
2. `Rule(...).with_pragma(Jaccard(...))` is accepted by the DSL
   (the typed-pragma registry sees the new handler).
3. `compile_to_mir(program)` runs end-to-end with a Jaccard pragma
   on a rule, producing a MIR program containing `JaccardIndex`
   wrap ops after `MirPragmaPass`.
4. The registered `@lowering(target=DIALECT, source=JaccardIndex)`
   rule emits well-formed IIR; the rendered output matches a stable
   golden snapshot.

## What this demo does NOT modify

- No files under `src/srdatalog/`.
- No entries in the main package's `[project.entry-points."srdatalog.plugins"]`
  block.
- No imports from framework-internal modules other than the public
  `srdatalog.ir.core` surface and one deferred call into the
  sorted_array lowering helper (`_lower_insert_into`) — flagged at
  the use site as the only cross-dialect reuse, intentionally taken
  so the demo can focus on the registration pathway rather than
  re-implementing well-tested codegen.

## Reference

Spec: [`docs/phase_e_plugin_extensibility.md`](../../docs/phase_e_plugin_extensibility.md)
§4 (worked example).
