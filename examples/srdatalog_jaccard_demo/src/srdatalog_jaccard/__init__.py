'''srdatalog_jaccard — external-plugin demo for srdatalog.

Spec: `docs/phase_e_plugin_extensibility.md` §4 (worked example).

What this package proves: a SEPARATE Python package, installed via
`pip install srdatalog-jaccard-demo`, can extend the srdatalog
compiler with a new dialect + typed Pragma + lowering rule WITHOUT
touching any file in `srdatalog/` core.

The plugin contract surface this demo exercises:

  - `Dialect(name='relation.jaccard', ops=[JaccardIndex])` — new
    dialect with a custom Op.
  - `Jaccard(Pragma)` — typed Pragma subclass.
  - `@pragma_handler(Jaccard, on=ExecutePipeline)` — materializes
    each Jaccard pragma into a `JaccardIndex(inner=InsertInto)`
    wrap op (parallel to the built-in `DedupHash` / `WorkStealing`
    pattern).
  - `@lowering(target=DIALECT, source=JaccardIndex)` — lowers the
    wrap op to IIR by delegating to the sorted_array dialect's
    `_lower_insert_into` helper. The delegation reuses an
    EXISTING IIR shape (the dedup-style emit, but parameterised
    by a Jaccard threshold side-channel) — the demo's point is
    the REGISTRATION pathway, not novel codegen semantics.
  - `register(compiler)` — entry-point callable invoked by
    `Compiler.with_default_plugins()`. Declared in this package's
    `pyproject.toml` under `[project.entry-points."srdatalog.plugins"]`
    as `jaccard = "srdatalog_jaccard:register"`.

What this demo does NOT modify:

  - Nothing in `src/srdatalog/`. Verified by the in-tree
    `tests/test_jaccard_external_plugin.py` smoke run.
  - Nothing in the main `pyproject.toml` entry-point block.

If a future change to srdatalog core breaks any of the above
imports, the plugin contract has leaked and needs a follow-up.
'''

from __future__ import annotations

from typing import Any

# Importing `lowerings` runs `@lowering(DIALECT, JaccardIndex)` as a
# side effect — registers the wrap-op lowering on DIALECT.lowerings.
# Importing `pragmas.jaccard` runs `@pragma_handler(Jaccard, ...)` as
# a side effect — populates the module-global pragma registry that
# `MirPragmaPass` consults and that `Rule.with_pragma` validates
# against. Both registrations are import-time + idempotent (Python's
# module cache makes the side effects fire exactly once per process).
from srdatalog_jaccard import lowerings
from srdatalog_jaccard.dialect import DIALECT
from srdatalog_jaccard.pragmas import jaccard

__all__ = ['DIALECT', 'register']


def register(compiler: Any) -> None:
  '''Plugin entry point — register the `relation.jaccard` dialect
  on `compiler`.

  Lowerings + pragma handler are wired by the import-time side
  effects above. This callable only performs the per-Compiler step:
  `compiler.register_dialect(DIALECT)`.

  Idempotent: F4's `Compiler.register_plugin` short-circuits
  re-registration of the same plugin name (`jaccard`).
  '''
  compiler.register_dialect(DIALECT)


# Plugin metadata read by F4's topo-sort + conflict detection
# (`src/srdatalog/ir/core/plugin.py::_plugin_attr`).
#
# `plugin_name` — entry-point identifier; must match the key in
#   pyproject.toml `[project.entry-points."srdatalog.plugins"]`.
# `provides` — the dialect name this plugin contributes. Other
#   plugins may declare `requires=('relation.jaccard',)` to load
#   after this one.
# `requires` — dialects that must be loaded first. We declare
#   `relation.sorted_array` because the lowering body delegates to
#   that dialect's `_lower_insert_into` helper at call time. The
#   helper itself is reachable via a deferred import regardless of
#   load order, but pinning the dependency in `requires` documents
#   the contract: this plugin would not function without
#   sorted_array's lowering helpers, so loading it standalone is a
#   misconfiguration.
register.plugin_name = 'jaccard'  # type: ignore[attr-defined]
register.provides = ('relation.jaccard',)  # type: ignore[attr-defined]
register.requires = ('relation.sorted_array',)  # type: ignore[attr-defined]
