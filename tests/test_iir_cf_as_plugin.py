'''Phase E discipline test — `iir.cf` is shipped as a discoverable
entry-point plugin AND remains usable via the legacy direct-import
pathway, byte-for-byte equivalent.

Spec: `docs/phase_e_plugin_extensibility.md` §2 (entry-point discovery)
and §5 (built-ins ARE plugins). Companion to
`tests/test_sorted_array_as_plugin.py` (which exercises the same
contract for the first plugin shipped) and `tests/test_core_plugin.py`
(which exercises the F4 framework with synthetic plugins). This file
covers the second real built-in dialect — the IIR control-flow
scaffold that sorted_array (and every future data-structure dialect)
lowers into.

Three things asserted:

  1. `Compiler.with_default_plugins()` discovers `iir.cf` through the
     `srdatalog.dialects` entry-point group and registers its dialect
     on the resulting Compiler.
  2. Calling `compiler.register_plugin(register)` twice is a no-op the
     second time (F4 idempotency, exercised against the real register
     callable).
  3. End-to-end `compile_kernel_body(ep)` returns byte-identical text
     whether or not the caller first went through
     `with_default_plugins()` — the legacy direct-import pathway and
     the plugin-discovered pathway produce the same artifact.

The byte-equivalence anchor (#3) is the load-bearing assertion: if a
future change inadvertently makes the plugin path diverge from the
legacy path, this test catches it before it reaches the full
`test_runner_byte_equivalence.py` suite.
'''

from __future__ import annotations

import srdatalog.ir.mir.types as mir
from srdatalog.dsl import Program, Relation, Var
from srdatalog.ir.codegen.cuda.api import compile_kernel_body
from srdatalog.ir.core import Compiler
from srdatalog.ir.core.plugin import DIALECT_ENTRY_POINT_GROUP
from srdatalog.ir.default_pipelines import (
  DEFAULT_PROGRAM_PIPELINE,
  InitialProg,
)
from srdatalog.ir.dialects.iir.cf import (
  DIALECT as IIR_CF_DIALECT,
)
from srdatalog.ir.dialects.iir.cf import (
  register as register_iir_cf,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _tc_program() -> Program:
  '''Tiny TC-style program — same shape as
  `tests/test_sorted_array_as_plugin.py::_tc_program`. Two rules give
  the MIR plan an actual `ExecutePipeline` to feed
  `compile_kernel_body`.
  '''
  X, Y, Z = Var('x'), Var('y'), Var('z')
  arc = Relation('ArcInput', 2)
  edge = Relation('Edge', 2)
  path = Relation('Path', 2)
  return Program(
    rules=[
      (edge(X, Y) <= arc(X, Y)).named('EdgeLoad'),
      (path(X, Y) <= edge(X, Y)).named('TCBase'),
      (path(X, Z) <= path(X, Y) & edge(Y, Z)).named('TCRec'),
    ],
  )


def _find_execute_pipelines(node: object) -> list[mir.ExecutePipeline]:
  '''Walk a MIR plan and collect every ExecutePipeline. Mirrors
  `_find_execute_pipelines` in `test_sorted_array_as_plugin.py`.
  '''
  out: list[mir.ExecutePipeline] = []
  if isinstance(node, mir.ExecutePipeline):
    out.append(node)
  elif isinstance(node, mir.FixpointPlan):
    for inst in node.instructions:
      out.extend(_find_execute_pipelines(inst))
  elif isinstance(node, mir.ParallelGroup):
    for op in node.ops:
      out.extend(_find_execute_pipelines(op))
  return out


def _first_ep(mir_prog: mir.Program) -> mir.ExecutePipeline:
  for step, _is_rec in mir_prog.steps:
    eps = _find_execute_pipelines(step)
    if eps:
      return eps[0]
  raise AssertionError('no ExecutePipeline in mir_prog')


def _build_first_ep_via(compiler: Compiler) -> mir.ExecutePipeline:
  '''Drive the program-level pipeline on `compiler` to get a real
  `ExecutePipeline` we can hand to `compile_kernel_body`.'''
  state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert state.mir_program is not None
  return _first_ep(state.mir_program)


# -----------------------------------------------------------------------------
# 1. Plugin discovery — `with_default_plugins()` picks up iir.cf
# -----------------------------------------------------------------------------


def test_iir_cf_is_discovered_by_with_default_plugins() -> None:
  '''`Compiler.with_default_plugins()` walks the `srdatalog.dialects`
  entry-point group; the `iir_cf` entry point shipped in this repo's
  `pyproject.toml` is loaded and registers the `iir.cf` dialect.

  If this test fails with "no plugins loaded", the most likely cause
  is that the editable install's `.egg-info` is stale — run
  `pip install -e . --no-deps --force-reinstall` (or `uv sync`) to
  refresh the entry-point metadata.
  '''
  compiler = Compiler.with_default_plugins()

  # The plugin name comes from the entry-point name in pyproject.toml
  # (`[project.entry-points."srdatalog.dialects"] iir_cf = "..."`)
  # — F4's `_resolve_plugin` records that name in `_plugins_loaded`.
  assert 'iir_cf' in compiler._plugins_loaded, (
    f'expected iir_cf in loaded plugins; got {list(compiler._plugins_loaded)!r}. '
    f'If empty, the editable install metadata is stale: '
    f'`pip install -e . --no-deps --force-reinstall`.'
  )

  # The dialect itself is registered on the Compiler.
  dialect = compiler.get_dialect('iir.cf')
  assert dialect is IIR_CF_DIALECT, (
    'with_default_plugins registered a different Dialect instance than '
    'the module-level singleton — the plugin path is not reusing DIALECT'
  )

  # The dialect's plugin attribution is the entry-point name.
  assert compiler._dialects_by_plugin['iir.cf'] == 'iir_cf'


def test_entry_point_group_name_matches_f4() -> None:
  '''The entry-point group name for built-in data dialects is exactly
  the one F4's plugin discovery walks. Locked at `srdatalog.dialects`
  per PR-1a (`docs/phase_decomposition_redesign.md` §3.3.4); the
  legacy `srdatalog.plugins` group remains as a back-compat shim and
  is covered separately by `tests/test_plugin_group_split.py`.
  '''
  assert DIALECT_ENTRY_POINT_GROUP == 'srdatalog.dialects'


# -----------------------------------------------------------------------------
# 2. Idempotency — re-registering the real callable is a no-op
# -----------------------------------------------------------------------------


def test_register_iir_cf_plugin_idempotent() -> None:
  '''Calling `register_plugin(register_iir_cf)` twice on the same
  Compiler does not double-register the dialect and does not raise.
  F4's `register_plugin` short-circuits on the plugin name
  (`register.plugin_name = 'iir_cf'`) — this test verifies the
  short-circuit fires for the real register callable, not just the
  synthetic ones in `test_core_plugin.py`.
  '''
  compiler = Compiler()
  compiler.register_plugin(register_iir_cf)
  first_loaded = dict(compiler._plugins_loaded)
  first_dialect_count = len(compiler.dialects)

  # Second call must be a no-op (not raise PluginConflictError, not
  # double-register the dialect, not append a second lowering, etc.).
  compiler.register_plugin(register_iir_cf)

  assert compiler._plugins_loaded == first_loaded
  assert len(compiler.dialects) == first_dialect_count
  assert compiler.get_dialect('iir.cf') is IIR_CF_DIALECT


def test_register_iir_cf_metadata_matches_entry_point() -> None:
  '''The `register` callable's `plugin_name` / `provides` / `requires`
  attributes match the entry-point declaration in `pyproject.toml`.
  These attributes are what F4 topo-sorts on (see
  `core/plugin.py::_topo_sort_plugins`) — getting them wrong silently
  breaks dependency ordering, so we pin them here.
  '''
  assert register_iir_cf.plugin_name == 'iir_cf'  # type: ignore[attr-defined]
  assert register_iir_cf.provides == ('iir.cf',)  # type: ignore[attr-defined]
  assert register_iir_cf.requires == ()  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# 3. Byte-equivalence — plugin path and legacy path produce same output
# -----------------------------------------------------------------------------


def test_compile_kernel_body_byte_equivalent_across_plugin_paths() -> None:
  '''The full `compile_kernel_body(ep)` text is byte-identical whether
  the caller obtains its `ep` via a plugin-discovered Compiler
  (`with_default_plugins`) or via a Compiler that explicitly invokes
  the register callable. This is the load-bearing assertion: it pins
  that wrapping the dialect as a plugin did not perturb the codegen
  output.

  We deliberately check both phases (count + materialize) since the
  kernel-body pipeline branches on `is_counting` and a subtle plugin
  ordering bug could surface in only one phase.
  '''
  # Path A: full plugin discovery via the entry-point group.
  compiler_via_plugin = Compiler.with_default_plugins()
  ep_via_plugin = _build_first_ep_via(compiler_via_plugin)

  # Path B: explicit plugin registration (skips entry-point walk).
  compiler_explicit = Compiler()
  compiler_explicit.register_plugin(register_iir_cf)
  ep_explicit = _build_first_ep_via(compiler_explicit)

  # Materialize phase.
  text_a = compile_kernel_body(ep_via_plugin, is_counting=False)
  text_b = compile_kernel_body(ep_explicit, is_counting=False)
  assert text_a == text_b, (
    'compile_kernel_body diverged between plugin-discovered and '
    'explicit-register Compilers (materialize phase)'
  )

  # Count phase — mirrors the runner emit's per-kernel knobs.
  text_a_ct = compile_kernel_body(ep_via_plugin, is_counting=True, output_var_name='output_ctx')
  text_b_ct = compile_kernel_body(ep_explicit, is_counting=True, output_var_name='output_ctx')
  assert text_a_ct == text_b_ct, (
    'compile_kernel_body diverged between plugin-discovered and '
    'explicit-register Compilers (count phase)'
  )
