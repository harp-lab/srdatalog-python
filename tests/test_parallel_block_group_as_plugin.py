'''Phase E discipline test — `parallel.block_group` is shipped as a
discoverable entry-point plugin AND remains usable via the legacy
direct-import pathway, byte-for-byte equivalent.

Spec: `docs/phase_e_plugin_extensibility.md` §2 (entry-point discovery)
and §5 (built-ins ARE plugins). Companion to
`tests/test_sorted_array_as_plugin.py`,
`tests/test_iir_cf_as_plugin.py`,
`tests/test_parallel_data_as_plugin.py`, and
`tests/test_d2l_as_plugin.py` (which exercise the same contract for
the first four built-in dialects shipped as plugins). This file
covers the fifth real built-in dialect — the typed-pragma
materialization sub-dialect that hosts the `BlockGroup` Pragma and the
`mir.BlockGroupRoot` wrap-op lowering (introduced by C3, PR #45).

Three things asserted:

  1. `Compiler.with_default_plugins()` discovers `parallel_block_group`
     through the `srdatalog.dialects` entry-point group and registers
     its dialect on the resulting Compiler.
  2. Calling `compiler.register_plugin(register)` twice is a no-op the
     second time (F4 idempotency, exercised against the real register
     callable).
  3. End-to-end `compile_kernel_body(ep)` returns byte-identical text
     whether or not the caller first went through
     `with_default_plugins()` — for BOTH a plain (non-BG) program AND
     a program whose recursive rule carries `with_pragma(BlockGroup())`.
     The BG-pragma variant is the load-bearing assertion for THIS PR:
     it pins that wrapping the BG sub-dialect as a plugin did not
     perturb codegen for the dual-write transition path
     (`ep.block_group=True` AND a `BlockGroup` typed pragma both set,
     which is exactly what the DSL produces today).

If a future change inadvertently makes the plugin path diverge from
the legacy path, this test catches it before it reaches the full
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
from srdatalog.ir.dialects.parallel.block_group import (
  DIALECT as PARALLEL_BLOCK_GROUP_DIALECT,
)
from srdatalog.ir.dialects.parallel.block_group import (
  register as register_parallel_block_group,
)
from srdatalog.ir.dialects.parallel.block_group.pragmas.block_group import (
  BlockGroup,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _tc_program() -> Program:
  '''Tiny TC-style program — same shape as
  `tests/test_parallel_data_as_plugin.py::_tc_program`. Two rules give
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


def _tc_program_with_bg_pragma() -> Program:
  '''Same TC shape, but the recursive rule carries
  `with_pragma(BlockGroup())`. Includes an explicit `with_plan(delta=0,
  var_order=...)` so the join becomes a multi-source `ColumnJoin` at
  the pipeline root — the only shape the legacy BG path supports
  (`jit_root_column_join_block_group` only fires on root CJ multi).

  This exercises the C3 dual-write transition: the DSL sets BOTH the
  typed `BlockGroup` pragma on the rule's plan AND the legacy
  `block_group=True` bool. `MirPragmaPass` then strips the pragma but
  leaves `pipeline` untouched (dual-write skip branch); the legacy
  bool-driven emit path produces the actual BG CUDA. The byte-equiv
  assertion in `test_compile_kernel_body_byte_equivalent_with_bg_pragma`
  pins that this entire chain is invariant under
  plugin-vs-explicit-register paths.

  Mirrors `tests/test_integration_tc_2level_bg.py::build_tc_2level_bg`
  but applies the pragma via `with_pragma(BlockGroup())` (the typed
  surface this PR is validating) rather than the legacy
  `with_plan(block_group=True)` kwarg.
  '''
  X, Y, Z = Var('x'), Var('y'), Var('z')
  arc = Relation('ArcInput', 2)
  edge = Relation('Edge', 2)
  path = Relation('Path', 2)
  return Program(
    rules=[
      (edge(X, Y) <= arc(X, Y)).named('EdgeLoad'),
      (path(X, Y) <= edge(X, Y)).named('TCBase'),
      (
        (path(X, Z) <= path(X, Y) & edge(Y, Z))
        .named('TCRec')
        .with_plan(delta=0, var_order=['y', 'x', 'z'])
        .with_pragma(BlockGroup())
      ),
    ],
  )


def _find_execute_pipelines(node: object) -> list[mir.ExecutePipeline]:
  '''Walk a MIR plan and collect every ExecutePipeline. Mirrors the
  helper of the same name in `test_parallel_data_as_plugin.py`.
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


def _first_bg_ep(mir_prog: mir.Program) -> mir.ExecutePipeline:
  '''Find the first EP with `block_group=True`. Asserts that BG was
  actually materialized (sanity for the BG-pragma program fixture).'''
  for step, _is_rec in mir_prog.steps:
    for ep in _find_execute_pipelines(step):
      if ep.block_group:
        return ep
  raise AssertionError('no block_group ExecutePipeline in mir_prog')


def _build_first_ep_via(compiler: Compiler) -> mir.ExecutePipeline:
  '''Drive the program-level pipeline on `compiler` to get a real
  `ExecutePipeline` we can hand to `compile_kernel_body`.'''
  state = compiler.run(InitialProg(program=_tc_program()), pipeline=DEFAULT_PROGRAM_PIPELINE)
  assert state.mir_program is not None
  return _first_ep(state.mir_program)


def _build_first_bg_ep_via(compiler: Compiler) -> mir.ExecutePipeline:
  '''Same as `_build_first_ep_via` but uses the BG-pragma program and
  returns the first `block_group=True` EP. Used for the BG-specific
  byte-equivalence assertion.'''
  state = compiler.run(
    InitialProg(program=_tc_program_with_bg_pragma()),
    pipeline=DEFAULT_PROGRAM_PIPELINE,
  )
  assert state.mir_program is not None
  return _first_bg_ep(state.mir_program)


# -----------------------------------------------------------------------------
# 1. Plugin discovery — `with_default_plugins()` picks up parallel.block_group
# -----------------------------------------------------------------------------


def test_parallel_block_group_is_discovered_by_with_default_plugins() -> None:
  '''`Compiler.with_default_plugins()` walks the `srdatalog.dialects`
  entry-point group; the `parallel_block_group` entry point shipped in
  this repo's `pyproject.toml` is loaded and registers the
  `parallel.block_group` dialect.

  If this test fails with "no plugins loaded", the most likely cause
  is that the editable install's `.egg-info` is stale — run
  `pip install -e . --no-deps --force-reinstall` (or `uv sync`) to
  refresh the entry-point metadata.
  '''
  compiler = Compiler.with_default_plugins()

  # The plugin name comes from the entry-point name in pyproject.toml
  # (`[project.entry-points."srdatalog.dialects"] parallel_block_group = "..."`)
  # — F4's `_resolve_plugin` records that name in `_plugins_loaded`.
  assert 'parallel_block_group' in compiler._plugins_loaded, (
    f'expected parallel_block_group in loaded plugins; got '
    f'{list(compiler._plugins_loaded)!r}. '
    f'If empty, the editable install metadata is stale: '
    f'`pip install -e . --no-deps --force-reinstall`.'
  )

  # The dialect itself is registered on the Compiler.
  dialect = compiler.get_dialect('parallel.block_group')
  assert dialect is PARALLEL_BLOCK_GROUP_DIALECT, (
    'with_default_plugins registered a different Dialect instance than '
    'the module-level singleton — the plugin path is not reusing DIALECT'
  )

  # The dialect's plugin attribution is the entry-point name.
  assert compiler._dialects_by_plugin['parallel.block_group'] == 'parallel_block_group'


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


def test_register_parallel_block_group_plugin_idempotent() -> None:
  '''Calling `register_plugin(register_parallel_block_group)` twice on
  the same Compiler does not double-register the dialect and does not
  raise. F4's `register_plugin` short-circuits on the plugin name
  (`register.plugin_name = 'parallel_block_group'`) — this test
  verifies the short-circuit fires for the real register callable, not
  just the synthetic ones in `test_core_plugin.py`.
  '''
  compiler = Compiler()
  compiler.register_plugin(register_parallel_block_group)
  first_loaded = dict(compiler._plugins_loaded)
  first_dialect_count = len(compiler.dialects)

  # Second call must be a no-op (not raise PluginConflictError, not
  # double-register the dialect, not append a second lowering, etc.).
  compiler.register_plugin(register_parallel_block_group)

  assert compiler._plugins_loaded == first_loaded
  assert len(compiler.dialects) == first_dialect_count
  assert compiler.get_dialect('parallel.block_group') is PARALLEL_BLOCK_GROUP_DIALECT


def test_register_parallel_block_group_metadata_matches_entry_point() -> None:
  '''The `register` callable's `plugin_name` / `provides` / `requires`
  attributes match the entry-point declaration in `pyproject.toml`.
  These attributes are what F4 topo-sorts on (see
  `core/plugin.py::_topo_sort_plugins`) — getting them wrong silently
  breaks dependency ordering, so we pin them here.
  '''
  assert register_parallel_block_group.plugin_name == 'parallel_block_group'  # type: ignore[attr-defined]
  assert register_parallel_block_group.provides == ('parallel.block_group',)  # type: ignore[attr-defined]
  assert register_parallel_block_group.requires == ()  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# 3. Byte-equivalence — plugin path and legacy path produce same output
# -----------------------------------------------------------------------------


def test_compile_kernel_body_byte_equivalent_across_plugin_paths() -> None:
  '''The full `compile_kernel_body(ep)` text is byte-identical whether
  the caller obtains its `ep` via a plugin-discovered Compiler
  (`with_default_plugins`) or via a Compiler that explicitly invokes
  the register callable. Baseline (non-BG) variant: ensures that
  registering the BG sub-dialect as a plugin doesn't perturb codegen
  for programs that don't use it.

  We deliberately check both phases (count + materialize) since the
  kernel-body pipeline branches on `is_counting` and a subtle plugin
  ordering bug could surface in only one phase.
  '''
  # Path A: full plugin discovery via the entry-point group.
  compiler_via_plugin = Compiler.with_default_plugins()
  ep_via_plugin = _build_first_ep_via(compiler_via_plugin)

  # Path B: explicit plugin registration (skips entry-point walk).
  compiler_explicit = Compiler()
  compiler_explicit.register_plugin(register_parallel_block_group)
  ep_explicit = _build_first_ep_via(compiler_explicit)

  # Materialize phase.
  text_a = compile_kernel_body(ep_via_plugin, is_counting=False)
  text_b = compile_kernel_body(ep_explicit, is_counting=False)
  assert text_a == text_b, (
    'compile_kernel_body diverged between plugin-discovered and '
    'explicit-register Compilers (materialize phase, non-BG)'
  )

  # Count phase — mirrors the runner emit's per-kernel knobs.
  text_a_ct = compile_kernel_body(ep_via_plugin, is_counting=True, output_var_name='output_ctx')
  text_b_ct = compile_kernel_body(ep_explicit, is_counting=True, output_var_name='output_ctx')
  assert text_a_ct == text_b_ct, (
    'compile_kernel_body diverged between plugin-discovered and '
    'explicit-register Compilers (count phase, non-BG)'
  )


def test_compile_kernel_body_byte_equivalent_with_bg_pragma() -> None:
  '''Load-bearing assertion for THIS PR: with a rule carrying
  `with_pragma(BlockGroup())`, `compile_kernel_body` produces
  byte-identical CUDA whether the BG sub-dialect arrives via the
  plugin path or via an explicit register call.

  The DSL dual-writes BOTH the typed `BlockGroup` pragma onto the
  rule's plan AND the legacy `block_group=True` bool field, so the
  legacy emission path drives the kernel-body output. This test
  verifies that the plugin-vs-explicit choice doesn't perturb that
  output — i.e. that wrapping `parallel.block_group` as a plugin
  preserved the byte-equiv contract for the C3 transition state.

  Both count and materialize phases are checked since BG emits a
  distinct count-kernel (`kernel_bg_*` family in the legacy emitter)
  whose codegen path is structurally different from the materialize
  kernel.
  '''
  # Path A: full plugin discovery via the entry-point group.
  compiler_via_plugin = Compiler.with_default_plugins()
  ep_via_plugin = _build_first_bg_ep_via(compiler_via_plugin)

  # Path B: explicit plugin registration (skips entry-point walk).
  # We explicitly register parallel_block_group; the rest of the
  # pipeline (sorted_array, iir_cf, parallel_data, d2l) is what
  # `compile_kernel_body` needs to produce CUDA — those come from the
  # module-level dialect import side effects, not from per-Compiler
  # registration, so a bare Compiler() + register(BG) is enough to
  # drive the full path.
  compiler_explicit = Compiler()
  compiler_explicit.register_plugin(register_parallel_block_group)
  ep_explicit = _build_first_bg_ep_via(compiler_explicit)

  # Sanity: both EPs carry the BG flag (the DSL dual-write set the
  # legacy bool field, and MirPragmaPass took the skip branch leaving
  # the bool untouched — see `test_pragma_block_group_end_to_end.py
  # ::test_pragma_pass_skips_wrap_in_dual_write_mode`).
  assert ep_via_plugin.block_group is True
  assert ep_explicit.block_group is True

  # Materialize phase.
  text_a = compile_kernel_body(ep_via_plugin, is_counting=False)
  text_b = compile_kernel_body(ep_explicit, is_counting=False)
  assert text_a == text_b, (
    'compile_kernel_body diverged between plugin-discovered and '
    'explicit-register Compilers (materialize phase, BG pragma)'
  )

  # Count phase — BG's count kernel has its own emission path
  # (`_gen_kernel_bg_count`); the plugin-vs-explicit choice must not
  # perturb it either.
  text_a_ct = compile_kernel_body(ep_via_plugin, is_counting=True, output_var_name='output_ctx')
  text_b_ct = compile_kernel_body(ep_explicit, is_counting=True, output_var_name='output_ctx')
  assert text_a_ct == text_b_ct, (
    'compile_kernel_body diverged between plugin-discovered and '
    'explicit-register Compilers (count phase, BG pragma)'
  )
