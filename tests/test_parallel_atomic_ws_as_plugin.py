'''Phase E discipline test — `parallel.atomic_ws` is shipped as a
discoverable entry-point plugin AND remains usable via the legacy
direct-import pathway, byte-for-byte equivalent.

Spec: `docs/phase_e_plugin_extensibility.md` §2 (entry-point discovery)
and §5 (built-ins ARE plugins). Companion to
`tests/test_sorted_array_as_plugin.py`, `tests/test_iir_cf_as_plugin.py`,
`tests/test_d2l_as_plugin.py`, and `tests/test_parallel_data_as_plugin.py`
(which exercise the same contract for prior built-in dialects shipped
as plugins). This file covers the sixth real built-in dialect — the
atomic work-stealing parallelism strategy that hosts the `WSScope`
MIR wrap op + the `WorkStealing` typed pragma's lowering surface
(introduced by Phase C4, see
`docs/phase_c_pragma_materialization.md` §4.2).

Four things asserted:

  1. `Compiler.with_default_plugins()` discovers `parallel.atomic_ws`
     through the `srdatalog.plugins` entry-point group and registers
     its dialect on the resulting Compiler.
  2. Calling `compiler.register_plugin(register)` twice is a no-op the
     second time (F4 idempotency, exercised against the real register
     callable).
  3. `register.plugin_name` / `provides` / `requires` match the
     entry-point declaration in `pyproject.toml`.
  4. End-to-end `compile_kernel_body(ep)` returns byte-identical text
     whether or not the caller first went through
     `with_default_plugins()` — the legacy direct-import pathway and
     the plugin-discovered pathway produce the same artifact. This
     test runs the byte-equivalence assertion with the WS-specific
     trigger present (`Rule(...).with_pragma(WorkStealing())`) so
     the WS materialization + `WSScope` lowering are actually
     exercised on both sides.

The byte-equivalence anchor (#4) is the load-bearing assertion: if a
future change inadvertently makes the plugin path diverge from the
legacy path, this test catches it before it reaches the full
`test_runner_byte_equivalence.py` suite. The WS pragma is the
discriminating element — without it, the EP shape would not exercise
the C4 lowering surface this sub-dialect owns.
'''

from __future__ import annotations

import srdatalog.ir.mir.types as mir
from srdatalog.dsl import Program, Relation, Var
from srdatalog.ir.codegen.cuda.api import compile_kernel_body
from srdatalog.ir.core import Compiler
from srdatalog.ir.core.plugin import ENTRY_POINT_GROUP
from srdatalog.ir.default_pipelines import (
  DEFAULT_PROGRAM_PIPELINE,
  InitialProg,
)
from srdatalog.ir.dialects.parallel.atomic_ws import (
  DIALECT as ATOMIC_WS_DIALECT,
)
from srdatalog.ir.dialects.parallel.atomic_ws import (
  register as register_atomic_ws,
)
from srdatalog.ir.dialects.parallel.atomic_ws.pragmas.work_stealing import (
  WorkStealing,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _tc_program_with_ws() -> Program:
  '''Tiny TC-style program — same shape as
  `tests/test_parallel_data_as_plugin.py::_tc_program`, but with the
  recursive `TCRec` rule carrying `with_pragma(WorkStealing())` so
  the resulting `ExecutePipeline` carries the WS dual-write signal
  (typed `WorkStealing` instance on each plan's `pragmas` tuple AND
  the legacy `PlanEntry.work_stealing=True` shadow field; see
  `dsl._BUILTIN_BOOL_SHADOW_PRAGMAS`).

  Why the explicit `.with_plan(delta=0)`: the planner's variant-
  level propagation (`_plan_variant` in `ir/hir/plan.py`) keys on
  `PlanEntry.delta` matching `HirRuleVariant.delta_idx`. Recursive
  rule variants have `delta_idx >= 0` (one per recursive body atom),
  while `Rule.with_pragma` on a plan-less rule attaches a default
  `PlanEntry(delta=-1)` (the base-variant convention; see
  `dsl.Rule.with_pragma`). The TCRec rule has a single recursive
  body atom (`path`), so `delta=0` is the variant we need to tag;
  pinning it explicitly here ensures the WS dual-write actually
  reaches the recursive `ExecutePipeline`.

  The pragma is what differentiates this fixture from the sibling
  plugin tests: it forces `MirPragmaPass` to fire the
  `@pragma_handler(WorkStealing, on=ExecutePipeline)` callback and
  the legacy `_lower_insert_into` WS branch to fire on the kernel
  side. Byte-equivalence across paths is meaningful only when the
  sub-dialect's surface is actually exercised.
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
        .with_plan(delta=0)
        .with_pragma(WorkStealing())
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


def _ws_ep(mir_prog: mir.Program) -> mir.ExecutePipeline:
  '''Return the first `ExecutePipeline` whose `work_stealing` bool is
  True (i.e. the WS-tagged rule). The DSL's `.with_pragma(
  WorkStealing())` dual-writes this bool, so picking by it is robust
  to MIR pass reorderings (e.g. if `MirPragmaPass` consumed the
  typed pragma already, the bool is still the unambiguous marker
  during the dual-write window).
  '''
  for step, _is_rec in mir_prog.steps:
    for ep in _find_execute_pipelines(step):
      if ep.work_stealing:
        return ep
  raise AssertionError('no work-stealing ExecutePipeline in mir_prog')


def _build_ws_ep_via(compiler: Compiler) -> mir.ExecutePipeline:
  '''Drive the program-level pipeline on `compiler` to get a real
  `ExecutePipeline` (the WS-tagged one) we can hand to
  `compile_kernel_body`.'''
  state = compiler.run(
    InitialProg(program=_tc_program_with_ws()),
    pipeline=DEFAULT_PROGRAM_PIPELINE,
  )
  assert state.mir_program is not None
  return _ws_ep(state.mir_program)


# -----------------------------------------------------------------------------
# 1. Plugin discovery — `with_default_plugins()` picks up parallel.atomic_ws
# -----------------------------------------------------------------------------


def test_parallel_atomic_ws_is_discovered_by_with_default_plugins() -> None:
  '''`Compiler.with_default_plugins()` walks the `srdatalog.plugins`
  entry-point group; the `parallel_atomic_ws` entry point shipped in
  this repo's `pyproject.toml` is loaded and registers the
  `parallel.atomic_ws` dialect.

  If this test fails with "no plugins loaded", the most likely cause
  is that the editable install's `.egg-info` is stale — run
  `pip install -e . --no-deps --force-reinstall` (or `uv sync`) to
  refresh the entry-point metadata.
  '''
  compiler = Compiler.with_default_plugins()

  # The plugin name comes from the entry-point name in pyproject.toml
  # (`[project.entry-points."srdatalog.plugins"] parallel_atomic_ws = "..."`)
  # — F4's `_resolve_plugin` records that name in `_plugins_loaded`.
  assert 'parallel_atomic_ws' in compiler._plugins_loaded, (
    f'expected parallel_atomic_ws in loaded plugins; got '
    f'{list(compiler._plugins_loaded)!r}. If empty, the editable install '
    f'metadata is stale: `pip install -e . --no-deps --force-reinstall`.'
  )

  # The dialect itself is registered on the Compiler.
  dialect = compiler.get_dialect('parallel.atomic_ws')
  assert dialect is ATOMIC_WS_DIALECT, (
    'with_default_plugins registered a different Dialect instance than '
    'the module-level singleton — the plugin path is not reusing DIALECT'
  )

  # The dialect's plugin attribution is the entry-point name.
  assert compiler._dialects_by_plugin['parallel.atomic_ws'] == 'parallel_atomic_ws'


def test_entry_point_group_name_matches_f4() -> None:
  '''The entry-point group name in `pyproject.toml` is exactly the
  one F4's plugin discovery walks. Locked at `srdatalog.plugins` per
  the spec (`docs/phase_e_plugin_extensibility.md` §2).
  '''
  assert ENTRY_POINT_GROUP == 'srdatalog.plugins'


# -----------------------------------------------------------------------------
# 2. Idempotency — re-registering the real callable is a no-op
# -----------------------------------------------------------------------------


def test_register_parallel_atomic_ws_plugin_idempotent() -> None:
  '''Calling `register_plugin(register_atomic_ws)` twice on the same
  Compiler does not double-register the dialect and does not raise.
  F4's `register_plugin` short-circuits on the plugin name
  (`register.plugin_name = 'parallel_atomic_ws'`) — this test
  verifies the short-circuit fires for the real register callable,
  not just the synthetic ones in `test_core_plugin.py`.
  '''
  compiler = Compiler()
  compiler.register_plugin(register_atomic_ws)
  first_loaded = dict(compiler._plugins_loaded)
  first_dialect_count = len(compiler.dialects)

  # Second call must be a no-op (not raise PluginConflictError, not
  # double-register the dialect, not append a second lowering, etc.).
  compiler.register_plugin(register_atomic_ws)

  assert compiler._plugins_loaded == first_loaded
  assert len(compiler.dialects) == first_dialect_count
  assert compiler.get_dialect('parallel.atomic_ws') is ATOMIC_WS_DIALECT


def test_register_parallel_atomic_ws_metadata_matches_entry_point() -> None:
  '''The `register` callable's `plugin_name` / `provides` / `requires`
  attributes match the entry-point declaration in `pyproject.toml`.
  These attributes are what F4 topo-sorts on (see
  `core/plugin.py::_topo_sort_plugins`) — getting them wrong silently
  breaks dependency ordering, so we pin them here.
  '''
  assert register_atomic_ws.plugin_name == 'parallel_atomic_ws'  # type: ignore[attr-defined]
  assert register_atomic_ws.provides == ('parallel.atomic_ws',)  # type: ignore[attr-defined]
  assert register_atomic_ws.requires == ()  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# 3. Byte-equivalence — plugin path and legacy path produce same output,
#    with the WorkStealing pragma actually triggering on both sides
# -----------------------------------------------------------------------------


def test_compile_kernel_body_byte_equivalent_across_plugin_paths_with_ws() -> None:
  '''The full `compile_kernel_body(ep)` text is byte-identical whether
  the caller obtains its `ep` via a plugin-discovered Compiler
  (`with_default_plugins`) or via a Compiler that explicitly invokes
  the register callable. This is the load-bearing assertion: it pins
  that wrapping the dialect as a plugin did not perturb the codegen
  output.

  The fixture program `_tc_program_with_ws` tags `TCRec` with
  `Rule.with_pragma(WorkStealing())`, so the dual-write contract
  (`docs/phase_c_pragma_materialization.md` §4.2) fires on each path:

    - `MirPragmaPass` strips the typed pragma instance off the EP
      (per `materialize_work_stealing`'s bool-shadow short-circuit)
      but leaves the `pipeline` shape untouched; the legacy
      `work_stealing=True` PlanEntry bool drives the WS-count emit
      inside `_lower_insert_into`.
    - The plugin-discovered Compiler and the explicit-register
      Compiler both see the same MIR + run the same kernel-pipeline,
      so the rendered CUDA must match byte-for-byte.

  We deliberately check both phases (count + materialize) since the
  kernel-body pipeline branches on `is_counting` and a subtle plugin
  ordering bug could surface in only one phase. The WS-count emit
  in particular only fires when `is_counting=True`, so omitting that
  branch would silently let WS-specific plugin regressions through.
  '''
  # Path A: full plugin discovery via the entry-point group.
  compiler_via_plugin = Compiler.with_default_plugins()
  ep_via_plugin = _build_ws_ep_via(compiler_via_plugin)

  # Path B: explicit plugin registration (skips entry-point walk).
  compiler_explicit = Compiler()
  compiler_explicit.register_plugin(register_atomic_ws)
  ep_explicit = _build_ws_ep_via(compiler_explicit)

  # Sanity: both EPs are the WS-tagged one (the fixture's TCRec).
  assert ep_via_plugin.work_stealing is True
  assert ep_explicit.work_stealing is True

  # Materialize phase.
  text_a = compile_kernel_body(ep_via_plugin, is_counting=False)
  text_b = compile_kernel_body(ep_explicit, is_counting=False)
  assert text_a == text_b, (
    'compile_kernel_body diverged between plugin-discovered and '
    'explicit-register Compilers (materialize phase, WS pragma)'
  )

  # Count phase — mirrors the runner emit's per-kernel knobs. This is
  # the phase where the legacy `if ctx.ws_enabled:` branch in
  # `_lower_insert_into` emits the per-thread `<out>++` shape; if the
  # plugin path failed to register the WS lowering surface, the two
  # outputs would diverge here.
  text_a_ct = compile_kernel_body(ep_via_plugin, is_counting=True, output_var_name='output_ctx')
  text_b_ct = compile_kernel_body(ep_explicit, is_counting=True, output_var_name='output_ctx')
  assert text_a_ct == text_b_ct, (
    'compile_kernel_body diverged between plugin-discovered and '
    'explicit-register Compilers (count phase, WS pragma)'
  )
