'''External-plugin demo — end-to-end test.

Spec: `docs/phase_e_plugin_extensibility.md` §4 (worked example).

This test is the load-bearing assertion that the plugin contract is
honoured: a SEPARATE package (this directory's `srdatalog_jaccard`),
installed via `pip install -e .`, can extend the compiler with a
new dialect + typed Pragma + lowering rule WITHOUT touching any
file in `src/srdatalog/`.

Sections:

  1. Plugin discovery — `Compiler.with_default_plugins()` finds the
     `jaccard` entry point shipped by this package's `pyproject.toml`,
     calls its `register(compiler)`, and registers the
     `relation.jaccard` dialect.

  2. Pragma registration — `@pragma_handler(Jaccard, on=
     ExecutePipeline)` ran as an import-time side effect, so the
     DSL's `Rule.with_pragma(Jaccard())` validator accepts the typed
     pragma. DSL-time config validation
     (`Jaccard.__post_init__` raising `PragmaConfigError`) fires on
     out-of-range thresholds.

  3. End-to-end compile_to_mir — `MirPragmaPass` consumes the
     `Jaccard` instance and inserts the `JaccardIndex` wrap op into
     the resulting MIR program. The pragma is stripped from
     `ep.pragmas` post-pass (per the `MirPragmaPass` post-flight
     invariant).

  4. Lowering rule — the registered
     `@lowering(target=DIALECT, source=JaccardIndex)` rule fires and
     produces well-formed IIR; the rendered C++ output matches a
     stable golden snapshot.

Tests skip cleanly if the demo package is not installed — the
`srdatalog_jaccard` import sits inside a `pytest.importorskip` at
top of file so users running the full test suite without first
`pip install -e examples/srdatalog_jaccard_demo` see one
SKIPPED line instead of an ImportError crash.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import pytest

# Skip cleanly if the demo package is not installed. The most common
# path here is a fresh checkout where the user has not yet run
# `pip install -e examples/srdatalog_jaccard_demo --no-deps`. Per the
# README, the install step is a one-liner; the importorskip turns a
# hard ImportError into a single SKIPPED row in the pytest summary.
srdatalog_jaccard = pytest.importorskip('srdatalog_jaccard')

from srdatalog_jaccard import DIALECT as JACCARD_DIALECT
from srdatalog_jaccard import register as register_jaccard
from srdatalog_jaccard.dialect import JaccardIndex
from srdatalog_jaccard.pragmas.jaccard import Jaccard

import srdatalog.ir.mir.types as m
from srdatalog.dsl import Program, Relation, Var
from srdatalog.ir.core import Compiler, Pragma
from srdatalog.ir.core.plugin import ENTRY_POINT_GROUP
from srdatalog.ir.core.pragma import PragmaConfigError
from srdatalog.ir.hir import compile_to_mir

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _similar_program(threshold: float = 0.7) -> Program:
  '''Build a one-rule program: `Similar(x, y) :- Arc(x, y)` with a
  `Jaccard(threshold=...)` pragma attached.

  This is the minimal shape that exercises the full pipeline: HIR
  planning, MIR lowering, `MirPragmaPass`, and (in section 4) the
  wrap-op lowering. The rule is intentionally non-recursive so the
  resulting MIR contains a single `ExecutePipeline` — easier to
  introspect than a fixpoint.
  '''
  x, y = Var('x'), Var('y')
  arc = Relation('Arc', 2)
  similar = Relation('Similar', 2)
  return Program(
    rules=[
      (similar(x, y) <= arc(x, y)).with_pragma(Jaccard(threshold=threshold)),
    ],
  )


def _find_execute_pipelines(node: object) -> list[m.ExecutePipeline]:
  '''Walk a MIR step tree, collecting every `ExecutePipeline`. The
  MIR plan can nest EPs inside `FixpointPlan.instructions` /
  `ParallelGroup.ops`; this helper flattens them out.
  '''
  out: list[m.ExecutePipeline] = []
  if isinstance(node, m.ExecutePipeline):
    out.append(node)
  elif isinstance(node, m.FixpointPlan):
    for inst in node.instructions:
      out.extend(_find_execute_pipelines(inst))
  elif isinstance(node, m.ParallelGroup):
    for op in node.ops:
      out.extend(_find_execute_pipelines(op))
  return out


def _first_ep(mir_prog: m.Program) -> m.ExecutePipeline:
  '''Return the first `ExecutePipeline` in a MIR program (linear walk
  over `steps`). Asserts at least one EP exists — every test fixture
  in this file produces one.
  '''
  for step, _is_rec in mir_prog.steps:
    eps = _find_execute_pipelines(step)
    if eps:
      return eps[0]
  raise AssertionError('no ExecutePipeline in mir_prog')


# -----------------------------------------------------------------------------
# 1. Plugin discovery — `with_default_plugins()` picks up jaccard
# -----------------------------------------------------------------------------


def test_jaccard_is_discovered_by_with_default_plugins() -> None:
  '''`Compiler.with_default_plugins()` walks the
  `srdatalog.plugins` entry-point group; the `jaccard` entry point
  shipped by this package's `pyproject.toml` is loaded and registers
  the `relation.jaccard` dialect on the resulting Compiler.

  If this fails with "no plugins loaded named jaccard", check the
  install: `pip install -e examples/srdatalog_jaccard_demo --no-deps`.
  '''
  compiler = Compiler.with_default_plugins()

  assert 'jaccard' in compiler._plugins_loaded, (
    f'expected jaccard in loaded plugins; got '
    f'{sorted(compiler._plugins_loaded)!r}. If missing, the editable '
    f'install of examples/srdatalog_jaccard_demo is stale.'
  )

  dialect = compiler.get_dialect('relation.jaccard')
  assert dialect is JACCARD_DIALECT, (
    'with_default_plugins registered a different Dialect instance '
    'than the module-level singleton — the plugin path is not '
    'reusing DIALECT'
  )

  # The dialect's plugin attribution is the entry-point name.
  assert compiler._dialects_by_plugin['relation.jaccard'] == 'jaccard'


def test_jaccard_entry_point_shape() -> None:
  '''The entry-point declaration in `pyproject.toml` exposes the
  expected (name, target) tuple. Catches typos in the
  `[project.entry-points."srdatalog.plugins"]` block before a downstream
  load-order test reports a more confusing failure.
  '''
  import importlib.metadata

  eps = list(importlib.metadata.entry_points(group=ENTRY_POINT_GROUP))
  jaccard_eps = [ep for ep in eps if ep.name == 'jaccard']
  assert len(jaccard_eps) == 1, (
    f'expected exactly one jaccard entry point; got {jaccard_eps!r}. '
    f'Multiple entries indicates a stale install left in place; '
    f'rerun `pip install -e examples/srdatalog_jaccard_demo --no-deps '
    f'--force-reinstall`.'
  )
  ep = jaccard_eps[0]
  assert ep.value == 'srdatalog_jaccard:register'


def test_jaccard_register_metadata_matches_entry_point() -> None:
  '''The `register` callable's `plugin_name` / `provides` / `requires`
  attributes match the entry-point declaration and the F4 topo-sort
  semantics. Pin them here so a future contributor renaming the
  entry point doesn't silently break the topo-sort.
  '''
  assert register_jaccard.plugin_name == 'jaccard'  # type: ignore[attr-defined]
  assert register_jaccard.provides == ('relation.jaccard',)  # type: ignore[attr-defined]
  assert register_jaccard.requires == ('relation.sorted_array',)  # type: ignore[attr-defined]


def test_register_jaccard_plugin_idempotent() -> None:
  '''Calling `register_plugin(register_jaccard)` twice on the same
  Compiler is a no-op the second time — F4's `register_plugin`
  short-circuits on the plugin name (`register.plugin_name = "jaccard"`).
  Mirrors `tests/test_sorted_array_as_plugin.py::
  test_register_sorted_array_plugin_idempotent` for the external case.
  '''
  compiler = Compiler()
  # Register the dependency first (required by the topo-sort declaration).
  from srdatalog.ir.dialects.relation.sorted_array import register as register_sa

  compiler.register_plugin(register_sa)
  compiler.register_plugin(register_jaccard)
  first_loaded = dict(compiler._plugins_loaded)
  first_dialect_count = len(compiler.dialects)

  compiler.register_plugin(register_jaccard)
  assert compiler._plugins_loaded == first_loaded
  assert len(compiler.dialects) == first_dialect_count
  assert compiler.get_dialect('relation.jaccard') is JACCARD_DIALECT


# -----------------------------------------------------------------------------
# 2. Pragma registration — DSL accepts Jaccard, validates threshold
# -----------------------------------------------------------------------------


def test_dsl_accepts_jaccard_pragma() -> None:
  '''`Rule(...).with_pragma(Jaccard())` does not raise — proves the
  `@pragma_handler(Jaccard, on=ExecutePipeline)` decoration registered
  on import (via `srdatalog_jaccard.__init__`'s side-effect chain).
  '''
  prog = _similar_program(threshold=0.5)
  assert len(prog.rules) == 1
  rule = prog.rules[0]
  # The DSL `with_pragma` appends to the rule's plans tuple; for
  # a rule with no prior plans, exactly one plan carries the pragma.
  assert len(rule.plans) == 1
  attached = rule.plans[0].pragmas
  assert len(attached) == 1
  assert isinstance(attached[0], Jaccard)
  assert attached[0].threshold == 0.5


def test_jaccard_rejects_out_of_range_threshold() -> None:
  '''`Jaccard(threshold=...)` validates at construction; out-of-range
  values raise `PragmaConfigError` (subclass of `ValueError`) so the
  user sees the error at their `.with_pragma(...)` keystroke rather
  than deep in `MirPragmaPass`.
  '''
  with pytest.raises(PragmaConfigError, match=r'threshold'):
    Jaccard(threshold=0.0)
  with pytest.raises(PragmaConfigError, match=r'threshold'):
    Jaccard(threshold=1.5)
  with pytest.raises(PragmaConfigError, match=r'threshold'):
    Jaccard(threshold=-0.1)


def test_with_pragma_rejects_non_jaccard_unregistered_pragma() -> None:
  '''A typed `Pragma` subclass with no `@pragma_handler` registration
  is rejected at DSL time. This is the parallel discipline test to
  `tests/test_pragma_dedup_hash_end_to_end.py::
  test_with_pragma_rejects_unregistered_pragma` — proves the
  did-you-mean validator covers the external case too.
  '''
  from srdatalog.ir.core import UnregisteredPragmaError

  @final
  @dataclass(frozen=True, slots=True)
  class _GhostJaccard(Pragma):
    pass

  prog = _similar_program()
  rule = prog.rules[0]
  with pytest.raises(UnregisteredPragmaError, match=r'_GhostJaccard'):
    rule.with_pragma(_GhostJaccard())


# -----------------------------------------------------------------------------
# 3. End-to-end — compile_to_mir materializes the wrap op
# -----------------------------------------------------------------------------


def test_compile_to_mir_inserts_jaccard_index_wrap_op() -> None:
  '''`compile_to_mir(program)` runs the full HIR -> MIR pipeline
  INCLUDING `MirPragmaPass`. After the pass, the EP carrying a
  `Jaccard()` pragma:

    - Has `pragmas == ()` (pragma consumed per the post-flight
      invariant).
    - Has every trailing `InsertInto` in its `pipeline` replaced by
      `JaccardIndex(inner=that_insert, threshold=...)`.

  This is the load-bearing assertion that the plugin's
  materialization handler ran inside the production
  `compile_to_mir` entry point — not via a test-only shim.
  '''
  prog = _similar_program(threshold=0.7)
  mir_prog = compile_to_mir(prog)

  ep = _first_ep(mir_prog)
  assert ep.pragmas == (), f'expected empty pragmas after MirPragmaPass; got {ep.pragmas!r}'

  # Find the wrap op.
  wrap_ops = [op for op in ep.pipeline if isinstance(op, JaccardIndex)]
  assert len(wrap_ops) == 1, (
    f'expected exactly one JaccardIndex in pipeline; got '
    f'{[type(o).__name__ for o in ep.pipeline]!r}'
  )
  gate = wrap_ops[0]
  assert isinstance(gate.inner, m.InsertInto)
  assert gate.threshold == 0.7

  # The original InsertInto must NOT also be in the pipeline (it
  # was REPLACED by the wrap op, not appended).
  bare_inserts = [op for op in ep.pipeline if isinstance(op, m.InsertInto)]
  assert bare_inserts == [], (
    f'expected zero bare InsertIntos in pipeline (all wrapped); got {bare_inserts!r}'
  )


def test_compile_to_mir_no_pragma_is_unchanged() -> None:
  '''Sanity: a program WITHOUT the Jaccard pragma compiles to MIR
  with no `JaccardIndex` ops. Anchors the assertion above —
  catches the case where the materialization handler runs on every
  EP regardless of pragma presence (a registration-side bug).
  '''
  x, y = Var('x'), Var('y')
  arc = Relation('Arc', 2)
  similar = Relation('Similar', 2)
  prog = Program(rules=[similar(x, y) <= arc(x, y)])

  mir_prog = compile_to_mir(prog)
  ep = _first_ep(mir_prog)
  assert all(not isinstance(op, JaccardIndex) for op in ep.pipeline)


# -----------------------------------------------------------------------------
# 4. Lowering rule fires + golden snapshot
# -----------------------------------------------------------------------------


# Stable golden snapshot for the lowered IIR rendered to C++. This is
# a NEW path (no Nim reference exists for Jaccard), so we anchor the
# output here. If the snapshot drifts, the test fails loudly — at
# which point the new output should be inspected by hand before
# updating this constant.
#
# The exact text is derived from running the lowering against the
# `_similar_program(threshold=0.7)` fixture and rendering via
# `EmitCtx(indent_level=4)`. The leading marker comment is the
# discriminator for this plugin's output; the rest is the byte-
# equivalent dedup-hash gate the lowering delegates to.
_EXPECTED_RENDER = (
  '        // jaccard threshold=0.7\n'
  '        // Emit: Similar(x, y)\n'
  '        { bool _p = dedup_table.try_insert(thread_id, x, y);\n'
  '          if (_p) {\n'
  '        if (tile.thread_rank() == 0) {\n'
  '          uint32_t pos = atomicAdd(atomic_write_pos, 1u);\n'
  '          out_data_0[(pos + out_base_0) + 0 * out_stride_0] = x;\n'
  '          out_data_0[(pos + out_base_0) + 1 * out_stride_0] = y;\n'
  '        }\n'
  '        } }\n'
)


def test_jaccard_lowering_fires_and_emits_marker() -> None:
  '''The registered `@lowering(target=DIALECT, source=JaccardIndex)`
  rule is looked up through the compiler's dialect registry (the
  same mechanism the production dispatcher would use) and applied to
  a real `JaccardIndex` instance materialized by `compile_to_mir`.

  Three assertions:

    - The rule object reachable via `dialect.lowerings` claims our
      `JaccardIndex` source type.
    - Calling its `.apply(gate, ctx)` returns a `Block` IIR op (i.e.,
      the lowering produced well-formed IIR, not raised an exception
      or returned `None`).
    - The rendered C++ matches the byte-stable snapshot above
      (anchors the threshold marker + the inner dedup gate).
  '''
  from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit
  from srdatalog.ir.dialects.iir.cf import Block
  from srdatalog.ir.dialects.relation.sorted_array.lowerings import LoweringCtx

  compiler = Compiler.with_default_plugins()
  dialect = compiler.get_dialect('relation.jaccard')

  jaccard_lowerings = [lo for lo in dialect.lowerings if lo.matches is JaccardIndex]
  assert len(jaccard_lowerings) == 1, (
    f'expected exactly one JaccardIndex lowering registered; got {len(jaccard_lowerings)}'
  )
  rule = jaccard_lowerings[0]
  assert rule.produces == ('iir.cf', 'relation.sorted_array')

  # Materialize the wrap op via the production compile_to_mir path.
  prog = _similar_program(threshold=0.7)
  mir_prog = compile_to_mir(prog)
  ep = _first_ep(mir_prog)
  gate = next(op for op in ep.pipeline if isinstance(op, JaccardIndex))

  # Build a minimally-populated LoweringCtx matching the view-var
  # shape lower_scan_pipeline would set up. (We bypass
  # `compile_pipeline` because the sorted_array chain dispatcher
  # does not know how to skip past our external wrap op; this
  # discovery is documented in the test-run-report — addressing it
  # would require a contract addition in the main package, out of
  # scope for the demo.)
  ctx = LoweringCtx(
    view_var_names={'0': 'view_Arc_0_FULL'},
    is_counting=False,
    output_var='output',
    dedup_hash=False,
  )
  iir = rule.apply(gate, ctx)
  assert isinstance(iir, Block)

  # Threshold marker is the discriminator for this plugin's output.
  marker_stmts = [s for s in iir.stmts if type(s).__name__ == 'Comment']
  assert any('jaccard threshold=0.7' in s.text for s in marker_stmts), (
    f'expected jaccard threshold marker in lowered IIR; got comments '
    f'{[s.text for s in marker_stmts]!r}'
  )

  emit_ctx = EmitCtx(indent_level=4)
  rendered = emit(iir, emit_ctx)
  assert rendered == _EXPECTED_RENDER, (
    f'rendered output diverged from golden snapshot.\n'
    f'--- expected ---\n{_EXPECTED_RENDER}'
    f'--- got ---\n{rendered}'
  )
