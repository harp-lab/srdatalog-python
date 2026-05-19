'''Declarative pipeline shims (F5.1).

Spec: docs/phase_f5_declarative_pipeline.md.

Two pipelines, each a `list[Pass]` of thin `ProgramPass` shims that
wrap the existing imperative entry points without changing behavior:

  DEFAULT_PROGRAM_PIPELINE — Program -> mir.Program
    HirPlanningShim, HirToMirShim, MirOptShim, MirProgramAssembly

  DEFAULT_KERNEL_PIPELINE  — KernelCtx(ep, ...) -> body string
    AssignHandlesShim, CollectViewSpecsShim, EmitViewDeclsShim,
    LowerKernelBodyShim, RenderShim

Each shim is a `ProgramPass` subclass mirroring one block of either
`compile_to_mir` (src/srdatalog/ir/hir/__init__.py:96-124) or
`compile_kernel_body` (src/srdatalog/ir/codegen/cuda/api.py:98-185).

The through-state types `InitialProg` and `KernelCtx` are
frozen+slots dataclasses; each shim returns a new instance via
`dataclasses.replace`. F5.1 is pure addition — F5.2 + F5.3 cut the
existing imperative entry points to call `Compiler.run` over these
pipelines.
'''

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from srdatalog.ir.core import Pass, ProgramPass

if TYPE_CHECKING:
  from srdatalog.dsl import Program
  from srdatalog.ir.codegen.cuda.envelope import ViewSpec
  from srdatalog.ir.hir.types import HirProgram
  from srdatalog.ir.mir import types as mir
  from srdatalog.ir.mir.types import ExecutePipeline


# -----------------------------------------------------------------------------
# Through-state dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InitialProg:
  '''Program-pipeline through-state. Threaded by the four
  `DEFAULT_PROGRAM_PIPELINE` shims; each fills one field.

  `verbose` mirrors the legacy `compile_to_mir(verbose=...)` kwarg.
  Read by `HirPlanningShim` when computing HIR from scratch; ignored
  if `hir` is pre-populated by the caller. Added in F5.2 so the
  declarative pipeline can preserve the imperative entry point's
  verbosity knob byte-equivalently.

  `target` carries the render target chosen by the caller. PR-1c
  (Phase B2-1, foundation) adds the field so downstream pipeline
  stages can read it; default ``'cuda'`` preserves all existing call
  sites byte-equivalently. Actual per-target dispatch lands in Phase
  B2-2. See docs/phase_decomposition_redesign.md §3.3.1.'''

  program: Program
  hir: HirProgram | None = None
  steps: list[tuple[mir.MirNode, bool]] | None = None
  mir_program: mir.Program | None = None
  verbose: bool = False
  target: str = 'cuda'


@dataclass(frozen=True, slots=True)
class KernelCtx:
  '''Kernel-pipeline through-state. Carries the original
  `compile_kernel_body` kwargs plus the progressively populated
  derived state (handle-assigned ep, view_specs, view_decls + var
  maps, iir, body_text).

  `target` carries the render target chosen by the caller. PR-1c
  (Phase B2-1, foundation) adds the field so kernel-pipeline shims
  (notably `VerifyRenderabilityShim`) can read it; default ``'cuda'``
  preserves all existing call sites byte-equivalently. Actual
  per-target render dispatch lands in Phase B2-2. See
  docs/phase_decomposition_redesign.md §3.3.1.'''

  ep: ExecutePipeline
  is_counting: bool
  output_var_name: str = 'output'
  output_vars: dict[str, str] | None = None
  slot_mode: str = 'positional'
  rel_index_types: dict[str, str] | None = None
  tiled_cartesian: bool = False
  bg_enabled: bool = False
  target: str = 'cuda'
  # populated by shims:
  view_specs: tuple[ViewSpec, ...] | None = None
  view_decls: str | None = None
  name_map: dict[str, str] | None = None
  base_map: dict[str, int] | None = None
  iir: Any = None
  body_text: str | None = None


# -----------------------------------------------------------------------------
# Program-pipeline shims (mirror src/srdatalog/ir/hir/__init__.py:96-124)
# -----------------------------------------------------------------------------


class HirPlanningShim(ProgramPass):
  '''Wraps `srdatalog.ir.hir.compile_to_hir`. Adds `hir` to state.'''

  def __init__(self) -> None:
    super().__init__(
      name='hir_planning',
      consumes=(),
      produces=('hir',),
      fn=self._fn,
    )

  @staticmethod
  def _fn(state: InitialProg, _compiler: Any) -> InitialProg:
    if state.hir is not None:
      return state
    from srdatalog.ir.hir import compile_to_hir

    return dataclasses.replace(state, hir=compile_to_hir(state.program, verbose=state.verbose))


class HirToMirShim(ProgramPass):
  '''Wraps `srdatalog.ir.hir.lower.lower_hir_to_mir_steps`. Adds
  `steps` to state.'''

  def __init__(self) -> None:
    super().__init__(
      name='hir_to_mir',
      consumes=('hir',),
      produces=('mir_steps',),
      fn=self._fn,
    )

  @staticmethod
  def _fn(state: InitialProg, _compiler: Any) -> InitialProg:
    from srdatalog.ir.hir.lower import lower_hir_to_mir_steps

    assert state.hir is not None, 'HirToMirShim: hir not set'
    return dataclasses.replace(state, steps=lower_hir_to_mir_steps(state.hir))


class MirOptShim(ProgramPass):
  '''Wraps `srdatalog.ir.mir.passes.apply_all_mir_passes`. Updates
  `steps` in place.'''

  def __init__(self) -> None:
    super().__init__(
      name='mir_opt',
      consumes=('mir_steps',),
      produces=('mir_steps',),
      fn=self._fn,
    )

  @staticmethod
  def _fn(state: InitialProg, _compiler: Any) -> InitialProg:
    from srdatalog.ir.mir.passes import apply_all_mir_passes

    assert state.steps is not None, 'MirOptShim: steps not set'
    return dataclasses.replace(state, steps=apply_all_mir_passes(state.steps))


class MirProgramAssembly(ProgramPass):
  '''Wraps the final `mir.Program(steps=[...])` assembly. Adds
  `mir_program` to state.'''

  def __init__(self) -> None:
    super().__init__(
      name='mir_program_assembly',
      consumes=('mir_steps',),
      produces=('mir_program',),
      fn=self._fn,
    )

  @staticmethod
  def _fn(state: InitialProg, _compiler: Any) -> InitialProg:
    from srdatalog.ir.mir import types as mir_types

    assert state.steps is not None, 'MirProgramAssembly: steps not set'
    prog = mir_types.Program(steps=[(node, is_rec) for node, is_rec in state.steps])
    return dataclasses.replace(state, mir_program=prog)


# -----------------------------------------------------------------------------
# Kernel-pipeline shims (mirror src/srdatalog/ir/codegen/cuda/api.py:98-185)
# -----------------------------------------------------------------------------


class AssignHandlesShim(ProgramPass):
  '''Wraps `srdatalog.ir.codegen.cuda.envelope.assign_handle_positions`.
  Rebuilds `state.ep` with handle_starts assigned across the pipeline
  body.'''

  def __init__(self) -> None:
    super().__init__(
      name='assign_handles',
      consumes=(),
      produces=('ep_with_handles',),
      fn=self._fn,
    )

  @staticmethod
  def _fn(state: KernelCtx, _compiler: Any) -> KernelCtx:
    from srdatalog.ir.codegen.cuda.envelope import assign_handle_positions

    new_pipeline = assign_handle_positions(list(state.ep.pipeline))
    new_ep = dataclasses.replace(state.ep, pipeline=new_pipeline)
    return dataclasses.replace(state, ep=new_ep)


class CollectViewSpecsShim(ProgramPass):
  '''Wraps `collect_unique_view_specs` (envelope) — also exists to
  realize that the per-spec view_count computation is consumed
  exclusively by the next shim, so it lives there rather than on
  the through-state.'''

  def __init__(self) -> None:
    super().__init__(
      name='collect_view_specs',
      consumes=('ep_with_handles',),
      produces=('view_specs',),
      fn=self._fn,
    )

  @staticmethod
  def _fn(state: KernelCtx, _compiler: Any) -> KernelCtx:
    from srdatalog.ir.codegen.cuda.envelope import collect_unique_view_specs

    specs = collect_unique_view_specs(list(state.ep.pipeline))
    return dataclasses.replace(state, view_specs=tuple(specs))


class EmitViewDeclsShim(ProgramPass):
  '''Wraps `emit_view_declarations` (envelope) plus `view_counts_for_specs`
  (relation.d2l). Splits the combined `view_vars` dict into a name-only
  map (handle_idx string -> view var) and a base-slot map (handle_idx
  string -> int) per the `__base__` sentinel convention.'''

  def __init__(self) -> None:
    super().__init__(
      name='emit_view_decls',
      consumes=('view_specs',),
      produces=('view_decls',),
      fn=self._fn,
    )

  @staticmethod
  def _fn(state: KernelCtx, _compiler: Any) -> KernelCtx:
    from srdatalog.ir.codegen.cuda.envelope import emit_view_declarations
    from srdatalog.ir.dialects.relation.d2l import view_counts_for_specs

    assert state.view_specs is not None, 'EmitViewDeclsShim: view_specs not set'
    specs = list(state.view_specs)
    view_counts = view_counts_for_specs(specs, state.rel_index_types or {})
    view_decls, view_vars = emit_view_declarations(
      specs,
      list(state.ep.pipeline),
      slot_mode=state.slot_mode,
      view_counts=view_counts,
    )
    name_map = {k: v for k, v in view_vars.items() if k.isdigit()}
    base_map = {
      k.removeprefix('__base__'): int(v) for k, v in view_vars.items() if k.startswith('__base__')
    }
    return dataclasses.replace(
      state,
      view_decls=view_decls,
      name_map=name_map,
      base_map=base_map,
    )


class LowerKernelBodyShim(ProgramPass):
  '''Wraps `lower_scan_pipeline` (sorted_array dialect). Builds a
  `LoweringCtx` from the populated state and runs the MIR -> IIR
  walk. Adds `iir` to state.'''

  def __init__(self) -> None:
    super().__init__(
      name='lower_kernel_body',
      consumes=('view_decls',),
      produces=('iir',),
      fn=self._fn,
    )

  @staticmethod
  def _fn(state: KernelCtx, _compiler: Any) -> KernelCtx:
    from srdatalog.ir.dialects.relation.sorted_array.lowerings import (
      LoweringCtx,
      lower_scan_pipeline,
    )

    assert state.name_map is not None, 'LowerKernelBodyShim: name_map not set'
    assert state.base_map is not None, 'LowerKernelBodyShim: base_map not set'
    lower_ctx = LoweringCtx(
      view_var_names=state.name_map,
      is_counting=state.is_counting,
      output_var=state.output_var_name,
      output_var_overrides=dict(state.output_vars) if state.output_vars else {},
      rel_index_types=dict(state.rel_index_types) if state.rel_index_types else {},
      view_slot_bases=state.base_map,
      dedup_hash=state.ep.dedup_hash,
      tiled_cartesian=state.tiled_cartesian,
      bg_enabled=state.bg_enabled,
    )
    iir = lower_scan_pipeline(list(state.ep.pipeline), lower_ctx)
    return dataclasses.replace(state, iir=iir)


class VerifyRenderabilityShim(ProgramPass):
  '''R3 closure check (topology check #4 per
  docs/concept_glossary.md section 13.3).

  Runs `verify_renderability` on the post-fixpoint IIR tree. For every
  op type reachable in the tree, asserts either a registered
  `@register_render(op_type, target=state.target)` OR a `@rewrite` is
  present. On failure, raises `UnrenderableOpError` naming the op
  type + target — long before the renderer's own `KeyError` would
  surface, and with a precise message pointing at the missing plugin
  contribution.

  Today's IIR rewrite step is conceptual — `LowerKernelBodyShim`
  produces the final IIR directly. So this shim runs AFTER
  `LowerKernelBodyShim` (which produces the IIR) and BEFORE
  `RenderShim` (which consumes it). The shim is structurally a
  pass-through on `KernelCtx` — it neither inspects nor mutates the
  state beyond the closure check.

  `consumes=('iir',)` + `produces=()` because the check doesn't
  introduce a new pseudo-dialect; it's a gate, not a transformation.
  Per the spec note in `code_discipline.md` R3, this is exactly the
  "pipeline-stage closure verification" that prevents silent
  fall-through in `RenderShim`.

  PR-1c (Phase B2-1, foundation) replaced the hardcoded
  ``target='cuda'`` with ``state.target`` so the check follows the
  caller-chosen render target. See
  docs/phase_decomposition_redesign.md §2.3 and §3.3.1.'''

  def __init__(self) -> None:
    super().__init__(
      name='verify_renderability',
      consumes=('iir',),
      produces=(),
      fn=self._fn,
    )

  @staticmethod
  def _fn(state: KernelCtx, compiler: Any) -> KernelCtx:
    from srdatalog.ir.core.verifier import verify_renderability

    assert state.iir is not None, 'VerifyRenderabilityShim: iir not set'
    verify_renderability(state.iir, target=state.target, compiler=compiler)
    return state


class RenderShim(ProgramPass):
  '''Wraps `srdatalog.ir.codegen.cuda.emit.emit` with an
  `EmitCtx(indent_level=4)`. Concatenates `view_decls` + emitted IIR
  into the final `body_text`.'''

  def __init__(self) -> None:
    super().__init__(
      name='render',
      consumes=('iir',),
      produces=('body_text',),
      fn=self._fn,
    )

  @staticmethod
  def _fn(state: KernelCtx, _compiler: Any) -> KernelCtx:
    from srdatalog.ir.codegen.cuda.emit import EmitCtx, emit

    assert state.iir is not None, 'RenderShim: iir not set'
    assert state.view_decls is not None, 'RenderShim: view_decls not set'
    emit_ctx = EmitCtx(indent_level=4)
    body_text = state.view_decls + emit(state.iir, emit_ctx)
    return dataclasses.replace(state, body_text=body_text)


# -----------------------------------------------------------------------------
# Pipeline singletons
# -----------------------------------------------------------------------------


DEFAULT_PROGRAM_PIPELINE: list[Pass] = [
  HirPlanningShim(),
  HirToMirShim(),
  MirOptShim(),
  MirProgramAssembly(),
]


DEFAULT_KERNEL_PIPELINE: list[Pass] = [
  AssignHandlesShim(),
  CollectViewSpecsShim(),
  EmitViewDeclsShim(),
  LowerKernelBodyShim(),
  VerifyRenderabilityShim(),
  RenderShim(),
]


__all__ = [
  'DEFAULT_KERNEL_PIPELINE',
  'DEFAULT_PROGRAM_PIPELINE',
  'AssignHandlesShim',
  'CollectViewSpecsShim',
  'EmitViewDeclsShim',
  'HirPlanningShim',
  'HirToMirShim',
  'InitialProg',
  'KernelCtx',
  'LowerKernelBodyShim',
  'MirOptShim',
  'MirProgramAssembly',
  'RenderShim',
  'VerifyRenderabilityShim',
]
