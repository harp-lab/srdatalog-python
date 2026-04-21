'''HIR main orchestrator. Mirrors src/srdatalog/hir/hir.nim.

Entry point: `compile_to_hir(program)` runs the default pipeline
(currently: stratification only; future passes will be appended here as
they are ported from Nim).
'''

from __future__ import annotations

from srdatalog.dsl import Program
from srdatalog.hir.pass_ import Pipeline
from srdatalog.hir.types import HirProgram


def default_pipeline(verbose: bool = False) -> Pipeline:
  '''Build the standard HIR pipeline.

  Nim reference (compileToHir):
    Pass 0   rewriteConstants          [rule-rewrite, ported]
    Pass 1   rewriteHeadConstants      [rule-rewrite, ported]
    Pass 1.5 optimizeSemiJoins         [rule-rewrite, ported]
    Pass 2   stratify                  [HIR entry — fixed, built in]
    Pass 3   generateVariants          [HIR transform, ported]
    Pass 4   planJoins                 [HIR transform, ported]
    Pass 5   selectIndices             [HIR transform, ported]
  '''
  from srdatalog.hir.index import IndexSelectionPass
  from srdatalog.hir.plan import JoinPlannerPass
  from srdatalog.hir.semi_naive import SemiNaiveVariantPass
  from srdatalog.hir.split import TempIndexRegistrationPass, TempRelSynthesisPass
  from srdatalog.rule_rewrite import ConstantRewritePass, HeadConstantRewritePass, SemiJoinPass

  p = Pipeline(verbose=verbose)
  p.add_rule_rewrite(ConstantRewritePass())
  p.add_rule_rewrite(HeadConstantRewritePass())
  p.add_rule_rewrite(SemiJoinPass())
  p.add_hir_transform(SemiNaiveVariantPass())
  p.add_hir_transform(JoinPlannerPass())
  p.add_hir_transform(TempRelSynthesisPass())  # Pass 4.5
  p.add_hir_transform(IndexSelectionPass())
  p.add_hir_transform(TempIndexRegistrationPass())  # Pass 5.5
  return p


def compile_to_hir(program: Program, verbose: bool = False) -> HirProgram:
  '''Run the default HIR pipeline on a Program.'''
  return default_pipeline(verbose=verbose).compile_to_hir(program)


def compile_to_mir(
  program: Program,
  verbose: bool = False,
  apply_mir_passes: bool = True,
):
  '''End-to-end: Program -> HIR -> MIR. Returns a mir_types.Program.

  By default runs the ported MIR optimization passes (pre_reconstruct_rebuild,
  clause_order_reorder, prefix_source_reorder). Pass `apply_mir_passes=False`
  to stop at the raw output of `lower_hir_to_mir_steps`.
  '''
  import srdatalog.mir.types as mir
  from srdatalog.hir.lower import lower_hir_to_mir_steps

  hir = compile_to_hir(program, verbose=verbose)
  steps = lower_hir_to_mir_steps(hir)
  if apply_mir_passes:
    from srdatalog.mir.passes import apply_all_mir_passes

    steps = apply_all_mir_passes(steps)
  return mir.Program(steps=[(node, is_rec) for node, is_rec in steps])
