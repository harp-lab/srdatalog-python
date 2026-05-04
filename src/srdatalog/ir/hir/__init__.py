'''HIR main orchestrator. Mirrors src/srdatalog/hir/hir.nim.

Entry point: `compile_to_hir(program)` runs the default pipeline
(currently: stratification only; future passes will be appended here as
they are ported from Nim).

A `DIALECT` catalog is also exposed (see `srdatalog.ir.core.dialect`)
so HIR participates in the framework registry alongside `iir.cf`,
`relation.sorted_array`, etc. The HIR types here are not yet
`Op`-subclassed or frozen — that's the next standardization step.
The catalog is purely metadata today.
'''

from __future__ import annotations

from srdatalog.dsl import Program
from srdatalog.ir.core import Dialect
from srdatalog.ir.hir.pass_ import Pipeline
from srdatalog.ir.hir.types import (
  AccessPattern,
  HirProgram,
  HirRuleVariant,
  HirStratum,
  RelationDecl,
  Version,
)

DIALECT = Dialect(
  name='hir',
  types=[
    AccessPattern,
    HirProgram,
    HirRuleVariant,
    HirStratum,
    RelationDecl,
    Version,
  ],
)


# Verifier scaffolding — HIR invariants (stratum dependency acyclicity,
# every relation decl appears in at most one stratum, etc.) land
# incrementally as we encode them.
def _register_passes() -> None:
  from srdatalog.ir.core.passes import verifier

  @verifier(DIALECT)
  def _verify(_prog):
    return []


_register_passes()


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
  from srdatalog.ir.hir.index import IndexSelectionPass
  from srdatalog.ir.hir.plan import JoinPlannerPass
  from srdatalog.ir.hir.rule_rewrite import (
    ConstantRewritePass,
    HeadConstantRewritePass,
    SemiJoinPass,
    WildcardRewritePass,
  )
  from srdatalog.ir.hir.semi_naive import SemiNaiveVariantPass
  from srdatalog.ir.hir.split import TempIndexRegistrationPass, TempRelSynthesisPass

  p = Pipeline(verbose=verbose)
  p.add_rule_rewrite(WildcardRewritePass())
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
  *,
  hir: HirProgram | None = None,
  verbose: bool = False,
  apply_mir_passes: bool = True,
):
  '''End-to-end: Program -> HIR -> MIR. Returns a mir_types.Program.

  By default runs the ported MIR optimization passes (pre_reconstruct_rebuild,
  clause_order_reorder, prefix_source_reorder). Pass `apply_mir_passes=False`
  to stop at the raw output of `lower_hir_to_mir_steps`.

  Pass `hir=...` to reuse an already-computed HirProgram. The default
  (`hir=None`) runs `compile_to_hir(program, verbose=verbose)` internally.
  Callers that already have HIR (e.g. `ir.pipeline.compile_program`)
  should pass it to avoid the redundant pass.
  '''
  import srdatalog.ir.mir.types as mir
  from srdatalog.ir.hir.lower import lower_hir_to_mir_steps

  if hir is None:
    hir = compile_to_hir(program, verbose=verbose)
  steps = lower_hir_to_mir_steps(hir)
  if apply_mir_passes:
    from srdatalog.ir.mir.passes import apply_all_mir_passes

    steps = apply_all_mir_passes(steps)
  return mir.Program(steps=[(node, is_rec) for node, is_rec in steps])
