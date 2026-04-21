'''HIR pass infrastructure: minimal Python port of
src/srdatalog/hir/pass_infrastructure.nim + pass_manager.nim.

Ports the parts that are actually load-bearing today:
  - PassLevel enum (stage-in-pipeline classification)
  - Dialect enum (IR abstraction level, for metadata)
  - PassInfo dataclass (name/level/order/source-target dialect)
  - RuleRewritePass and HirTransformPass protocols
  - Pipeline class that runs registered passes in order

Deliberately DOES NOT port yet:
  - ConversionTarget / RewritePattern / PatternSet (MIR-level rewrites;
    Nim's verifyLegality is itself a stub)
  - Compile-time macro registry (Python has no macros; runtime registration
    on a Pipeline instance is simpler and achieves the same goal)

When a new pass is ported, give it a PassInfo and register it via
`Pipeline.add_rule_rewrite` or `Pipeline.add_hir_transform`.
'''
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from srdatalog.dsl import Program, Rule
from srdatalog.hir_types import HirProgram, RelationDecl


class PassLevel(Enum):
  '''Stage in the compilation pipeline. Mirrors Nim pass_manager.PassLevel.'''
  RULE_REWRITE = "plRuleRewrite"       # (rules, decls) -> (rules, decls)
  HIR_TRANSFORM = "plHirTransform"     # HirProgram -> HirProgram
  LOWERING = "plLowering"              # HirProgram -> MirNode
  MIR_OPTIMIZE = "plMirOptimize"       # MirNode -> MirNode
  DIALECT_UNIFY = "plDialectUnify"     # MirNode -> MirNode (unified dialect)


class Dialect(Enum):
  '''IR abstraction level. Mirrors Nim pass_infrastructure.Dialect.'''
  HIR = "HIR"
  MIR_WCOJ = "MIR_WCOJ"
  MIR_BINARY = "MIR_Binary"
  MIR_MATERIALIZED = "MIR_Materialized"
  MIR_UNIFIED = "MIR_Unified"


@dataclass(frozen=True)
class PassInfo:
  '''Metadata for a registered pass.'''
  name: str
  level: PassLevel
  order: int                 # Lower runs first within the same level
  source_dialect: Dialect
  target_dialect: Dialect


@runtime_checkable
class RuleRewritePass(Protocol):
  '''Rule-level transform that runs BEFORE stratification.

  Implementations must expose an `info: PassInfo` attribute and a `run`
  method. Use Protocol not ABC so existing bare functions can be adapted
  with a thin wrapper class.
  '''
  info: PassInfo

  def run(
    self, rules: list[Rule], decls: list[RelationDecl]
  ) -> tuple[list[Rule], list[RelationDecl]]: ...


@runtime_checkable
class HirTransformPass(Protocol):
  '''HIR-level transform that runs AFTER stratification. Typically mutates
  the HirProgram in place but must return it for uniformity.
  '''
  info: PassInfo

  def run(self, hir: HirProgram) -> HirProgram: ...


def program_to_decls(program: Program) -> list[RelationDecl]:
  '''Extract RelationDecls from a user-facing Program. Pipeline entry helper.'''
  decls: list[RelationDecl] = []
  for rel in program.relations:
    decls.append(
      RelationDecl(
        rel_name=rel.name,
        types=[t.__name__ for t in rel.column_types],
        semiring="NoProvenance",
      )
    )
  return decls


class Pipeline:
  '''HIR pipeline orchestrator. Mirrors the role of compileToHir in hir.nim.

  Stratification is a fixed entry step (not registered) because it's the
  point where (rules, decls) becomes HirProgram; every HIR pipeline needs
  it exactly once. Rule-rewrite passes run before it; HIR transform passes
  run after. Within each category, passes run in order-ascending order.
  '''
  def __init__(self, verbose: bool = False) -> None:
    self.rule_rewrites: list[RuleRewritePass] = []
    self.hir_transforms: list[HirTransformPass] = []
    self.verbose = verbose

  def add_rule_rewrite(self, p: RuleRewritePass) -> "Pipeline":
    if p.info.level is not PassLevel.RULE_REWRITE:
      raise ValueError(
        f"pass {p.info.name} registered as rule rewrite but level is {p.info.level}"
      )
    self.rule_rewrites.append(p)
    return self

  def add_hir_transform(self, p: HirTransformPass) -> "Pipeline":
    if p.info.level is not PassLevel.HIR_TRANSFORM:
      raise ValueError(
        f"pass {p.info.name} registered as hir transform but level is {p.info.level}"
      )
    self.hir_transforms.append(p)
    return self

  def compile_to_hir(self, program: Program) -> HirProgram:
    '''Run the full HIR pipeline on a Program. Returns the resulting HirProgram.'''
    from srdatalog.hir_stratify import stratify  # deferred to avoid import cycle

    rules = list(program.rules)
    decls = program_to_decls(program)

    for p in sorted(self.rule_rewrites, key=lambda q: q.info.order):
      if self.verbose:
        print(f"[HIR] {p.info.name} (order {p.info.order})")
      rules, decls = p.run(rules, decls)

    hir = stratify(rules, decls)
    if self.verbose:
      print(f"[HIR] Stratification -> {len(hir.strata)} strata")

    for p in sorted(self.hir_transforms, key=lambda q: q.info.order):
      if self.verbose:
        print(f"[HIR] {p.info.name} (order {p.info.order})")
      hir = p.run(hir)

    return hir
