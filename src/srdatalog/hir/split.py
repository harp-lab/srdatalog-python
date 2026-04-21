'''Split-rule HIR transforms: Pass 4.5 (temp relation decl synthesis) and
Pass 5.5 (temp index registration). Mirror the inline blocks in
src/srdatalog/hir/hir.nim's compileToHir.

Both passes run as regular HIR transforms, positioned around
IndexSelectionPass:
  JoinPlannerPass        (order 200)
    TempRelSynthesisPass    (order 250)  <-- this file
  IndexSelectionPass     (order 300)
    TempIndexRegistrationPass (order 350)  <-- this file

Split metadata (split_at, temp_vars, temp_rel_name) is populated by
JoinPlannerPass from hir_plan._plan_variant.
'''
from __future__ import annotations

from srdatalog.dsl import Atom, ArgKind
from srdatalog.hir.types import HirProgram, RelationDecl
from srdatalog.hir.pass_ import PassInfo, PassLevel, Dialect


def _infer_temp_types(variant, decls: list[RelationDecl]) -> list[str]:
  '''Pick each temp var's type from the first above-split positive clause
  that contains it (source decl lookup). Falls back to "int". Mirrors
  the inference loop in compileToHir's Pass 4.5.
  '''
  rule = variant.original_rule
  decls_map = {d.rel_name: d for d in decls}
  types: list[str] = []
  for tv in variant.temp_vars:
    found = False
    for clause_idx in range(variant.split_at):
      clause = rule.body[clause_idx]
      if not isinstance(clause, Atom):
        continue
      for arg_idx, arg in enumerate(clause.args):
        if arg.kind is ArgKind.LVAR and arg.var_name == tv:
          if clause.rel in decls_map:
            orig = decls_map[clause.rel]
            if arg_idx < len(orig.types):
              types.append(orig.types[arg_idx])
              found = True
          break
      if found:
        break
    if not found:
      types.append("int")
  return types


class TempRelSynthesisPass:
  '''Pass 4.5: for each split variant, add a `_temp_<RuleName>` RelationDecl
  to the HirProgram (if not already declared). Temp is marked
  `is_generated=True, is_temp=True`. Column types inferred from above-split
  source relations.
  '''
  info = PassInfo(
    name="TempRelSynthesis",
    level=PassLevel.HIR_TRANSFORM,
    order=250,
    source_dialect=Dialect.HIR,
    target_dialect=Dialect.HIR,
  )

  def run(self, hir: HirProgram) -> HirProgram:
    existing = {d.rel_name for d in hir.relation_decls}
    for stratum in hir.strata:
      for v in list(stratum.base_variants) + list(stratum.recursive_variants):
        if (
          v.split_at >= 0
          and v.temp_rel_name
          and v.temp_rel_name not in existing
        ):
          types = _infer_temp_types(v, hir.relation_decls)
          hir.relation_decls.append(RelationDecl(
            rel_name=v.temp_rel_name,
            types=types,
            semiring="NoProvenance",
            is_generated=True,
            is_temp=True,
          ))
          existing.add(v.temp_rel_name)
    return hir


class TempIndexRegistrationPass:
  '''Pass 5.5: register the identity index [0..arity-1] for each split
  variant's temp relation in its enclosing stratum's required_indices
  and canonical_index. Runs after the main IndexSelectionPass so it
  only affects temp rels (which weren't seen by the selection pass
  because they're head-only here).
  '''
  info = PassInfo(
    name="TempIndexRegistration",
    level=PassLevel.HIR_TRANSFORM,
    order=350,
    source_dialect=Dialect.HIR,
    target_dialect=Dialect.HIR,
  )

  def run(self, hir: HirProgram) -> HirProgram:
    for stratum in hir.strata:
      for v in list(stratum.base_variants) + list(stratum.recursive_variants):
        if v.split_at < 0 or not v.temp_rel_name:
          continue
        arity = len(v.temp_vars)
        temp_idx = list(range(arity))
        if v.temp_rel_name not in stratum.required_indices:
          stratum.required_indices[v.temp_rel_name] = [temp_idx]
          stratum.canonical_index[v.temp_rel_name] = temp_idx
    return hir
