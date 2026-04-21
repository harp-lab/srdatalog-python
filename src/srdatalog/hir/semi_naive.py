'''HIR Pass 3: Semi-Naive Variant Generation.

Populates `base_variants` / `recursive_variants` on each HirStratum.

For each stratum:
  - Non-recursive: one HirRuleVariant per rule (base, delta_idx=-1).
  - Recursive: for each rule, one DELTA variant per body clause whose
    relation is an SCC member. The variant pins `delta_idx` to that clause
    index and sets `clause_versions[idx] = DELTA` (others FULL).

Mirrors `generateVariants` in src/srdatalog/hir/semi_naive.nim.

Negations are NOT delta candidates (only positive occurrences of SCC
members get incrementalized).
'''
from __future__ import annotations

from srdatalog.dsl import Rule, Atom
from srdatalog.hir.types import HirProgram, HirRuleVariant, Version
from srdatalog.hir.pass_ import PassInfo, PassLevel, Dialect


def find_scc_clause_indices(rule: Rule, scc_members: set[str]) -> list[int]:
  '''Indices of positive body clauses (Atoms) whose relation is in scc_members.'''
  return [
    i for i, b in enumerate(rule.body)
    if isinstance(b, Atom) and b.rel in scc_members
  ]


def create_base_variant(rule: Rule) -> HirRuleVariant:
  return HirRuleVariant(
    original_rule=rule,
    delta_idx=-1,
    clause_versions=[Version.FULL] * len(rule.body),
  )


def create_delta_variant(rule: Rule, delta_idx: int) -> HirRuleVariant:
  cvs = [Version.FULL] * len(rule.body)
  cvs[delta_idx] = Version.DELTA
  return HirRuleVariant(
    original_rule=rule,
    delta_idx=delta_idx,
    clause_versions=cvs,
  )


def generate_variants(hir: HirProgram) -> HirProgram:
  '''Populate base_variants and recursive_variants per stratum. Mutates
  and returns the same HirProgram.
  '''
  for stratum in hir.strata:
    scc_members = stratum.scc_members
    if stratum.is_recursive:
      for rule in stratum.stratum_rules:
        for idx in find_scc_clause_indices(rule, scc_members):
          stratum.recursive_variants.append(create_delta_variant(rule, idx))
    else:
      for rule in stratum.stratum_rules:
        stratum.base_variants.append(create_base_variant(rule))
  return hir


class SemiNaiveVariantPass:
  '''Pipeline wrapper. Runs right after stratify (order=100) so downstream
  HIR transforms that care about variants can assume they exist.
  '''
  info = PassInfo(
    name="SemiNaiveVariants",
    level=PassLevel.HIR_TRANSFORM,
    order=100,
    source_dialect=Dialect.HIR,
    target_dialect=Dialect.HIR,
  )

  def run(self, hir: HirProgram) -> HirProgram:
    return generate_variants(hir)
