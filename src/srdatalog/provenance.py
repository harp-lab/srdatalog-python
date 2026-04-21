'''Provenance tracking for rule/clause rewriting passes.

Mirrors src/srdatalog/provenance.nim. The HIR JSON emitter writes a
`"provenance": {...}` sub-object only when the kind is `pkCompilerGen`,
so user-written rules and clauses keep the default `UserProvenance`
sentinel and emit no extra JSON keys.
'''
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class ProvenanceKind(Enum):
  USER = "pkUser"
  COMPILER_GEN = "pkCompilerGen"
  DERIVED = "pkDerived"


@dataclass(frozen=True)
class Provenance:
  kind: ProvenanceKind = ProvenanceKind.USER
  parent_rule: str = ""
  derived_from: str = ""
  transform_pass: str = ""


USER_PROVENANCE: Provenance = Provenance()


def compiler_gen(parent_rule: str, derived_from: str, transform_pass: str) -> Provenance:
  '''Construct a Provenance for a compiler-generated rule or clause.'''
  return Provenance(
    kind=ProvenanceKind.COMPILER_GEN,
    parent_rule=parent_rule,
    derived_from=derived_from,
    transform_pass=transform_pass,
  )
