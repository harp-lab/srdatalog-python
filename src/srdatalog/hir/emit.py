'''Canonical JSON emitter for HIR, matching src/srdatalog/hir/json_printer.nim
output byte-for-byte (minus the hirSExpr debug field).

Strategy:
  - Build a dict tree that matches the key ordering json_printer emits.
  - Serialize with json.dumps(indent=2, ensure_ascii=False); tests compare
    after structural canonicalization so field order matters but
    pretty-print formatting does not.

The hirSExpr field is intentionally skipped: it's a debug text that Nim
produces for VSCode display, and reproducing it in Python adds surface
area with no verification value. Tests strip it from the golden fixture
before comparing.
'''
from __future__ import annotations
import json

from srdatalog.dsl import Atom, Negation, Filter, Let, Agg, Split, ClauseArg, ArgKind, Rule
from srdatalog.provenance import Provenance, ProvenanceKind
from srdatalog.hir.types import (
  Version,
  AccessPattern,
  HirRuleVariant,
  HirStratum,
  HirProgram,
  RelationDecl,
)


def _arg_obj(a: ClauseArg) -> dict:
  '''Mirrors toJson(ClauseArg) in json_printer.nim.'''
  if a.kind is ArgKind.LVAR:
    return {"kind": "var", "name": a.var_name}
  if a.kind is ArgKind.CONST:
    return {"kind": "const", "val": a.const_cpp_expr}
  if a.kind is ArgKind.CPP_CODE:
    return {"kind": "code", "code": a.cpp_code}
  raise ValueError(f"Unknown ArgKind: {a.kind}")


def _head_obj(atom: Atom) -> dict:
  '''Mirrors toJson(HeadClause).'''
  return {
    "rel": atom.rel,
    "args": [_arg_obj(a) for a in atom.args],
  }


def _prov_obj(prov: Provenance) -> dict:
  '''Mirror Nim's provenance JSON for pkCompilerGen. Fields are elided
  when they carry the default empty string — matches Nim's %-formatting.
  '''
  return {
    "kind": "compilerGen",
    "parentRule": prov.parent_rule,
    "derivedFrom": prov.derived_from,
    "transformPass": prov.transform_pass,
  }


def _body_obj(clause) -> dict:
  '''Mirrors toJson(BodyClause). Covers RelClause, NegClause, IfClause, LetClause.

  Aggregation and Split clauses are still not expressible in the Python DSL.
  Emits `"provenance": {...}` ONLY for compiler-generated Atoms (RelClause),
  matching the Nim side where user-written clauses stay prov-less.
  '''
  if isinstance(clause, Negation):
    a = clause.atom
    return {
      "kind": "negation",
      "rel": a.rel,
      "args": [_arg_obj(arg) for arg in a.args],
    }
  if isinstance(clause, Filter):
    return {"kind": "filter", "code": clause.code}
  if isinstance(clause, Let):
    return {"kind": "bind", "var": clause.var_name, "code": clause.code}
  if isinstance(clause, Agg):
    return {
      "kind": "aggregation",
      "rel": clause.rel,
      "func": clause.func,
      "result": clause.result_var,
      "args": [_arg_obj(a) for a in clause.args],
    }
  if isinstance(clause, Split):
    return {"kind": "split"}
  # Atom (RelClause)
  out: dict = {
    "kind": "relation",
    "rel": clause.rel,
    "args": [_arg_obj(arg) for arg in clause.args],
  }
  if clause.prov.kind is ProvenanceKind.COMPILER_GEN:
    out["provenance"] = _prov_obj(clause.prov)
  return out


def _rule_obj(rule: Rule) -> dict:
  '''Mirrors toJson(Rule). Head is emitted as a single-element list to
  match the Nim side's seq[HeadClause]. Emits `"provenance": {...}`
  ONLY for compiler-generated rules.
  '''
  out: dict = {
    "name": rule.name,
    "head": [_head_obj(rule.head)],
    "body": [_body_obj(b) for b in rule.body],
  }
  if rule.prov.kind is ProvenanceKind.COMPILER_GEN:
    out["provenance"] = _prov_obj(rule.prov)
  return out


def _arg_text(a: ClauseArg) -> str:
  '''Mirrors argToText in json_printer.nim.'''
  if a.kind is ArgKind.LVAR:
    return a.var_name
  if a.kind is ArgKind.CONST:
    return a.const_cpp_expr
  if a.kind is ArgKind.CPP_CODE:
    return "{" + a.cpp_code + "}"
  raise ValueError(f"Unknown ArgKind: {a.kind}")


def _rule_text(rule: Rule) -> str:
  '''Mirrors ruleToText. Format: "<name>: <head> <-\n      <body1>,\n      <body2>".'''
  head = "(" + rule.head.rel + " " + " ".join(_arg_text(a) for a in rule.head.args) + ")"
  body_parts: list[str] = []
  for b in rule.body:
    if isinstance(b, Negation):
      atom = b.atom
      body_parts.append(
        "(NOT (" + atom.rel + " " + " ".join(_arg_text(x) for x in atom.args) + "))"
      )
    elif isinstance(b, Filter):
      body_parts.append("(FILTER " + b.code + ")")
    elif isinstance(b, Let):
      body_parts.append("(LET " + b.var_name + " := " + b.code + ")")
    elif isinstance(b, Agg):
      args = " ".join(_arg_text(a) for a in b.args)
      body_parts.append(
        "(AGG " + b.result_var + " := " + b.func + " (" + b.rel + " " + args + "))"
      )
    elif isinstance(b, Split):
      body_parts.append("---")
    else:
      body_parts.append(
        "(" + b.rel + " " + " ".join(_arg_text(x) for x in b.args) + ")"
      )
  return rule.name + ": " + head + " <-\n      " + ",\n      ".join(body_parts)


def _access_obj(ap: AccessPattern) -> dict:
  '''Mirrors toJson(AccessPattern). Note the `index` key is conditional on
  non-empty indexCols.
  '''
  obj: dict = {
    "rel": ap.rel_name,
    "version": ap.version.value,
    "accessOrder": list(ap.access_order),
    "prefixLen": ap.prefix_len,
  }
  if ap.index_cols:
    obj["index"] = list(ap.index_cols)
  obj["clauseIdx"] = ap.clause_idx
  return obj


def _variant_obj(v: HirRuleVariant, is_recursive: bool) -> dict:
  '''Mirrors toJson(HirRuleVariant). deltaIdx only appears for recursive variants.'''
  obj: dict = {
    "rule": _rule_obj(v.original_rule),
    "hirText": _rule_text(v.original_rule),
  }
  if is_recursive:
    obj["deltaIdx"] = v.delta_idx
  obj["clauseOrder"] = list(v.clause_order)
  obj["varOrder"] = list(v.var_order)
  obj["joins"] = [_access_obj(ap) for ap in v.access_patterns]
  obj["negations"] = [_access_obj(ap) for ap in v.negation_patterns]
  return obj


def _stratum_obj(s: HirStratum, idx: int) -> dict:
  '''Mirrors toJson(HirStratum). sccMembers is sorted for determinism (same
  fix that was applied to the Nim side in the previous commit).
  '''
  return {
    "id": idx,
    "isRecursive": s.is_recursive,
    "relations": sorted(s.scc_members),
    "base": [_variant_obj(v, False) for v in s.base_variants],
    "recursive": [_variant_obj(v, True) for v in s.recursive_variants],
  }


def _relations_obj(decls: list[RelationDecl]) -> dict:
  '''Mirrors the `relations` block in dumpHirToJson. Insertion order follows
  decl order (matches Nim's seq iteration).
  '''
  out: dict = {}
  for d in decls:
    out[d.rel_name] = {
      "semiring": d.semiring,
      "types": list(d.types),
    }
  return out


def hir_to_obj(prog: HirProgram) -> dict:
  '''Build the canonical dict tree for a HirProgram. Does not include
  hirSExpr (strip that from the Nim fixture when comparing).
  '''
  return {
    "strata": [_stratum_obj(s, i) for i, s in enumerate(prog.strata)],
    "relations": _relations_obj(prog.relation_decls),
  }


def emit_hir_json(prog: HirProgram) -> str:
  '''Emit canonical JSON string.'''
  return json.dumps(hir_to_obj(prog), indent=2, ensure_ascii=False)
