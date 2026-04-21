'''Dataset-specific interned constants.

Mirrors the `dataset_const NAME: "symbol"` macro in `src/srdatalog/lang.nim`
(lines ~126-265). In Nim, SCREAMING_SNAKE identifiers declared in a rule
set are resolved at macro-expansion time by reading a JSON file that maps
lower-case symbol names to int64 IDs (produced by the domain's
string-intern table — e.g., Java-symbol dictionary for doop):

    dataset_const MAIN: "main"        # NAME in rule body
    dataset_const CLASS_INIT_METHOD: "class_init_method"

The resolver replaces each NAME with the corresponding int everywhere:
  1. In atom args where the NAME appears as a bare identifier.
  2. Inside `Filter.code` C++ bodies via word-boundary substitution.

The Python port does the same at rule-build time (when building the
`Program`) via `resolve_dataset_consts(program, consts)`.

Typical usage:

    from srdatalog.dataset_const import load_meta, resolve_program_consts

    consts = load_meta("integration_tests/examples/doop/batik_meta.json", {
      "ABSTRACT":           "abstract",
      "CLASS_INIT_METHOD":  "class_init_method",
      "JAVA_LANG_OBJECT":   "java_lang_Object",
    })
    prog = resolve_program_consts(build_doop_program(consts), consts)
'''

from __future__ import annotations

import json
import re
from pathlib import Path

from srdatalog.dsl import (
  ArgKind,
  Atom,
  ClauseArg,
  Filter,
  Negation,
  Program,
  Rule,
)


def load_meta(
  meta_path: str | Path,
  declarations: dict[str, str],
) -> dict[str, int]:
  '''Read `meta_path` (JSON map: symbol-name → int) and return
  `{const_name: int_value}` for each declared entry.

  Args:
    meta_path: path to the JSON file (e.g. batik_meta.json).
    declarations: mapping from UPPER_CASE const name (as used in rules)
      to the lower-case JSON key. Mirrors the
      `dataset_const NAME: "key"` pairs from Nim's rules_def.

  Raises KeyError on any missing declaration — we fail fast rather
  than silently resolving to a garbage value (Nim aborts the compile
  with a similar error).
  '''
  data = json.loads(Path(meta_path).read_text())
  out: dict[str, int] = {}
  for name, key in declarations.items():
    if key not in data:
      raise KeyError(
        f"dataset_const {name!r} refers to missing key {key!r} in {meta_path}. "
        f"Available keys: {sorted(data.keys())[:10]}..."
      )
    value = data[key]
    if not isinstance(value, int):
      raise TypeError(
        f"dataset_const {name!r}: expected int value, got {type(value).__name__} ({value!r})"
      )
    out[name] = value
  return out


# Word-boundary regex for substituting constant names inside C++ filter
# bodies. Matches NAME only when it's a standalone identifier so e.g.
# "MAIN" doesn't clobber "MAIN_DESCRIPTOR" (the longer name substitutes
# first because we sort by length below).
def _compile_subst_pattern(consts: dict[str, int]) -> tuple[re.Pattern, dict[str, str]]:
  if not consts:
    return re.compile(r"(?!)"), {}  # Matches nothing
  # Longest-first avoids partial overlaps (MAIN vs MAIN_DESCRIPTOR).
  names_sorted = sorted(consts.keys(), key=len, reverse=True)
  pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names_sorted) + r")\b")
  mapping = {n: str(consts[n]) for n in consts}
  return pattern, mapping


def _rewrite_cpp_code(code: str, consts: dict[str, int]) -> str:
  '''Substitute NAME → int inside a filter/let C++ body, respecting
  identifier word boundaries. No-op when `consts` is empty.'''
  if not code or not consts:
    return code
  pat, mp = _compile_subst_pattern(consts)
  return pat.sub(lambda m: mp[m.group(1)], code)


def _rewrite_arg(arg: ClauseArg, consts: dict[str, int]) -> ClauseArg:
  '''Substitute a single clause argument. If it's a variable whose
  name is a declared const, convert to a CONST(int). CPP_CODE args
  get word-boundary substitution.'''
  if arg.kind is ArgKind.LVAR and arg.var_name in consts:
    n = consts[arg.var_name]
    return ClauseArg(kind=ArgKind.CONST, const_value=n, const_cpp_expr=str(n))
  if arg.kind is ArgKind.CPP_CODE and arg.cpp_code:
    return ClauseArg(
      kind=ArgKind.CPP_CODE,
      cpp_code=_rewrite_cpp_code(arg.cpp_code, consts),
    )
  return arg


def _rewrite_atom(atom: Atom, consts: dict[str, int]) -> Atom:
  return Atom(
    rel=atom.rel,
    args=tuple(_rewrite_arg(a, consts) for a in atom.args),
    prov=atom.prov,
  )


def _rewrite_clause(clause, consts: dict[str, int]):
  if isinstance(clause, Atom):
    return _rewrite_atom(clause, consts)
  if isinstance(clause, Negation):
    return Negation(atom=_rewrite_atom(clause.atom, consts))
  if isinstance(clause, Filter):
    return Filter(vars=clause.vars, code=_rewrite_cpp_code(clause.code, consts))
  # Let / Agg / Split passthrough (add Let rewrite if it ever carries user cpp).
  if hasattr(clause, "code") and isinstance(getattr(clause, "code", None), str):
    # Generic best-effort for Let-like nodes with a `code` field.
    import dataclasses

    if dataclasses.is_dataclass(clause):
      return dataclasses.replace(
        clause,
        code=_rewrite_cpp_code(clause.code, consts),
      )
  return clause


def resolve_rule_consts(rule: Rule, consts: dict[str, int]) -> Rule:
  '''Rewrite a single Rule: substitute dataset consts in head, body,
  and debug_code. Returns a new Rule (dataclass is frozen).'''
  import dataclasses

  new_heads = tuple(_rewrite_atom(h, consts) for h in rule.heads)
  new_body = tuple(_rewrite_clause(c, consts) for c in rule.body)
  new_debug = _rewrite_cpp_code(rule.debug_code, consts) if rule.debug_code else ""
  return dataclasses.replace(
    rule,
    heads=new_heads,
    body=new_body,
    debug_code=new_debug,
  )


def resolve_program_consts(program: Program, consts: dict[str, int]) -> Program:
  '''Apply dataset_const substitution across every rule in a Program.
  `Relation` decls don't carry constant references, so they pass
  through unchanged.'''
  if not consts:
    return program
  return Program(
    relations=list(program.relations),
    rules=[resolve_rule_consts(r, consts) for r in program.rules],
  )


__all__ = [
  "load_meta",
  "resolve_program_consts",
  "resolve_rule_consts",
]
