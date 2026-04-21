'''Structural validator: compare every translated benchmark against
its Nim source, rule-by-rule.

Runs `tools.nim_to_dsl.translate_file` to get the canonical parse of
each `.nim` file, then imports the corresponding `examples/<name>.py`
and extracts the same fields from the built `Program`. Fails loudly
on any mismatch — name, head atom, body clause count/shape, plan
entries, count/semi_join flags.

Usage:

    python tools/validate_translation.py

Exits nonzero if any benchmark has a diverging rule. Expected output
on a clean checkout:

    [OK] doop                76 rels / 84 rules
    [OK] triangle            4 rels / 1 rule
    [OK] ...
    All 17 benchmarks structurally equivalent.

This is the "the translator produced something that matches Nim"
invariant — distinct from the HIR-compile and byte-match tests which
check semantics further downstream.
'''

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "examples"))
sys.path.insert(0, str(_REPO / "tools"))

from nim_to_dsl import BodyAtom, FilterClause, Plan, RuleSpec, translate_file

# ---------------------------------------------------------------------------
# Benchmark catalog
# ---------------------------------------------------------------------------

_UPSTREAM = Path("/home/stargazermiao/workspace/SRDatalog/integration_tests/examples")
BENCHMARKS: list[tuple[str, Path, str | None]] = [
  # (python module name, path to .nim, optional meta json for dataset_consts)
  ("andersen", _UPSTREAM / "andersen_gpu/andersen.nim", None),
  ("cspa", _UPSTREAM / "andersen_gpu/cspa.nim", None),
  ("crdt", _UPSTREAM / "crdt/crdt.nim", None),
  ("ddisasm", _UPSTREAM / "ddisasm/ddisasm.nim", str(_UPSTREAM / "ddisasm/cvc5_consts.json")),
  ("reg_scc", _UPSTREAM / "ddisasm/reg_scc.nim", None),
  ("doop", _UPSTREAM / "doop/doop.nim", str(_UPSTREAM / "doop/batik_meta.json")),
  ("galen", _UPSTREAM / "galen/galen.nim", None),
  ("polonius_test", _UPSTREAM / "polonius/polonius_test.nim", None),
  ("sg", _UPSTREAM / "sg/sg.nim", None),
  ("tc", _UPSTREAM / "tc/tc.nim", None),
  ("triangle", _UPSTREAM / "triangle/triangle.nim", None),
  ("lsqb_q3_triangle", _UPSTREAM / "triangle/lsqb_q3_triangle.nim", None),
  ("lsqb_q6_2hop", _UPSTREAM / "triangle/lsqb_q6_2hop.nim", None),
  ("lsqb_q6_count", _UPSTREAM / "triangle/lsqb_q6_count.nim", None),
  ("lsqb_q7_optional", _UPSTREAM / "triangle/lsqb_q7_optional.nim", None),
  ("lsqb_q9_neg2hop", _UPSTREAM / "triangle/lsqb_q9_neg2hop.nim", None),
  ("lsqb_triangle_count", _UPSTREAM / "triangle/lsqb_triangle_count.nim", None),
]


# ---------------------------------------------------------------------------
# Canonicalization — convert both representations to a single dict shape
# ---------------------------------------------------------------------------


def _canon_plan(p: Plan) -> dict[str, Any]:
  return {
    "delta": p.delta,
    "var_order": list(p.var_order),
    "block_group": p.block_group,
    "work_stealing": p.work_stealing,
    "dedup_hash": p.dedup_hash,
  }


def _canon_atom_nim(a: BodyAtom) -> dict[str, Any]:
  return {
    "kind": "atom",
    "rel": a.rel,
    "args": list(a.args),  # raw Nim identifiers / ints / _
    "negated": a.negated,
  }


def _canon_body_clause_nim(c) -> dict[str, Any]:
  if isinstance(c, BodyAtom):
    return _canon_atom_nim(c)
  if isinstance(c, FilterClause):
    return {
      "kind": "filter",
      "vars": list(c.vars),
      "code": _normalize_filter_code(c.code),
    }
  if c == "__SPLIT__":
    return {"kind": "split"}
  raise TypeError(f"unknown Nim body clause: {c!r}")


def _normalize_filter_code(code: str) -> str:
  '''Collapse whitespace so minor formatting differences don't flag as
  divergence. Nim's pragma-body comes in with stripped newlines already;
  Python DSL re-emits it escaped. Both should normalize to the same
  trimmed token stream.'''
  return " ".join(code.split())


def _canon_rule_nim(r: RuleSpec) -> dict[str, Any]:
  out: dict[str, Any] = {
    "name": r.name,
    "heads": [_canon_atom_nim(h) for h in r.heads],
    "body": [_canon_body_clause_nim(c) for c in r.body],
    "plans": [_canon_plan(p) for p in r.plans],
    "count": r.count,
    "semi_join": r.semi_join,
  }
  return out


# ---------------------------------------------------------------------------
# Python-side canonicalization: walk the built Program
# ---------------------------------------------------------------------------


def _canon_arg_py(arg) -> str:
  '''Python DSL args are ClauseArg dataclasses. Squash to the same raw
  token form the Nim side uses: variable → name, int const → str(int),
  wildcard → "_". CPP_CODE / CONST carry their emission string.'''
  from srdatalog.dsl import ArgKind

  if arg.kind is ArgKind.LVAR:
    # `Var("_")` in Python represents the Nim wildcard.
    return arg.var_name if arg.var_name != "_" else "_"
  if arg.kind is ArgKind.CONST:
    return arg.const_cpp_expr or str(arg.const_value)
  if arg.kind is ArgKind.CPP_CODE:
    return f"$${arg.cpp_code}"
  raise TypeError(f"unknown arg kind: {arg.kind!r}")


def _canon_atom_py(atom) -> dict[str, Any]:
  return {
    "kind": "atom",
    "rel": atom.rel,
    "args": [_canon_arg_py(a) for a in atom.args],
    "negated": False,
  }


def _canon_body_clause_py(clause) -> dict[str, Any]:
  from srdatalog.dsl import Atom, Filter, Negation, Split

  if isinstance(clause, Atom):
    return _canon_atom_py(clause)
  if isinstance(clause, Negation):
    d = _canon_atom_py(clause.atom)
    d["negated"] = True
    return d
  if isinstance(clause, Filter):
    return {
      "kind": "filter",
      "vars": list(clause.vars),
      "code": _normalize_filter_code(clause.code),
    }
  if isinstance(clause, Split):
    return {"kind": "split"}
  raise TypeError(f"unknown Python body clause: {clause!r}")


def _canon_plan_py(entry) -> dict[str, Any]:
  return {
    "delta": entry.delta,
    "var_order": list(entry.var_order),
    "block_group": entry.block_group,
    "work_stealing": entry.work_stealing,
    "dedup_hash": entry.dedup_hash,
  }


def _canon_rule_py(rule) -> dict[str, Any]:
  return {
    "name": rule.name,
    "heads": [_canon_atom_py(rule.head)],  # Python DSL is single-head per Rule
    "body": [_canon_body_clause_py(c) for c in rule.body],
    "plans": [_canon_plan_py(p) for p in rule.plans],
    "count": rule.count,
    "semi_join": rule.semi_join,
  }


def _normalize_py_multi_head(rules_py: list) -> list:
  '''Python-DSL multi-head rules are split by the translator into
  `<name>_h0`, `<name>_h1` single-head clones. Undo that for
  comparison so each Nim multi-head rule matches a *group* of
  single-head Python rules.
  '''
  out = []
  buf: dict[str, list] = {}
  for r in rules_py:
    name = r["name"]
    # Match `X_h<digits>` suffix.
    import re

    m = re.match(r"^(.*)_h(\d+)$", name)
    if m:
      base = m.group(1)
      buf.setdefault(base, []).append(r)
    else:
      out.append(r)
  for base, group in buf.items():
    merged = dict(group[0])
    merged["name"] = base
    merged["heads"] = [h for g in group for h in g["heads"]]
    # Body, plans, count, semi_join must all match across clones.
    for g in group[1:]:
      if g["body"] != merged["body"] or g["plans"] != merged["plans"]:
        raise AssertionError(
          f"Multi-head clones for {base!r} have divergent body/plan — "
          f"this indicates an earlier bug."
        )
    out.append(merged)
  return out


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def _find_build_program_fn(mod):
  '''Return the `build_*_program()` zero-arg builder. We deliberately
  skip the `build_<name>(meta_json)` wrapper (which resolves dataset
  consts to ints) — validation wants the SYMBOLIC form so args like
  `Var("CLASS_INIT_METHOD")` match the Nim source's identifier
  `CLASS_INIT_METHOD`.'''
  for n in dir(mod):
    if n.startswith("build_") and n.endswith("_program"):
      fn = getattr(mod, n)
      if callable(fn):
        return fn
  raise RuntimeError(f"{mod.__name__}: no build_*_program() function")


def _build_python_program(mod_name: str):
  mod = importlib.import_module(mod_name)
  return _find_build_program_fn(mod)()


def _compare_rules(nim_rules, py_rules, module: str) -> list[str]:
  errors = []
  if len(nim_rules) != len(py_rules):
    errors.append(f"{module}: rule count mismatch — Nim={len(nim_rules)}, Python={len(py_rules)}")
  by_name_nim = {r["name"]: r for r in nim_rules}
  by_name_py = {r["name"]: r for r in py_rules}
  for nim_name, nim_r in by_name_nim.items():
    if nim_name not in by_name_py:
      errors.append(f"{module}: rule {nim_name!r} missing from Python output")
      continue
    py_r = by_name_py[nim_name]
    for field in ("heads", "body", "plans", "count", "semi_join"):
      if nim_r[field] != py_r[field]:
        errors.append(
          f"{module}: {nim_name!r}.{field} differs:\n"
          f"  Nim:    {nim_r[field]}\n"
          f"  Python: {py_r[field]}"
        )
  for py_name in by_name_py:
    if py_name not in by_name_nim:
      errors.append(f"{module}: rule {py_name!r} in Python but not Nim source")
  return errors


def validate_one(mod_name: str, nim_path: Path, meta_json: str | None) -> list[str]:
  _py, nim_rels, nim_rules, _consts = translate_file(nim_path)
  # meta_json is ignored — we compare symbolic form (pre-substitution).
  prog = _build_python_program(mod_name)

  # Canonicalize both sides.
  nim_canon = [_canon_rule_nim(r) for r in nim_rules]
  py_canon = [_canon_rule_py(r) for r in prog.rules]
  py_canon = _normalize_py_multi_head(py_canon)

  errors = _compare_rules(nim_canon, py_canon, mod_name)
  # Also verify relation-count parity (relations don't have structural
  # subtleties worth diffing here beyond counts + names).
  nim_rel_names = {r.name for r in nim_rels}
  py_rel_names = {r.name for r in prog.relations}
  missing_py = nim_rel_names - py_rel_names
  extra_py = py_rel_names - nim_rel_names
  if missing_py:
    errors.append(f"{mod_name}: relations missing from Python: {sorted(missing_py)}")
  if extra_py:
    errors.append(f"{mod_name}: relations only in Python: {sorted(extra_py)}")
  return errors


def main() -> int:
  all_errors: list[str] = []
  for mod_name, nim_path, meta in BENCHMARKS:
    try:
      errors = validate_one(mod_name, nim_path, meta)
    except Exception as e:
      print(f"[FAIL] {mod_name:<24} crashed: {type(e).__name__}: {e}")
      all_errors.append(f"{mod_name}: {e}")
      continue
    prog = _build_python_program(mod_name)
    if errors:
      print(f"[FAIL] {mod_name:<24} {len(errors)} divergence(s)")
      for e in errors[:10]:
        print(f"         {e}")
      if len(errors) > 10:
        print(f"         ... {len(errors) - 10} more")
      all_errors.extend(errors)
    else:
      print(f"[ OK ] {mod_name:<24} {len(prog.relations)} rels / {len(prog.rules)} rules")
  if all_errors:
    print(f"\n{len(all_errors)} divergence(s) across benchmarks")
    return 1
  print(f"\nAll {len(BENCHMARKS)} benchmarks structurally equivalent to Nim source.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
