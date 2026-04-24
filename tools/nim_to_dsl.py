'''Translate a SRDatalog Nim `.nim` source file to a Python DSL program.

Handles the doop-shaped subset (which is a strict superset of most
benchmarks in `integration_tests/examples/`):

  - `schema <Name>:` block with `RelName:\\n  Relation[t1, t2] {.pragmas.}`
    where pragmas may include `input: "X.csv"`, `print_size`,
    `index: "SRDatalog::GPU::Device2LevelIndex"`.
  - `rules_def <Name>:` block with
    - `dataset_const NAME: "key"` lines
    - Rule forms:
        [{(head ...)} <-- {(body1 ...), (body2 ...)}]
        [{(head1 ...), (head2 ...)} <-- {body}]
      each followed by `{.name: X [, plan: [...]].}`.
    - Body clauses: `(Rel args)`, `~(Rel args)` (negation),
      `(filter((vars)) {.cpp: """..."""})`.
    - Plan pragmas: `plan: [(delta: N, var_order: [a, b], block_group: true)]`.

It emits a Python file with:
  - `Relation(...)` decls with pragma kwargs
  - A `build_<lower>()` function returning `Program(...)`
  - Imports from `srdatalog.dsl`, `srdatalog.dataset_const`
  - A comment block with the dataset_const declarations (caller maps
    them via `load_meta`).

Run:

    python tools/nim_to_dsl.py integration_tests/examples/doop/doop.nim \\
        --out examples/doop/doop.py

Returns nonzero if the Nim file uses syntax outside the supported
subset (unrecognized body form, etc.) — better to fail loudly than
emit subtly wrong Python.
'''

from __future__ import annotations

import argparse
import keyword
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Lexing helpers — work with blank-line-separated pragma-aware paragraphs.
# ---------------------------------------------------------------------------


def _strip_comments(src: str) -> str:
  '''Drop `#`-prefixed line comments. Nim block comments `#[ ... ]#`
  do appear in doop — strip those too (single-pass, no nesting).'''
  src = re.sub(r"#\[.*?\]#", "", src, flags=re.DOTALL)
  # Line comments — preserve leading whitespace to keep indent structure.
  return re.sub(r"(^|\s)#[^\n]*", r"\1", src)


# ---------------------------------------------------------------------------
# Schema block
# ---------------------------------------------------------------------------


@dataclass
class RelationSpec:
  name: str
  types: list[str]  # e.g. ["int", "int"]
  input_file: str = ""
  print_size: bool = False
  index_type: str = ""


_PRAGMA_INPUT = re.compile(r'input\s*:\s*"([^"]+)"')
_PRAGMA_INDEX = re.compile(r'index\s*:\s*"([^"]+)"')
_PRAGMA_PRINT = re.compile(r'\bprint_size\b')


def _parse_schema_block(src: str) -> list[RelationSpec]:
  '''Find `schema <Name>:` then collect `RelName:\\n  Relation[types] {.pragmas.}`
  groups until the block ends (a line at column-0 that isn't the schema body).

  Nim indent-based — we loosely detect block end when we hit a line
  that starts with `rules_def`, `datalog_db`, `proc`, or ` ` column-0
  non-schema content. The schema body is what matters; we don't
  validate structure beyond that.'''
  m = re.search(r"^schema\s+(\w+)\s*:", src, flags=re.MULTILINE)
  if not m:
    return []
  start = m.end()
  # End of block: next top-level (non-indented) declaration.
  tail = src[start:]
  end_m = re.search(r"^(rules_def|datalog_db|proc|fixpoint_rule)\b", tail, flags=re.MULTILINE)
  block = tail[: end_m.start()] if end_m else tail

  specs: list[RelationSpec] = []
  # Match `RelName:\n  Relation[types] {.pragmas.}` — allow multi-line pragmas.
  # The pragma block is `{. ... .}` with newlines inside.
  relation_pat = re.compile(
    r"^\s*(\w+)\s*:\s*\n"
    r"\s*Relation\[([^\]]+)\]"  # types
    r"(\s*\{\.\s*(.*?)\s*\.\})?",  # optional pragma block
    re.MULTILINE | re.DOTALL,
  )
  for rm in relation_pat.finditer(block):
    name, types_raw, _pragma_wrap, pragma_body = rm.groups()
    types = [t.strip() for t in types_raw.split(",")]
    spec = RelationSpec(name=name, types=types)
    if pragma_body:
      if mm := _PRAGMA_INPUT.search(pragma_body):
        spec.input_file = mm.group(1)
      if _PRAGMA_PRINT.search(pragma_body):
        spec.print_size = True
      if mm := _PRAGMA_INDEX.search(pragma_body):
        spec.index_type = mm.group(1)
    specs.append(spec)
  return specs


# ---------------------------------------------------------------------------
# rules_def block — dataset_consts + rules
# ---------------------------------------------------------------------------


@dataclass
class DatasetConst:
  name: str  # UPPER_SNAKE
  key: str  # lower_snake JSON key


@dataclass
class FilterClause:
  vars: list[str]
  code: str  # raw C++


@dataclass
class BodyAtom:
  rel: str
  args: list[str]  # raw idents / ints / underscore
  negated: bool = False


BodyClauseT = "BodyAtom | FilterClause"


@dataclass
class Plan:
  delta: int = -1
  var_order: list[str] = field(default_factory=list)
  block_group: bool = False
  work_stealing: bool = False
  dedup_hash: bool = False


@dataclass
class RuleSpec:
  name: str
  heads: list[BodyAtom]  # head atoms (usually 1, sometimes >1)
  body: list[BodyClauseT]
  plans: list[Plan] = field(default_factory=list)
  count: bool = False  # rule-level `count: true` pragma
  semi_join: bool = False  # rule-level `semi_join: true` pragma
  var_order: list[str] = field(default_factory=list)  # rule-level `var_order: [...]`
  inject_cpp: str = ""  # rule-level `inject_cpp: """..."""` pragma body


def _extract_rules_block(src: str) -> str:
  m = re.search(r"^rules_def\s+\w+\s*:", src, flags=re.MULTILINE)
  if not m:
    return ""
  start = m.end()
  tail = src[start:]
  end_m = re.search(
    r"^(datalog_db|fixpoint_rule|proc|when\s+isMainModule)\b", tail, flags=re.MULTILINE
  )
  return tail[: end_m.start()] if end_m else tail


def _parse_dataset_consts(block: str) -> list[DatasetConst]:
  # `dataset_const NAME:\n  "key"` (indented string) — two-line form.
  pat = re.compile(
    r"^\s*dataset_const\s+(\w+)\s*:\s*\n\s*\"([^\"]+)\"",
    re.MULTILINE,
  )
  return [DatasetConst(name=m.group(1), key=m.group(2)) for m in pat.finditer(block)]


def _strip_outer_pragma(s: str) -> tuple[str, str]:
  '''Split `[rule] {. pragmas .}` into (rule, pragmas). Returns ("", "") if
  no match. Uses a final-`{.` scan so a rule containing filter's
  inner `{.cpp:...}` doesn't confuse us: we find the LAST `{.` in the
  string and assume it opens the outer pragma block.'''
  s = s.strip()
  # Find matching `.}` for the last `{.` — but nested {. .} in filters
  # make this tricky. A robust shortcut: rightmost `.}` closes the
  # outermost pragma; find `{.` immediately before it matched at brace
  # depth 0. Since pragmas are balanced, scan backwards.
  if not s.endswith(".}"):
    return s, ""
  # Walk backwards counting `.}` vs `{.`
  depth = 0
  i = len(s) - 2  # position of '.' in ".}"
  while i >= 1:
    if s[i] == "." and i + 1 < len(s) and s[i + 1] == "}":
      depth += 1
      i -= 1
    elif s[i] == "{" and i + 1 < len(s) and s[i + 1] == ".":
      depth -= 1
      if depth == 0:
        return s[:i].rstrip(), s[i + 2 : -2].strip()
      i -= 1
    i -= 1
  return s, ""


def _split_top_level_commas(s: str) -> list[str]:
  '''Split on commas not nested inside parens/braces/brackets/quotes.'''
  parts, buf, depth, in_str = [], [], 0, False
  i = 0
  while i < len(s):
    ch = s[i]
    if in_str:
      buf.append(ch)
      if ch == '"' and s[i - 1] != "\\":
        in_str = False
      i += 1
      continue
    if ch == '"':
      in_str = True
      buf.append(ch)
    elif ch in "([{":
      depth += 1
      buf.append(ch)
    elif ch in ")]}":
      depth -= 1
      buf.append(ch)
    elif ch == "," and depth == 0:
      parts.append("".join(buf).strip())
      buf = []
    else:
      buf.append(ch)
    i += 1
  tail = "".join(buf).strip()
  if tail:
    parts.append(tail)
  return parts


def _parse_atom(s: str) -> BodyAtom:
  '''Parse `Rel arg1 arg2 ...` (already stripped of parens).'''
  parts = s.strip().split()
  if not parts:
    raise ValueError(f"empty atom: {s!r}")
  return BodyAtom(rel=parts[0], args=parts[1:])


_FILTER_CPP_RE = re.compile(
  r'filter\(\(\s*([^)]*)\s*\)\)\s*\{\.\s*cpp\s*:\s*"""(.*?)"""',
  re.DOTALL,
)

# Shorthand filter syntax: `?(v1, v2, ...) """return ...;"""` — Nim macro
# sugar for `filter((v1, v2, ...)) {.cpp: """..."""}`. Appears in the
# LSQB triangle variants.
_SHORTHAND_FILTER_RE = re.compile(
  r'^\?\(\s*([^)]*)\s*\)\s*"""(.*?)"""\s*$',
  re.DOTALL,
)


def _parse_body_clause(s: str) -> BodyClauseT:
  s = s.strip().rstrip(",").strip()
  # Split marker — partitions the rule body for pipeline-A/B splits
  # (used by ddisasm's negation-pushdown variants).
  if s == "split":
    return "__SPLIT__"  # sentinel; handled in body emit
  # Shorthand filter: `?(v1, v2) """cpp"""`
  if s.startswith("?"):
    m = _SHORTHAND_FILTER_RE.match(s)
    if m:
      vars_ = [v.strip() for v in m.group(1).split(",") if v.strip()]
      return FilterClause(vars=vars_, code=m.group(2).strip())
    raise ValueError(f"can't parse shorthand filter: {s!r}")
  # Top-level `~(Rel ...)` or `!(Rel ...)` negation forms (Nim accepts both)
  if s.startswith("~(") or s.startswith("!("):
    inner = s[2:]
    if inner.endswith(")"):
      inner = inner[:-1]
    atom = _parse_atom(inner)
    atom.negated = True
    return atom
  # Parenthesised forms — atom, nested filter, or nested negation.
  if s.startswith("("):
    s_inner = s[1:].strip()
    if s_inner.endswith(")"):
      s_inner = s_inner[:-1].strip()
    if s_inner.startswith("filter"):
      m = _FILTER_CPP_RE.search(s_inner)
      if not m:
        raise ValueError(f"can't parse filter: {s!r}")
      raw_vars = m.group(1).strip()
      vars_ = [v.strip() for v in raw_vars.split(",") if v.strip()]
      return FilterClause(vars=vars_, code=m.group(2).strip())
    # `(~rel ...)` / `(!rel ...)` form
    if s_inner.startswith("~(") or s_inner.startswith("!("):
      inner = s_inner[2:].rstrip(")").strip()
      atom = _parse_atom(inner)
      atom.negated = True
      return atom
    # Plain atom `(Rel ...)`
    return _parse_atom(s_inner)
  raise ValueError(f"unrecognized body clause: {s!r}")


_RULE_OUTER = re.compile(
  r"\[\s*\{(.*?)\}\s*<--\s*\{(.*?)\}\s*\](?:\s*\{\.(.*?)\.\})?",
  re.DOTALL,
)


def _parse_rule_pragma(body: str) -> tuple[str, list[Plan], bool, bool, list[str], str]:
  '''Parse `name: X, plan: [...], count: true, ...` from a rule pragma body.
  Returns `(name, plans, count, semi_join, rule_level_var_order, inject_cpp)`.'''
  name = ""
  m = re.search(r"\bname\s*:\s*(\w+)", body)
  if m:
    name = m.group(1)

  count = bool(re.search(r"\bcount\s*:\s*true\b", body))
  semi_join = bool(re.search(r"\bsemi_join\s*:\s*true\b", body))

  inject_cpp = ""
  im = re.search(r'\binject_cpp\s*:\s*"""(.*?)"""', body, re.DOTALL)
  if im:
    inject_cpp = im.group(1)

  # Rule-level var_order (distinct from plan-entry var_order — some
  # benchmarks tag it on the rule directly).
  rule_var_order: list[str] = []
  # Match `var_order: [a, b, c]` NOT inside a `plan: [(...)]` block.
  # Heuristic: strip the plan block first, then scan for var_order.
  stripped = re.sub(r"plan\s*:\s*\[.*?\](?=\s*[,\.]|$)", "", body, flags=re.DOTALL)
  vm = re.search(r"\bvar_order\s*:\s*\[([^\]]*)\]", stripped)
  if vm:
    rule_var_order = [v.strip() for v in vm.group(1).split(",") if v.strip()]

  plans: list[Plan] = []
  # plan: [ (entry1), (entry2), ... ]
  # Non-greedy regex would stop at the FIRST `]`, which is the inner `]`
  # of a nested `var_order: [...]` list. Instead find `plan: [` and scan
  # forward at depth 0 for the matching outer `]`.
  pm = re.search(r"\bplan\s*:\s*\[", body)
  if pm:
    start = pm.end()  # first char after the outer `[`
    depth = 1
    i = start
    while i < len(body) and depth > 0:
      ch = body[i]
      if ch == "[":
        depth += 1
      elif ch == "]":
        depth -= 1
      i += 1
    # `i` now points one past the matching outer `]` (or end-of-string
    # if the source is malformed — we fall through safely below).
    plan_body = body[start : i - 1].strip() if depth == 0 else ""
    entries = _split_top_level_commas(plan_body)
    for e in entries:
      e = e.strip().lstrip("(").rstrip(")").strip()
      if not e:
        continue
      p = Plan()
      if dm := re.search(r"\bdelta\s*:\s*(-?\d+)", e):
        p.delta = int(dm.group(1))
      if vm := re.search(r"\bvar_order\s*:\s*\[([^\]]*)\]", e):
        p.var_order = [v.strip() for v in vm.group(1).split(",") if v.strip()]
      if re.search(r"\bblock_group\s*:\s*true", e):
        p.block_group = True
      if re.search(r"\bwork_stealing\s*:\s*true", e):
        p.work_stealing = True
      if re.search(r"\bdedup_hash\s*:\s*true", e):
        p.dedup_hash = True
      plans.append(p)
  return name, plans, count, semi_join, rule_var_order, inject_cpp


def _parse_rules(block: str) -> list[RuleSpec]:
  rules: list[RuleSpec] = []
  for m in _RULE_OUTER.finditer(block):
    heads_raw, body_raw, pragma_raw = m.groups()
    # Heads: split on top-level commas, each head is `(Rel args)`.
    head_parts = _split_top_level_commas(heads_raw)
    heads: list[BodyAtom] = []
    for hp in head_parts:
      hp = hp.strip()
      if not (hp.startswith("(") and hp.endswith(")")):
        raise ValueError(f"can't parse head: {hp!r}")
      heads.append(_parse_atom(hp[1:-1]))
    # Body: top-level comma split, each part is a body clause.
    body_parts = _split_top_level_commas(body_raw)
    body: list[BodyClauseT] = [_parse_body_clause(p) for p in body_parts]
    # Pragma
    name, plans, count, semi_join, rule_var_order, inject_cpp = (
      "",
      [],
      False,
      False,
      [],
      "",
    )
    if pragma_raw is not None:
      name, plans, count, semi_join, rule_var_order, inject_cpp = _parse_rule_pragma(pragma_raw)
    if not name:
      raise ValueError(f"rule missing name pragma: {m.group(0)[:80]}")
    # If the rule has a top-level var_order (no explicit plan block),
    # synthesize a single Plan with that var_order so codegen picks it up.
    if rule_var_order and not plans:
      plans = [Plan(var_order=rule_var_order)]
    rules.append(
      RuleSpec(
        name=name,
        heads=heads,
        body=body,
        plans=plans,
        count=count,
        semi_join=semi_join,
        var_order=rule_var_order,
        inject_cpp=inject_cpp,
      )
    )
  return rules


# ---------------------------------------------------------------------------
# Python code emission
# ---------------------------------------------------------------------------


def _py_ident(n: str) -> str:
  '''Turn a Nim identifier into a Python-legal one.
  `class` → `class_`, `isStaticMethodInvocation_Insn` → unchanged.'''
  if keyword.iskeyword(n):
    return n + "_"
  return n


def _emit_arg(a: str) -> str:
  '''Emit one atom arg as Python-DSL source:
  - `_`   → `Wild()` (handled below — actually we use `Var('_')`)
  - int literal → literal
  - otherwise a Var reference by identifier.'''
  if a == "_":
    return 'Var("_")'
  if re.match(r"^-?\d+$", a):
    return a
  return _py_ident(a)


def _all_vars(rules: list[RuleSpec], dataset_const_names: set[str] | None = None) -> set[str]:
  '''Collect every unique var identifier across rules. Used to generate
  the `x = Var("x")` block inside `build_*()`.

  `dataset_const_names` is excluded from the result — those get bound
  as `Const(meta[...])` separately, not `Var(...)`. Without this, LOAD
  (etc.) would be double-bound and the `Const` binding would shadow
  silently depending on declaration order.'''
  excluded = dataset_const_names or set()
  names: set[str] = set()

  def collect_atom(a: BodyAtom) -> None:
    for arg in a.args:
      if arg == "_":
        continue
      if re.match(r"^-?\d+$", arg):
        continue
      if arg in excluded:
        continue
      names.add(arg)

  for r in rules:
    for h in r.heads:
      collect_atom(h)
    for c in r.body:
      if isinstance(c, BodyAtom):
        collect_atom(c)
      elif isinstance(c, FilterClause):
        for v in c.vars:
          if v not in excluded:
            names.add(v)
  return names


def _emit_atom(a: BodyAtom) -> str:
  args = ", ".join(_emit_arg(x) for x in a.args)
  rel = _py_ident(a.rel)
  return f"{rel}({args})"


def _emit_plan_kwargs(p: Plan) -> str:
  parts = []
  if p.delta != -1:
    parts.append(f"delta={p.delta}")
  if p.var_order:
    vos = ", ".join(repr(v) for v in p.var_order)
    parts.append(f"var_order=[{vos}]")
  if p.block_group:
    parts.append("block_group=True")
  if p.work_stealing:
    parts.append("work_stealing=True")
  if p.dedup_hash:
    parts.append("dedup_hash=True")
  return ", ".join(parts)


def _emit_body_conj(body: list[BodyClauseT], dataset_const_names: set[str] | None = None) -> str:
  '''Emit a body as `A & B & C` — negations use `~A`, filters use
  `Filter((...,), "...")`, the split marker becomes `SPLIT`.

  Filter C++ bodies can reference dataset_const names (e.g.
  `return x != ABSTRACT;`). With the Const-based build signature, those
  names are bound to local `Const(meta[...])` values; we rewrite the
  filter code into an f-string so `{ABSTRACT.value}` interpolates the
  resolved integer at build-time. Word-boundary match avoids touching
  identifiers that merely contain a const name as a substring.
  '''
  const_names = dataset_const_names or set()
  parts: list[str] = []
  for c in body:
    if isinstance(c, BodyAtom):
      if c.negated:
        parts.append(f"~{_emit_atom(c)}")
      else:
        parts.append(_emit_atom(c))
    elif isinstance(c, FilterClause):
      vs = ", ".join(repr(v) for v in c.vars)
      code_escaped = c.code.replace('\\', '\\\\').replace('"', '\\"').replace("\n", " ")
      uses_const = False
      if const_names:
        # Replace word-bounded occurrences with f-string placeholders.
        pattern = r"\b(" + "|".join(re.escape(n) for n in const_names) + r")\b"

        def _sub(m: re.Match) -> str:
          nonlocal uses_const
          uses_const = True
          return "{" + _py_ident(m.group(1)) + ".value}"

        code_escaped = re.sub(pattern, _sub, code_escaped)
      prefix = "f" if uses_const else ""
      parts.append(f'Filter(({vs},), {prefix}"{code_escaped}")')
    elif c == "__SPLIT__":
      parts.append("SPLIT")
  return " & ".join(parts)


def _emit_rule(r: RuleSpec, dataset_const_names: set[str] | None = None) -> list[str]:
  '''Emit one Python-DSL rule expression.

  Single-head: `(head <= body).named(...)`.
  Multi-head:  `((h0 | h1 | h2) <= body).named(...)` — the DSL `|`
  operator composes atoms into a HeadGroup, producing one Rule whose
  `heads` carries all atoms. Mirrors Nim's `Rule.head: seq[HeadClause]`
  and lowers to one pipeline with N insert-intos.

  `dataset_const_names` is threaded to `_emit_body_conj` so Filter code
  containing const references gets rewritten to an f-string.
  '''
  body_str = _emit_body_conj(r.body, dataset_const_names)
  if len(r.heads) == 1:
    head_str = _emit_atom(r.heads[0])
  else:
    head_str = "(" + " | ".join(_emit_atom(h) for h in r.heads) + ")"
  rule_expr = f"({head_str} <= {body_str}).named({r.name!r})"
  if r.plans:
    for p in r.plans:
      kw = _emit_plan_kwargs(p)
      rule_expr = f"{rule_expr}.with_plan({kw})"
  if r.count:
    rule_expr = f"{rule_expr}.with_count()"
  if r.semi_join:
    rule_expr = f"{rule_expr}.with_semi_join()"
  if r.inject_cpp:
    code = r.inject_cpp.replace('"""', '\\"\\"\\"')
    rule_expr = f'{rule_expr}.with_inject_cpp("""{code}""")'
  return [rule_expr]


def _emit_program(
  schema_name: str,
  relations: list[RelationSpec],
  rules: list[RuleSpec],
  dataset_consts: list[DatasetConst],
  source_file: str,
) -> str:
  out: list[str] = []
  out.append(f'''"""Auto-generated from {source_file} by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py {source_file} --out <this file>
"""''')
  out.append("from __future__ import annotations")
  out.append("")
  # isort order: all-caps (constants) first, then CamelCase, each alphabetical.
  # Keep in sync with ruff's default sort so generated files pass CI lint.
  mixed = ["Filter", "Program", "Relation", "Var"]
  if dataset_consts:
    mixed.insert(0, "Const")
  imports = ["SPLIT"] + mixed
  out.append(f"from srdatalog.dsl import {', '.join(imports)}")
  out.append("")
  out.append("# ----- Relations ----------------------------------------------")
  out.append("")
  for r in relations:
    pyname = _py_ident(r.name)
    types_tuple = ", ".join(r.types)
    kwargs = [f'"{r.name}"', str(len(r.types)), f"column_types=({types_tuple},)"]
    if r.input_file:
      kwargs.append(f'input_file="{r.input_file}"')
    if r.print_size:
      kwargs.append("print_size=True")
    if r.index_type:
      kwargs.append(f'index_type="{r.index_type}"')
    out.append(f"{pyname} = Relation({', '.join(kwargs)})")
  out.append("")
  out.append(f"# ----- Rules: {schema_name} -----")
  out.append("")
  lower = schema_name.lower()
  if dataset_consts:
    out.append(f"def build_{lower}_program(meta: dict[str, int]) -> Program:")
    out.append('  """Build the program, consuming `meta` for dataset_const values.')
    out.append("")
    out.append("  `meta` is a `{json_key: int_value}` dict — typically")
    out.append('  `json.load(open("batik_meta.json"))` or similar. Each declared')
    out.append("  dataset_const binds to a Python-local `Const(meta[key])` at the top")
    out.append("  of this function; any missing key raises KeyError loudly here")
    out.append("  instead of surfacing as silent wrong integers downstream.")
    out.append('  """')
  else:
    out.append(f"def build_{lower}_program() -> Program:")
  # Vars (excluding dataset_const names — those become `Const(meta[...])` bindings).
  const_names_set = {dc.name for dc in dataset_consts}
  all_vars = sorted(_all_vars(rules, const_names_set))
  for v in all_vars:
    py = _py_ident(v)
    out.append(f'  {py} = Var("{v}")')
  # Dataset consts — concrete Python bindings resolved from `meta`.
  sorted_consts = sorted(dataset_consts, key=lambda dc: dc.name)
  if sorted_consts:
    out.append("")
    out.append("  # dataset_consts — Python bindings, resolved from meta.json keys.")
    for dc in sorted_consts:
      py = _py_ident(dc.name)
      out.append(f'  {py} = Const(meta["{dc.key}"])')
  out.append("")
  out.append("  return Program(")
  out.append("    rules=[")
  for r in rules:
    for line in _emit_rule(r, const_names_set):
      out.append(f"      {line},")
  out.append("    ],")
  out.append("  )")
  out.append("")
  return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def translate_file(
  nim_path: Path,
) -> tuple[str, list[RelationSpec], list[RuleSpec], list[DatasetConst]]:
  src = _strip_comments(nim_path.read_text())
  m = re.search(r"^schema\s+(\w+)\s*:", src, flags=re.MULTILINE)
  schema_name = m.group(1) if m else nim_path.stem
  relations = _parse_schema_block(src)
  rules_block = _extract_rules_block(src)
  consts = _parse_dataset_consts(rules_block)
  rules = _parse_rules(rules_block)
  py = _emit_program(schema_name, relations, rules, consts, str(nim_path))
  return py, relations, rules, consts


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("nim_source", help="Path to the Nim source file")
  p.add_argument("--out", "-o", default=None, help="Output .py path (default: stdout)")
  args = p.parse_args(argv)

  py, rels, rules, consts = translate_file(Path(args.nim_source))
  sys.stderr.write(
    f"[nim_to_dsl] {len(rels)} relations, {len(rules)} rules, {len(consts)} dataset_consts\n"
  )
  if args.out:
    Path(args.out).write_text(py)
    sys.stderr.write(f"[nim_to_dsl] wrote {args.out}\n")
  else:
    sys.stdout.write(py)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
