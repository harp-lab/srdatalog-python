'''Python front-end DSL for SRDatalog, replacing the Nim macro DSL in lang.nim/syntax.nim.

Rules are constructed with Python objects and operator overloading:

  X, Y, Z = Var("X"), Var("Y"), Var("Z")
  edge = Relation("Edge", 2)
  path = Relation("Path", 2)

  # Path(X, Y) :- Edge(X, Y)
  r1 = Rule(heads=(path(X, Y),), body=[edge(X, Y)], name="TCBase")

  # Path(X, Z) :- Path(X, Y), Edge(Y, Z)
  r2 = (path(X, Z) <= path(X, Y) & edge(Y, Z)).named("TCRec")

This module defines only the DSL surface; lowering to HIR is in hir_passes.py (TBD).
'''

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Union

from srdatalog.provenance import USER_PROVENANCE, Provenance


class ArgKind(Enum):
  '''Mirrors Nim ClauseArgKind in syntax.nim.'''

  LVAR = "var"
  CONST = "const"
  CPP_CODE = "code"


@dataclass(frozen=True)
class ClauseArg:
  '''An argument slot in an atom: a logic var, a compile-time constant, or raw C++ code.'''

  kind: ArgKind
  # Exactly one of the following is populated:
  var_name: str | None = None  # LVAR
  const_value: int | None = None  # CONST
  const_cpp_expr: str | None = None  # CONST (C++ representation)
  cpp_code: str | None = None  # CPP_CODE


class Var:
  '''A logic variable. Distinct from Python values; used by operator overloads to build AST.'''

  __slots__ = ("name",)

  def __init__(self, name: str):
    self.name = name

  def __repr__(self) -> str:
    return f"Var({self.name!r})"

  def _to_arg(self) -> ClauseArg:
    return ClauseArg(kind=ArgKind.LVAR, var_name=self.name)


class Const:
  '''A compile-time constant argument wrapping a Python value.

  Prefer this over bare `int` arguments when you want the intent explicit
  at the call site — e.g., `Method_Modifier(Const(abstract_id), meth)`
  instead of `Method_Modifier(abstract_id, meth)` where `abstract_id` is
  a Python int that readers can't tell apart from a pure-Python value.

  For dataset-resolved constants (read from a meta.json at program
  construction time), this is the recommended shape:

      meta = load_meta("batik_meta.json")
      ABSTRACT = Const(meta["abstract"])   # Python binding, value baked in
      Method_Modifier(ABSTRACT, meth)

  `cpp_expr` overrides the auto-derived C++ literal. For `int` it
  defaults to `str(value)`. Other types require an explicit `cpp_expr`
  until we need them.
  '''

  __slots__ = ("cpp_expr", "value")

  def __init__(self, value, cpp_expr: str | None = None):
    self.value = value
    if cpp_expr is not None:
      self.cpp_expr = cpp_expr
    elif isinstance(value, int) and not isinstance(value, bool):
      self.cpp_expr = str(value)
    else:
      raise TypeError(
        f"Const({value!r}): cpp_expr required for non-int values (type {type(value).__name__})"
      )

  def __repr__(self) -> str:
    return f"Const({self.value!r})"

  def _to_arg(self) -> ClauseArg:
    return ClauseArg(kind=ArgKind.CONST, const_value=self.value, const_cpp_expr=self.cpp_expr)


def _coerce_arg(x) -> ClauseArg:
  '''Convert a Python value to a ClauseArg.

  Accepts: Var, Const, a bare int (short-hand for `Const(int)`), or a
  pre-built ClauseArg. Anything else raises — prefer `Const(...)` or
  `Var(...)` over relying on implicit coercion.
  '''
  if isinstance(x, Var):
    return x._to_arg()
  if isinstance(x, Const):
    return x._to_arg()
  if isinstance(x, ClauseArg):
    return x
  if isinstance(x, int) and not isinstance(x, bool):
    return ClauseArg(kind=ArgKind.CONST, const_value=x, const_cpp_expr=str(x))
  raise TypeError(f"Unsupported atom argument: {x!r} (expected Var, Const, int, or ClauseArg)")


def cpp(code: str) -> ClauseArg:
  '''Raw C++ code as a clause argument (rare; mirrors the `$"..."` Nim syntax).'''
  return ClauseArg(kind=ArgKind.CPP_CODE, cpp_code=code)


@dataclass(frozen=True)
class Atom:
  '''A relation application, used as head or body clause.

  Build via `Relation.__call__`, never directly. Supports `&` to chain into a body
  conjunction and `<=` to form a rule with this atom as head.

  `prov` carries rewrite provenance: set by passes like semi-join
  optimization when a rewritten body clause is emitted in place of
  the original. Defaults to user-written.
  '''

  rel: str
  args: tuple[ClauseArg, ...]
  prov: Provenance = USER_PROVENANCE
  # Back-reference to the Relation this atom was built from. Set by
  # Relation.__call__; None for Atoms hand-constructed from just a name
  # string (rewrite passes that don't have the Relation in scope). Used
  # by Program to auto-derive its relations list from rules, so users
  # no longer have to pass relations= in parallel with rules= and risk
  # drift. Not part of equality / hash: two atoms are equal iff they
  # have the same rel name and args, regardless of how they were built.
  relation: Relation | None = field(default=None, compare=False, repr=False)

  def __and__(self, other) -> Conjunction:
    '''Compose with Atom / Negation / Filter / Let / Conjunction.'''
    if isinstance(other, Conjunction):
      return Conjunction((self,) + other.clauses)
    return Conjunction((self, other))

  def __invert__(self) -> Negation:
    '''`~atom` = negation.'''
    return Negation(self)

  def __le__(self, body) -> Rule:
    '''`head <= body` → Rule. Anonymous; call `.named(name)` to label.
    `body` can be any BodyClauseT or a Conjunction of them.
    '''
    clauses = body.clauses if isinstance(body, Conjunction) else (body,)
    return Rule(heads=(self,), body=clauses, name=None)

  def __or__(self, other: Atom | HeadGroup) -> HeadGroup:
    '''Compose atoms into a multi-head group: `A | B | C <= body`.'''
    if isinstance(other, HeadGroup):
      return HeadGroup((self,) + other.atoms)
    return HeadGroup((self, other))


@dataclass(frozen=True)
class HeadGroup:
  '''Intermediate: accumulates head atoms under `|`. Mirrors Nim's
  `{(A args), (B args)} <-- body` multi-head rule form.
  '''

  atoms: tuple[Atom, ...]

  def __or__(self, other: Atom | HeadGroup) -> HeadGroup:
    extra = other.atoms if isinstance(other, HeadGroup) else (other,)
    return HeadGroup(self.atoms + extra)

  def __le__(self, body) -> Rule:
    clauses = body.clauses if isinstance(body, Conjunction) else (body,)
    return Rule(heads=self.atoms, body=clauses, name=None)


@dataclass(frozen=True)
class Negation:
  '''Negated atom (`~rel(...)`). Appears only in rule bodies.'''

  atom: Atom

  def __and__(self, other: BodyClauseT | Conjunction) -> Conjunction:
    if isinstance(other, Conjunction):
      return Conjunction((self,) + other.clauses)
    return Conjunction((self, other))


@dataclass(frozen=True)
class Filter:
  '''Inline filter — `return <cpp_code>` against bound vars. Mostly produced
  by the constant-rewriting pass (where e.g. `R(1, x)` becomes `R(_c0, x)`
  + `Filter((_c0,), "return _c0 == 1;")`), but available in the surface
  DSL too.
  '''

  vars: tuple[str, ...]
  code: str

  def __and__(self, other: BodyClauseT | Conjunction) -> Conjunction:
    if isinstance(other, Conjunction):
      return Conjunction((self,) + other.clauses)
    return Conjunction((self, other))


@dataclass(frozen=True)
class Let:
  '''Bind a fresh variable to a C++ expression. Produced by the head-
  constant-rewriting pass when a head has literal args; the head arg is
  replaced by a fresh variable and a corresponding `Let` is appended to
  the body (so the fresh variable is bound before InsertInto reads it).
  '''

  var_name: str
  code: str
  deps: tuple[str, ...] = ()

  def __and__(self, other: BodyClauseT | Conjunction) -> Conjunction:
    if isinstance(other, Conjunction):
      return Conjunction((self,) + other.clauses)
    return Conjunction((self, other))


@dataclass(frozen=True)
class Agg:
  '''Aggregation body clause. Binds `result_var` to the aggregate of
  `rel(args...)` using `func` (C++ aggregator name; "agg" + `cpp_type`
  for custom aggregators, mirrors Nim's AggClause).

  Example: count of R(x, y) bound to `c`:
      agg(c, "count", r(x, y))

  Nim's HIR emits these into JSON as `{"kind": "aggregation", ...}` but
  its lowering pipeline does not construct moAggregate nodes from
  AggClause (zero such constructions in src/srdatalog). Python mirrors
  that behavior: Agg round-trips through HIR but does not appear in MIR.
  '''

  result_var: str
  func: str
  rel: str
  args: tuple[ClauseArg, ...]
  cpp_type: str = ""
  # Back-reference to the aggregated Relation, so Program can auto-
  # derive its decls list. Mirrors Atom.relation — populated by the
  # `agg()` helper from its rel_atom argument.
  relation: Relation | None = field(default=None, compare=False, repr=False)

  def __and__(self, other: BodyClauseT | Conjunction) -> Conjunction:
    if isinstance(other, Conjunction):
      return Conjunction((self,) + other.clauses)
    return Conjunction((self, other))


@dataclass(frozen=True)
class Split:
  '''Split marker — partitions a rule body into above-split and below-split
  sections. Mirrors Nim's SplitClause (`split` keyword).

  Pipeline A writes the above-split output to a temp relation;
  Pipeline B scans the temp and joins with below-split clauses to
  produce the head. Useful for negation pushdown / selective join
  evaluation. At most one Split per rule body.
  '''

  def __and__(self, other):
    if isinstance(other, Conjunction):
      return Conjunction((self,) + other.clauses)
    return Conjunction((self, other))


SPLIT = Split()


BodyClauseT = Union[Atom, Negation, Filter, Let, Agg, Split]


def agg(result_var, func: str, rel_atom: Atom, cpp_type: str = "") -> Agg:
  '''Build an aggregation body clause.

  `result_var` may be a Var instance or a bare string var name.
  `rel_atom` is the output of `Relation(...)(...)` — its rel + args
  become the aggregation's relation reference.
  '''
  name = result_var.name if isinstance(result_var, Var) else result_var
  return Agg(
    result_var=name,
    func=func,
    rel=rel_atom.rel,
    args=rel_atom.args,
    cpp_type=cpp_type,
    relation=rel_atom.relation,
  )


def count(result_var, rel_atom: Atom) -> Agg:
  '''Convenience: count(v, R(x, y)) → (v = count(R(x, y))).'''
  return agg(result_var, "count", rel_atom)


def sum(result_var, rel_atom: Atom) -> Agg:
  return agg(result_var, "sum", rel_atom)


@dataclass(frozen=True)
class Conjunction:
  '''Intermediate: accumulates body clauses under `&`. Not emitted directly.'''

  clauses: tuple[BodyClauseT, ...]

  def __and__(self, other: BodyClauseT | Conjunction) -> Conjunction:
    extra = other.clauses if isinstance(other, Conjunction) else (other,)
    return Conjunction(self.clauses + extra)


@dataclass(frozen=True)
class PlanEntry:
  '''User-specified plan for a rule variant. Mirrors PlanEntry in syntax.nim.

  `delta == -1` targets the base (non-recursive) variant; otherwise it is
  the body-clause index used as the delta seed for semi-naive evaluation.
  `var_order` and `clause_order` override the default planning heuristic;
  when only `var_order` is given, `clause_order` is derived from it.

  The pragma flags flow through to HirRuleVariant so codegen sees them:
    - fanout          -> fan-out work-stealing for Cartesian products
    - work_stealing   -> mid-level work-stealing (task queue + steal loop)
    - block_group     -> block-group work partitioning
    - dedup_hash      -> GPU hash table for in-kernel existential dedup
  `balanced_root` / `balanced_sources` drive balanced partitioning for
  skewed joins (not yet lowered in Python).
  '''

  delta: int = -1
  var_order: tuple[str, ...] = ()
  clause_order: tuple[int, ...] = ()
  fanout: bool = False
  work_stealing: bool = False
  block_group: bool = False
  dedup_hash: bool = False
  balanced_root: tuple[str, ...] = ()
  balanced_sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class Rule:
  '''A Datalog rule: `head_1, head_2, ... :- body_1, body_2, ...`.

  `heads` is always a tuple of one or more Atoms (mirrors Nim's
  `Rule.head: seq[HeadClause]`). Build multi-head rules with
  `(A | B | C) <= body`; single-head still reads `A <= body`.

  `plans` holds user-provided PlanEntry overrides (one per delta position).
  `count` marks a rule as count-only: no materialization, just the cardinality.
  `semi_join` opts the rule into the Pass 1.5 semi-join optimization.
  `is_generated` is True for compiler-synthesised rules (e.g. the
  `_SJ_Target_Filter_...` helpers emitted by semi-join optimization).
  `prov` carries rewrite provenance (user vs compiler-gen) — mirrors
  syntax.nim's `Rule.prov`.
  '''

  heads: tuple[Atom, ...]
  body: tuple[BodyClauseT, ...]
  name: str | None = None
  plans: tuple[PlanEntry, ...] = ()
  count: bool = False
  semi_join: bool = False
  is_generated: bool = False
  prov: Provenance = USER_PROVENANCE
  # `debug_code` carries the `inject_cpp: "..."` rule pragma. When non-empty,
  # the lowering pass emits an InjectCppHook MIR node per variant (after
  # pipelines, before maintenance).
  debug_code: str = ""

  @property
  def head(self) -> Atom:
    '''First head (convenience for single-head rules). For multi-head,
    iterate `self.heads`.'''
    return self.heads[0]

  def named(self, name: str) -> Rule:
    return dataclasses.replace(self, name=name)

  def with_plan(
    self,
    *,
    delta: int = -1,
    var_order: tuple[str, ...] | list[str] | None = None,
    clause_order: tuple[int, ...] | list[int] | None = None,
    fanout: bool = False,
    work_stealing: bool = False,
    block_group: bool = False,
    dedup_hash: bool = False,
    balanced_root: tuple[str, ...] | list[str] | None = None,
    balanced_sources: tuple[str, ...] | list[str] | None = None,
  ) -> Rule:
    '''Append a single PlanEntry. Can be called multiple times to add
    entries for different deltas (or use .with_plans(entries) to replace).
    '''
    entry = PlanEntry(
      delta=delta,
      var_order=tuple(var_order) if var_order is not None else (),
      clause_order=tuple(clause_order) if clause_order is not None else (),
      fanout=fanout,
      work_stealing=work_stealing,
      block_group=block_group,
      dedup_hash=dedup_hash,
      balanced_root=tuple(balanced_root) if balanced_root is not None else (),
      balanced_sources=tuple(balanced_sources) if balanced_sources is not None else (),
    )
    return dataclasses.replace(self, plans=self.plans + (entry,))

  def with_plans(self, entries: list[PlanEntry] | tuple[PlanEntry, ...]) -> Rule:
    '''Replace all plans with the given sequence.'''
    return dataclasses.replace(self, plans=tuple(entries))

  def with_count(self) -> Rule:
    '''Mark this rule as count-only.'''
    return dataclasses.replace(self, count=True)

  def with_semi_join(self) -> Rule:
    '''Opt into semi-join optimization (Pass 1.5). Ignored on rules with
    <= 2 body clauses (the pass skips them per Nim's semantics).
    '''
    return dataclasses.replace(self, semi_join=True)

  def with_inject_cpp(self, code: str) -> Rule:
    '''Attach a C++ debug hook to be emitted as an InjectCppHook MIR node
    once per variant (after the rule's pipeline runs). Mirrors Nim's
    `inject_cpp: "..."` rule pragma.
    '''
    return dataclasses.replace(self, debug_code=code)


class Relation:
  '''A relation declaration. Callable to build atoms.

  Arity + column_types are structural metadata.
  Pragma fields (all optional) mirror Nim's Relation[...] pragmas:
    - input_file   → CSV the load-data block reads into this relation
    - print_size   → runner emits a size-readback line after the fixpoint
    - output_file  → runner writes the final contents to this path
    - index_type   → C++ index template (e.g. "SRDatalog::GPU::Device2LevelIndex")
    - semiring     → override "NoProvenance" (rare — provenance semirings)
  '''

  __slots__ = (
    "arity",
    "column_types",
    "index_type",
    "input_file",
    "name",
    "output_file",
    "print_size",
    "semiring",
  )

  def __init__(
    self,
    name: str,
    arity: int,
    column_types: tuple[type, ...] | None = None,
    *,
    input_file: str = "",
    print_size: bool = False,
    output_file: str = "",
    index_type: str = "",
    semiring: str = "NoProvenance",
  ):
    self.name = name
    self.arity = arity
    self.column_types = column_types or tuple([int] * arity)
    self.input_file = input_file
    self.print_size = print_size
    self.output_file = output_file
    self.index_type = index_type
    self.semiring = semiring

  def __call__(self, *args) -> Atom:
    if len(args) != self.arity:
      raise ValueError(f"{self.name} expects arity {self.arity}, got {len(args)}")
    return Atom(
      rel=self.name,
      args=tuple(_coerce_arg(a) for a in args),
      relation=self,
    )

  def __repr__(self) -> str:
    return f"Relation({self.name!r}, arity={self.arity})"


@dataclass
class Program:
  '''A Datalog program. Takes rules; the relations list is derived from
  them via the Relation back-ref on each Atom.

  The previous API took `relations=[...]` in parallel with `rules=[...]`.
  That was a pure bug generator — if a relation was declared but never
  used, or used but never declared, the downstream passes silently
  generated wrong code. With the derived list, the schema is exactly the
  set of relations referenced by some rule, in rule-first-occurrence
  order (heads before body, body in source order). This matches the
  Nim-side normalization in hir.nim:normalizeDecls and keeps byte-match
  across the two ports.
  '''

  rules: list[Rule] = field(default_factory=list)
  relations: list[Relation] = field(init=False)

  def __post_init__(self) -> None:
    self.relations = _derive_relations(self.rules)

  def add(self, *items: Rule) -> Program:
    for it in items:
      if isinstance(it, Rule):
        self.rules.append(it)
      else:
        raise TypeError(f"Program.add: unsupported item {it!r}")
    self.relations = _derive_relations(self.rules)
    return self

  def _repr_mimebundle_(self, include: object = None, exclude: object = None) -> dict[str, object]:
    '''Jupyter display hook.

    Returns a dict mapping mime type → payload. Jupyter / IPython picks
    the richest renderer available for the mime types present.

    We emit two:
      - application/vnd.srdatalog.viz+json — the visualization bundle.
        A Jupyter labextension or VS Code webview registers a renderer
        for this mime type; without one, Jupyter falls back to text/plain.
      - text/plain — a one-line summary so the cell isn't blank in
        non-visualizing UIs (terminal IPython, plain `print(prog)`).

    Jupyter default omits the JIT C++ block — on doop that's the
    difference between a 300 KB and a 3 MB cell output, which matters
    when re-running cells. Use `prog.show(include_jit=True)` to
    explicitly request kernels.

    `include` / `exclude` follow the IPython display protocol — when
    provided, restrict / suppress entries from the returned dict.
    '''
    # Local import to avoid cycles (viz.bundle imports pipeline which
    # imports dsl indirectly via hir).
    from srdatalog.viz.bundle import get_visualization_bundle

    bundle = get_visualization_bundle(self, include_jit=False)
    out = {
      "application/vnd.srdatalog.viz+json": bundle,
      "text/plain": (f"<Program: {len(self.relations)} relation(s), {len(self.rules)} rule(s)>"),
    }
    if include:
      out = {k: v for k, v in out.items() if k in include}
    if exclude:
      out = {k: v for k, v in out.items() if k not in exclude}
    return out

  def show(self, *, include_jit: bool = True) -> None:
    '''Render this program in Jupyter, optionally with the JIT block.

    The default `_repr_mimebundle_` (triggered by leaving `prog` as the
    last expression of a cell) emits a JIT-less bundle for speed. Call
    `prog.show()` when you want the full bundle including per-rule
    generated C++ kernels — typically when you're inspecting codegen
    rather than just iterating on the rule structure.

    Requires IPython (only meaningful inside Jupyter / IPython).
    '''
    from srdatalog.viz.bundle import get_visualization_bundle

    try:
      from IPython.display import publish_display_data
    except ImportError as e:
      raise RuntimeError("Program.show() requires IPython") from e
    bundle = get_visualization_bundle(self, include_jit=include_jit)
    publish_display_data(
      {
        "application/vnd.srdatalog.viz+json": bundle,
        "text/plain": (
          f"<Program: {len(self.relations)} relation(s), {len(self.rules)} rule(s)"
          f", jit={'on' if include_jit else 'off'}>"
        ),
      }
    )


def _derive_relations(rules: list[Rule]) -> list[Relation]:
  '''Walk rules in order, yield each Relation the first time it appears.

  Order: for each rule, heads (declaration order) then body clauses
  (source order, unwrapping Negation/Agg). Atoms without a Relation
  back-ref (legacy hand-constructed or produced by rewrite passes that
  lack the Relation in scope) are skipped — those only show up after
  HIR transforms, never in user-authored top-level programs.
  '''
  out: list[Relation] = []
  seen: set[str] = set()

  def take(rel: Relation | None) -> None:
    if rel is None or rel.name in seen:
      return
    seen.add(rel.name)
    out.append(rel)

  for rule in rules:
    for h in rule.heads:
      take(h.relation)
    for clause in rule.body:
      if isinstance(clause, Atom):
        take(clause.relation)
      elif isinstance(clause, Negation):
        take(clause.atom.relation)
      elif isinstance(clause, Agg):
        take(clause.relation)
  return out
