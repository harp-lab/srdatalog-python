'''Python front-end DSL for SRDatalog, replacing the Nim macro DSL in lang.nim/syntax.nim.

Rules are constructed with Python objects and operator overloading:

  X, Y, Z = Var("X"), Var("Y"), Var("Z")
  edge = Relation("Edge", 2)
  path = Relation("Path", 2)

  # Path(X, Y) :- Edge(X, Y)
  r1 = Rule(head=path(X, Y), body=[edge(X, Y)], name="TCBase")

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


def _coerce_arg(x) -> ClauseArg:
  '''Convert a Python value to a ClauseArg. Vars stay vars; ints become consts.'''
  if isinstance(x, Var):
    return x._to_arg()
  if isinstance(x, ClauseArg):
    return x
  if isinstance(x, int):
    return ClauseArg(kind=ArgKind.CONST, const_value=x, const_cpp_expr=str(x))
  raise TypeError(f"Unsupported atom argument: {x!r} (expected Var, int, or ClauseArg)")


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
    return Rule(head=self, body=clauses, name=None)


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
  '''A Datalog rule: `head :- body_1, body_2, ...`.

  `plans` holds user-provided PlanEntry overrides (one per delta position).
  `count` marks a rule as count-only: no materialization, just the cardinality.
  `semi_join` opts the rule into the Pass 1.5 semi-join optimization.
  `is_generated` is True for compiler-synthesised rules (e.g. the
  `_SJ_Target_Filter_...` helpers emitted by semi-join optimization).
  `prov` carries rewrite provenance (user vs compiler-gen) — mirrors
  syntax.nim's `Rule.prov`.
  '''

  head: Atom
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
    return Atom(rel=self.name, args=tuple(_coerce_arg(a) for a in args))

  def __repr__(self) -> str:
    return f"Relation({self.name!r}, arity={self.arity})"


@dataclass
class Program:
  '''A Datalog program: relation decls + rules. Input to the HIR pipeline.'''

  relations: list[Relation] = field(default_factory=list)
  rules: list[Rule] = field(default_factory=list)

  def add(self, *items: Relation | Rule) -> Program:
    for it in items:
      if isinstance(it, Relation):
        self.relations.append(it)
      elif isinstance(it, Rule):
        self.rules.append(it)
      else:
        raise TypeError(f"Program.add: unsupported item {it!r}")
    return self
