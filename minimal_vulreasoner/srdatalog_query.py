'''SRDatalog encoding of VulReasoner's provenance-aware analyst aggregate.

In database terms, each satisfied PyReason rule body is a join result and a
derivation witness: one set of input tuples that jointly supports an
``AnalystAt`` head.  ``paired_minimum_bounds_ann_fn`` is a rule-local, grouped
``ARG MAX`` over those why-provenance candidates.  It computes one interval per
witness, selects the witness with greatest lower bound, and carries that same
witness's upper bound.  Connector rank makes PyReason's first-witness tie
policy explicit; it is a tie-break key, not evidence strength.

Each ``*Candidate`` relation materializes the grouped-aggregate boundary for
one source rule.  The connector join inserts all witness rows into
``Candidate.NEW``; keyed maintenance selects the winner and places only a new
or changed winner in ``Candidate.DELTA``.  The promotion rule is only a
projection into ``AnalystAt``--it performs no selection.  ``AnalystAt`` then
uses interval intersection to combine the already-selected winners from
different rules.  Direct insertion of raw witnesses into ``AnalystAt`` would
incorrectly intersect losing witnesses.

This reproduces a provenance-aware selection policy but does not materialize
complete semiring provenance: the parity query retains the selected witness's
rank and value, not its full derivation polynomial.  A head ``ARG MAX`` could
remove the named candidate relations in a future DSL while preserving the same
logical grouping boundary.

PyReason's ``<-1`` is an ordinary logical-time shift here:
``Successor(t,t1)`` joins an ``AnalystAt`` state at ``t`` to the head at ``t1``.
Bounds are positive IEEE-754 float32 bit patterns in integer columns, so
integer comparison and ``std::min`` preserve probability order over ``[0,1]``.
'''

from __future__ import annotations

from srdatalog import (
  Program,
  Relation,
  Var,
  float32_to_u32,
  interval_lattice,
  max_lower_lattice,
)
from srdatalog.dsl import Filter, Let

try:
  from .analyst_rule_loader import AnalystRuleSpec, load_analyst_rules
  from .example_config import RULES_FILE
except ImportError:
  from analyst_rule_loader import AnalystRuleSpec, load_analyst_rules
  from example_config import RULES_FILE

RULE_SPECS = load_analyst_rules(RULES_FILE)
CONNECTORS = tuple((spec.relation_name, spec.input_file) for spec in RULE_SPECS)


def _minimum3(left: str, middle: str, right: str) -> str:
  return f"std::min(std::min({left}, {middle}), {right})"


def build_analyst_program(
  rule_specs: tuple[AnalystRuleSpec, ...] = RULE_SPECS,
) -> Program:
  if any(spec.delay != 1 for spec in rule_specs):
    raise ValueError("the explicit Successor encoding currently supports only <-1")
  if any(
    spec.annotation_function != "paired_minimum_bounds_ann_fn"
    for spec in rule_specs
  ):
    raise ValueError("unsupported analyst annotation function")
  analyst_seed = Relation(
    "AnalystSeed",
    4,
    column_types=(int, int, int, int),
    input_file="analyst_seed.csv",
  )
  has_label = Relation(
    "HasLabel",
    4,
    column_types=(int, int, int, int),
    input_file="has_label.csv",
  )
  step_from = Relation(
    "StepFrom",
    3,
    column_types=(int, int, int),
    input_file="step_from.csv",
  )
  successor = Relation(
    "Successor",
    2,
    column_types=(int, int),
    input_file="successor.csv",
  )
  analyst_at = Relation(
    "AnalystAt",
    4,
    column_types=(int, int, int, int),
    print_size=True,
    output_file="analyst_at.csv",
    value_spec=interval_lattice(
      key_columns=(0, 1),
      lower_column=2,
      upper_column=3,
    ),
  )
  connectors = tuple((spec.relation_name, spec.input_file) for spec in rule_specs)
  connector_relations = [
    Relation(
      name,
      5,
      column_types=(int, int, int, int, int),
      input_file=filename,
    )
    for name, filename in connectors
  ]
  # Materialized state for one rule-local grouped ARG MAX.  These are not six
  # additional logical inference steps; they preserve which join witness owns
  # the interval selected by the PyReason annotation callback.
  candidate_relations = [
    Relation(
      f"{name}Candidate",
      5,
      column_types=(int, int, int, int, int),
      value_spec=max_lower_lattice(
        key_columns=(0, 1),
        rank_column=2,
        lower_column=3,
        upper_column=4,
      ),
    )
    for name, _ in connectors
  ]

  cb1, cb2 = Var("cb1"), Var("cb2")
  time, next_time = Var("time"), Var("next_time")
  cause_label, effect_label = Var("cause_label"), Var("effect_label")
  analyst_lower, analyst_upper = Var("analyst_lower"), Var("analyst_upper")
  cause_lower, cause_upper = Var("cause_lower"), Var("cause_upper")
  effect_lower, effect_upper = Var("effect_lower"), Var("effect_upper")
  connector_lower, connector_upper = Var("connector_lower"), Var("connector_upper")
  connector_rank = Var("connector_rank")
  result_lower, result_upper = Var("result_lower"), Var("result_upper")

  rules = [
    (
      analyst_at(cb1, time, analyst_lower, analyst_upper)
      <= analyst_seed(cb1, time, analyst_lower, analyst_upper)
    ).named("AnalystSeed")
  ]

  for connector, candidate, rule_spec in zip(
    connector_relations,
    candidate_relations,
    rule_specs,
  ):
    analyst_lower_threshold = float32_to_u32(rule_spec.analyst_bound.lower)
    analyst_upper_threshold = float32_to_u32(rule_spec.analyst_bound.upper)
    cause_lower_threshold = float32_to_u32(rule_spec.cause_label_bound.lower)
    cause_upper_threshold = float32_to_u32(rule_spec.cause_label_bound.upper)
    effect_lower_threshold = float32_to_u32(rule_spec.effect_label_bound.lower)
    effect_upper_threshold = float32_to_u32(rule_spec.effect_label_bound.upper)
    connector_lower_threshold = float32_to_u32(rule_spec.connector_bound.lower)
    connector_upper_threshold = float32_to_u32(rule_spec.connector_bound.upper)
    body = (
      analyst_at(cb1, time, analyst_lower, analyst_upper)
      & has_label(cb1, cause_label, cause_lower, cause_upper)
      & has_label(cb2, effect_label, effect_lower, effect_upper)
      & connector(
        cause_label,
        effect_label,
        connector_rank,
        connector_lower,
        connector_upper,
      )
      & step_from(cb1, cb2, time)
      & successor(time, next_time)
      & Filter(
        vars=(
          analyst_lower.name,
          analyst_upper.name,
          cause_lower.name,
          cause_upper.name,
          effect_lower.name,
          effect_upper.name,
          connector_lower.name,
          connector_upper.name,
        ),
        code=(
          f"return {analyst_lower.name} >= {analyst_lower_threshold} && "
          f"{analyst_upper.name} <= {analyst_upper_threshold} && "
          f"{cause_lower.name} >= {cause_lower_threshold} && "
          f"{cause_upper.name} <= {cause_upper_threshold} && "
          f"{effect_lower.name} >= {effect_lower_threshold} && "
          f"{effect_upper.name} <= {effect_upper_threshold} && "
          f"{connector_lower.name} >= {connector_lower_threshold} && "
          f"{connector_upper.name} <= {connector_upper_threshold};"
        ),
      )
      & Let(
        result_lower.name,
        _minimum3(cause_lower.name, effect_lower.name, connector_lower.name),
        deps=(cause_lower.name, effect_lower.name, connector_lower.name),
      )
      & Let(
        result_upper.name,
        _minimum3(cause_upper.name, effect_upper.name, connector_upper.name),
        deps=(cause_upper.name, effect_upper.name, connector_upper.name),
      )
    )
    # The connector join emits raw derivation witnesses.  Candidate relation
    # maintenance, not this join pipeline, performs the keyed ARG MAX.
    rules.append(
      (
        candidate(cb2, next_time, connector_rank, result_lower, result_upper) <= body
      ).named(f"Analyst{connector.name}")
    )
    # This is a projection/copy of the already-selected witness.  Insertion
    # into AnalystAt applies the distinct cross-rule interval-intersection join.
    rules.append(
      (
        analyst_at(cb2, next_time, result_lower, result_upper)
        <= candidate(cb2, next_time, connector_rank, result_lower, result_upper)
      ).named(f"Promote{connector.name}Candidate")
    )

  return Program(rules=rules)


PROGRAM = build_analyst_program()
