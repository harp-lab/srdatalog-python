'''Parse the constrained PyReason analyst-rule CSV into source-level specs.'''

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

_HEAD = re.compile(
  r"^analystAt\((?P<target>[A-Za-z_]\w*)\):(?P<annotation>[A-Za-z_]\w*)$"
)
_ATOM = re.compile(
  r"(?P<predicate>[A-Za-z_]\w*)\((?P<arguments>[^)]*)\)"
  r"(?:\:\[(?P<lower>[0-9.]+),(?P<upper>[0-9.]+)\])?"
)


@dataclass(frozen=True)
class Bound:
  lower: float
  upper: float

  def __post_init__(self) -> None:
    if not 0.0 <= self.lower <= self.upper <= 1.0:
      raise ValueError(f"invalid probability interval [{self.lower},{self.upper}]")


@dataclass(frozen=True)
class AnalystRuleSpec:
  name: str
  connector_predicate: str
  annotation_function: str
  delay: int
  analyst_bound: Bound
  cause_label_bound: Bound
  effect_label_bound: Bound
  connector_bound: Bound

  @property
  def relation_name(self) -> str:
    return "".join(part.title() for part in self.connector_predicate.split("_"))

  @property
  def input_file(self) -> str:
    return f"{self.connector_predicate}.csv"


@dataclass(frozen=True)
class _AtomSpec:
  predicate: str
  arguments: tuple[str, ...]
  bound: Bound | None


def _atom_specs(body: str) -> tuple[_AtomSpec, ...]:
  atoms = []
  for match in _ATOM.finditer(body):
    lower = match.group("lower")
    upper = match.group("upper")
    bound = None if lower is None else Bound(float(lower), float(upper))
    atoms.append(
      _AtomSpec(
        match.group("predicate"),
        tuple(arg.strip() for arg in match.group("arguments").split(",")),
        bound,
      )
    )
  return tuple(atoms)


def parse_analyst_rule(rule_text: str, name: str) -> AnalystRuleSpec:
  '''Parse and validate the one connector-rule shape supported by this query.'''
  pieces = re.split(r"\s+<-(\d+)\s+", rule_text.strip(), maxsplit=1)
  if len(pieces) != 3:
    raise ValueError(f"{name}: expected a delayed '<-N' rule")
  head_text, delay_text, body_text = pieces
  head = _HEAD.fullmatch(head_text)
  if head is None or head.group("target") != "CB2":
    raise ValueError(f"{name}: expected analystAt(CB2):annotation head")

  atoms = _atom_specs(body_text)
  analyst = [atom for atom in atoms if atom.predicate == "analystAt"]
  labels = [atom for atom in atoms if atom.predicate == "hasLabel"]
  steps = [atom for atom in atoms if atom.predicate == "stepFrom"]
  connectors = [
    atom
    for atom in atoms
    if atom.predicate not in {"analystAt", "hasLabel", "stepFrom"}
  ]
  if len(analyst) != 1 or len(labels) != 2 or len(steps) != 1 or len(connectors) != 1:
    raise ValueError(
      f"{name}: expected analystAt, two hasLabel clauses, one connector, and stepFrom"
    )
  if analyst[0].arguments != ("CB1",) or steps[0].arguments != ("CB1", "CB2"):
    raise ValueError(f"{name}: unsupported analystAt/stepFrom variables")
  label_by_block = {atom.arguments[0]: atom for atom in labels}
  if set(label_by_block) != {"CB1", "CB2"}:
    raise ValueError(f"{name}: hasLabel clauses must be keyed by CB1 and CB2")
  connector = connectors[0]
  expected_connector_args = (
    label_by_block["CB1"].arguments[1],
    label_by_block["CB2"].arguments[1],
  )
  if connector.arguments != expected_connector_args:
    raise ValueError(f"{name}: connector variables do not pair the hasLabel clauses")
  bounded = (analyst[0], label_by_block["CB1"], label_by_block["CB2"], connector)
  if any(atom.bound is None for atom in bounded):
    raise ValueError(f"{name}: every bounded body atom requires [lower,upper]")

  return AnalystRuleSpec(
    name=name,
    connector_predicate=connector.predicate,
    annotation_function=head.group("annotation"),
    delay=int(delay_text),
    analyst_bound=analyst[0].bound,
    cause_label_bound=label_by_block["CB1"].bound,
    effect_label_bound=label_by_block["CB2"].bound,
    connector_bound=connector.bound,
  )


def load_analyst_rules(path: str | Path) -> tuple[AnalystRuleSpec, ...]:
  with Path(path).open(newline="") as handle:
    rows = csv.DictReader(handle)
    required = {"rule_text", "name", "infer_edges", "set_static"}
    if rows.fieldnames is None or not required <= set(rows.fieldnames):
      raise ValueError(f"analyst rule CSV requires columns {sorted(required)}")
    specs = []
    for row in rows:
      if row["infer_edges"].strip().lower() != "true":
        raise ValueError(f"{row['name']}: analyst connector rule must infer edges")
      if row["set_static"].strip().lower() != "false":
        raise ValueError(f"{row['name']}: delayed analyst result must not be static")
      specs.append(parse_analyst_rule(row["rule_text"], row["name"]))
  if not specs:
    raise ValueError(f"analyst rule CSV contains no rules: {path}")
  predicates = [spec.connector_predicate for spec in specs]
  if len(predicates) != len(set(predicates)):
    raise ValueError("analyst rule CSV contains duplicate connector predicates")
  return tuple(specs)
