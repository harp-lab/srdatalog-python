'''Stream GraphML into the relation files consumed by the analyst query.

GraphML is an ingestion format, not part of Datalog semantics.  This adapter
matches the subset of PyReason's GraphML attribute interpretation used by the
minimal VulReasoner example and emits deterministic, headerless relation CSVs.
'''

from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if REPO_SRC.exists() and str(REPO_SRC) not in sys.path:
  sys.path.insert(0, str(REPO_SRC))

from srdatalog.value_semantics import float32_to_u32

CONNECTOR_PREDICATES = (
  "can_cause",
  "contributes_to",
  "derives",
  "unsafe_variant_of",
  "manifestation_of",
  "implements",
)


@dataclass(frozen=True)
class GraphMLKey:
  name: str
  domain: str
  value_type: str


@dataclass(frozen=True)
class ConnectorFact:
  predicate: str
  source: str
  target: str
  rank: int
  lower: float
  upper: float


@dataclass(frozen=True)
class ParsedGraphML:
  nodes: tuple[str, ...]
  connectors: tuple[ConnectorFact, ...]
  edge_count: int


def _local_name(tag: str) -> str:
  return tag.rsplit("}", 1)[-1]


def _typed_value(text: str, value_type: str):
  if value_type in {"int", "long"}:
    return int(text)
  if value_type in {"float", "double"}:
    return float(text)
  if value_type == "boolean":
    normalized = text.strip().lower()
    if normalized not in {"true", "false", "1", "0"}:
      raise ValueError(f"invalid GraphML boolean {text!r}")
    return normalized in {"true", "1"}
  return text


def pyreason_attribute(
  name: str,
  text: str,
  value_type: str = "string",
) -> tuple[str, float, float]:
  '''Return PyReason's logical label and interval for one GraphML attribute.'''
  value = _typed_value(text.strip(), value_type)
  numeric = isinstance(value, (int, float)) and 0 <= value <= 1
  numeric_string = (
    isinstance(value, str)
    and value.replace(".", "").isdigit()
    and 0 <= float(value) <= 1
  )
  if numeric or numeric_string:
    label = name
    lower = float(value)
    upper = 1.0
  else:
    label = f"{name}-{value}"
    lower = 1.0
    upper = 1.0

  if isinstance(value, str):
    pieces = value.split(",")
    if len(pieces) == 2:
      try:
        candidate_lower = int(pieces[0])
        candidate_upper = int(pieces[1])
      except (TypeError, ValueError):
        pass
      else:
        if 0 <= candidate_lower <= 1 and 0 <= candidate_upper <= 1:
          label = name
          lower = float(candidate_lower)
          upper = float(candidate_upper)
  return label, lower, upper


def parse_graphml(
  path: str | Path,
  *,
  connector_predicates: Iterable[str] = CONNECTOR_PREDICATES,
) -> ParsedGraphML:
  '''Stream selected edge attributes from a directed GraphML document.'''
  selected = frozenset(connector_predicates)
  keys: dict[str, GraphMLKey] = {}
  nodes: list[str] = []
  connectors: list[ConnectorFact] = []
  edge_rank = 0

  for event, elem in ET.iterparse(path, events=("start", "end")):
    kind = _local_name(elem.tag)
    if event == "start" and kind == "graph":
      if elem.attrib.get("edgedefault", "directed") != "directed":
        raise ValueError("minimal VulReasoner requires directed GraphML")
      continue
    if event != "end":
      continue
    if kind == "key":
      key_id = elem.attrib.get("id")
      name = elem.attrib.get("attr.name", key_id)
      if key_id is None or name is None:
        raise ValueError("GraphML key requires id and attr.name")
      keys[key_id] = GraphMLKey(
        name=name,
        domain=elem.attrib.get("for", "all"),
        value_type=elem.attrib.get("attr.type", "string"),
      )
      elem.clear()
    elif kind == "node":
      node_id = elem.attrib.get("id")
      if node_id is None:
        raise ValueError("GraphML node requires id")
      nodes.append(node_id)
      elem.clear()
    elif kind == "edge":
      source = elem.attrib.get("source")
      target = elem.attrib.get("target")
      if source is None or target is None:
        raise ValueError("GraphML edge requires source and target")
      for data in elem:
        if _local_name(data.tag) != "data":
          continue
        key_id = data.attrib.get("key")
        if key_id not in keys:
          raise ValueError(f"GraphML edge references unknown key {key_id!r}")
        key = keys[key_id]
        if key.domain not in {"edge", "all"}:
          continue
        label, lower, upper = pyreason_attribute(
          key.name,
          data.text or "",
          key.value_type,
        )
        if label in selected:
          connectors.append(
            ConnectorFact(label, source, target, edge_rank, lower, upper)
          )
      edge_rank += 1
      elem.clear()

  if not nodes:
    raise ValueError(f"GraphML document contains no nodes: {path}")
  return ParsedGraphML(tuple(nodes), tuple(connectors), edge_rank)


def _write_csv(path: Path, rows) -> None:
  with path.open("w", newline="") as handle:
    csv.writer(handle).writerows(rows)


def emit_minimal_dataset(
  graphml_path: str | Path,
  output_dir: str | Path,
  *,
  workflow: Iterable[tuple[str, str]],
  initial_node: str,
  end_time: int,
  connector_predicates: Iterable[str] = CONNECTOR_PREDICATES,
) -> Path:
  '''Emit the exact extensional relations used by the minimal example.'''
  if end_time < 1:
    raise ValueError("end_time must be positive")
  workflow = tuple(workflow)
  if len(workflow) < 2:
    raise ValueError("workflow must contain at least two blocks")
  block_ids = [block for block, _ in workflow]
  if len(set(block_ids)) != len(block_ids):
    raise ValueError("workflow block ids must be unique")
  if initial_node not in block_ids:
    raise ValueError(f"initial node {initial_node!r} is not in the workflow")

  connector_predicates = tuple(connector_predicates)
  if not connector_predicates or len(set(connector_predicates)) != len(
    connector_predicates
  ):
    raise ValueError("connector predicates must be non-empty and unique")
  graph = parse_graphml(
    graphml_path,
    connector_predicates=connector_predicates,
  )
  graph_nodes = set(graph.nodes)
  missing_labels = sorted(label for _, label in workflow if label not in graph_nodes)
  if missing_labels:
    raise ValueError(f"workflow labels missing from GraphML: {missing_labels}")

  symbols = sorted(graph_nodes | set(block_ids))
  symbol_ids = {symbol: index for index, symbol in enumerate(symbols)}
  one = float32_to_u32(1.0)
  out = Path(output_dir)
  out.mkdir(parents=True, exist_ok=True)

  # The PyReason seed Fact is valid on the inclusive interval [0, 1].
  _write_csv(
    out / "analyst_seed.csv",
    ((symbol_ids[initial_node], time, one, one) for time in (0, 1)),
  )
  _write_csv(
    out / "has_label.csv",
    (
      (symbol_ids[block], symbol_ids[label], one, one)
      for block, label in workflow
    ),
  )

  # PyReason asserts workflow edge i on inclusive times [i+1, i+2].
  step_rows = []
  for index, ((source, _), (target, _)) in enumerate(pairwise(workflow)):
    for time in (index + 1, index + 2):
      step_rows.append((symbol_ids[source], symbol_ids[target], time))
  _write_csv(out / "step_from.csv", step_rows)
  _write_csv(out / "successor.csv", ((time, time + 1) for time in range(end_time)))

  by_predicate: dict[str, list[ConnectorFact]] = {
    predicate: [] for predicate in connector_predicates
  }
  for connector in graph.connectors:
    by_predicate[connector.predicate].append(connector)
  for predicate, facts in by_predicate.items():
    _write_csv(
      out / f"{predicate}.csv",
      (
        (
          symbol_ids[fact.source],
          symbol_ids[fact.target],
          fact.rank,
          float32_to_u32(fact.lower),
          float32_to_u32(fact.upper),
        )
        for fact in facts
      ),
    )

  manifest = {
    "source_graphml": str(Path(graphml_path).resolve()),
    "graph_nodes": len(graph.nodes),
    "graph_edges": graph.edge_count,
    "selected_connector_facts": len(graph.connectors),
    "connector_predicates": connector_predicates,
    "workflow": workflow,
    "initial_node": initial_node,
    "end_time": end_time,
    "encoding": "IEEE-754 binary32 bit pattern stored as uint32-compatible integer",
    "edge_rank": "zero-based GraphML document edge order",
    "fact_lifetimes": {
      "analyst_seed": [0, 1],
      "workflow_step_i": "inclusive [i+1, i+2]",
    },
  }
  (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  (out / "symbols.json").write_text(json.dumps(symbol_ids, indent=2) + "\n")
  return out
