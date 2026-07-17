from __future__ import annotations

import csv
import json
from pathlib import Path

from minimal_vulreasoner.example_config import (
  END_TIME,
  INITIAL_NODE,
  KG_FILE,
  WORKFLOW,
)
from minimal_vulreasoner.graphml_ingest import (
  CONNECTOR_PREDICATES,
  emit_minimal_dataset,
  parse_graphml,
  pyreason_attribute,
)
from srdatalog import float32_to_u32


def _rows(path: Path) -> list[list[str]]:
  with path.open(newline="") as handle:
    return list(csv.reader(handle))


def test_pyreason_graph_attribute_compatibility():
  assert pyreason_attribute("can_cause", "1", "long") == (
    "can_cause",
    1.0,
    1.0,
  )
  assert pyreason_attribute("confidence", "0.25", "double") == (
    "confidence",
    0.25,
    1.0,
  )
  assert pyreason_attribute("bounded", "0,1") == ("bounded", 0.0, 1.0)
  assert pyreason_attribute("kind", "CodeEntity") == (
    "kind-CodeEntity",
    1.0,
    1.0,
  )


def test_actual_graphml_extracts_required_connector_edges_in_document_order():
  graph = parse_graphml(KG_FILE)
  facts = {
    (fact.predicate, fact.source, fact.target): fact for fact in graph.connectors
  }

  assert graph.edge_count > 0
  assert set(CONNECTOR_PREDICATES) <= {fact.predicate for fact in graph.connectors}
  assert facts[
    ("contributes_to", "computed_write_length", "incorrect_length_calculation")
  ].lower == 1.0
  assert facts[
    ("can_cause", "incorrect_length_calculation", "return_address_overwrite")
  ].upper == 1.0
  assert (
    "manifestation_of",
    "return_address_overwrite",
    "CWE_121",
  ) in facts
  assert [fact.rank for fact in graph.connectors] == sorted(
    fact.rank for fact in graph.connectors
  )


def test_emit_exact_minimal_extensional_relations(tmp_path):
  out = emit_minimal_dataset(
    KG_FILE,
    tmp_path,
    workflow=WORKFLOW,
    initial_node=INITIAL_NODE,
    end_time=END_TIME,
  )
  symbols = json.loads((out / "symbols.json").read_text())
  manifest = json.loads((out / "manifest.json").read_text())
  one = str(float32_to_u32(1.0))

  assert _rows(out / "analyst_seed.csv") == [
    [str(symbols["b1"]), "0", one, one],
    [str(symbols["b1"]), "1", one, one],
  ]
  assert len(_rows(out / "has_label.csv")) == 4
  assert _rows(out / "step_from.csv") == [
    [str(symbols["b1"]), str(symbols["b2"]), "1"],
    [str(symbols["b1"]), str(symbols["b2"]), "2"],
    [str(symbols["b2"]), str(symbols["b3"]), "2"],
    [str(symbols["b2"]), str(symbols["b3"]), "3"],
    [str(symbols["b3"]), str(symbols["b4"]), "3"],
    [str(symbols["b3"]), str(symbols["b4"]), "4"],
  ]
  assert len(_rows(out / "successor.csv")) == END_TIME
  assert manifest["selected_connector_facts"] == sum(
    len(_rows(out / f"{predicate}.csv")) for predicate in CONNECTOR_PREDICATES
  )
