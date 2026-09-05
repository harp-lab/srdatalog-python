from __future__ import annotations

import pytest

from minimal_vulreasoner.analyst_rule_loader import (
  load_analyst_rules,
  parse_analyst_rule,
)
from minimal_vulreasoner.example_config import RULES_FILE


def test_loads_all_existing_analyst_rules_as_query_specs():
  specs = load_analyst_rules(RULES_FILE)

  assert [spec.connector_predicate for spec in specs] == [
    "can_cause",
    "contributes_to",
    "derives",
    "unsafe_variant_of",
    "manifestation_of",
    "implements",
  ]
  assert all(spec.delay == 1 for spec in specs)
  assert all(spec.annotation_function == "paired_minimum_bounds_ann_fn" for spec in specs)
  assert all(spec.analyst_bound.lower == 0.25 for spec in specs)
  assert all(spec.connector_bound.lower == 0.1 for spec in specs)
  assert specs[0].relation_name == "CanCause"
  assert specs[3].input_file == "unsafe_variant_of.csv"


def test_rejects_a_rule_whose_connector_does_not_pair_labels():
  with pytest.raises(ValueError, match="pair"):
    parse_analyst_rule(
      "analystAt(CB2):paired_minimum_bounds_ann_fn <-1 "
      "analystAt(CB1):[0.25,1], hasLabel(CB1,L1):[0.1,1], "
      "hasLabel(CB2,L2):[0.1,1], can_cause(L2,L1):[0.1,1], "
      "stepFrom(CB1,CB2)",
      "bad",
    )
