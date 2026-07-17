from __future__ import annotations

import csv
import json

import pytest

from minimal_vulreasoner.stress_workload import (
  CONNECTOR_PREDICATES,
  decode_float32_bits,
  emit_csv_dataset,
  encode_float32_bits,
  generate_stress_workload,
  summarize_target_bounds,
)


def test_layered_shape_is_deterministic_and_bounded():
  workload = generate_stress_workload(depth=3, width=4, fanout=2)

  assert workload.node_count == 16
  assert workload.edge_count == 24
  assert len(workload.labels) == 16
  assert len(workload.connectors) == 24
  assert len(workload.seeds) == 4
  assert workload.successors == ((0, 1), (1, 2), (2, 3))
  assert workload.target_nodes == ("b_3_0", "b_3_1", "b_3_2", "b_3_3")
  assert {item.predicate for item in workload.connectors} == set(CONNECTOR_PREDICATES)


def test_fanout_is_capped_at_width_without_duplicate_steps():
  workload = generate_stress_workload(depth=2, width=3, fanout=99)

  assert workload.fanout == 3
  assert workload.edge_count == 18
  assert len(set(workload.steps)) == len(workload.steps)


@pytest.mark.parametrize("value", [0.0, 0.1, 0.25, 0.5, 1.0])
def test_float32_bit_encoding_round_trips(value):
  decoded = decode_float32_bits(encode_float32_bits(value))
  assert decoded == pytest.approx(value, abs=1e-7)


def test_positive_float_bits_preserve_bound_order():
  values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
  assert [encode_float32_bits(v) for v in values] == sorted(encode_float32_bits(v) for v in values)


def test_target_bound_summary_covers_all_results_with_stable_digest():
  bounds = {"b": (0.5, 0.9), "a": (0.25, 1.0)}
  summary = summarize_target_bounds(bounds, sample_size=1)
  assert summary["target_bounds_sample"] == {"a": (0.25, 1.0)}
  assert summary["target_bounds_sha256"] == summarize_target_bounds(bounds)[
    "target_bounds_sha256"
  ]


def test_emit_common_csv_dataset(tmp_path):
  workload = generate_stress_workload(depth=2, width=2, fanout=1)
  out = emit_csv_dataset(workload, tmp_path)

  manifest = json.loads((out / "manifest.json").read_text())
  assert manifest["node_count"] == 6
  assert manifest["edge_count"] == 4

  with (out / "analyst_seed.csv").open(newline="") as f:
    rows = list(csv.reader(f))
  assert len(rows) == 2
  assert int(rows[0][2]) == encode_float32_bits(1.0)
  assert manifest["headerless"] is True
  assert (out / "symbols.json").exists()

  for predicate in CONNECTOR_PREDICATES:
    assert (out / f"{predicate}.csv").exists()


@pytest.mark.parametrize(
  "kwargs",
  [
    {"depth": 0, "width": 1, "fanout": 1},
    {"depth": 1, "width": 0, "fanout": 1},
    {"depth": 1, "width": 1, "fanout": 0},
  ],
)
def test_invalid_shapes_are_rejected(kwargs):
  with pytest.raises(ValueError):
    generate_stress_workload(**kwargs)
