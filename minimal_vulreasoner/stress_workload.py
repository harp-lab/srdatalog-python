"""Deterministic VulReasoner-shaped stress datasets.

The generated relations are shared by the PyReason oracle and the future
SRDatalog implementation.  Logical interval bounds remain floats here.  CSV
export bit-casts each float32 bound to a uint32-compatible integer so the GPU
path can keep lower and upper as separate 32-bit columns without parsing
floating-point text.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if REPO_SRC.exists() and str(REPO_SRC) not in sys.path:
  sys.path.insert(0, str(REPO_SRC))

from srdatalog.value_semantics import float32_to_u32, u32_to_float32

CONNECTOR_PREDICATES = (
  "can_cause",
  "contributes_to",
  "derives",
  "unsafe_variant_of",
  "manifestation_of",
  "implements",
)


@dataclass(frozen=True)
class BoundedNode:
  node: str
  time: int
  lower: float
  upper: float


@dataclass(frozen=True)
class BoundedLabel:
  block: str
  label: str
  lower: float
  upper: float


@dataclass(frozen=True)
class TimedStep:
  source: str
  target: str
  time: int


@dataclass(frozen=True)
class BoundedConnector:
  predicate: str
  source_label: str
  target_label: str
  lower: float
  upper: float


@dataclass(frozen=True)
class StressWorkload:
  depth: int
  width: int
  fanout: int
  seeds: tuple[BoundedNode, ...]
  labels: tuple[BoundedLabel, ...]
  steps: tuple[TimedStep, ...]
  connectors: tuple[BoundedConnector, ...]
  successors: tuple[tuple[int, int], ...]
  target_nodes: tuple[str, ...]

  @property
  def node_count(self) -> int:
    return (self.depth + 1) * self.width

  @property
  def edge_count(self) -> int:
    return len(self.steps)


def encode_float32_bits(value: float) -> int:
  """Bit-cast a [0,1] float to the corresponding unsigned 32-bit integer."""
  return float32_to_u32(value)


def decode_float32_bits(bits: int) -> float:
  return u32_to_float32(bits)


def _node(layer: int, slot: int) -> str:
  return f"b_{layer}_{slot}"


def _label(layer: int, slot: int) -> str:
  return f"l_{layer}_{slot}"


def generate_stress_workload(depth: int, width: int, fanout: int) -> StressWorkload:
  """Build a layered workflow with deterministic bounded fanout.

  All nodes in layer zero are seeds.  A step from layer ``t`` to ``t+1``
  is active exactly at time ``t``; the analyst rules' delay of one therefore
  produces the next layer at time ``t+1``.  Target selection is cyclic, so
  every layer has the same width and the generated edge count is exactly
  ``depth * width * min(width, fanout)``.
  """
  if depth < 1:
    raise ValueError("depth must be at least 1")
  if width < 1:
    raise ValueError("width must be at least 1")
  if fanout < 1:
    raise ValueError("fanout must be at least 1")

  effective_fanout = min(width, fanout)
  labels: list[BoundedLabel] = []
  for layer in range(depth + 1):
    for slot in range(width):
      ordinal = layer * width + slot
      lower = 0.50 + 0.01 * (ordinal % 10)
      upper = 0.90 + 0.01 * (ordinal % 5)
      labels.append(BoundedLabel(_node(layer, slot), _label(layer, slot), lower, upper))

  steps: list[TimedStep] = []
  connectors: list[BoundedConnector] = []
  for layer in range(depth):
    for source_slot in range(width):
      for offset in range(effective_fanout):
        target_slot = (source_slot + offset) % width
        edge_ordinal = len(steps)
        predicate = CONNECTOR_PREDICATES[edge_ordinal % len(CONNECTOR_PREDICATES)]
        connector_lower = 0.45 + 0.01 * (edge_ordinal % 15)
        connector_upper = 0.85 + 0.01 * (edge_ordinal % 10)
        steps.append(
          TimedStep(
            source=_node(layer, source_slot),
            target=_node(layer + 1, target_slot),
            time=layer,
          )
        )
        connectors.append(
          BoundedConnector(
            predicate=predicate,
            source_label=_label(layer, source_slot),
            target_label=_label(layer + 1, target_slot),
            lower=connector_lower,
            upper=connector_upper,
          )
        )

  seeds = tuple(BoundedNode(_node(0, slot), 0, 1.0, 1.0) for slot in range(width))
  targets = tuple(_node(depth, slot) for slot in range(width))
  return StressWorkload(
    depth=depth,
    width=width,
    fanout=effective_fanout,
    seeds=seeds,
    labels=tuple(labels),
    steps=tuple(steps),
    connectors=tuple(connectors),
    successors=tuple((t, t + 1) for t in range(depth)),
    target_nodes=targets,
  )


def _write_csv(path: Path, rows) -> None:
  with path.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)


def emit_csv_dataset(workload: StressWorkload, output_dir: str | Path) -> Path:
  """Emit one common relation-oriented dataset and return its directory."""
  out = Path(output_dir)
  out.mkdir(parents=True, exist_ok=True)

  symbols = sorted(
    {item.node for item in workload.seeds}
    | {item.block for item in workload.labels}
    | {item.label for item in workload.labels}
    | {item.source for item in workload.steps}
    | {item.target for item in workload.steps}
    | {item.source_label for item in workload.connectors}
    | {item.target_label for item in workload.connectors}
    | set(workload.target_nodes)
  )
  symbol_ids = {symbol: idx for idx, symbol in enumerate(symbols)}

  _write_csv(
    out / "analyst_seed.csv",
    (
      (
        symbol_ids[seed.node],
        seed.time,
        encode_float32_bits(seed.lower),
        encode_float32_bits(seed.upper),
      )
      for seed in workload.seeds
    ),
  )
  _write_csv(
    out / "has_label.csv",
    (
      (
        symbol_ids[item.block],
        symbol_ids[item.label],
        encode_float32_bits(item.lower),
        encode_float32_bits(item.upper),
      )
      for item in workload.labels
    ),
  )
  _write_csv(
    out / "step_from.csv",
    (
      (symbol_ids[step.source], symbol_ids[step.target], step.time)
      for step in workload.steps
    ),
  )
  _write_csv(
    out / "successor.csv",
    workload.successors,
  )

  by_predicate: dict[str, list[tuple[int, BoundedConnector]]] = {
    predicate: [] for predicate in CONNECTOR_PREDICATES
  }
  for rank, connector in enumerate(workload.connectors):
    by_predicate[connector.predicate].append((rank, connector))
  for predicate, connectors in by_predicate.items():
    _write_csv(
      out / f"{predicate}.csv",
      (
        (
          symbol_ids[item.source_label],
          symbol_ids[item.target_label],
          rank,
          encode_float32_bits(item.lower),
          encode_float32_bits(item.upper),
        )
        for rank, item in connectors
      ),
    )

  manifest = {
    "depth": workload.depth,
    "width": workload.width,
    "fanout": workload.fanout,
    "node_count": workload.node_count,
    "edge_count": workload.edge_count,
    "target_nodes": workload.target_nodes,
    "target_node_ids": [symbol_ids[node] for node in workload.target_nodes],
    "encoding": "IEEE-754 binary32 bit pattern stored as uint32-compatible integer",
    "headerless": True,
    "schemas": {
      "analyst_seed.csv": ["node", "time", "lower_bits", "upper_bits"],
      "has_label.csv": ["block", "label", "lower_bits", "upper_bits"],
      "step_from.csv": ["source", "target", "time"],
      "successor.csv": ["time", "next_time"],
      **{
        f"{predicate}.csv": [
          "source_label",
          "target_label",
          "rank",
          "lower_bits",
          "upper_bits",
        ]
        for predicate in CONNECTOR_PREDICATES
      },
    },
  }
  (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  (out / "symbols.json").write_text(json.dumps(symbol_ids, indent=2) + "\n")
  return out


def workload_summary(workload: StressWorkload) -> dict[str, object]:
  """JSON-friendly structural summary used by benchmark reports."""
  return {
    "depth": workload.depth,
    "width": workload.width,
    "fanout": workload.fanout,
    "nodes": workload.node_count,
    "steps": len(workload.steps),
    "labels": len(workload.labels),
    "connectors": len(workload.connectors),
    "seeds": len(workload.seeds),
    "targets": len(workload.target_nodes),
  }


def summarize_target_bounds(
  target_bounds: dict[str, tuple[float, float]],
  *,
  sample_size: int = 8,
) -> dict[str, object]:
  '''Return a compact sample plus a digest over every float32 result.'''
  ordered = sorted(target_bounds.items())
  digest = hashlib.sha256()
  for target, (lower, upper) in ordered:
    digest.update(
      f"{target}:{encode_float32_bits(lower)}:{encode_float32_bits(upper)}\n".encode()
    )
  return {
    "target_bounds_sample": dict(ordered[:sample_size]),
    "target_bounds_sha256": digest.hexdigest(),
  }
