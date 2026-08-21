#!/usr/bin/env python3
'''Run one real VulReasoner trajectory's phase-2 fragment on SRDatalog.

Hybrid step-4 runner (see ~/Projects/HANDOFF.md and TRACE_PARITY_CONTRACT.md):
consumes a phase-2 input JSON extracted from a /reason capture
({initial_node, edges, end_time, has_label:[[entity,label,lo,hi],...]}),
the CWE_121_MVP2 GraphML, and the six analyst rules; runs the GPU fixpoint;
prints RESULT_JSON with:
  - analyst_rows: the temporal analystAt map {node@time: [lower, upper]}
  - witnesses: per analyst rule, every satisfying body grounding
    [cb2, next_time, cb1, cause_label, effect_label] (full qualified sets per
    contract Tier 1, not the ARG-MAX winner only)
  - tc_facts: connector facts derived by native SRDatalog transitive closure
    (kg_property_rules semantics: can_cause/contributes_to/derives, [1,1])

Connector transitivity is native SRDatalog TC via per-predicate Closed
relations; base connector facts are asserted [1,1] at ingest (true for this
KG), so TC rows inherit first-hop rank/bounds and still read [1,1].

Usage (from the srdatalog repo root):
  uv run --project minimal_vulreasoner --frozen \
    python minimal_vulreasoner/run_vr_phase2.py --inputs t0.json [--no-compile]
'''

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
  sys.path.insert(0, str(REPO_SRC))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
  sys.path.insert(0, str(HERE))

import csv

from srdatalog import (
  Program,
  Relation,
  Var,
  build_project,
  compile_jit_project,
  float32_to_u32,
  interval_lattice,
  max_lower_lattice,
)
from srdatalog.dsl import Filter, Let
from stress_workload import decode_float32_bits

from benchmark_srdatalog import _bind, _compiler_config
from example_config import KG_FILE, RULES_FILE
from graphml_ingest import parse_graphml
from srdatalog_query import _minimum3
from analyst_rule_loader import load_analyst_rules

import ctypes  # noqa: E402  (after _bind import for parity with peers)

# kg_property_rules.csv transitive predicates that are also analyst connectors.
TRANSITIVE_CONNECTORS = frozenset({"can_cause", "contributes_to", "derives"})


def emit_trajectory_dataset(inputs: dict, output_dir: Path, rule_specs,
                            kg_file=KG_FILE) -> dict:
  '''Emit the extensional relations for one real trajectory.'''
  connector_predicates = tuple(spec.connector_predicate for spec in rule_specs)
  graph = parse_graphml(kg_file, connector_predicates=connector_predicates)
  bad = [c for c in graph.connectors if (c.lower, c.upper) != (1.0, 1.0)]
  if bad:
    raise ValueError(
      f"TC inheritance assumes [1,1] base connector facts; found {bad[:3]}"
    )

  blocks = {inputs["initial_node"]}
  for src, dst in inputs["edges"]:
    blocks.add(src)
    blocks.add(dst)
  labels = {label for _, label, _, _ in inputs["has_label"]}
  entities = {entity for entity, _, _, _ in inputs["has_label"]}
  symbols = sorted(set(graph.nodes) | blocks | labels | entities)
  symbol_ids = {symbol: index for index, symbol in enumerate(symbols)}
  one = float32_to_u32(1.0)

  output_dir.mkdir(parents=True, exist_ok=True)

  def write(name, rows):
    with (output_dir / name).open("w", newline="") as handle:
      csv.writer(handle).writerows(rows)

  # PyReason initial-control Fact is valid on inclusive [0, 1] at [1,1].
  write("analyst_seed.csv",
        ((symbol_ids[inputs["initial_node"]], t, one, one) for t in (0, 1)))
  # Static hasLabel facts with augmenter bounds (timeless = static).
  write("has_label.csv",
        ((symbol_ids[e], symbol_ids[l], float32_to_u32(lo), float32_to_u32(hi))
         for e, l, lo, hi in inputs["has_label"]))
  # PyReason asserts workflow edge i on inclusive times [i+1, i+2].
  step_rows = []
  for index, (src, dst) in enumerate(inputs["edges"]):
    for t in (index + 1, index + 2):
      step_rows.append((symbol_ids[src], symbol_ids[dst], t))
  write("step_from.csv", step_rows)
  write("successor.csv", ((t, t + 1) for t in range(inputs["end_time"])))

  by_predicate = {predicate: [] for predicate in connector_predicates}
  for fact in graph.connectors:
    by_predicate[fact.predicate].append(fact)
  for predicate, facts in by_predicate.items():
    write(f"{predicate}.csv",
          ((symbol_ids[f.source], symbol_ids[f.target], f.rank,
            float32_to_u32(f.lower), float32_to_u32(f.upper)) for f in facts))

  return {
    "symbols": symbol_ids,
    "graph_nodes": len(graph.nodes),
    "graph_edges": graph.edge_count,
    "selected_connector_facts": len(graph.connectors),
    "base_connector_pairs": {
      predicate: sorted({(f.source, f.target) for f in facts})
      for predicate, facts in by_predicate.items()
    },
  }


def build_phase2_program(rule_specs) -> Program:
  '''srdatalog_query.build_analyst_program plus native TC + witness outputs.'''
  if any(spec.delay != 1 for spec in rule_specs):
    raise ValueError("the explicit Successor encoding currently supports only <-1")
  if any(spec.annotation_function != "paired_minimum_bounds_ann_fn"
         for spec in rule_specs):
    raise ValueError("unsupported analyst annotation function")

  analyst_seed = Relation("AnalystSeed", 4, column_types=(int,) * 4,
                          input_file="analyst_seed.csv")
  has_label = Relation("HasLabel", 4, column_types=(int,) * 4,
                       input_file="has_label.csv")
  step_from = Relation("StepFrom", 3, column_types=(int,) * 3,
                       input_file="step_from.csv")
  successor = Relation("Successor", 2, column_types=(int,) * 2,
                       input_file="successor.csv")
  analyst_at = Relation(
    "AnalystAt", 4, column_types=(int,) * 4, print_size=True,
    output_file="analyst_at.csv",
    value_spec=interval_lattice(key_columns=(0, 1), lower_column=2, upper_column=3),
  )

  base_relations = {}
  joined_relations = {}
  rules = []
  tcx, tcy, tcz = Var("tcx"), Var("tcy"), Var("tcz")
  tcr1, tcl1, tcu1 = Var("tcr1"), Var("tcl1"), Var("tcu1")
  tcr2, tcl2, tcu2 = Var("tcr2"), Var("tcl2"), Var("tcu2")
  for spec in rule_specs:
    base = Relation(spec.relation_name, 5, column_types=(int,) * 5,
                    input_file=spec.input_file)
    base_relations[spec.connector_predicate] = base
    if spec.connector_predicate in TRANSITIVE_CONNECTORS:
      closed = Relation(f"{spec.relation_name}Closed", 5, column_types=(int,) * 5,
                        print_size=True)
      rules.append(
        (closed(tcx, tcy, tcr1, tcl1, tcu1)
         <= base(tcx, tcy, tcr1, tcl1, tcu1)).named(f"Copy{spec.relation_name}"))
      # kg_property_rules transitivity; [1,1] inherited from the first hop
      # (asserted at ingest), rank likewise — witness selection is unaffected
      # because every derived interval is identical.
      rules.append(
        (closed(tcx, tcz, tcr1, tcl1, tcu1)
         <= closed(tcx, tcy, tcr1, tcl1, tcu1)
         & closed(tcy, tcz, tcr2, tcl2, tcu2)).named(f"TC{spec.relation_name}"))
      joined_relations[spec.connector_predicate] = closed
    else:
      joined_relations[spec.connector_predicate] = base

  cb1, cb2 = Var("cb1"), Var("cb2")
  time_, next_time = Var("time"), Var("next_time")
  cause_label, effect_label = Var("cause_label"), Var("effect_label")
  analyst_lower, analyst_upper = Var("analyst_lower"), Var("analyst_upper")
  cause_lower, cause_upper = Var("cause_lower"), Var("cause_upper")
  effect_lower, effect_upper = Var("effect_lower"), Var("effect_upper")
  connector_lower, connector_upper = Var("connector_lower"), Var("connector_upper")
  connector_rank = Var("connector_rank")
  result_lower, result_upper = Var("result_lower"), Var("result_upper")

  rules.append(
    (analyst_at(cb1, time_, analyst_lower, analyst_upper)
     <= analyst_seed(cb1, time_, analyst_lower, analyst_upper)).named("AnalystSeed"))

  witness_relations = {}
  for spec in rule_specs:
    connector = joined_relations[spec.connector_predicate]
    candidate = Relation(
      f"{spec.relation_name}Candidate", 5, column_types=(int,) * 5,
      value_spec=max_lower_lattice(key_columns=(0, 1), rank_column=2,
                                   lower_column=3, upper_column=4),
    )
    witness = Relation(f"{spec.relation_name}Witness", 5, column_types=(int,) * 5,
                       print_size=True)
    witness_relations[spec.name] = witness

    def body():
      return (
        analyst_at(cb1, time_, analyst_lower, analyst_upper)
        & has_label(cb1, cause_label, cause_lower, cause_upper)
        & has_label(cb2, effect_label, effect_lower, effect_upper)
        & connector(cause_label, effect_label, connector_rank,
                    connector_lower, connector_upper)
        & step_from(cb1, cb2, time_)
        & successor(time_, next_time)
        & Filter(
          vars=(analyst_lower.name, analyst_upper.name,
                cause_lower.name, cause_upper.name,
                effect_lower.name, effect_upper.name,
                connector_lower.name, connector_upper.name),
          code=(
            f"return {analyst_lower.name} >= {float32_to_u32(spec.analyst_bound.lower)} && "
            f"{analyst_upper.name} <= {float32_to_u32(spec.analyst_bound.upper)} && "
            f"{cause_lower.name} >= {float32_to_u32(spec.cause_label_bound.lower)} && "
            f"{cause_upper.name} <= {float32_to_u32(spec.cause_label_bound.upper)} && "
            f"{effect_lower.name} >= {float32_to_u32(spec.effect_label_bound.lower)} && "
            f"{effect_upper.name} <= {float32_to_u32(spec.effect_label_bound.upper)} && "
            f"{connector_lower.name} >= {float32_to_u32(spec.connector_bound.lower)} && "
            f"{connector_upper.name} <= {float32_to_u32(spec.connector_bound.upper)};"
          ),
        )
        & Let(result_lower.name,
              _minimum3(cause_lower.name, effect_lower.name, connector_lower.name),
              deps=(cause_lower.name, effect_lower.name, connector_lower.name))
        & Let(result_upper.name,
              _minimum3(cause_upper.name, effect_upper.name, connector_upper.name),
              deps=(cause_upper.name, effect_upper.name, connector_upper.name))
      )

    rules.append(
      (candidate(cb2, next_time, connector_rank, result_lower, result_upper)
       <= body()).named(f"Analyst{spec.relation_name}"))
    # Full qualified groundings for contract Tier 1 — every satisfying body
    # assignment, independent of the candidate ARG-MAX selection.
    rules.append(
      (witness(cb2, next_time, cb1, cause_label, effect_label)
       <= body()).named(f"Witness{spec.relation_name}"))
    rules.append(
      (analyst_at(cb2, next_time, result_lower, result_upper)
       <= candidate(cb2, next_time, connector_rank, result_lower, result_upper)
       ).named(f"Promote{spec.relation_name}Candidate"))

  return Program(rules=rules), witness_relations


def _copy_rows(lib, name: str, columns: int):
  import cupy as cp
  count = int(lib.srdatalog_dev_count(name.encode()))
  host = []
  for column in range(columns):
    pointer = int(lib.srdatalog_dev_ptr(name.encode(), column))
    memory = cp.cuda.UnownedMemory(pointer, count * 4, lib)
    device = cp.ndarray((count,), dtype=cp.uint32,
                        memptr=cp.cuda.MemoryPointer(memory, 0))
    host.append(device.get())
  cp.cuda.get_current_stream().synchronize()
  return [tuple(int(host[c][row]) for c in range(columns)) for row in range(count)]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--inputs", required=True, help="phase-2 input JSON")
  parser.add_argument("--kg", default=str(KG_FILE),
                      help="GraphML KG to ingest connectors from (default: the "
                           "minimal example's CWE_121_MVP2 copy)")
  parser.add_argument("--end-time", type=int, default=None,
                      help="override the input JSON's end_time (sizes the "
                           "successor relation)")
  parser.add_argument("--cache-base", default="./build")
  parser.add_argument("--data-dir", default=None)
  parser.add_argument("--jobs", type=int, default=8)
  parser.add_argument("--no-compile", action="store_true")
  parser.add_argument("--repeat", type=int, default=1,
                      help="extra srdatalog_run calls after the first (the "
                           "first pays GPU/RMM init; repeats measure cached "
                           "execution, same convention as benchmark_srdatalog)")
  args = parser.parse_args()

  inputs = json.loads(Path(args.inputs).read_text())
  if args.end_time is not None:
    inputs["end_time"] = args.end_time
  rule_specs = load_analyst_rules(RULES_FILE)
  data_dir = Path(args.data_dir or f"./build/vr_phase2_data_t{inputs.get('trajectory', 'x')}")

  ingest_started = time.perf_counter()
  manifest = emit_trajectory_dataset(inputs, data_dir, rule_specs,
                                     kg_file=Path(args.kg))
  ingest_seconds = time.perf_counter() - ingest_started
  symbol_names = {sid: name for name, sid in manifest["symbols"].items()}

  program, witness_relations = build_phase2_program(rule_specs)
  emit_started = time.perf_counter()
  project = build_project(program, "VulReasonerPhase2Plan", cache_base=args.cache_base)
  emit_seconds = time.perf_counter() - emit_started

  compile_seconds = 0.0
  if args.no_compile:
    artifacts = list(Path(project["dir"]).glob("*.so"))
    if not artifacts:
      raise RuntimeError(f"no cached shared library in {project['dir']}; omit --no-compile")
    artifact = str(artifacts[0].resolve())
  else:
    started = time.perf_counter()
    build = compile_jit_project(project, _compiler_config(args.jobs))
    compile_seconds = time.perf_counter() - started
    if not build.ok():
      failure = next(r for r in build.compile_results if r.returncode)
      raise RuntimeError((failure.stderr or failure.stdout)[-12000:])
    artifact = str(Path(build.artifact).resolve())

  lib = _bind(artifact)
  if lib.srdatalog_init() != 0:
    raise RuntimeError("srdatalog_init failed")
  load_started = time.perf_counter()
  if lib.srdatalog_load_all(str(data_dir.resolve()).encode()) != 0:
    raise RuntimeError("srdatalog_load_all failed")
  load_seconds = time.perf_counter() - load_started
  run_seconds_all = []
  for _ in range(max(1, args.repeat)):
    run_started = time.perf_counter()
    if lib.srdatalog_run(0) != 0:
      raise RuntimeError("srdatalog_run failed")
    run_seconds_all.append(time.perf_counter() - run_started)
  run_seconds = run_seconds_all[0]

  raw = _copy_rows(lib, "AnalystAt", 4)
  analyst_rows = dict(sorted(
    (f"{symbol_names[node]}@{t}", [decode_float32_bits(lo), decode_float32_bits(hi)])
    for node, t, lo, hi in raw))

  witnesses = {}
  for rule_name, relation in witness_relations.items():
    rows = _copy_rows(lib, relation.name, 5)
    witnesses[rule_name] = sorted(
      [symbol_names[cb2], t, symbol_names[cb1],
       symbol_names[cause], symbol_names[effect]]
      for cb2, t, cb1, cause, effect in rows)

  tc_facts = {}
  for spec in rule_specs:
    if spec.connector_predicate not in TRANSITIVE_CONNECTORS:
      continue
    rows = _copy_rows(lib, f"{spec.relation_name}Closed", 5)
    base_pairs = set(map(tuple, manifest["base_connector_pairs"][spec.connector_predicate]))
    derived = sorted(
      {(symbol_names[x], symbol_names[y]) for x, y, _, _, _ in rows}
      - base_pairs)
    tc_facts[spec.connector_predicate] = {
      "derived_pairs": [list(p) for p in derived],
      "derived_bounds": sorted({
        (decode_float32_bits(lo), decode_float32_bits(hi))
        for x, y, _, lo, hi in rows
        if (symbol_names[x], symbol_names[y]) not in base_pairs}),
    }

  result = {
    "engine": "srdatalog",
    "trajectory": inputs.get("trajectory"),
    "graph_nodes": manifest["graph_nodes"],
    "graph_edges": manifest["graph_edges"],
    "selected_connector_facts": manifest["selected_connector_facts"],
    "analyst_rows": analyst_rows,
    "witnesses": witnesses,
    "tc_facts": tc_facts,
    "ingest_seconds": ingest_seconds,
    "emit_seconds": emit_seconds,
    "compile_seconds": compile_seconds,
    "load_seconds": load_seconds,
    "run_seconds": run_seconds,
    "run_seconds_all": run_seconds_all,
  }
  print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
  lib.srdatalog_shutdown()
  sys.stdout.flush()
  sys.stderr.flush()
  # Same ctypes/CUDA teardown-order caveat as run_srdatalog_reasoner.py.
  os._exit(0)


if __name__ == "__main__":
  main()
