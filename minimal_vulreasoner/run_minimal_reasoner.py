#!/usr/bin/env python3
"""Minimal, self-contained VulReasoner PyReason example.

  1. reset + load a GraphML knowledge graph            (pr.reset / pr.load_graphml)
  2. the four settings VulReasoner toggles              (pr.settings.*)
  3. register the paired-minimum annotation function    (pr.add_annotation_function)
  4. load one ruleset from CSV                          (pr.add_rule_from_csv)
  5. declare a closed-world predicate                   (pr.add_closed_world_predicate)
  6. add facts: hasLabel / analystAt / stepFrom         (pr.add_fact / pr.Fact)
  7. run the fixpoint                                   (pr.reason)
  8. save the rule trace                                (pr.save_rule_trace)

The example workflow is a 4-block chain whose labels are connected in the KG by
``contributes_to`` -> ``can_cause`` -> ``manifestation_of`` edges. That lets the
analyst rules propagate the ``analystAt`` "control" atom one hop per timestep
from the first block all the way to the ``CWE_121`` vulnerability class:

    b1 (computed_write_length)
      --contributes_to-->  b2 (incorrect_length_calculation)
      --can_cause-->        b3 (return_address_overwrite)
      --manifestation_of--> b4 (CWE_121)

Run it:
    python run_minimal_reasoner.py
"""
from __future__ import annotations

import pyreason as pr

try:
    from .annotation_fn import paired_minimum_bounds_ann_fn
    from .example_config import (
        END_TIME,
        INITIAL_NODE,
        KG_FILE,
        OUTPUT_DIR,
        RULES_FILE,
        WORKFLOW,
    )
except ImportError:
    from annotation_fn import paired_minimum_bounds_ann_fn
    from example_config import (
        END_TIME,
        INITIAL_NODE,
        KG_FILE,
        OUTPUT_DIR,
        RULES_FILE,
        WORKFLOW,
    )

# The annotation function computes the [lower, upper] bound for a fired analyst
# rule by pairing the cause/effect hasLabel bounds.

# The minimal "workflow": a list of (code_block_id, label) pairs. Each label is a
# real node in CWE_121_MVP2.graphml, and consecutive labels are joined by a KG
# connector edge that one analyst rule keys off of.
#
# NOTE: the block ids (b1..b4) are deliberately NOT named CB1/CB2 — those are the
# *rule variable* names inside analyst_rules.csv (and the head vars the annotation
# function looks for), not graph nodes. Keeping them distinct avoids confusion.
def configure_engine() -> None:
    """Steps 1-3 & 5: load the KG, set engine flags, register the ann fn + CWA."""
    # 1. Reset any prior state and load the knowledge graph. The GraphML carries
    #    the label->label ontology edges (can_cause, contributes_to, ...) that the
    #    analyst rules match against.
    pr.reset()
    pr.load_graphml(str(KG_FILE))

    # 2. The four settings VulReasoner enables.
    pr.settings.atom_trace = True                    # record which clauses justified each atom
    pr.settings.allow_ground_rules = True            # treat label constants as ground atoms
    pr.settings.save_graph_attributes_to_trace = True

    # 3. Register the annotation function referenced by the analyst rule heads
    #    (``analystAt(CB2):paired_minimum_bounds_ann_fn <- ...``). It must be
    #    registered before the rules that name it are loaded/run.
    pr.add_annotation_function(paired_minimum_bounds_ann_fn)

    # 5. Closed-world predicate: analystAt is False everywhere it is not proven,
    #    which lets the negative/threshold clauses behave correctly.
    pr.add_closed_world_predicate("analystAt")


def load_rules() -> None:
    """Step 4: load the single ruleset from CSV."""
    pr.add_rule_from_csv(str(RULES_FILE), raise_errors=False)


def add_facts() -> None:
    """Step 6: add the hasLabel, analystAt, and stepFrom facts.

    pr.Fact signature used here mirrors VulReasoner:
        Fact(fact_text, name, start_time, end_time)   # timed fact
        Fact(fact_text, name, static=True)            # holds for all timesteps
    """
    # a) hasLabel(block, label) — static, one per workflow block.
    for block_id, label in WORKFLOW:
        pr.add_fact(pr.Fact(f"hasLabel({block_id},{label})", f"label-{block_id}", static=True))

    # b) analystAt(seed) — the "control" atom the analyst starts with, valid at t=0..1.
    pr.add_fact(pr.Fact(f"analystAt({INITIAL_NODE})", "initial-control", 0, 1))

    # c) stepFrom(src,dst) — the workflow edges. Each edge is asserted one timestep
    #    later than the previous so analystAt propagates exactly one hop per step,
    #    matching VulReasoner's add_workflow_facts ordering.
    for i in range(len(WORKFLOW) - 1):
        src = WORKFLOW[i][0]
        dst = WORKFLOW[i + 1][0]
        pr.add_fact(pr.Fact(f"stepFrom({src},{dst})", f"edge-{i}", i + 1, i + 2))


def main() -> None:
    configure_engine()
    load_rules()
    add_facts()

    # 7. Run the fixpoint.
    interpretation = pr.reason(timesteps=END_TIME)

    # 8. Persist the per-rule trace CSVs (edges/nodes) to ./output.
    OUTPUT_DIR.mkdir(exist_ok=True)
    pr.save_rule_trace(interpretation, str(OUTPUT_DIR))

    # Print where analystAt ended up so the propagation is visible without opening
    # the CSVs. If everything wired up, b1..b4 should all become analystAt=True.
    print(f"Reasoning complete. Rule-trace CSVs written to: {OUTPUT_DIR}")
    print("Expected: analystAt propagates b1 -> b2 -> b3 -> b4 (CWE_121).")
    filtered = interpretation.query(pr.Query("analystAt(b4)"))
    print(f"analystAt(b4) reached: {filtered}")


if __name__ == "__main__":
    main()
