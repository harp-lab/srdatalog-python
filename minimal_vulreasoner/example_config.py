'''Source-of-truth inputs shared by the PyReason and SRDatalog runners.'''

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
KG_FILE = HERE / "graphml" / "CWE_121_MVP2.graphml"
RULES_FILE = HERE / "rules" / "analyst_rules.csv"
OUTPUT_DIR = HERE / "output"

# VulReasoner's default horizon.  The example settles after the third hop.
END_TIME = 20

WORKFLOW = (
  ("b1", "computed_write_length"),
  ("b2", "incorrect_length_calculation"),
  ("b3", "return_address_overwrite"),
  ("b4", "CWE_121"),
)
INITIAL_NODE = "b1"
