"""DOOP compiler lifetime checks using synthetic, distinct symbol IDs.

The metadata fixture supplies constants only; it is not paired with a
dataset or a generated-code snapshot.
"""

import json
import sys
from pathlib import Path

from integration_helpers import FIXTURES, check_mir_lifetimes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from doop import build_doopdb_program


def build_doop():
  meta = json.load((FIXTURES / "doop_meta.json").open())
  return build_doopdb_program(meta)


def test_doop_mir():
  check_mir_lifetimes(build_doop())


if __name__ == "__main__":
  test_doop_mir()
  print("doop: OK")
