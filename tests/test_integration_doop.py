'''doop_bg_2level.nim -- multi-head rules + dataset_consts.

The `meta` dict is loaded from batik_meta.json (same file used on the
Nim side at fixture-generation time) so the `Const(meta[...])` bindings
inside build_doopdb_program resolve to byte-identical integer literals.
'''

import json
import sys
from pathlib import Path

from integration_helpers import FIXTURES, diff_hir, diff_mir

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from doop import build_doopdb_program


def build_doop():
  meta = json.load((FIXTURES / "doop_meta.json").open())
  return build_doopdb_program(meta)


def test_doop_hir():
  diff_hir(build_doop(), "doop")


def test_doop_mir():
  diff_mir(build_doop(), "doop")


if __name__ == "__main__":
  test_doop_hir()
  test_doop_mir()
  print("doop: OK")
