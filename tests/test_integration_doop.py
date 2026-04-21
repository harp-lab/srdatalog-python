'''doop_bg_2level.nim -- multi-head rules + dataset_consts.

Uses tests/fixtures/integration/doop_meta.json to resolve dataset_const
values. The same meta.json was used on the Nim side to generate the
goldens (via SRDATALOG_DATASET_CONFIG=<path> srdatalog_plan ...), so
the resolved int literals line up byte-for-byte.
'''

import sys
from pathlib import Path

from integration_helpers import FIXTURES, diff_hir, diff_mir

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from doop import build_doopdb  # noqa: E402


def build_doop():
  prog, _ = build_doopdb(str(FIXTURES / "doop_meta.json"))
  return prog


def test_doop_hir():
  diff_hir(build_doop(), "doop")


def test_doop_mir():
  diff_mir(build_doop(), "doop")


if __name__ == "__main__":
  test_doop_hir()
  test_doop_mir()
  print("doop: OK")
