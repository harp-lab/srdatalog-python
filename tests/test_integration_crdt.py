'''crdt.nim -- negation, inline filters, anonymous vars, recursive strata'''

import sys
from pathlib import Path

from integration_helpers import diff_hir, diff_mir

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from crdt import build_crdtdb_program  # noqa: E402


def test_crdt_hir():
  diff_hir(build_crdtdb_program(), "crdt")


def test_crdt_mir():
  diff_mir(build_crdtdb_program(), "crdt")


if __name__ == "__main__":
  test_crdt_hir()
  test_crdt_mir()
  print("crdt: OK")
