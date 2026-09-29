'''crdt.nim -- negation, inline filters, anonymous vars, recursive strata'''

import sys
from pathlib import Path

from integration_helpers import check_mir_lifetimes, diff_hir

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from crdt import build_crdtdb_program


def test_crdt_hir():
  diff_hir(build_crdtdb_program(), "crdt")


def test_crdt_mir():
  check_mir_lifetimes(build_crdtdb_program())


if __name__ == "__main__":
  test_crdt_hir()
  test_crdt_mir()
  print("crdt: OK")
