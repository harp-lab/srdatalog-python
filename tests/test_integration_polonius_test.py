'''polonius_test.nim -- borrow-checker analysis (~40 rules, 2-level indexes).

The Program builder is taken from the auto-generated example so the
`index_type="SRDatalog::GPU::Device2LevelIndex"` pragmas flow through.
A hand-rolled builder that omits those kwargs silently undercounts
NumSources on FULL_VER reads of 2-level-indexed relations and diverges
from the Nim golden at the JIT-runner layer.
'''

import sys
from pathlib import Path

from integration_helpers import diff_hir, diff_mir

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from polonius_test import build_poloniusdb_program as build_polonius_test


def test_polonius_test_hir():
  diff_hir(build_polonius_test(), "polonius_test")


def test_polonius_test_mir():
  diff_mir(build_polonius_test(), "polonius_test")
