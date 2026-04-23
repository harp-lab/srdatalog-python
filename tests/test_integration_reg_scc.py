'''reg_scc.nim -- 10 rules, 13 wildcards, plan arrays.

Builder imported from the auto-generated example so `index_type=` pragmas
on any 2-level-indexed relation flow through to gen_complete_runner.
Hand-rolled builders are easy to leave incomplete and drift from Nim.
'''

import sys
from pathlib import Path

from integration_helpers import diff_hir, diff_mir

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from reg_scc import build_regsccdb_program as build_reg_scc


def test_reg_scc_hir():
  diff_hir(build_reg_scc(), "reg_scc")


def test_reg_scc_mir():
  diff_mir(build_reg_scc(), "reg_scc")
