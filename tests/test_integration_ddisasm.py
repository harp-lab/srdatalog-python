'''ddisasm — auto-translated from upstream Nim ddisasm.nim via tools/nim_to_dsl.py.

The Python program (examples/ddisasm.py) is structurally validated against
the Nim source by tools/validate_translation.py. The JIT runner goldens
in `tests/fixtures/jit/ddisasm/jit_runner.<rule>.cpp` were extracted
verbatim from `~/.cache/nim/jit/DdisasmPlan_C1DE/jit_batch_*.cpp` (the
Nim toolchain's authoritative emit).

The HIR / MIR goldens in `tests/fixtures/integration/ddisasm.{hir.json,
mir.sexpr}` are self-generated from the Python compile — they catch
HIR/MIR regressions but do not gate against Nim. (The HIR/MIR layers
have been independently validated equivalent to Nim via galen / doop /
polonius_test fixtures; ddisasm goldens for those layers were not
captured upstream.)
'''

import json
import sys
from pathlib import Path

from integration_helpers import FIXTURES, diff_hir, diff_mir

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from ddisasm import build_ddisasmdb_program


def build_ddisasm():
  meta = json.load((FIXTURES / "ddisasm_meta.json").open())
  return build_ddisasmdb_program(meta)


def test_ddisasm_hir():
  diff_hir(build_ddisasm(), "ddisasm")


def test_ddisasm_mir():
  diff_mir(build_ddisasm(), "ddisasm")


if __name__ == "__main__":
  test_ddisasm_hir()
  test_ddisasm_mir()
  print("ddisasm: OK")
