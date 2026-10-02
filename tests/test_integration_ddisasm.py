'''ddisasm — auto-translated from upstream Nim ddisasm.nim via tools/nim_to_dsl.py.

The Python program (examples/ddisasm.py) is structurally validated against
the Nim source by tools/validate_translation.py. The JIT runner goldens
in `tests/fixtures/jit/ddisasm/jit_runner.<rule>.cpp` were extracted
verbatim from `~/.cache/nim/jit/DdisasmPlan_C1DE/jit_batch_*.cpp` (the
Nim toolchain's authoritative emit).

'''

import json
import sys
from pathlib import Path

import pytest
from integration_helpers import FIXTURES, check_mir_lifetimes

from srdatalog import build_project

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from ddisasm import build_ddisasmdb_program


def build_ddisasm():
  meta = json.load((FIXTURES / "ddisasm_meta.json").open())
  return build_ddisasmdb_program(meta)


def test_ddisasm_mir():
  check_mir_lifetimes(build_ddisasm())


@pytest.mark.parametrize("layout", ["split", "sharded", "unity"])
def test_ddisasm_dedup_type_defined_before_use(tmp_path, layout):
  project = build_project(
    build_ddisasm(),
    "DdisasmPlan",
    cache_base=str(tmp_path),
    shard_step_bodies=layout == "sharded",
    unity=layout == "unity",
  )
  checked = 0
  for path in [project["main"], *project["batches"]]:
    cpp = Path(path).read_text()
    if "DedupTable dedup_table{};" not in cpp:
      continue
    assert cpp.count("struct DedupTable {") == 1, path
    assert cpp.index("struct DedupTable {") < cpp.index("DedupTable dedup_table{};"), path
    checked += 1
  assert checked > 0


if __name__ == "__main__":
  test_ddisasm_mir()
  print("ddisasm: OK")
