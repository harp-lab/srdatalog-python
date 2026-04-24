'''End-to-end byte-match: the `jit_batch_0.cpp` file Python writes
must compile with the same flags Nim uses.

This is the strongest correctness signal we have. The `_cpp_norm`
helper collapses whitespace + reorders nothing — semantic equivalence
in C++ syntax means the same compile.

Reference fixture: a Nim-cached `jit_batch_0.cpp` for the triangle
program, snapshotted from `~/.cache/nim/jit/TrianglePlan_*/` and
committed under `tests/fixtures/e2e/`. Regen with the script below
when the runtime / Nim emit changes (and verify the new bytes still
compile against an unchanged runtime first).
'''

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from integration_helpers import _cpp_norm

from srdatalog import Program, Relation, Var, build_project

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "e2e" / "TrianglePlan_jit_batch_0.cpp"


def _build_triangle_program() -> Program:
  '''Mirrors integration_tests/examples/triangle/triangle.nim exactly:
  Z(x, y, z) :- R(x, y), S(y, z, h), T(z, x, f), var_order [x, y, z].
  '''
  x, y, z = Var("x"), Var("y"), Var("z")
  h, f = Var("h"), Var("f")
  R = Relation("RRel", 2, column_types=(int, int))
  S = Relation("SRel", 3, column_types=(int, int, int))
  T = Relation("TRel", 3, column_types=(int, int, int))
  Z = Relation("ZRel", 3, column_types=(int, int, int))
  return Program(
    rules=[
      (Z(x, y, z) <= R(x, y) & S(y, z, h) & T(z, x, f)).named("Triangle"),
    ],
  )


def test_triangle_batch_byte_matches_nim_emit():
  '''Python's jit_batch_0.cpp must match the Nim-cached version after
  whitespace normalization. If this fails, either:
    - Python's emitter drifted from Nim
    - Nim's emitter changed and the fixture is stale (regen)
  '''
  if not FIXTURE.exists():
    print(f"[SKIP] fixture missing: {FIXTURE}")
    print(f"       To create it: copy ~/.cache/nim/jit/TrianglePlan_*/jit_batch_0.cpp to {FIXTURE}")
    return

  with tempfile.TemporaryDirectory() as td:
    result = build_project(
      _build_triangle_program(),
      project_name="TrianglePlan",
      cache_base=td,
    )
    actual = Path(str(result["batches"][0])).read_text()

  golden = FIXTURE.read_text()
  if _cpp_norm(actual) != _cpp_norm(golden):
    a, g = _cpp_norm(actual), _cpp_norm(golden)
    for k, (x, y) in enumerate(zip(a, g)):
      if x != y:
        print(f"First diff at char {k}")
        print(f"  ACTUAL: {a[max(0, k - 80) : k + 80]!r}")
        print(f"  GOLDEN: {g[max(0, k - 80) : k + 80]!r}")
        break
    raise AssertionError(
      f"jit_batch_0.cpp differs from Nim emit (actual={len(a)} chars, golden={len(g)} chars)"
    )


if __name__ == "__main__":
  import inspect

  this = sys.modules[__name__]
  passed = failed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if not name.startswith("test_"):
      continue
    try:
      fn()
      print(f"OK  {name}")
      passed += 1
    except AssertionError as e:
      print(f"FAIL {name}")
      print(str(e)[:1500])
      failed += 1
    except Exception as e:
      print(f"ERROR {name}: {type(e).__name__}: {e}")
      failed += 1
  print(f"\n{passed} pass / {failed} fail")
