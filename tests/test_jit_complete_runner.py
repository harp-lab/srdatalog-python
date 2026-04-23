'''Byte-match tests for codegen/jit/complete_runner.py.

Phase 2 baseline: covers non-recursive, non-BG, non-WS, non-dedup_hash
pipelines with kernel_count + kernel_materialize + kernel_fused +
all phase methods (setup, launch_count, scan_and_resize, scan_only,
read_total, launch_materialize, launch_fused, read_fused_result) +
execute + execute_fused.

First target: gen_jit_code/Triangle — simplest fixture (no tiled
Cartesian, single ColumnJoin-only pipeline). Second: triangle/Triangle
which DOES have a tiled Cartesian leaf (3-level).
'''

import sys
from pathlib import Path

from integration_helpers import diff_jit_runner

from srdatalog.codegen.batchfile import _collect_pipelines
from srdatalog.codegen.jit.complete_runner import gen_complete_runner
from srdatalog.codegen.jit.orchestrator_jit import gen_step_body
from srdatalog.hir import compile_to_mir


def _python_jit_runner(prog, rule_name: str, db_type_name: str) -> str:
  '''Emit the `full` output of gen_complete_runner for one rule — this
  is what Nim's jitCompleteRunner returns as `full` and what the
  srdatalog_plan fixture-dump tool writes to jit_runner.<rule>.cpp.

  We run gen_step_body for every step first — orchestrator processing
  mutates ExecutePipeline nodes with `concurrent_write=True` when
  multiple rules in a ParallelGroup share a destination. The
  complete_runner disables tiled Cartesian on concurrent-write rules,
  so the flag must be set before runner emission.
  '''
  mir = compile_to_mir(prog)
  for i, (step, is_rec) in enumerate(mir.steps):
    gen_step_body(step, db_type_name, is_rec, i)
  for ep in _collect_pipelines(mir):
    if ep.rule_name == rule_name:
      _decl, full = gen_complete_runner(ep, db_type_name)
      return full
  raise ValueError(f"no pipeline named {rule_name!r}")


# -----------------------------------------------------------------------------
# gen_jit_code/Triangle — simplest baseline (no tiled Cartesian)
# -----------------------------------------------------------------------------


def test_gen_jit_code_triangle_runner_byte_match():
  from test_integration_gen_jit_code import build_gen_jit_code

  actual = _python_jit_runner(
    build_gen_jit_code(),
    "Triangle",
    "TriangleRules_DB_DeviceDB",
  )
  diff_jit_runner("gen_jit_code", "Triangle", actual)


# -----------------------------------------------------------------------------
# triangle/Triangle — has tiled Cartesian leaf (3-level)
# -----------------------------------------------------------------------------


def test_triangle_triangle_runner_byte_match():
  from test_integration_triangle import build_triangle

  actual = _python_jit_runner(
    build_triangle(),
    "Triangle",
    "TriangleRules_DB_DeviceDB",
  )
  diff_jit_runner("triangle", "Triangle", actual)


# -----------------------------------------------------------------------------
# Sweep: every fixture whose runner we can emit. Skips BG/WS/dedup-hash
# pipelines (still NotImplementedError) plus the 3 polonius_test rules with
# negation-pre-narrow tiled fallback — future phases will close those.
# -----------------------------------------------------------------------------


def _build_fn_for(stem: str):
  mod = __import__(f"test_integration_{stem}")
  for name in dir(mod):
    if name.startswith("build_"):
      return getattr(mod, name)
  return None


def _db_name_from_fixture(stem: str, rule: str) -> str | None:
  import re as _re
  from pathlib import Path as _P

  path = _P(__file__).resolve().parent / "fixtures" / "jit" / stem / f"jit_runner.{rule}.cpp"
  if not path.exists():
    return None
  m = _re.search(r"using DB = (\w+);", path.read_text())
  return m.group(1) if m else None


# Fixtures where our current port disagrees with the Nim golden — one
# entry per (stem, rule) until the relevant feature branch lands.
RUNNER_BYTE_MATCH_SKIPS = {
  # Work-stealing runner variants — Phase 5 port.
  ("lsqb_q6_nosj", "TwoHopPath"),
  ("polonius_test", "subset_trans_D0"),
}


def test_all_runner_fixtures_byte_match():
  '''Sweep: emit every per-fixture jit_runner.<rule>.cpp via
  gen_complete_runner and byte-match against the Nim golden. Skipped
  entries are listed in RUNNER_BYTE_MATCH_SKIPS with the feature-branch
  gap each represents.'''

  from srdatalog.codegen.batchfile import _collect_pipelines
  from srdatalog.hir import compile_to_mir

  fixture_root = Path(__file__).resolve().parent / "fixtures" / "jit"
  errors: list[str] = []
  checked = 0

  for stem_dir in sorted(fixture_root.iterdir()):
    if not stem_dir.is_dir():
      continue
    stem = stem_dir.name
    try:
      build_fn = _build_fn_for(stem)
    except Exception:
      continue
    if build_fn is None:
      continue
    try:
      prog = build_fn()
      mir = compile_to_mir(prog)
    except Exception as e:
      errors.append(f"{stem}: MIR build failed: {e}")
      continue
    # Build the same rel_index_types map that build.py:95 assembles for the
    # live compile, so gen_complete_runner's pluginViewCount resolves custom
    # index types (e.g., SRDatalog::GPU::Device2LevelIndex) — without this
    # the runner falls back to the default plugin's viewCount=1 everywhere
    # and diverges from the Nim golden by 1 view per FULL_VER reference on
    # any 2-level-indexed relation.
    rel_index_types = {r.name: r.index_type for r in prog.relations if getattr(r, "index_type", "")}
    db_name = None
    for ep in _collect_pipelines(mir):
      db_name = _db_name_from_fixture(stem, ep.rule_name)
      if db_name:
        break
    if not db_name:
      continue
    try:
      for i, (step, is_rec) in enumerate(mir.steps):
        gen_step_body(step, db_name, is_rec, i)
    except Exception:
      pass  # tolerate orchestrator failures; runner test still runs
    for ep in _collect_pipelines(mir):
      rule = ep.rule_name
      if (stem, rule) in RUNNER_BYTE_MATCH_SKIPS:
        continue
      fx = fixture_root / stem / f"jit_runner.{rule}.cpp"
      if not fx.exists():
        continue
      try:
        _decl, full = gen_complete_runner(ep, db_name, rel_index_types=rel_index_types)
      except NotImplementedError:
        errors.append(
          f"{stem}/{rule}: unexpected NotImplementedError — add to "
          f"RUNNER_BYTE_MATCH_SKIPS if intentional."
        )
        continue
      checked += 1
      try:
        diff_jit_runner(stem, rule, full)
      except AssertionError as e:
        errors.append(f"{stem}/{rule}: {str(e)[:400]}")

  assert not errors, f"{len(errors)} errors across {checked} fixtures:\n" + "\n".join(errors[:20])
  # Sanity: expect to check a substantial portion of the 125 matchable fixtures.
  assert checked >= 120, f"only checked {checked} fixtures — RUNNER_BYTE_MATCH_SKIPS too broad?"


if __name__ == "__main__":
  import inspect

  this = sys.modules[__name__]
  passed = 0
  failed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if not name.startswith("test_"):
      continue
    try:
      fn()
      print(f"OK  {name}")
      passed += 1
    except AssertionError as e:
      print(f"FAIL {name}")
      print(str(e)[:3000])
      failed += 1
    except Exception as e:
      print(f"ERROR {name}: {type(e).__name__}: {e}")
      failed += 1
  print(f"\n{passed} pass / {failed} fail")
