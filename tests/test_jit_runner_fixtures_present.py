'''Smoke test: verify all JIT runner fixtures are present and non-empty.

This test guards Phase 1 of the jit_complete_runner.nim port:
  - srdatalog_plan.nim now dumps jit_runner.<rule>.cpp alongside
    jit_batch.<rule>.cpp for every integration program.
  - These fixtures are the byte-match target for Phases 2-6 (the actual
    Python runner emitter).

The test just asserts the fixtures exist so a missing fixture is caught
early (e.g., if someone regenerates without the runner-dump flag).
'''

import sys
from pathlib import Path

JIT_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "jit"


def _stems_and_rules() -> list[tuple[str, list[str]]]:
  '''For every fixture dir, list its jit_batch.<rule>.cpp files and
  return (stem, [rule_name,...]). We use jit_batch presence as the
  ground truth — every batch file must have a matching runner file.'''
  out: list[tuple[str, list[str]]] = []
  for d in sorted(JIT_FIXTURES.iterdir()):
    if not d.is_dir():
      continue
    rules = sorted(
      p.name.removeprefix("jit_batch.").removesuffix(".cpp") for p in d.glob("jit_batch.*.cpp")
    )
    out.append((d.name, rules))
  return out


def test_every_batch_has_runner_fixture():
  '''Every jit_batch.<rule>.cpp must have a sibling jit_runner.<rule>.cpp.'''
  missing: list[str] = []
  for stem, rules in _stems_and_rules():
    for rule in rules:
      p = JIT_FIXTURES / stem / f"jit_runner.{rule}.cpp"
      if not p.exists():
        missing.append(f"{stem}/{p.name}")
  assert not missing, (
    f"{len(missing)} runner fixtures missing — regenerate with "
    f"srdatalog_plan.nim (jit_output_dir arg). First 5: {missing[:5]}"
  )


def test_runner_fixtures_non_empty_and_struct_headed():
  '''Each runner fixture must start with the expected banner and
  contain the struct definition — a trivial shape guard against
  future regressions in the dump tool.'''
  bad: list[str] = []
  for stem, rules in _stems_and_rules():
    for rule in rules:
      p = JIT_FIXTURES / stem / f"jit_runner.{rule}.cpp"
      text = p.read_text()
      if len(text) < 200:
        bad.append(f"{stem}/{p.name}: too short ({len(text)} bytes)")
        continue
      if "// JIT-Generated Complete Runner:" not in text:
        bad.append(f"{stem}/{p.name}: missing banner")
        continue
      if f"struct JitRunner_{rule} {{" not in text:
        bad.append(f"{stem}/{p.name}: missing struct header")
  assert not bad, "\n".join(bad[:10])


def test_fixture_count_sanity():
  '''Sanity check: expect ~20 fixture dirs and 100+ total runner files.'''
  entries = _stems_and_rules()
  assert len(entries) >= 18, f"expected >=18 fixture dirs, got {len(entries)}"
  total = sum(len(rules) for _, rules in entries)
  assert total >= 100, f"expected >=100 runner fixtures, got {total}"


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
      print(f"OK   {name}")
      passed += 1
    except AssertionError as e:
      print(f"FAIL {name}")
      print(str(e)[:1000])
      failed += 1
  print(f"\n{passed} pass / {failed} fail")
