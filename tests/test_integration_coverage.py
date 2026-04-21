'''Guard-rail: every Nim byte-diff fixture has a matching integration test.

We use the integration suite (HIR + MIR byte-diff against Nim goldens) as
the cheap correctness gate that lets us drop slow end-to-end runs. That
only works if every fixture on disk is actually being diffed by *some*
test. This catches the case where a fixture gets added (or an old test
gets deleted) and the coverage quietly rots.

Matching is by fixture-stem appearing as a quoted string anywhere under
`tests/test_integration_*.py` — loose on purpose so tests can factor
shared builders without having to name-match their file.
'''

from __future__ import annotations

from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
JIT_FIXTURES = TESTS_DIR / "fixtures" / "jit"
INTEGRATION_FIXTURES = TESTS_DIR / "fixtures" / "integration"


def _fixture_stems_jit() -> set[str]:
  return {p.name for p in JIT_FIXTURES.iterdir() if p.is_dir()}


def _fixture_stems_integration() -> set[str]:
  # Stems come in {stem}.hir.json / {stem}.mir.sexpr pairs.
  return {p.name.split(".")[0] for p in INTEGRATION_FIXTURES.glob("*.hir.json")}


def _integration_test_sources() -> str:
  return "\n".join(p.read_text() for p in TESTS_DIR.glob("test_integration_*.py"))


def test_every_jit_fixture_has_an_integration_test():
  sources = _integration_test_sources()
  orphans = [s for s in sorted(_fixture_stems_jit()) if f'"{s}"' not in sources]
  assert not orphans, (
    f"JIT fixture dirs without a matching integration test: {orphans}.\n"
    f"Add tests/test_integration_<name>.py that calls diff_hir/diff_mir on each."
  )


def test_every_integration_fixture_has_an_integration_test():
  sources = _integration_test_sources()
  orphans = [s for s in sorted(_fixture_stems_integration()) if f'"{s}"' not in sources]
  assert not orphans, f"Integration fixtures (.hir.json/.mir.sexpr) without a test: {orphans}"


def test_jit_and_integration_fixture_sets_agree():
  # Both fixture trees should track the same set of benchmarks. Divergence
  # means a fixture got added to one tree and forgotten in the other.
  jit = _fixture_stems_jit()
  intg = _fixture_stems_integration()
  jit_only = sorted(jit - intg)
  intg_only = sorted(intg - jit)
  assert not jit_only and not intg_only, (
    f"Fixture trees disagree. jit-only={jit_only}, integration-only={intg_only}"
  )
