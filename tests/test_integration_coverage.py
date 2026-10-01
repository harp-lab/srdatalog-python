'''Keep the two integration fixture catalogs consistent.'''

from __future__ import annotations

from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
JIT_FIXTURES = TESTS_DIR / "fixtures" / "jit"
INTEGRATION_FIXTURES = TESTS_DIR / "fixtures" / "integration"


def _fixture_stems_jit() -> set[str]:
  return {p.name for p in JIT_FIXTURES.iterdir() if p.is_dir()}


def _fixture_stems_integration() -> set[str]:
  # HIR schema fixtures identify the integration-program catalog.
  return {p.name.split(".")[0] for p in INTEGRATION_FIXTURES.glob("*.hir.json")}


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
