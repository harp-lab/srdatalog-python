'''Sweep: byte-diff orchestrator.cpp + every jit_batch.<rule>.cpp for
every fixture dir under tests/fixtures/jit/ against what the Python
codegen emits. Parallels test_jit_complete_runner.py which only covers
jit_runner.<rule>.cpp.

Same discovery path: each fixture stem must have a matching
tests/test_integration_<stem>.py with a `build_*` factory. Skips are
tracked in ORCH_BYTE_MATCH_SKIPS / BATCH_BYTE_MATCH_SKIPS so each
entry documents a known codegen-port gap.
'''

from __future__ import annotations

import re
from pathlib import Path

import pytest
from integration_helpers import diff_jit_batch, diff_orchestrator

from srdatalog.codegen.batchfile import _collect_pipelines
from srdatalog.codegen.jit.file import gen_jit_file_content_from_execute_pipeline
from srdatalog.codegen.jit.orchestrator_jit import gen_step_body
from srdatalog.hir import compile_to_mir

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "jit"

# (stem, rule) pairs where the Python codegen currently diverges from the
# Nim golden. Add with a one-line reason per entry.
ORCH_BYTE_MATCH_SKIPS: set[str] = set()
BATCH_BYTE_MATCH_SKIPS: set[tuple[str, str]] = set()


def _build_fn_for(stem: str):
  mod = __import__(f"test_integration_{stem}")
  for name in dir(mod):
    if name.startswith("build_"):
      return getattr(mod, name)
  return None


def _db_name_for(stem: str) -> str | None:
  '''Scrape `using DB = <name>;` from any jit_runner fixture in the dir.
  Falls back to reading orchestrator.cpp, which references the DB type
  in `get_relation_by_schema<...>(db)` wrapped handles.
  '''
  d = FIXTURE_ROOT / stem
  for f in d.glob("jit_runner.*.cpp"):
    m = re.search(r"using DB = (\w+);", f.read_text())
    if m:
      return m.group(1)
  orch = d / "orchestrator.cpp"
  if orch.exists():
    # Orchestrator bodies reference the DB type via templated calls; grab
    # the first `<AnythingRules_DB_DeviceDB, ...>` — same naming convention.
    m = re.search(r"(\w+_DB_DeviceDB)", orch.read_text())
    if m:
      return m.group(1)
  return None


def _discover_stems() -> list[str]:
  return sorted(p.name for p in FIXTURE_ROOT.iterdir() if p.is_dir())


@pytest.mark.parametrize("stem", _discover_stems())
def test_orchestrator_byte_match(stem: str):
  if stem in ORCH_BYTE_MATCH_SKIPS:
    pytest.skip(f"{stem} orchestrator skipped (see ORCH_BYTE_MATCH_SKIPS)")
  build_fn = _build_fn_for(stem)
  if build_fn is None:
    pytest.skip(f"no build_* factory in test_integration_{stem}")
  db_name = _db_name_for(stem)
  if db_name is None:
    pytest.skip(f"cannot infer DB type name for {stem}")
  prog = build_fn()
  mir = compile_to_mir(prog)
  actual = "".join(
    gen_step_body(node, db_name, is_rec, i) for i, (node, is_rec) in enumerate(mir.steps)
  )
  diff_orchestrator(stem, actual)


def _batch_params() -> list[tuple[str, str]]:
  out: list[tuple[str, str]] = []
  for p in sorted(FIXTURE_ROOT.iterdir()):
    if not p.is_dir():
      continue
    for f in sorted(p.glob("jit_batch.*.cpp")):
      m = re.match(r"jit_batch\.(.+)\.cpp$", f.name)
      if m:
        out.append((p.name, m.group(1)))
  return out


@pytest.mark.parametrize("stem,rule", _batch_params())
def test_jit_batch_byte_match(stem: str, rule: str):
  if (stem, rule) in BATCH_BYTE_MATCH_SKIPS:
    pytest.skip(f"{stem}/{rule} jit_batch skipped (see BATCH_BYTE_MATCH_SKIPS)")
  build_fn = _build_fn_for(stem)
  if build_fn is None:
    pytest.skip(f"no build_* factory in test_integration_{stem}")
  prog = build_fn()
  mir = compile_to_mir(prog)
  for ep in _collect_pipelines(mir):
    if ep.rule_name == rule:
      actual = gen_jit_file_content_from_execute_pipeline(ep)
      diff_jit_batch(stem, rule, actual)
      return
  pytest.skip(f"no pipeline named {rule!r} in {stem} MIR — upstream gap")
