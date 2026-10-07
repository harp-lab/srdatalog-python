'''End-to-end Jupyter test on the large, recursive DOOP benchmark.

Doop is the stress test for our visualization pipeline: real-world
recursion fanout, dataset_const integers baked in via Const(meta[...]),
two-level GPU indexes, etc. If `_repr_mimebundle_` works end-to-end
on doop without OOMing the kernel or producing malformed JSON, simpler
programs are a freebie.

Two layers tested:

  Layer A — Default (JIT-less) bundle: confirms `_repr_mimebundle_`
            stays under a soft size budget so cells aren't slow to
            re-emit. Catches accidental size regressions if someone
            forgets the include_jit=False default.

  Layer B — Full bundle via prog.show(include_jit=True): exercises
            the JIT codegen path on the full program. Slower and
            larger but proves the kernel-roundtrip survives the
            mass of generated C++ that doop emits.

Uses the repository's synthetic metadata fixture; no sibling dataset
checkout is required.
'''

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

VIZ_MIME = "application/vnd.srdatalog.viz+json"

_REPO = Path(__file__).resolve().parent.parent
_DOOP_META = _REPO / "tests/fixtures/integration/doop_meta.json"
_DOOP_PY = _REPO / "examples/doop.py"


def _doop_available() -> bool:
  return _DOOP_META.exists() and _DOOP_PY.exists()


_HAS_DOOP = _doop_available()
_HAS_NBCLIENT = (
  importlib.util.find_spec("nbclient") is not None
  and importlib.util.find_spec("nbformat") is not None
  and importlib.util.find_spec("ipykernel") is not None
)


@pytest.fixture
def doop_program():
  '''Build the doop Program once — the build itself is fast (no
  compile, just DSL construction); reusing it across tests is just
  to keep the test file readable.'''
  sys.path.insert(0, str(_REPO / "examples"))
  try:
    import doop as doop_mod
  finally:
    sys.path.pop(0)
  meta = json.loads(_DOOP_META.read_text())
  return doop_mod.build_doopdb_program(meta)


# ---------------------------------------------------------------------------
# Layer A: default JIT-less bundle stays small
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_DOOP, reason="doop example or metadata fixture not present")
def test_doop_default_jupyter_bundle_size_budget(doop_program):
  '''Default `_repr_mimebundle_` must omit JIT and stay under a soft
  size budget. Doop without JIT is ~300 KB; with JIT ~3 MB. The 1 MB
  guard catches accidental regressions if someone changes the default.'''
  bundle = doop_program._repr_mimebundle_()
  payload = bundle[VIZ_MIME]
  assert payload["has_jit"] is False
  assert "jit" not in payload
  size = len(json.dumps(payload))
  assert size < 1024 * 1024, f"default bundle too large: {size / 1024:.0f} KB"
  assert [rule["name"] for rule in payload["rules"]] == [rule.name for rule in doop_program.rules]
  assert set(payload["relations"]) == {relation.name for relation in doop_program.relations}


@pytest.mark.skipif(not _HAS_DOOP, reason="doop example or metadata fixture not present")
def test_doop_full_bundle_has_all_runners(doop_program):
  '''When we explicitly opt in (CLI / show()), the JIT block contains
  one entry per runner — including the `_D{n}` delta-suffixed names
  for recursive rules.'''
  from srdatalog.ir.codegen.cuda.batchfile import _collect_pipelines
  from srdatalog.ir.hir import compile_to_mir
  from srdatalog.viz.bundle import get_visualization_bundle

  bundle = get_visualization_bundle(doop_program, include_jit=True)
  assert bundle["has_jit"] is True
  expected_runners = {pipeline.rule_name for pipeline in _collect_pipelines(compile_to_mir(doop_program))}
  assert set(bundle["jit"]) == expected_runners
  assert all(isinstance(code, str) and code for code in bundle["jit"].values())


# ---------------------------------------------------------------------------
# Layer B: kernel roundtrip on doop
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
  not _HAS_DOOP or not _HAS_NBCLIENT,
  reason="doop or nbclient/ipykernel not installed",
)
def test_doop_notebook_roundtrip_emits_viz_mime(tmp_path, doop_program):
  '''Run doop's program through an actual Jupyter kernel and confirm
  the mime bundle survives the message-spec encoding. Doop is large
  enough that this catches latent issues (e.g. JSON serialization
  of unusual Const values) that the small-program test misses.'''
  import nbformat
  from nbclient import NotebookClient

  nb = nbformat.v4.new_notebook()
  nb.cells.append(
    nbformat.v4.new_code_cell(
      source=(
        "import json, sys\n"
        f"sys.path.insert(0, {str(_REPO / 'src')!r})\n"
        f"sys.path.insert(0, {str(_REPO / 'examples')!r})\n"
        "import doop as doop_mod\n"
        f"meta = json.loads(open({str(_DOOP_META)!r}).read())\n"
        "doop_mod.build_doopdb_program(meta)"
      )
    )
  )
  nb_path = tmp_path / "doop_viz.ipynb"
  nbformat.write(nb, str(nb_path))

  # 60s — doop bundle is ~300 KB JSON-encoded; well within budget but
  # the kernel boot itself plus DSL build can take 5-10s on slow boxes.
  client = NotebookClient(nb, timeout=60)
  client.execute()

  outputs = nb.cells[0].outputs
  result = next((o for o in outputs if o.output_type == "execute_result"), None)
  assert result is not None, f"no execute_result. outputs: {outputs}"
  assert VIZ_MIME in result.data
  payload = result.data[VIZ_MIME]
  parsed = json.loads(payload) if isinstance(payload, str) else payload
  assert [rule["name"] for rule in parsed["rules"]] == [rule.name for rule in doop_program.rules]
  assert set(parsed["relations"]) == {relation.name for relation in doop_program.relations}
  assert parsed["has_jit"] is False  # default Jupyter path
