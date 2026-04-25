'''Tests for Program._repr_mimebundle_ — the Jupyter display hook.

Three layers tested here:

  Layer 1 — Direct call: invoke `_repr_mimebundle_` and inspect the
            returned dict. Pure unit test, no IPython needed.

  Layer 2 — IPython display formatter integration. Boots an InteractiveShell,
            evaluates a cell whose last expression is a Program, and
            inspects the formatter output to confirm Jupyter would
            dispatch our mime type. Skipped if IPython isn't installed.

  Layer 3 — Notebook execution end-to-end via nbclient. Runs an actual
            .ipynb through the kernel and inspects cell outputs. Skipped
            if nbclient isn't installed.

Layers 2 and 3 catch wiring problems Layer 1 misses (e.g. wrong method
name, signature mismatch with IPython's display protocol). Layer 3
catches issues that only show up across the kernel boundary (pickling
of mime bundle keys, message-spec encoding, etc).

The actual JS rendering of `application/vnd.srdatalog.viz+json` is
tested in the srdatalog-viz repo — that's a separate concern from
"does Jupyter receive the bundle correctly?".
'''

from __future__ import annotations

import importlib.util

import pytest

from srdatalog.dsl import Program, Relation, Var

VIZ_MIME = "application/vnd.srdatalog.viz+json"


def _tc_program() -> Program:
  X, Y, Z = Var("x"), Var("y"), Var("z")
  edge = Relation("Edge", 2)
  path = Relation("Path", 2)
  return Program(
    rules=[
      (path(X, Y) <= edge(X, Y)).named("TCBase"),
      (path(X, Z) <= path(X, Y) & edge(Y, Z)).named("TCRec").with_plan(var_order=["x", "y", "z"]),
    ]
  )


# ---------------------------------------------------------------------------
# Layer 1: direct call
# ---------------------------------------------------------------------------


def test_mimebundle_returns_viz_json_and_text_plain():
  prog = _tc_program()
  bundle = prog._repr_mimebundle_()
  assert set(bundle) == {VIZ_MIME, "text/plain"}
  payload = bundle[VIZ_MIME]
  # Default Jupyter bundle: HIR + MIR + rule summary, no JIT.
  assert "hir" in payload
  assert "mir" in payload
  assert payload["has_jit"] is False
  assert "jit" not in payload
  assert payload["rules"][0]["name"] == "TCBase"


def test_mimebundle_text_plain_is_summary():
  prog = _tc_program()
  bundle = prog._repr_mimebundle_()
  assert bundle["text/plain"] == "<Program: 2 relation(s), 2 rule(s)>"


def test_mimebundle_include_restricts():
  prog = _tc_program()
  only_viz = prog._repr_mimebundle_(include={VIZ_MIME})
  assert set(only_viz) == {VIZ_MIME}
  only_text = prog._repr_mimebundle_(include={"text/plain"})
  assert set(only_text) == {"text/plain"}


def test_mimebundle_exclude_drops():
  prog = _tc_program()
  no_text = prog._repr_mimebundle_(exclude={"text/plain"})
  assert set(no_text) == {VIZ_MIME}


def test_mimebundle_empty_program():
  bundle = Program(rules=[])._repr_mimebundle_()
  assert bundle["text/plain"] == "<Program: 0 relation(s), 0 rule(s)>"
  assert bundle[VIZ_MIME]["rules"] == []
  assert bundle[VIZ_MIME]["relations"] == []


# ---------------------------------------------------------------------------
# Layer 2: IPython display formatter
# ---------------------------------------------------------------------------


_HAS_IPYTHON = importlib.util.find_spec("IPython") is not None


@pytest.mark.skipif(not _HAS_IPYTHON, reason="IPython not installed")
def test_ipython_display_formatter_dispatches_our_mime_type():
  '''Boot an InteractiveShell, run a cell whose value is a Program,
  and confirm `display_formatter.format(value)` returns our mime type.

  This is the integration point Jupyter uses when a Program is the last
  expression of a cell — it goes through the same formatter path. If
  this passes, real notebook cells will deliver our bundle to
  registered renderers.
  '''
  from IPython.testing.globalipapp import get_ipython

  ip = get_ipython()
  ip.run_cell(
    "from srdatalog.dsl import Program, Relation, Var\n"
    "X = Var('x')\n"
    "e = Relation('Edge', 1)\n"
    "p = Relation('Path', 1)\n"
    "_prog = Program(rules=[(p(X) <= e(X)).named('R')])"
  )
  prog = ip.user_ns["_prog"]
  format_dict, _metadata = ip.display_formatter.format(prog)
  assert VIZ_MIME in format_dict
  assert "text/plain" in format_dict
  # The viz payload is JSON-serialized for transmission to Jupyter
  # frontends; format() returns it as a JSON string.
  import json

  payload = format_dict[VIZ_MIME]
  parsed = json.loads(payload) if isinstance(payload, str) else payload
  assert parsed["rules"][0]["name"] == "R"


# ---------------------------------------------------------------------------
# Layer 3: notebook execution end-to-end
# ---------------------------------------------------------------------------


_HAS_NBCLIENT = (
  importlib.util.find_spec("nbclient") is not None
  and importlib.util.find_spec("nbformat") is not None
  and importlib.util.find_spec("ipykernel") is not None
)


@pytest.mark.skipif(not _HAS_NBCLIENT, reason="nbclient/nbformat/ipykernel not installed")
def test_notebook_execution_emits_viz_mime_type(tmp_path):
  '''Build a .ipynb on the fly, execute it via nbclient, inspect the
  cell output. This exercises the full message-spec path: cell exec →
  shell.send → display_pub → output messages → notebook output.

  If this passes, we've verified the mime bundle survives the kernel
  boundary. The next step (extension renders the JSON) is tested on
  the srdatalog-viz side.
  '''
  import nbformat
  from nbclient import NotebookClient

  nb = nbformat.v4.new_notebook()
  nb.cells.append(
    nbformat.v4.new_code_cell(
      source=(
        "from srdatalog.dsl import Program, Relation, Var\n"
        "X = Var('x')\n"
        "e = Relation('Edge', 1)\n"
        "p = Relation('Path', 1)\n"
        "Program(rules=[(p(X) <= e(X)).named('R')])"
      )
    )
  )
  nb_path = tmp_path / "viz_test.ipynb"
  nbformat.write(nb, str(nb_path))

  client = NotebookClient(nb, timeout=30)
  client.execute()

  outputs = nb.cells[0].outputs
  assert outputs, "cell produced no output"
  # Find the execute_result output — that's where _repr_mimebundle_ lands.
  result = next((o for o in outputs if o.output_type == "execute_result"), None)
  assert result is not None, f"no execute_result in outputs: {outputs}"
  assert VIZ_MIME in result.data, f"viz mime not in {list(result.data)}"
  assert "text/plain" in result.data
