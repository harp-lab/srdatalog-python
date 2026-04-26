'''Tests for srdatalog.viz.html.program_to_html.

We can't run a headless browser in CI, so these tests verify the
WIRING — the iframe srcdoc must contain the bundle JS, the data
dispatch, and the right structural pieces. If a real browser would
fail to mount the renderer, these tests at least catch the obvious
shape regressions.
'''

from __future__ import annotations

import html as html_lib

from srdatalog.dsl import Program, Relation, Var
from srdatalog.viz.html import _make_ruleset_payload, program_to_html


def _tc_program() -> Program:
  X, Y, Z = Var("x"), Var("y"), Var("z")
  edge = Relation("Edge", 2)
  path = Relation("Path", 2)
  return Program(
    rules=[
      (path(X, Y) <= edge(X, Y)).named("TCBase"),
      (path(X, Z) <= path(X, Y) & edge(Y, Z)).named("TCRec"),
    ]
  )


def test_iframe_is_self_contained():
  '''The iframe must carry srcdoc with the bundle inlined — no
  external <script src=...> fetches that would break in offline
  notebooks.'''
  out = program_to_html(_tc_program())
  assert "<iframe" in out
  assert "srcdoc=" in out
  assert 'sandbox="allow-scripts"' in out
  assert "<script src=" not in out


def test_each_call_uses_unique_iframe_id():
  '''Multiple cells rendering the same program must not collide on
  iframe id — otherwise the second cell would clobber the first in
  some legacy notebook UIs.'''
  a = program_to_html(_tc_program())
  b = program_to_html(_tc_program())
  # Pull the id="srdv-..." substring out and assert they differ.
  id_a = a.split('id="', 1)[1].split('"', 1)[0]
  id_b = b.split('id="', 1)[1].split('"', 1)[0]
  assert id_a.startswith("srdv-")
  assert id_b.startswith("srdv-")
  assert id_a != id_b


def test_bundle_js_is_inlined():
  '''The 418 KB renderer script must end up inside the srcdoc.'''
  out = program_to_html(_tc_program())
  # srcdoc is HTML-escaped; the bundle is large so even after escape
  # the iframe payload is multi-100KB. Look for a known chunk.
  assert len(out) > 300_000  # bundle + CSS + html scaffolding
  # Renderer references reactflow at runtime; the minified bundle
  # contains its module name.
  assert "reactflow" in out.lower() or "react-flow" in out.lower()


def test_dispatched_data_carries_rules():
  '''The bootstrap script dispatches a `setRuleset` message with the
  rules. Names must be in the srcdoc (as a JSON-in-HTML string).'''
  out = program_to_html(_tc_program())
  decoded = html_lib.unescape(out)
  assert '"setRuleset"' in decoded
  assert '"TCBase"' in decoded
  assert '"TCRec"' in decoded
  # ruleStratumMap maps each rule to a stratum id (int).
  assert "ruleStratumMap" in decoded


def test_vscode_global_is_stubbed():
  '''Without our stub, the renderer's postMessage calls throw inside
  a Jupyter iframe (no acquireVsCodeApi). The srcdoc must define
  window.vscode before loading the bundle.'''
  out = program_to_html(_tc_program())
  decoded = html_lib.unescape(out)
  assert "window.vscode" in decoded
  assert "postMessage" in decoded


# ---------------------------------------------------------------------------
# Direct test of _make_ruleset_payload
# ---------------------------------------------------------------------------


def test_payload_flattens_strata_into_rules_list():
  '''Renderer wants a flat `rules: [...]` plus a stratum lookup map.
  The HIR JSON is grouped by stratum; we flatten while preserving
  the stratum mapping.'''
  from srdatalog.viz.bundle import get_visualization_bundle

  bundle = get_visualization_bundle(_tc_program())
  payload = _make_ruleset_payload(bundle)
  assert payload["command"] == "setRuleset"
  names = [r["name"] for r in payload["rules"]]
  assert names == ["TCBase", "TCRec"]
  # TCBase lives in stratum 0 (non-recursive); TCRec in stratum 1
  # (recursive). Auto-derived from rule structure by HIR stratify.
  assert payload["ruleStratumMap"]["TCBase"] == 0
  assert payload["ruleStratumMap"]["TCRec"] == 1
  # Each rule entry has the head/body relation names.
  tcbase = next(r for r in payload["rules"] if r["name"] == "TCBase")
  assert tcbase["head"] == [{"relName": "Path"}]
  assert tcbase["body"] == [{"relName": "Edge"}]


def test_empty_program_payload_is_well_formed():
  '''Edge case: no rules, no strata. Must still produce a valid
  setRuleset message (with empty rules) so the iframe doesn't crash.'''
  from srdatalog.viz.bundle import get_visualization_bundle

  bundle = get_visualization_bundle(Program(rules=[]))
  payload = _make_ruleset_payload(bundle)
  assert payload == {"command": "setRuleset", "rules": [], "ruleStratumMap": {}}
