'''Tests for theme + per-rule view options on program_to_html.'''

from __future__ import annotations

import html as html_lib
import json

from srdatalog.dsl import Program, Relation, Var
from srdatalog.viz.html import _make_plan_payload, program_to_html


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
# Theme
# ---------------------------------------------------------------------------


def test_theme_default_is_dark():
  out = program_to_html(_tc_program())
  decoded = html_lib.unescape(out)
  # The dispatched setTheme call references `var theme = "dark"`.
  assert 'command: "setTheme"' in decoded
  assert 'var theme = "dark"' in decoded
  assert "background: #1e1e1e" in decoded


def test_theme_light_mode():
  out = program_to_html(_tc_program(), theme="light")
  decoded = html_lib.unescape(out)
  assert 'var theme = "light"' in decoded
  # Light bg color in body fallback CSS
  assert "background: #ffffff" in decoded


def test_theme_dispatches_before_data():
  '''The setTheme message must dispatch BEFORE setRuleset/setPlan so
  the initial paint uses the chosen palette — otherwise the renderer
  flashes dark for one frame then re-paints.

  The data JSON literal is ASSIGNED before the send() function body
  in source, but EXECUTED after — what matters is dispatch order
  inside send(). Test the structural order of the two dispatch calls
  inside the function.'''
  out = program_to_html(_tc_program(), theme="light")
  decoded = html_lib.unescape(out)
  send_start = decoded.find("function send()")
  assert send_start > 0
  send_body = decoded[send_start : decoded.find("}", send_start + 200) + 1]
  # The setTheme dispatch must precede the data dispatch within send().
  theme_in_body = send_body.find('command: "setTheme"')
  data_in_body = send_body.find("data: data")
  assert theme_in_body > 0, f"setTheme not in send(): {send_body[:300]}"
  assert data_in_body > 0
  assert theme_in_body < data_in_body


# ---------------------------------------------------------------------------
# Per-rule view
# ---------------------------------------------------------------------------


def test_rule_name_switches_to_setplan():
  '''When rule_name is provided, dispatch setPlan instead of
  setRuleset — that's the per-rule plan view in the renderer.

  Both literal command strings appear in the bundle JS (the renderer's
  message switch), so we test the JSON-payload form which is unique
  to the dispatch site.'''
  out = program_to_html(_tc_program(), rule_name="TCRec")
  decoded = html_lib.unescape(out)
  assert '"command": "setPlan"' in decoded
  assert '"command": "setRuleset"' not in decoded


def test_rule_name_carries_only_matching_variants():
  '''The plan payload must contain VariantInfo entries only for the
  named rule. Other rules' variants must be excluded.'''
  from srdatalog.viz.bundle import get_visualization_bundle

  bundle = get_visualization_bundle(_tc_program())
  payload = _make_plan_payload(bundle, "TCRec")
  variants = payload["variants"]["variants"]
  assert all(v["variant"]["rule"]["name"] == "TCRec" for v in variants)
  # TCRec is the recursive rule, so we expect at least one variant
  # in a recursive stratum.
  assert any(v["isRecursiveStratum"] for v in variants)


def test_rule_name_unknown_returns_empty_variants():
  '''Asking for a non-existent rule must not crash — render an empty
  plan view instead. Useful for compiler-generated names that haven't
  been emitted yet during incremental development.'''
  from srdatalog.viz.bundle import get_visualization_bundle

  bundle = get_visualization_bundle(_tc_program())
  payload = _make_plan_payload(bundle, "DoesNotExist")
  assert payload["variants"]["variants"] == []
  # Still well-formed: the renderer just shows an empty plan.
  assert payload["command"] == "setPlan"


def test_plan_payload_carries_hir_and_mir_text():
  '''The renderer's plan view has tabs for HIR S-expr / MIR / JIT.
  The setPlan message must wire those payloads through.'''
  from srdatalog.viz.bundle import get_visualization_bundle

  bundle = get_visualization_bundle(_tc_program())
  payload = _make_plan_payload(bundle, "TCBase")
  inner = payload["variants"]
  assert "hirSExpr" in inner
  assert "mirSExpr" in inner
  assert inner["mirSExpr"].startswith("(program")  # MIR S-expr
  # hirSExpr is the HIR JSON pretty-printed for the renderer's HIR tab.
  hir_obj = json.loads(inner["hirSExpr"])
  assert "strata" in hir_obj


def test_per_rule_iframe_size():
  '''Per-rule view with include_jit=False stays small (HIR + MIR text
  only, no kernels). Sanity bound to catch payload regressions.'''
  out = program_to_html(_tc_program(), rule_name="TCRec", include_jit=False)
  # 425KB bundle + 7KB CSS + ~5KB scaffolding + few KB of payload
  assert 400_000 < len(out) < 800_000


def test_default_show_invocation_smoke():
  '''Calling Program.show() without IPython installed in this test env
  raises a RuntimeError, but we can at least import-check the function.
  The program_to_html call inside is what actually does the work; it
  must work without IPython.'''
  out = program_to_html(_tc_program(), rule_name="TCBase", theme="light")
  assert "<iframe" in out
