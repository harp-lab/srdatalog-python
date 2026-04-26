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


# ---------------------------------------------------------------------------
# Delta-filtered (per-version) view
# ---------------------------------------------------------------------------


def _self_join_program() -> Program:
  '''A rule whose body has TWO references to a recursive relation —
  semi-naive evaluation produces one delta variant per such reference,
  so this gives us deltaIdx ∈ {0, 1} to test against.'''
  X, Y, Z = Var("x"), Var("y"), Var("z")
  e = Relation("Edge", 2)
  tc = Relation("TC", 2)
  return Program(
    rules=[
      (tc(X, Y) <= e(X, Y)).named("TCBase"),
      # Body[0] = TC(x, y), Body[1] = TC(y, z) — both recursive,
      # so semi-naive emits two variants: delta seeded on body 0,
      # then delta seeded on body 1.
      (tc(X, Z) <= tc(X, Y) & tc(Y, Z)).named("TCSelfJoin"),
    ]
  )


def test_delta_filter_isolates_single_variant():
  '''A self-join recursive rule has body length 2 with both sides
  referencing the recursive relation → 2 delta variants. Asking for
  delta=0 must return only the deltaIdx=0 variant.'''
  from srdatalog.viz.bundle import get_visualization_bundle

  bundle = get_visualization_bundle(_self_join_program())
  payload = _make_plan_payload(bundle, "TCSelfJoin", delta=0)
  variants = payload["variants"]["variants"]
  assert len(variants) == 1
  assert variants[0]["variant"]["deltaIdx"] == 0


def test_delta_filter_other_index():
  '''delta=1 must select the OTHER variant — confirms we're filtering
  by deltaIdx and not just clamping to 0.'''
  from srdatalog.viz.bundle import get_visualization_bundle

  bundle = get_visualization_bundle(_self_join_program())
  payload = _make_plan_payload(bundle, "TCSelfJoin", delta=1)
  variants = payload["variants"]["variants"]
  assert len(variants) == 1
  assert variants[0]["variant"]["deltaIdx"] == 1


def test_delta_none_returns_all_variants():
  '''Default `delta=None` keeps every variant — preserves the
  pre-delta-filter behavior so existing callers keep working.'''
  from srdatalog.viz.bundle import get_visualization_bundle

  bundle = get_visualization_bundle(_self_join_program())
  payload = _make_plan_payload(bundle, "TCSelfJoin")
  variants = payload["variants"]["variants"]
  # Both body atoms reference the recursive relation TC → 2 deltas.
  assert len(variants) == 2
  assert {v["variant"]["deltaIdx"] for v in variants} == {0, 1}


def test_delta_filter_unknown_returns_empty():
  '''Asking for a delta that doesn't exist (e.g. delta=99 on a body of
  length 2) is not an error — just an empty plan view. Same UX as
  asking for an unknown rule name.'''
  from srdatalog.viz.bundle import get_visualization_bundle

  bundle = get_visualization_bundle(_self_join_program())
  payload = _make_plan_payload(bundle, "TCSelfJoin", delta=99)
  assert payload["variants"]["variants"] == []


def test_delta_minus_one_selects_base_variants():
  '''Base (non-recursive) variants don't carry deltaIdx in the JSON.
  We default the missing field to -1, so `delta=-1` is the way to
  explicitly select base variants.'''
  from srdatalog.viz.bundle import get_visualization_bundle

  bundle = get_visualization_bundle(_tc_program())
  payload = _make_plan_payload(bundle, "TCBase", delta=-1)
  variants = payload["variants"]["variants"]
  assert len(variants) == 1
  assert variants[0]["type"] == "Base"


def test_delta_threads_through_program_to_html():
  '''Smoke: passing delta through the public entry point produces a
  setPlan iframe with only one variant in the dispatched data.

  The bundle JSON contains many `"deltaIdx": N` substrings (every
  variant in every recursive stratum), but the dispatched ruleset
  is in a separate JSON literal. We grep the bootstrap script's
  `var data = ...` line specifically.'''
  out = program_to_html(_self_join_program(), rule_name="TCSelfJoin", delta=0)
  decoded = html_lib.unescape(out)
  # Isolate the data JSON — it's the line starting `var data = {` in
  # the bootstrap. Slice from there to the next `;`.
  i = decoded.find("var data = {")
  assert i > 0, "bootstrap data line not found"
  data_line = decoded[i : decoded.find(";", i)]
  assert '"command": "setPlan"' in data_line
  assert '"deltaIdx": 0' in data_line
  assert '"deltaIdx": 1' not in data_line
