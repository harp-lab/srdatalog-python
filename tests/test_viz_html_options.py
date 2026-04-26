'''Tests for theme + per-rule view options on program_to_html.'''

from __future__ import annotations

import html as html_lib

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
  # send() body has nested `{}` (setTheme dispatch's data object literal),
  # so slice up to the closing brace at the function's depth — find the
  # next `if (document.readyState` which is right after send() ends.
  send_end = decoded.find("if (document.readyState", send_start)
  send_body = decoded[send_start:send_end]
  theme_in_body = send_body.find('command: "setTheme"')
  data_in_body = send_body.find("data: data")
  assert theme_in_body > 0, f"setTheme not in send(): {send_body[:400]}"
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


def test_plan_payload_hir_is_per_variant_sexpr():
  '''The renderer's HIR tab parses hirSExpr via splitSExprs and matches
  `:name X` to pick a variant. We must emit per-variant S-expressions
  with `:name` markers, not the full HIR JSON.'''
  from srdatalog.viz.bundle import get_visualization_bundle

  bundle = get_visualization_bundle(_tc_program())
  payload = _make_plan_payload(bundle, "TCBase")
  hir_sexpr = payload["variants"]["hirSExpr"]
  assert hir_sexpr.startswith("(variant"), f"expected S-expr, got: {hir_sexpr[:80]!r}"
  assert ":name TCBase" in hir_sexpr
  # Must NOT contain other rules' markers — TCBase only.
  assert ":name TCRec" not in hir_sexpr


def test_plan_payload_mir_is_filtered_to_rule():
  '''MIR tab must only show execute-pipeline blocks for the requested
  rule (and its _DN delta variants), NOT the whole-program MIR.'''
  from srdatalog.viz.bundle import get_visualization_bundle

  bundle = get_visualization_bundle(_tc_program())
  payload = _make_plan_payload(bundle, "TCRec")
  mir = payload["variants"]["mirSExpr"]
  # Only TCRec's pipelines (TCRec or TCRec_D0/_D1/...) should appear.
  assert ":rule TCRec" in mir or ":rule TCRec_D" in mir
  # Other rules must be excluded from this view.
  assert ":rule TCBase" not in mir
  # And no `(program ...)` wrapper — that's whole-program output.
  assert not mir.lstrip().startswith("(program")


def test_plan_payload_jit_is_filtered_to_rule():
  '''JIT tab must only carry runners for the requested rule (matching
  base name or `<name>_D<n>` delta-suffixed). Other rules' kernels
  are excluded so the user isn't drowning in unrelated code.'''
  from srdatalog.viz.bundle import get_visualization_bundle

  bundle = get_visualization_bundle(_tc_program(), include_jit=True)
  payload = _make_plan_payload(bundle, "TCBase")
  jit = payload["variants"]["jitByRule"]
  for key in jit:
    assert key == "TCBase" or key.startswith("TCBase_D") or key.startswith("_SJ_"), (
      f"unexpected JIT key in TCBase view: {key!r}"
    )
  assert "TCBase" in jit  # the base rule's runner must be present


def test_setrule_message_dispatched_before_setplan():
  '''The renderer's JIT lookup keys off `rule.name`, which only the
  setRule message populates. We must dispatch setRule before setPlan
  in the bootstrap so `rule` state is set when the JIT tab renders.'''
  out = program_to_html(_tc_program(), rule_name="TCBase")
  decoded = html_lib.unescape(out)
  set_rule_idx = decoded.find('"command": "setRule"')
  set_plan_idx = decoded.find('"command": "setPlan"')
  assert set_rule_idx > 0, "setRule message not dispatched"
  assert set_plan_idx > 0
  assert set_rule_idx < set_plan_idx


def test_ruleset_view_does_not_dispatch_setrule():
  '''Whole-program (overview) view doesn't need setRule — it uses the
  setRuleset graph which works without rule state.'''
  out = program_to_html(_tc_program())  # no rule_name
  decoded = html_lib.unescape(out)
  # We default the ruleMsg JS variable to null when no rule. Verify.
  assert "var ruleMsg = null" in decoded
  assert '"command": "setRuleset"' in decoded
  # No setRule message dispatch path activates when ruleMsg is null.


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
