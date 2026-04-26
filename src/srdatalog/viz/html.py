'''Render a Program as a self-contained HTML iframe for notebook cells.

The renderer (TS / React, built from harp-lab/srdatalog-viz) lives as
a static asset under `srdatalog/viz/static/`. We embed it inline in
an `<iframe srcdoc=...>` so each cell gets DOM isolation: no
`document.getElementById('root')` collision between cells, no shared
React tree, no leaking state.

Inside the iframe we:
  1. Stub `window.vscode` so the renderer's postMessage calls no-op
     (the renderer was built for VS Code webviews; we don't have a
     bidirectional channel from a Jupyter cell)
  2. Load the bundle JS + CSS (inlined as data; no external fetch)
  3. After the React app mounts, dispatch a `message` event carrying
     a `setRuleset` command so the ruleset overview graph populates

The bundle expects HIR JSON in the same shape `srdatalog.hir.emit.hir_to_obj`
emits — which by design byte-matches the Nim emit, so the same
renderer serves both ports.
'''

from __future__ import annotations

import functools
import html
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from srdatalog.viz.bundle import get_visualization_bundle

if TYPE_CHECKING:
  from srdatalog.dsl import Program


_STATIC = Path(__file__).resolve().parent / "static"


@functools.lru_cache(maxsize=1)
def _renderer_js() -> str:
  '''Bundle source — cached on first read so multiple cells in one
  notebook session don't re-read the 418 KB file.'''
  return (_STATIC / "renderer.js").read_text()


@functools.lru_cache(maxsize=1)
def _renderer_css() -> str:
  return (_STATIC / "renderer.css").read_text()


def program_to_html(
  program: Program,
  *,
  rule_name: str | None = None,
  delta: int | None = None,
  theme: str = "dark",
  height_px: int = 600,
  include_jit: bool = True,
) -> str:
  '''Return a `<iframe srcdoc=...>` HTML string that renders `program`.

  The iframe is self-contained — no external network fetches, no
  reliance on labextensions. Drop it into any notebook cell output
  (Jupyter Lab, classic Notebook, VS Code Jupyter, Colab) and the
  renderer mounts.

  Args:
    program: the Program to visualize.
    rule_name: when None, render the ruleset overview (all rules,
      stratum-grouped). When a string, drill into that one rule's
      plan view — shows variant access patterns, clause order, var
      order with drag-to-reorder.
    delta: only meaningful with `rule_name`. When None (default),
      shows every variant of the rule (one per delta seed for
      recursive rules — semi-naive evaluation produces N variants
      for N body clauses). When an int, filters to just that
      variant (delta=0 means "delta seeded on body clause 0").
      Use to look at one specific version of a rule in isolation.
    theme: 'dark' (default), 'light', or 'high-contrast'. Maps to
      the renderer's setTheme message.
    height_px: iframe height. Default 600px works for most rulesets;
      bump for larger ones.
    include_jit: include per-rule JIT C++ kernels in the bundle.
      The renderer shows them under the JIT tab. Off costs ~2 MB
      on doop.

  Each call generates a fresh iframe with a unique element ID, so
  multiple cells render side-by-side without colliding.
  '''
  bundle = get_visualization_bundle(program, include_jit=include_jit)
  return _build_iframe(bundle, rule_name=rule_name, delta=delta, theme=theme, height_px=height_px)


def _build_iframe(
  bundle: dict, *, rule_name: str | None, delta: int | None, theme: str, height_px: int
) -> str:
  cell_id = f"srdv-{uuid.uuid4().hex[:12]}"
  if rule_name is None:
    payload = _make_ruleset_payload(bundle)
  else:
    payload = _make_plan_payload(bundle, rule_name, delta=delta)

  # The full HTML document inside the iframe. Order matters:
  #   <div id="root"> first (renderer mounts here)
  #   stub vscode global (silences the renderer's postMessage)
  #   bundle script (mounts React + registers message listener)
  #   data dispatch script (sends setTheme + setRuleset/setPlan after
  #     a tick so the listener is wired before the message arrives)
  light_bg = "#ffffff" if theme == "light" else "#1e1e1e"
  doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>{_renderer_css()}
html, body, #root {{ margin: 0; padding: 0; height: 100%; width: 100%; background: {light_bg}; }}
</style>
</head>
<body>
<div id="root"></div>
<script>
  // Stub the VS Code webview API so the renderer's postMessage calls
  // are no-ops in this iframe context. We don't yet have a way to
  // route plan-edit messages from a Jupyter cell back to the user's
  // .py file (that's the upcoming labextension / RemoteTrigger work).
  window.vscode = {{ postMessage: function () {{}} }};
  window.acquireVsCodeApi = function () {{ return window.vscode; }};
</script>
<script>{_renderer_js()}</script>
<script>
  (function () {{
    var theme = {json.dumps(theme)};
    var data = {json.dumps(payload)};
    // Renderer registers its `message` listener inside a useEffect on
    // mount, which runs after the React tree commits. Defer dispatch
    // by a frame so we hit a wired listener instead of the empty
    // window. setTheme first so initial paint is in the chosen palette.
    function send() {{
      window.dispatchEvent(new MessageEvent("message", {{
        data: {{ command: "setTheme", theme: theme }}
      }}));
      window.dispatchEvent(new MessageEvent("message", {{ data: data }}));
    }}
    if (document.readyState === "complete") {{
      requestAnimationFrame(send);
    }} else {{
      window.addEventListener("load", function () {{ requestAnimationFrame(send); }});
    }}
  }})();
</script>
</body>
</html>"""

  # Embed via srcdoc — escape the document so quotes don't break out
  # of the attribute. html.escape(quote=True) covers `"`, `<`, `>`, `&`.
  srcdoc = html.escape(doc, quote=True)
  return (
    f'<iframe id="{cell_id}" srcdoc="{srcdoc}" '
    f'style="width: 100%; height: {height_px}px; border: 1px solid #444; '
    f'border-radius: 6px;" sandbox="allow-scripts"></iframe>'
  )


def _make_plan_payload(bundle: dict, rule_name: str, *, delta: int | None = None) -> dict:
  '''Build the `setPlan` message focused on one rule.

  Walks every stratum's base + recursive variants, picks out the
  ones whose `rule.name == rule_name`, and packages them as
  `VariantInfo[]` with their stratum context. The renderer's plan
  view shows access patterns, clauseOrder, varOrder with drag-to-
  reorder for each variant.

  When `delta` is given, only the variant with `deltaIdx == delta`
  is included — useful for looking at one specific version of a
  recursive rule in isolation. Base (non-recursive) variants don't
  carry a deltaIdx; we treat their absent value as -1 for the
  comparison, so `delta=-1` selects base variants explicitly.

  If no variant matches the name (or delta filter), the renderer
  just shows an empty plan view — we don't raise, since the user
  might be poking at generated rule names that haven't been
  emitted yet, or at a delta that doesn't exist for the rule.
  '''
  variants: list[dict] = []
  hir = bundle.get("hir", {})
  for stratum in hir.get("strata", []):
    sid = stratum["id"]
    is_rec_stratum = bool(stratum.get("isRecursive", False))
    for vlist_key, vtype in (("base", "Base"), ("recursive", "Recursive")):
      for idx, v in enumerate(stratum.get(vlist_key, [])):
        if (v.get("rule") or {}).get("name") != rule_name:
          continue
        if delta is not None and v.get("deltaIdx", -1) != delta:
          continue
        variants.append(
          {
            "stratumId": sid,
            "isRecursiveStratum": is_rec_stratum,
            "type": vtype,
            "variantIdx": idx,
            "variant": v,
          }
        )
  return {
    "command": "setPlan",
    "variants": {
      "variants": variants,
      "hirSExpr": json.dumps(hir, indent=2),
      "mirSExpr": bundle.get("mir", ""),
      "jitByRule": bundle.get("jit", {}) if bundle.get("has_jit") else {},
    },
  }


def _make_ruleset_payload(bundle: dict) -> dict:
  '''Build the `setRuleset` message the renderer expects.

  The renderer wants a flat `rules: GraphRule[]` plus a
  `ruleStratumMap: Record<string, number>` mapping rule name to
  stratum id. We synthesize both from the bundle's HIR strata.
  '''
  rules: list[dict] = []
  stratum_map: dict[str, int] = {}
  hir = bundle.get("hir", {})
  for stratum in hir.get("strata", []):
    sid = stratum["id"]
    for variant_list_key in ("base", "recursive"):
      for v in stratum.get(variant_list_key, []):
        rule = v.get("rule") or {}
        name = rule.get("name") or ""
        if not name:
          continue
        stratum_map[name] = sid
        rules.append(
          {
            "name": name,
            "head": [{"relName": h.get("rel", "")} for h in rule.get("head", [])],
            "body": [
              {"relName": c.get("rel", "")}
              for c in rule.get("body", [])
              if c.get("kind") in (None, "relation", "negation")
            ],
            # We don't track source line numbers in the Python bundle
            # (that's a viz/source.py concern, separate from the HIR).
            # 0 means "no jump target" — the graph still renders.
            "startLine": 0,
            "fullText": rule.get("hirText", ""),
          }
        )
  return {
    "command": "setRuleset",
    "rules": rules,
    "ruleStratumMap": stratum_map,
  }
