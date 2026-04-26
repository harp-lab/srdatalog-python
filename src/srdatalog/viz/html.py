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


def program_to_html(program: Program, *, height_px: int = 600) -> str:
  '''Return a `<iframe srcdoc=...>` HTML string that renders `program`.

  The iframe is self-contained — no external network fetches, no
  reliance on labextensions. Drop it into any notebook cell output
  (Jupyter Lab, classic Notebook, VS Code Jupyter, Colab) and the
  renderer mounts.

  Each call generates a fresh iframe with a unique element ID, so
  multiple cells render side-by-side without colliding.
  '''
  bundle = get_visualization_bundle(program, include_jit=False)
  return _build_iframe(bundle, height_px=height_px)


def _build_iframe(bundle: dict, *, height_px: int) -> str:
  cell_id = f"srdv-{uuid.uuid4().hex[:12]}"
  ruleset_payload = _make_ruleset_payload(bundle)

  # The full HTML document inside the iframe. Order matters:
  #   <div id="root"> first (renderer mounts here)
  #   stub vscode global (silences the renderer's postMessage)
  #   bundle script (mounts React + registers message listener)
  #   data dispatch script (sends setRuleset after a tick so the
  #     listener is wired before the message arrives)
  doc = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>{_renderer_css()}
html, body, #root {{ margin: 0; padding: 0; height: 100%; width: 100%; }}
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
    var data = {json.dumps(ruleset_payload)};
    // Renderer registers its `message` listener inside a useEffect on
    // mount, which runs after the React tree commits. Defer dispatch
    // by a frame so we hit a wired listener instead of the empty
    // window.
    function send() {{
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
