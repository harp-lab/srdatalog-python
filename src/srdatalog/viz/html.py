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
    # We previously also dispatched setRule here so the renderer's JIT
    # lookup (which keys off rule.name) would work. But setRule also
    # triggers generateGraph(newRule), which crashes on our payload —
    # generateGraph needs `clauseOrder` and per-clause `id` fields that
    # only exist at the per-VARIANT level in the HIR JSON, not at the
    # rule level. We instead populate the legacy `jitCode` path inside
    # the plan payload (see _make_plan_payload), which doesn't need
    # rule state at all.

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
    // window. setTheme dispatched first so initial paint uses the
    // chosen palette.
    function send() {{
      // Theme first.
      window.dispatchEvent(new MessageEvent("message", {{
        data: {{ command: "setTheme", theme: theme }}
      }}));
      // Critical: wait for React to commit the theme state and run its
      // theme-tracking useEffect (which updates the renderer's internal
      // `themeRef`). The setRuleset/setPlan handlers read themeRef at
      // graph-generation time, so dispatching synchronously gives the
      // graph stale `dark` colors while panels render `light`. Two rAFs
      // is enough — first lets React commit the setState; second runs
      // after the post-commit effect updates themeRef.
      requestAnimationFrame(function () {{
        requestAnimationFrame(function () {{
          window.dispatchEvent(new MessageEvent("message", {{ data: data }}));
        }});
      }});
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

  hirSExpr / mirSExpr / jitByRule are FILTERED to the requested rule
  so the per-rule view's HIR / MIR / JIT tabs only show that rule's
  content. The renderer's `splitSExprs` parses each as top-level
  S-expressions and matches `:name X` markers to pick the variant
  for the selected sidebar entry.
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
  filtered_jit = _filter_jit_for_rule(bundle, rule_name)
  return {
    "command": "setPlan",
    "variants": {
      "variants": variants,
      "hirSExpr": _synthesize_hir_sexpr(hir, rule_name=rule_name, delta=delta),
      "mirSExpr": _filter_mir_for_rule(bundle.get("mir", ""), rule_name),
      "jitByRule": filtered_jit,
      # `jitCode` is the legacy single-string path the renderer falls
      # back to when `rule.name` isn't set (we don't dispatch setRule —
      # it crashes on our payload — so this is what actually lands in
      # the JIT tab). The renderer splits it on
      # `// =====\n// JIT-Generated` markers and shows one struct per
      # variant in the tab switcher.
      "jitCode": "\n\n".join(filtered_jit.values()),
    },
  }


# ---------------------------------------------------------------------------
# Per-rule HIR / MIR / JIT extractors
# ---------------------------------------------------------------------------


def _synthesize_hir_sexpr(
  hir_obj: dict, *, rule_name: str | None = None, delta: int | None = None
) -> str:
  '''Synthesize per-variant HIR S-expressions the renderer can parse.

  The renderer's HIR tab runs `splitSExprs` on this string and pulls
  out top-level S-expressions, then filters them by `:name X` markers
  to pick the variant for the selected sidebar entry.

  We don't have a true HIR S-expr printer on the Python side (the
  Nim port emitted one for debug purposes; we documented it as
  intentionally skipped because byte-match goldens don't include it).
  Instead we wrap each variant's `hirText` field — which IS the per-
  rule textual rep we already maintain — into a small enclosing
  S-expression carrying the metadata the renderer needs.

  Format: `(variant :name <name> :delta <int> :stratum <int> :type <Base|Recursive>
              <hir-text>)`

  Note `hirText` itself is NOT a valid S-expression (it has `:` and
  `<-`), but the renderer's tokenizer is permissive — it just collects
  the substring between matched parens and shows it verbatim.
  '''
  parts: list[str] = []
  for stratum in hir_obj.get("strata", []):
    sid = stratum["id"]
    for vlist_key, vtype in (("base", "Base"), ("recursive", "Recursive")):
      for v in stratum.get(vlist_key, []):
        rname = (v.get("rule") or {}).get("name", "")
        if rule_name is not None and rname != rule_name:
          continue
        if delta is not None and v.get("deltaIdx", -1) != delta:
          continue
        d = v.get("deltaIdx", -1)
        text = v.get("hirText", "")
        parts.append(f"(variant :name {rname} :delta {d} :stratum {sid} :type {vtype}\n  {text})")
  return "\n\n".join(parts)


def _filter_mir_for_rule(full_mir: str, rule_name: str) -> str:
  '''Extract `(execute-pipeline :rule X ...)` blocks for X (and X_DN
  delta variants) from the full MIR S-expression, joined with blank
  lines so `splitSExprs` returns one per variant.

  The full MIR has a tree like
      (program (step ... (fixpoint-plan (execute-pipeline :rule X ...) ...)) ...)
  We don't parse the whole tree — just walk it once, paren-balanced,
  and collect every `(execute-pipeline ...)` whose `:rule` keyword
  matches `rule_name` or `rule_name_D<digits>`.
  '''
  if not full_mir:
    return ""
  marker = "(execute-pipeline"
  parts: list[str] = []
  i = 0
  while True:
    start = full_mir.find(marker, i)
    if start < 0:
      break
    end = _matching_paren_end(full_mir, start)
    if end < 0:
      break
    block = full_mir[start:end]
    rule_for_block = _extract_rule_kw(block)
    if rule_for_block == rule_name or (
      rule_for_block.startswith(rule_name + "_D") and rule_for_block[len(rule_name) + 2 :].isdigit()
    ):
      parts.append(block)
    i = end
  return "\n\n".join(parts)


def _matching_paren_end(s: str, start: int) -> int:
  '''Index just past the `)` that matches the `(` at s[start].'''
  if start >= len(s) or s[start] != "(":
    return -1
  depth = 0
  in_string = False
  i = start
  while i < len(s):
    c = s[i]
    if in_string:
      if c == '"' and s[i - 1] != "\\":
        in_string = False
    elif c == '"':
      in_string = True
    elif c == "(":
      depth += 1
    elif c == ")":
      depth -= 1
      if depth == 0:
        return i + 1
    i += 1
  return -1


def _extract_rule_kw(block: str) -> str:
  '''Pull the value following ":rule " from an execute-pipeline block.'''
  marker = ":rule "
  idx = block.find(marker)
  if idx < 0:
    return ""
  rest = block[idx + len(marker) :]
  # Atom up to whitespace or `)`.
  end = 0
  while end < len(rest) and rest[end] not in " \t\n\r)":
    end += 1
  return rest[:end]


def _filter_jit_for_rule(bundle: dict, rule_name: str) -> dict:
  '''Restrict the bundle's per-runner JIT map to the requested rule.

  Recursive rules emit one runner per delta variant, named
  `<rule>_D<n>`. Non-recursive rules emit a single runner named
  `<rule>`. We keep both shapes plus any `_SJ_*` semi-join helpers
  the renderer follows downstream.
  '''
  if not bundle.get("has_jit"):
    return {}
  jit = bundle.get("jit") or {}
  out: dict[str, str] = {}
  for k, v in jit.items():
    if k == rule_name or (k.startswith(rule_name + "_D") and k[len(rule_name) + 2 :].isdigit()):
      out[k] = v
    elif k.startswith("_SJ_"):
      # Keep all SJ helpers — the renderer cross-references them and
      # will only display the ones the selected variant actually uses.
      out[k] = v
  return out


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
