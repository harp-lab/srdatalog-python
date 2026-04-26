# Renderer bundle provenance

`renderer.js` and `renderer.css` are prebuilt artifacts from
[harp-lab/srdatalog-viz](https://github.com/harp-lab/srdatalog-viz).

To regenerate:

```bash
git clone https://github.com/harp-lab/srdatalog-viz
cd srdatalog-viz
npm install
node build.js --production
cp dist/webview/index.js  /path/to/srdatalog-python/src/srdatalog/viz/static/renderer.js
cp dist/webview/index.css /path/to/srdatalog-python/src/srdatalog/viz/static/renderer.css
```

Last sync: `ca75cca4b64ddf8b247d703e2ac6e31fd909e190` (2026-04-25).

The renderer expects to be loaded into a DOM with `<div id="root"></div>`,
where it consumes `setPlan` / `setRuleset` messages dispatched on
`window` to populate its state. `srdatalog.viz.html.program_to_html`
constructs an iframe srcdoc that does exactly that, so the same
bundle works unchanged across the VS Code extension's webview and
Jupyter cell outputs.

Future work: extract the renderer into a standalone npm package and
fetch it at install time via `populate_vendor.py`-style script,
instead of vendoring a build artifact.
