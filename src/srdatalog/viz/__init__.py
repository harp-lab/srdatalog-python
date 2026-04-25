'''Visualization support for the srdatalog-viz VS Code extension.

The extension shells out to this package to get everything it needs
to render a rule graph: the HIR JSON, MIR S-expr, per-rule JIT code,
and a source-location map (which rule lives on which line, where its
plan kwargs are). Writing plans back edits the user's .py source via
`patch.py`.

Public API:
  - bundle.get_visualization_bundle(program) — in-memory JSON bundle
  - introspect.load_program(path) — import .py, find/build a Program
  - source.find_rule_locations(src) — ast walk for editor mapping
  - patch.patch_rule_plan(src, name, var_order=..., clause_order=...)

CLI:
  python -m srdatalog.viz dump <file.py> [--entry build_fn] [--meta meta.json]
  python -m srdatalog.viz patch <file.py> <RuleName> --var-order x,y,z [--clause-order 1,0,2]
'''
