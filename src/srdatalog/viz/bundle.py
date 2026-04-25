'''Build the JSON bundle the VS Code extension renders.

One function: `get_visualization_bundle(program)` returns a dict that
contains HIR, MIR, per-rule JIT, and a summary of the rules. Purely
in-memory — no disk I/O, no compile pipeline rerun on the C++ side.

The shape is stable across Nim and Python ports because both compilers
share the same HIR/MIR contract (the byte-match fixture tests guard
that invariant). The extension's language-agnostic renderer consumes
this bundle; only the source-manipulation front (parse/patch) is
language-specific.
'''

from __future__ import annotations

from typing import TYPE_CHECKING

from srdatalog.hir.emit import hir_to_obj
from srdatalog.mir.emit import print_mir_sexpr
from srdatalog.pipeline import compile_program

if TYPE_CHECKING:
  from srdatalog.dsl import Program


def get_visualization_bundle(
  program: Program,
  project_name: str = "VizProject",
  *,
  include_jit: bool = True,
) -> dict:
  '''Return an in-memory bundle of everything the extension renders.

  Shape:
    {
      "hir":  {...},           # HIR JSON (hir_to_obj output)
      "mir":  "(program ...)", # MIR S-expr
      "jit":  {name: cpp},     # per-rule complete runner code (omitted if include_jit=False)
      "rules": [ {...}, ... ], # rule summary for sidebar
      "project_name": str,     # echo of the arg (for titles)
      "has_jit": bool,         # whether the JIT block was included
    }

  `include_jit=False` skips emitting the per-rule C++ kernel code. On
  doop (78 rules, 96 runners) that drops the bundle from ~3 MB to
  ~300 KB — important for Jupyter cell rerun latency. The HIR / MIR
  / rule summary are always included since they're cheap and drive
  the graph view. Pass `include_jit=True` (the CLI default) when the
  user explicitly wants to inspect generated kernels.
  '''
  cr = compile_program(program, project_name)
  bundle = {
    "hir": hir_to_obj(cr.hir),
    "mir": print_mir_sexpr(cr.mir),
    "rules": [_rule_summary(rule) for rule in program.rules],
    "relations": [r.name for r in program.relations],
    "project_name": project_name,
    "has_jit": include_jit,
  }
  if include_jit:
    bundle["jit"] = dict(cr.per_rule_runners)
  return bundle


def _rule_summary(rule) -> dict:
  '''Per-rule sidebar entry — name, head rels, body size, plans.'''
  return {
    "name": rule.name,
    "heads": [h.rel for h in rule.heads],
    "body_size": len(rule.body),
    "plans": [
      {
        "delta": p.delta,
        "var_order": list(p.var_order),
        "clause_order": list(p.clause_order),
      }
      for p in rule.plans
    ],
    "count": rule.count,
    "semi_join": rule.semi_join,
  }
