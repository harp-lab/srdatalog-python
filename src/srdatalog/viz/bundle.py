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


def get_visualization_bundle(program: Program, project_name: str = "VizProject") -> dict:
  '''Return an in-memory bundle of everything the extension renders.

  Shape:
    {
      "hir":  {...},           # HIR JSON (hir_to_obj output)
      "mir":  "(program ...)", # MIR S-expr
      "jit":  {name: cpp},     # per-rule complete runner code
      "rules": [ {...}, ... ], # rule summary for sidebar
      "project_name": str,     # echo of the arg (for titles)
    }
  '''
  cr = compile_program(program, project_name)
  return {
    "hir": hir_to_obj(cr.hir),
    "mir": print_mir_sexpr(cr.mir),
    "jit": dict(cr.per_rule_runners),
    "rules": [_rule_summary(rule) for rule in program.rules],
    "relations": [r.name for r in program.relations],
    "project_name": project_name,
  }


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
