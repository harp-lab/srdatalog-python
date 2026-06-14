'''Positional s-expression emitter for Python MIR — the `parse-Lmir`-native
form (no keyword plist), so the Racket side ingests the `.sexpr` directly with
`parse-Lmir` and needs no keyword→positional reader.

Each form matches a production of the `Lmir` grammar in
private/dialects/mir.rkt exactly (field order, lowercase versions, `#t`/`#f`
booleans). Whitespace is irrelevant to `read`/`parse-Lmir`; two-space indent
is for human readability only.
'''

from __future__ import annotations

import srdatalog.mir.types as m


# -----------------------------------------------------------------------------
# Leaf formatters
# -----------------------------------------------------------------------------


def _cols(cols) -> str:
  '''"(0 1 2)" — bare column-index list.'''
  return "(" + " ".join(str(c) for c in cols) + ")"


def _vars(vs) -> str:
  '''"(x y z)" — variable list.'''
  return "(" + " ".join(vs) + ")"


def _const_args(args) -> str:
  '''"((0 42) (2 99))" — negation constant-prefix pairs.'''
  return "(" + " ".join("(%s %s)" % (col, val) for col, val in args) + ")"


def _v(ver) -> str:
  '''Version → lowercase symbol (FULL → full), per srdl-version?.'''
  return ver.value.lower()


def _summary_spec(node: m.MirNode) -> str:
  '''One `:sources`/`:dests` summary index-spec, flattening joins. This is the
  flattened source SUMMARY; the authoritative per-source structure lives in the
  pipeline body (column-join / cartesian-join / negation).'''
  if isinstance(node, (m.ColumnJoin, m.CartesianJoin)):
    return " ".join(_summary_spec(s) for s in node.sources)
  if isinstance(node, (m.ColumnSource, m.Negation, m.Scan)):
    return "(index-spec %s %s %s)" % (node.rel_name, _cols(node.index), _v(node.version))
  if isinstance(node, m.InsertInto):
    return "(index-spec %s %s full)" % (node.rel_name, _cols(node.index))
  return ""


def _tuple(nodes) -> str:
  return "(tuple (" + " ".join(_summary_spec(n) for n in nodes) + "))"


# -----------------------------------------------------------------------------
# Main dispatcher
# -----------------------------------------------------------------------------


def print_mir_sexpr(node: m.MirNode, indent: int = 0) -> str:
  p = "  " * indent
  P = print_mir_sexpr

  # --- Program structure ---

  if isinstance(node, m.Program):
    rels = " ".join(
      '(relation-schema %s (%s) %s "%s" "%s" %s)'
      % (nm, " ".join(types), semiring, input_file, index_type, "#t" if print_size else "#f")
      for (nm, types, semiring, input_file, index_type, print_size) in (getattr(node, "relations", None) or [])
    )
    steps = "\n".join(
      "%s  (step %s\n%s)" % (p, "#t" if is_rec else "#f", P(plan, indent + 2))
      for plan, is_rec in node.steps
    )
    return "%s(program (%s) (\n%s))" % (p, rels, steps)

  if isinstance(node, m.FixpointPlan):
    return p + "(fixpoint-plan (\n" + "\n".join(P(i, indent + 1) for i in node.instructions) + "))"

  if isinstance(node, m.ParallelGroup):
    return p + "(parallel-group (\n" + "\n".join(P(o, indent + 1) for o in node.ops) + "))"

  if isinstance(node, m.ExecutePipeline):
    body = "\n".join(P(pn, indent + 1) for pn in node.pipeline)
    return p + "(execute-pipeline %s %s %s (\n%s))" % (
      node.rule_name, _tuple(node.source_specs), _tuple(node.dest_specs), body
    )

  if isinstance(node, m.PostStratumReconstructInternCols):
    return p + "(post-stratum-reconstruct-intern-cols %s %s)" % (
      node.rel_name, _cols(node.canonical_index)
    )

  # --- Pipeline body ops ---

  if isinstance(node, m.Scan):
    return p + "(scan %s %s %s %s %s)" % (
      _vars(node.vars), node.rel_name, _cols(node.index), _v(node.version), _vars(node.prefix_vars)
    )

  if isinstance(node, m.ColumnSource):
    return p + "(column-source %s %s %s %s)" % (
      node.rel_name, _cols(node.index), _v(node.version), _vars(node.prefix_vars)
    )

  if isinstance(node, m.ColumnJoin):
    return p + "(column-join %s (%s))" % (
      node.var_name, " ".join(P(s, 0) for s in node.sources)
    )

  if isinstance(node, m.CartesianJoin):
    vfs = " ".join(_vars(vs) for vs in node.var_from_source)
    return p + "(cartesian-join %s (%s) (%s))" % (
      _vars(node.vars), vfs, " ".join(P(s, 0) for s in node.sources)
    )

  if isinstance(node, m.Negation):
    if node.const_args:
      return p + "(negation %s %s %s %s %s)" % (
        node.rel_name,
        _v(node.version),
        _cols(node.index),
        _vars(node.prefix_vars),
        _const_args(node.const_args),
      )
    return p + "(negation %s %s %s %s)" % (
      node.rel_name, _v(node.version), _cols(node.index), _vars(node.prefix_vars)
    )

  if isinstance(node, m.Filter):
    return p + '(filter %s "%s")' % (_vars(node.vars), node.code.replace("\\", "\\\\").replace('"', '\\"'))

  if isinstance(node, m.ConstantBind):
    return p + '(constant-bind %s "%s" %s)' % (
      node.var_name, node.code.replace("\\", "\\\\").replace('"', '\\"'), _vars(node.deps)
    )

  if isinstance(node, m.InsertInto):
    return p + "(insert-into %s %s %s %s)" % (
      node.rel_name, _v(node.version), _cols(node.index), _vars(node.vars)
    )

  # --- Fixpoint maintenance ---

  if isinstance(node, m.RebuildIndex):
    return p + "(rebuild-index %s %s %s)" % (node.rel_name, _cols(node.index), _v(node.version))

  if isinstance(node, m.ClearRelation):
    return p + "(clear-relation %s %s)" % (node.rel_name, _v(node.version))

  if isinstance(node, m.CheckSize):
    return p + "(check-size %s %s)" % (node.rel_name, _v(node.version))

  if isinstance(node, m.ComputeDeltaIndex):
    return p + "(compute-delta-index %s %s)" % (node.rel_name, _cols(node.canonical_index))

  if isinstance(node, m.MergeIndex):
    return p + "(merge-index %s %s)" % (node.rel_name, _cols(node.index))

  if isinstance(node, m.RebuildIndexFromIndex):
    return p + "(rebuild-index-from-index %s %s %s %s %s)" % (
      node.rel_name, _cols(node.source_index), node.rel_name, _cols(node.target_index), _v(node.version)
    )

  raise TypeError("Unsupported MIR node (positional printer): %s" % type(node).__name__)
