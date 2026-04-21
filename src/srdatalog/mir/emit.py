'''S-expression emitter for Python MIR, matching the output of
src/srdatalog/mir/printer.nim byte-for-byte.

The canonical text form is what Racket will ingest as the final metalang,
so S-expr is the target rather than JSON. Indentation uses two spaces per
level, matching the Nim printer's `repeat("  ", indent)` convention.

Every registered Nim MIR op kind now has a Python emitter case. For
ops where Nim's printer silently returns empty (no `of` branch —
CreateFlatView, ProbeJoin, GatherColumn), the Python formats are new
conventions; if byte-diff on those becomes necessary, add matching
cases to src/srdatalog/mir/printer.nim first.
'''
from __future__ import annotations

from srdatalog.hir.types import Version
import srdatalog.mir.types as m


# -----------------------------------------------------------------------------
# Small helpers (mirror Nim printVarTuple / printIndex / printVer)
# -----------------------------------------------------------------------------

def _var_tuple(vars: list[str]) -> str:
  '''"(x y z)" — Nim uses space-separated, no commas.'''
  return "(" + " ".join(vars) + ")"


def _index(rel_name: str, cols: list[int]) -> str:
  '''"(Rel 0 1 ...)"'''
  return "(" + rel_name + " " + " ".join(str(c) for c in cols) + ")"


def _ver(v: Version) -> str:
  return v.value


# -----------------------------------------------------------------------------
# index-spec helpers (used by ExecutePipeline's :sources / :dests)
# -----------------------------------------------------------------------------

def _index_spec(node: m.MirNode) -> str:
  '''Mirror printIndexSpec in printer.nim.

  Joins flatten: a ColumnJoin / CartesianJoin contributes specs from ALL
  its sources, space-separated. Everything else produces one spec.
  '''
  if isinstance(node, m.ColumnJoin):
    return " ".join(_flatten_specs(s) for s in node.sources)
  if isinstance(node, m.CartesianJoin):
    return " ".join(_flatten_specs(s) for s in node.sources)

  if isinstance(node, m.ColumnSource):
    rel, ver, idx = node.rel_name, node.version, node.index
  elif isinstance(node, m.Negation):
    rel, ver, idx = node.rel_name, node.version, node.index
  elif isinstance(node, m.InsertInto):
    # Dest always uses FULL index for dedup logic (matches Nim).
    rel, ver, idx = node.rel_name, Version.FULL, node.index
  elif isinstance(node, m.Scan):
    rel, ver, idx = node.rel_name, node.version, node.index
  else:
    return "void"

  return (
    "(index-spec :schema " + rel
    + " :index (" + " ".join(str(c) for c in idx) + ")"
    + " :ver " + _ver(ver) + ")"
  )


def _flatten_specs(node: m.MirNode) -> str:
  '''Flatten joins into concatenated leaf index-specs.'''
  if isinstance(node, m.ColumnJoin):
    return " ".join(_flatten_specs(s) for s in node.sources)
  if isinstance(node, m.CartesianJoin):
    return " ".join(_flatten_specs(s) for s in node.sources)
  return _index_spec(node)


def _index_specs_tuple(nodes: list[m.MirNode], indent: int = 0) -> str:
  '''Mirror printIndexSpecsTuple.'''
  prefix = "  " * indent
  if not nodes:
    return "(tuple)"
  parts: list[str] = []
  for n in nodes:
    if isinstance(n, (m.ColumnJoin, m.CartesianJoin)):
      for s in n.sources:
        parts.append(_index_spec(s))
    else:
      parts.append(_index_spec(n))
  return "(tuple\n" + prefix + ("\n" + prefix).join(parts) + ")"


# -----------------------------------------------------------------------------
# Main dispatcher
# -----------------------------------------------------------------------------

def print_mir_sexpr(node: m.MirNode, indent: int = 0) -> str:
  p = "  " * indent

  # --- Leaves ---

  if isinstance(node, m.ColumnSource):
    return (
      p + "(column-source"
      + " :index " + _index(node.rel_name, node.index)
      + " :ver " + _ver(node.version)
      + " :prefix " + _var_tuple(node.prefix_vars)
      + ")"
    )

  if isinstance(node, m.Scan):
    return (
      p + "(scan"
      + " :vars " + _var_tuple(node.vars)
      + " :index " + _index(node.rel_name, node.index)
      + " :ver " + _ver(node.version)
      + " :prefix " + _var_tuple(node.prefix_vars)
      + ")"
    )

  if isinstance(node, m.ColumnJoin):
    body = p + "(column-join :var " + node.var_name
    body += "\n" + p + "  :sources (\n"
    for src in node.sources:
      body += print_mir_sexpr(src, indent + 2) + "\n"
    body += p + "  ))"
    return body

  if isinstance(node, m.CartesianJoin):
    body = p + "(cartesian-join :vars " + _var_tuple(node.vars)
    if node.var_from_source:
      body += " :var-from-source ("
      body += " ".join(_var_tuple(vs) for vs in node.var_from_source)
      body += ")"
    body += "\n" + p + "  :sources (\n"
    for src in node.sources:
      body += print_mir_sexpr(src, indent + 2) + "\n"
    body += p + "  ))"
    return body

  if isinstance(node, m.Filter):
    return (
      p + "(filter"
      + " :vars " + _var_tuple(node.vars)
      + " :code \"" + node.code + "\")"
    )

  if isinstance(node, m.ConstantBind):
    return (
      p + "(constant-bind"
      + " :var " + node.var_name
      + " :code \"" + node.code + "\""
      + " :deps " + _var_tuple(node.deps)
      + ")"
    )

  if isinstance(node, m.Aggregate):
    return (
      p + "(aggregate"
      + " :var " + node.result_var
      + " :func " + node.agg_func
      + " :index " + _index(node.rel_name, node.index)
      + " :ver " + _ver(node.version)
      + " :prefix " + _var_tuple(node.prefix_vars)
      + ")"
    )

  if isinstance(node, m.CreateFlatView):
    # (create-flat-view :index (Rel cols) :ver V) — aligned with the
    # rebuild-index style and matches the Nim printer case added
    # alongside the Python split-rule lowering.
    return (
      p + "(create-flat-view"
      + " :index " + _index(node.rel_name, node.index)
      + " :ver " + _ver(node.version)
      + ")"
    )

  if isinstance(node, m.InnerPipeline):
    body = p + "(inner-pipeline"
    if node.rule_name:
      body += " :rule " + node.rule_name
    body += " :bound-vars " + _var_tuple(node.bound_vars)
    body += "\n" + p + "  :handles (\n"
    for h in node.input_handles:
      body += print_mir_sexpr(h, indent + 2) + "\n"
    body += p + "  )\n"
    body += p + "  :ops (\n"
    for op in node.inner_ops:
      body += print_mir_sexpr(op, indent + 2) + "\n"
    body += p + "  ))"
    return body

  if isinstance(node, m.ProbeJoin):
    # Nim printer has no case for moProbeJoin — Python convention.
    res = p + "(probe-join"
    if node.input_buffer:
      res += " :input-buffer " + node.input_buffer
    res += " :output-buffer " + node.output_buffer
    res += " :probe " + _index(node.probe_rel, node.probe_index)
    res += " :ver " + _ver(node.probe_version)
    res += " :key " + node.join_key
    res += ")"
    return res

  if isinstance(node, m.GatherColumn):
    # Nim printer has no case for moGatherColumn — Python convention.
    res = p + "(gather-column"
    if node.input_buffer:
      res += " :input-buffer " + node.input_buffer
    res += " :schema " + node.rel_name
    res += " :ver " + _ver(node.rel_version)
    res += " :col " + str(node.column)
    res += " :out " + node.output_var
    res += ")"
    return res

  if isinstance(node, m.Negation):
    return (
      p + "(negation"
      + " :schema " + node.rel_name
      + " :ver " + _ver(node.version)
      + " :index " + _index(node.rel_name, node.index)
      + " :prefix " + _var_tuple(node.prefix_vars)
      + ")"
    )

  if isinstance(node, m.InsertInto):
    return (
      p + "(insert-into"
      + " :schema " + node.rel_name
      + " :ver " + _ver(node.version)
      + " :dedup-index (" + " ".join(str(c) for c in node.index) + ")"
      + " :terms " + _var_tuple(node.vars)
      + ")"
    )

  # --- Fixpoint maintenance ---

  if isinstance(node, m.RebuildIndex):
    return (
      p + "(rebuild-index"
      + " :index " + _index(node.rel_name, node.index)
      + " :ver " + _ver(node.version)
      + ")"
    )

  if isinstance(node, m.ClearRelation):
    return (
      p + "(clear-relation"
      + " :schema " + node.rel_name
      + " :ver " + _ver(node.version)
      + ")"
    )

  if isinstance(node, m.CheckSize):
    return (
      p + "(check-size"
      + " :schema " + node.rel_name
      + " :ver " + _ver(node.version)
      + ")"
    )

  if isinstance(node, m.ComputeDelta):
    return p + "(compute-delta :schema " + node.rel_name + ")"

  if isinstance(node, m.ComputeDeltaIndex):
    return (
      p + "(compute-delta-index"
      + " :schema " + node.rel_name
      + " :canonical-index (" + " ".join(str(c) for c in node.canonical_index) + ")"
      + ")"
    )

  if isinstance(node, m.MergeIndex):
    return (
      p + "(merge-index :index " + _index(node.rel_name, node.index) + ")"
    )

  if isinstance(node, m.MergeRelation):
    return p + "(merge-relation :schema " + node.rel_name + ")"

  if isinstance(node, m.RebuildIndexFromIndex):
    return (
      p + "(rebuild-index-from-index"
      + " :source " + _index(node.rel_name, node.source_index)
      + " :target " + _index(node.rel_name, node.target_index)
      + " :ver " + _ver(node.version)
      + ")"
    )

  # --- Structural ---

  if isinstance(node, m.ExecutePipeline):
    body = p + "(execute-pipeline"
    if node.rule_name:
      body += " :rule " + node.rule_name
    body += "\n"
    body += p + "  :sources " + _index_specs_tuple(node.source_specs, indent + 2) + "\n"
    body += p + "  :dests " + _index_specs_tuple(node.dest_specs, indent + 2) + "\n"
    for pn in node.pipeline:
      body += print_mir_sexpr(pn, indent + 2) + "\n"
    body += p + ")"
    return body

  if isinstance(node, m.FixpointPlan):
    body = p + "(fixpoint-plan\n"
    for instr in node.instructions:
      body += print_mir_sexpr(instr, indent + 2) + "\n"
    body += p + ")"
    return body

  if isinstance(node, m.Block):
    body = p + "(block\n"
    for instr in node.instructions:
      body += print_mir_sexpr(instr, indent + 2) + "\n"
    body += p + ")"
    return body

  if isinstance(node, m.BalancedScan):
    # (balanced-scan :group-var v :source1 (...) :source2 (...)
    #                :vars1 (...) :vars2 (...))
    body = p + "(balanced-scan"
    body += " :group-var " + node.group_var
    body += "\n" + p + "  :source1 " + print_mir_sexpr(node.source1, 0)
    body += "\n" + p + "  :source2 " + print_mir_sexpr(node.source2, 0)
    if node.vars1:
      body += " :vars1 " + _var_tuple(node.vars1)
    if node.vars2:
      body += " :vars2 " + _var_tuple(node.vars2)
    body += ")"
    return body

  if isinstance(node, m.PositionedExtract):
    body = p + "(positioned-extract"
    body += " :var " + node.var_name
    body += " :sources ("
    for i, src in enumerate(node.sources):
      if i > 0:
        body += " "
      body += print_mir_sexpr(src, 0)
    body += ")"
    body += " :bind " + _var_tuple(node.bind_vars)
    body += ")"
    return body

  if isinstance(node, m.ParallelGroup):
    body = p + "(parallel-group  ;; " + str(len(node.ops)) + " independent ops\n"
    for op in node.ops:
      body += print_mir_sexpr(op, indent + 2) + "\n"
    body += p + ")"
    return body

  if isinstance(node, m.InjectCppHook):
    res = p + "(inject-cpp-hook"
    if node.rule_name:
      res += " :rule " + node.rule_name
    # Nim always writes :code "..." (literal ellipsis) rather than dumping
    # the raw body — keeps the S-expr readable.
    res += " :code \"...\")"
    return res

  if isinstance(node, m.PostStratumReconstructInternCols):
    return (
      p + "(post-stratum-reconstruct-intern-cols"
      + " :rel " + node.rel_name
      + " :canonical-index (" + " ".join(str(c) for c in node.canonical_index) + ")"
      + ")"
    )

  if isinstance(node, m.Program):
    # Nim uses indent+4 for the nested plan (unlike Block/FixpointPlan which use +2),
    # and bool printing in Nim is lowercase "true"/"false" — both affect byte-diff.
    body = p + "(program\n"
    for plan, is_rec in node.steps:
      body += p + "  (step :recursive " + ("true" if is_rec else "false") + "\n"
      body += print_mir_sexpr(plan, indent + 4) + "\n"
      body += p + "  )\n"
    body += p + ")"
    return body

  raise TypeError(f"Unsupported MIR node: {type(node).__name__}")
