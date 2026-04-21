'''Shared codegen helpers for the batchfile emitter.

Ported from `batchfile_gen_helpers.py` on python-api-notemplate. The
view-spec collection and pipeline walk now consume our `mir_types`
nodes directly (ColumnSource/Scan/etc. use `rel_name`+`index` rather
than a bundled `FactIndex`); `CodeGenContext` keeps the same state
shape so follow-up batchfile emission can slot in unchanged.

Also fixes a latent bug in the original single-source branch of
`collect_unique_view_specs`: `isinstance(op, Scan, Negation, Aggregate)`
was missing the tuple parens and would have raised `TypeError` the
moment the code actually ran. Rewritten as `isinstance(op, (m.Scan,
m.Negation, m.Aggregate))`.
'''

from __future__ import annotations

from dataclasses import dataclass, field

import srdatalog.mir.types as m
from srdatalog.hir.types import Version

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------


@dataclass
class ViewSpec:
  '''One distinct `(rel_name, version, index)` tuple read by a pipeline.
  `handle_idx` is the pipeline-local slot the node was assigned (carried
  over from the MIR node's `handle_start` field; -1 means unassigned).
  '''

  rel_name: str
  version: Version
  index: list[int]
  handle_idx: int = -1


@dataclass
class CodeGenContext:
  '''State threaded through batchfile emission.

  Mirrors mhk's `CodeGenContext` shape so handlers that get ported later
  can slot in without further adaptation. `view_vars` is intentionally
  typed as an unconstrained `dict` — mhk's code stores both spec-key
  strings and integer cursor handles in it.
  '''

  output_name: str
  is_counting: bool
  is_jit_mode: bool
  output_vars: dict[str, str] = field(default_factory=dict)
  inside_cartesian_join: bool = False
  cartesian_bound_vars: list[str] = field(default_factory=list)
  view_vars: dict = field(default_factory=dict)

  def set_output_vars(self, dests: list[m.MirNode]) -> None:
    '''Each dest relation points at an OutputContext var; the first
    dest is the "active" one named `output_ctx`, subsequent dests share
    the buffer and record nothing during the count phase.
    '''
    for i, d in enumerate(dests):
      rel = getattr(d, "rel_name", None)
      if not rel:
        continue
      self.output_vars[rel] = "output_ctx" if i == 0 else "__skip_counting__"


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def spec_key(rel_name: str, version: Version, index: list[int]) -> str:
  '''Canonical key for a (rel, ver, idx) triple used by the view-spec
  dedupe table and by downstream handle-to-view lookups.
  '''
  idx_str = "_".join(str(x) for x in index)
  return f"{rel_name}_{idx_str}_{version.code}"


def assign_handles(pipeline: m.ExecutePipeline) -> m.ExecutePipeline:
  '''Placeholder: assign `handle_start` slots to each source/op in the
  pipeline. Stub matches mhk's TODO-only version; a proper pass will
  live alongside batchfile codegen.
  '''
  return pipeline


def _source_rel_ver_idx(node: m.MirNode) -> tuple[str, Version, list[int]] | None:
  '''Pull the `(rel, ver, index)` tuple out of any source-bearing leaf.
  Returns None for non-source nodes (ColumnJoin / CartesianJoin / ...).'''
  if isinstance(node, (m.ColumnSource, m.Scan, m.Negation, m.Aggregate)):
    return node.rel_name, node.version, list(node.index)
  return None


# -----------------------------------------------------------------------------
# View-spec collection
# -----------------------------------------------------------------------------


def collect_unique_view_specs(ops: list[m.MirNode]) -> list[ViewSpec]:
  '''Walk pipeline body ops and extract a de-duplicated list of the
  source views they consume. Order is first-occurrence (keeps the
  view-declaration layout stable).
  '''
  seen: set[str] = set()
  result: list[ViewSpec] = []

  def add(rel: str, ver: Version, idx: list[int], handle: int) -> None:
    key = spec_key(rel, ver, idx)
    if key in seen:
      return
    seen.add(key)
    result.append(ViewSpec(rel_name=rel, version=ver, index=idx, handle_idx=handle))

  for op in ops:
    # multi-source ops
    if isinstance(op, (m.ColumnJoin, m.CartesianJoin, m.PositionedExtract)):
      for src in op.sources:
        info = _source_rel_ver_idx(src)
        if info is None:
          continue
        rel, ver, idx = info
        add(rel, ver, idx, getattr(src, "handle_start", -1))
    # single-source leaf ops
    elif isinstance(op, (m.Scan, m.Negation, m.Aggregate)):
      info = _source_rel_ver_idx(op)
      if info is None:
        continue
      rel, ver, idx = info
      add(rel, ver, idx, getattr(op, "handle_start", -1))
    # BalancedScan has two sources
    elif isinstance(op, m.BalancedScan):
      for src in (op.source1, op.source2):
        info = _source_rel_ver_idx(src)
        if info is None:
          continue
        rel, ver, idx = info
        add(rel, ver, idx, getattr(src, "handle_start", -1))

  return result


# -----------------------------------------------------------------------------
# Source lookup in ExecutePipeline.source_specs
# -----------------------------------------------------------------------------


def find_source_idx(
  specs: list[m.MirNode],
  rel_name: str,
  index: list[int],
  version: Version | None,
) -> int:
  '''Locate a source node in a pipeline's `source_specs`. `version=None`
  ignores version when matching. Returns -1 on miss.
  '''
  idx_tuple = tuple(index)
  for i, spec in enumerate(specs):
    if not isinstance(spec, (m.ColumnSource, m.Scan, m.Negation, m.Aggregate)):
      continue
    if spec.rel_name != rel_name:
      continue
    if tuple(spec.index) != idx_tuple:
      continue
    if version is not None and spec.version != version:
      continue
    return i
  return -1


# -----------------------------------------------------------------------------
# View declarations
# -----------------------------------------------------------------------------


def emit_view_declarations(pipeline: m.ExecutePipeline, ctx: CodeGenContext) -> str:
  '''Top-of-kernel view declarations: emit `using ViewType`/`HandleType`
  aliases then one `auto view_X = views[i];` per unique source spec.

  Side effects on `ctx.view_vars`:
    - `spec_key(rel, ver, idx)` -> `view_<rel>_<idx>_<VER>` for every
      unique source, so downstream handlers can resolve a view by spec.
    - Each body-op's `handle_start` int is mapped to its owning view_var,
      letting later handle-based codegen look up the C++ name directly.
  '''
  view_specs = collect_unique_view_specs(pipeline.pipeline)

  res = (
    "\n"
    "  using ViewType = std::remove_cvref_t<decltype(views[0])>;\n"
    "  using HandleType = ViewType::NodeHandle;\n\n\n"
    "  "
  )
  spec_to_view_var: dict[str, str] = {}

  for spec in view_specs:
    key = spec_key(spec.rel_name, spec.version, spec.index)
    idx_str = "_".join(str(v) for v in spec.index)
    view_var = f"view_{spec.rel_name}_{idx_str}_{spec.version.name}"
    view_idx = find_source_idx(
      pipeline.source_specs,
      spec.rel_name,
      spec.index,
      spec.version,
    )
    if view_idx == -1:
      # Fall back to the handle recorded on the node; warn so regressions
      # in source_specs population surface loudly in dev.
      print(f"Warning, could not find source for spec: {key}")
      view_idx = spec.handle_idx
    res += f"auto {view_var} = views[{view_idx}];\n"
    spec_to_view_var[key] = view_var
    ctx.view_vars[key] = view_var

  # Map each op's handle slot to the view var it reads from, so the
  # batchfile body emitter can resolve handle->view directly.
  for op in pipeline.pipeline:
    if isinstance(op, (m.ColumnJoin, m.CartesianJoin)):
      for src in op.sources:
        info = _source_rel_ver_idx(src)
        if info is None:
          continue
        rel, ver, idx = info
        key = spec_key(rel, ver, idx)
        if key in spec_to_view_var:
          h = getattr(op, "handle_start", None)
          if h is not None:
            ctx.view_vars[h] = spec_to_view_var[key]
    elif isinstance(op, (m.Scan, m.Negation, m.Aggregate)):
      info = _source_rel_ver_idx(op)
      if info is None:
        continue
      rel, ver, idx = info
      key = spec_key(rel, ver, idx)
      if key in spec_to_view_var:
        h = getattr(op, "handle_start", None)
        if h is not None:
          ctx.view_vars[h] = spec_to_view_var[key]

  return res
