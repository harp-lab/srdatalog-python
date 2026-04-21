'''View/handle slot mapping + view-declaration emission.

Port of src/srdatalog/codegen/target_jit/jit_view_management.nim.

Deduplicates pipeline source views by (rel_name, index, version),
computes view-slot offsets for multi-view sources (e.g. Device2LevelIndex
contributes >1 slot per source via the plugin), and emits the
`auto view_X = views[i];` block at the top of each generated kernel.

A "view spec" is the triple (rel_name, index cols, version) plus the
handle_idx of the first op that referenced it. Two uses of the same
relation with different index orderings are distinct views; two uses
with the same (rel, idx, ver) share one view slot even across nested
CJ / Cartesian handles.
'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import srdatalog.mir.types as m
from srdatalog.hir.types import Version
from srdatalog.codegen.jit.context import (
  CodeGenContext, ind, get_view_slot_base, gen_view_access,
)
from srdatalog.codegen.jit.plugin import plugin_view_count


# -----------------------------------------------------------------------------
# Source-spec helpers
# -----------------------------------------------------------------------------

_SOURCE_SPEC_TYPES = (m.ColumnSource, m.Scan, m.Negation, m.Aggregate)


def get_source_index(src_spec: m.MirNode) -> list[int]:
  '''Extract column ordering from a source-bearing node.'''
  if isinstance(src_spec, _SOURCE_SPEC_TYPES):
    return list(src_spec.index)
  return []


def _source_rel_version(src_spec: m.MirNode) -> tuple[str, str]:
  '''Pull (rel_name, version-as-string) off a source-bearing node.
  Version is converted to its Nim `*_VER` form via `.code` so spec keys
  match what the existing fixtures use.'''
  if isinstance(src_spec, _SOURCE_SPEC_TYPES):
    return src_spec.rel_name, src_spec.version.code
  return "", ""


def source_spec_key(src_spec: m.MirNode) -> str:
  '''Unique key for a source spec: `<relName>_<VERSION>_<cols_joined>`.
  Two uses of the same relation with different index orderings → distinct
  keys. Includes version to keep DELTA / FULL views distinct.
  '''
  rel, ver = _source_rel_version(src_spec)
  idx_str = "_".join(str(c) for c in get_source_index(src_spec))
  return f"{rel}_{ver}_{idx_str}"


def _handle_start_of(src_spec: m.MirNode) -> int:
  return getattr(src_spec, "handle_start", -1)


# -----------------------------------------------------------------------------
# View-slot computation
# -----------------------------------------------------------------------------

def compute_total_view_count(
  source_specs: list[m.MirNode], rel_index_types: dict[str, str],
) -> int:
  '''Total view slots needed for all unique sources. Nested CJ/Cart
  handles for the same (rel, version, index) share slots.
  '''
  total = 0
  seen: list[str] = []
  for src_spec in source_specs:
    rel, ver = _source_rel_version(src_spec)
    key = source_spec_key(src_spec)
    if key not in seen:
      seen.append(key)
      index_type = rel_index_types.get(rel, "")
      total += plugin_view_count(ver, index_type)
  return total


def compute_view_slot_offsets(
  source_specs: list[m.MirNode], rel_index_types: dict[str, str],
) -> dict[int, int]:
  '''Map `handle_idx` → base slot in `views[]`.

  Multiple handles for the same relation+version+index share one slot —
  nested CJ/Cart handles reference the same physical view as the root
  handle.
  '''
  out: dict[int, int] = {}
  slot = 0
  seen_slots: dict[str, int] = {}
  for src_spec in source_specs:
    handle_idx = _handle_start_of(src_spec)
    rel, ver = _source_rel_version(src_spec)
    key = source_spec_key(src_spec)
    if handle_idx >= 0:
      if key in seen_slots:
        out[handle_idx] = seen_slots[key]
      else:
        out[handle_idx] = slot
        seen_slots[key] = slot
        index_type = rel_index_types.get(rel, "")
        slot += plugin_view_count(ver, index_type)
  return out


def register_pipeline_handles(
  offsets: dict[int, int],
  pipeline: list[m.MirNode],
  rel_index_types: dict[str, str],
  root_slots: dict[str, int],
) -> None:
  '''Walk the pipeline body and register every ColumnSource `handle_start`
  against the `root_slots` table. Mutates `offsets` in place — matches
  Nim's `registerPipelineHandles` var-param signature.
  '''
  for node in pipeline:
    if isinstance(node, m.ColumnSource):
      if node.handle_start not in offsets:
        key = source_spec_key(node)
        if key in root_slots:
          offsets[node.handle_start] = root_slots[key]
    elif isinstance(node, m.ColumnJoin):
      for src in node.sources:
        if isinstance(src, m.ColumnSource) and src.handle_start not in offsets:
          key = source_spec_key(src)
          if key in root_slots:
            offsets[src.handle_start] = root_slots[key]
    elif isinstance(node, m.CartesianJoin):
      for src in node.sources:
        if isinstance(src, m.ColumnSource) and src.handle_start not in offsets:
          key = source_spec_key(src)
          if key in root_slots:
            offsets[src.handle_start] = root_slots[key]


def build_root_slot_map(
  source_specs: list[m.MirNode], rel_index_types: dict[str, str],
) -> dict[str, int]:
  '''Map `<relName>_<VER>_<cols>` → view-slot base for each root source.
  First occurrence wins; subsequent duplicates share the first's slot.
  '''
  out: dict[str, int] = {}
  slot = 0
  for src_spec in source_specs:
    rel, ver = _source_rel_version(src_spec)
    key = source_spec_key(src_spec)
    if key not in out:
      out[key] = slot
    index_type = rel_index_types.get(rel, "")
    slot += plugin_view_count(ver, index_type)
  return out


# -----------------------------------------------------------------------------
# ViewSpec + unique view collection
# -----------------------------------------------------------------------------

@dataclass
class ViewSpec:
  '''(rel_name, index, version, handle_idx) — handle_idx is the index
  of the FIRST op that referenced this view.'''
  rel_name: str
  index: list[int]
  version: str
  handle_idx: int


def spec_key(rel_name: str, index: list[int], version: str = "") -> str:
  '''`Rel_<cols>_<VER>` — version-suffixed so DELTA and FULL share-key
  doesn't collapse them.'''
  base = rel_name + "_" + "_".join(str(c) for c in index)
  return base + "_" + version if version else base


def _record_spec(
  specs: list[ViewSpec], seen: set[str], rel: str, idx: list[int],
  ver: str, handle: int,
) -> None:
  k = spec_key(rel, idx, ver)
  if k in seen:
    return
  seen.add(k)
  specs.append(ViewSpec(rel_name=rel, index=list(idx), version=ver, handle_idx=handle))


def collect_unique_view_specs(ops: list[m.MirNode]) -> list[ViewSpec]:
  '''Walk the pipeline body and collect a de-duplicated list of
  `ViewSpec`s, first-occurrence order. Covers every op kind that
  references a view: ColumnJoin, CartesianJoin, Scan, Negation,
  Aggregate, BalancedScan, PositionedExtract.
  '''
  specs: list[ViewSpec] = []
  seen: set[str] = set()
  for op in ops:
    if isinstance(op, m.ColumnJoin):
      for src in op.sources:
        if isinstance(src, m.ColumnSource):
          _record_spec(
            specs, seen, src.rel_name, list(src.index),
            src.version.code, src.handle_start,
          )
    elif isinstance(op, m.CartesianJoin):
      for src in op.sources:
        if isinstance(src, m.ColumnSource):
          _record_spec(
            specs, seen, src.rel_name, list(src.index),
            src.version.code, src.handle_start,
          )
    elif isinstance(op, m.Scan):
      _record_spec(
        specs, seen, op.rel_name, list(op.index),
        op.version.code, op.handle_start,
      )
    elif isinstance(op, m.Negation):
      _record_spec(
        specs, seen, op.rel_name, list(op.index),
        op.version.code, op.handle_start,
      )
    elif isinstance(op, m.Aggregate):
      _record_spec(
        specs, seen, op.rel_name, list(op.index),
        op.version.code, op.handle_start,
      )
    elif isinstance(op, m.BalancedScan):
      s1, s2 = op.source1, op.source2
      _record_spec(
        specs, seen, s1.rel_name, list(s1.index),
        s1.version.code, s1.handle_start,
      )
      _record_spec(
        specs, seen, s2.rel_name, list(s2.index),
        s2.version.code, s2.handle_start,
      )
    elif isinstance(op, m.PositionedExtract):
      for src in op.sources:
        if isinstance(src, m.ColumnSource):
          _record_spec(
            specs, seen, src.rel_name, list(src.index),
            src.version.code, src.handle_start,
          )
  return specs


# -----------------------------------------------------------------------------
# View declaration emission
# -----------------------------------------------------------------------------

def jit_emit_view_declarations(
  specs: list[ViewSpec],
  ops: list[m.MirNode],
  ep_source_specs: list[m.MirNode],
  ctx: CodeGenContext,
) -> str:
  '''Emit the top-of-kernel `auto view_X = views[i];` block.

  Populates `ctx.view_vars` with both:
    - spec key (`rel_idx_VER`) → view_var
    - str(handle_idx) → view_var   (so nested op emitters can resolve
      "which view is this handle referring to?" directly)
  '''
  if not specs:
    return ""

  code = ""
  i = ind(ctx)

  code += i + "using ViewType = std::remove_cvref_t<decltype(views[0])>;\n"
  code += i + "using HandleType = ViewType::NodeHandle;\n\n"

  if ctx.debug:
    code += (
      i + "// View declarations (deduplicated by spec, "
      + str(len(specs)) + " unique views)\n"
    )

  spec_to_view_var: list[tuple[str, str]] = []
  for sp in specs:
    key = spec_key(sp.rel_name, sp.index, sp.version)
    idx_str = "_".join(str(v) for v in sp.index)
    view_var = (
      f"view_{sp.rel_name}_{idx_str}"
      + (f"_{sp.version}" if sp.version else "")
    )
    view_idx = get_view_slot_base(ctx, sp.handle_idx)
    code += i + f"auto {view_var} = {gen_view_access(view_idx)};\n"
    spec_to_view_var.append((key, view_var))
    ctx.view_vars[key] = view_var

  # Map every op's handle_start to its view_var so later emitters can
  # resolve handle -> view quickly.
  for op in ops:
    if isinstance(op, m.ColumnJoin):
      for src in op.sources:
        if not isinstance(src, m.ColumnSource):
          continue
        k = spec_key(src.rel_name, list(src.index), src.version.code)
        for kv_key, view_var in spec_to_view_var:
          if kv_key == k:
            ctx.view_vars[str(src.handle_start)] = view_var
            break
    elif isinstance(op, m.CartesianJoin):
      for src in op.sources:
        if not isinstance(src, m.ColumnSource):
          continue
        k = spec_key(src.rel_name, list(src.index), src.version.code)
        for kv_key, view_var in spec_to_view_var:
          if kv_key == k:
            ctx.view_vars[str(src.handle_start)] = view_var
            break
    elif isinstance(op, m.Scan):
      k = spec_key(op.rel_name, list(op.index), op.version.code)
      for kv_key, view_var in spec_to_view_var:
        if kv_key == k:
          ctx.view_vars[str(op.handle_start)] = view_var
          break
    elif isinstance(op, m.Negation):
      k = spec_key(op.rel_name, list(op.index), op.version.code)
      for kv_key, view_var in spec_to_view_var:
        if kv_key == k:
          ctx.view_vars[str(op.handle_start)] = view_var
          break
    elif isinstance(op, m.Aggregate):
      k = spec_key(op.rel_name, list(op.index), op.version.code)
      for kv_key, view_var in spec_to_view_var:
        if kv_key == k:
          ctx.view_vars[str(op.handle_start)] = view_var
          break

  code += "\n"
  return code
