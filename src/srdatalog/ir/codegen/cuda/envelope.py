'''target.cuda — full-file envelope emission.

The dialect-emitted kernel body sits inside a fixed-shape envelope:

  JIT_FILE_PRELUDE                     (constant header)
  + banner(rule_name, num_handles)
  + functor_start(rule_name, ...)
  + [optional DedupTable struct]
  + view_declarations + <BODY>         (operator() body)
  + functor_end()
  + "\\n"
  + JIT_FILE_FOOTER                    (constant footer)

Pure string emitters — no algorithm dispatch, no feature-flag-driven
branching beyond the dedup-hash struct injection. Lives in the dialect
because the envelope shape is target-specific (CUDA cooperative-groups
signature, `__device__` qualifier, etc.); a `target.cpp_tbb` envelope
would emit a different shape from the same MIR pipeline.

The legacy `ir/dialects/target/cuda/` modules still own their own copies of these
helpers during the Stage 2 transition (they are imported by the
byte-equivalence harness which compares the legacy emitter against
the dialect path). Once the legacy inner-body emitters are deleted,
the legacy copies go with them.
'''

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import overload

import srdatalog.ir.mir.types as m

# -----------------------------------------------------------------------------
# File prelude / footer (byte-identical to the legacy strings)
# -----------------------------------------------------------------------------

JIT_FILE_PRELUDE = """\
// JIT-Generated Rule Kernel Batch
// This file is auto-generated - do not edit
#define SRDATALOG_JIT_BATCH  // Guard: exclude host-side helpers from JIT compilation

// Main project header - includes all necessary boost/hana, etc.
#include "srdatalog.h"

#include <cstdint>
#include <cooperative_groups.h>

// JIT-specific headers (relative to generalized_datalog/)
#include "gpu/device_sorted_array_index.h"
#include "gpu/runtime/output_context.h"
#include "gpu/runtime/jit/intersect_handles.h"
#include "gpu/runtime/jit/jit_executor.h"
#include "gpu/runtime/jit/materialized_join.h"
#include "gpu/runtime/jit/ws_infrastructure.h"  // WCOJTask, WCOJTaskQueue, ChunkedOutputContext
#include "gpu/runtime/query.h"  // For DeviceRelationType

namespace cg = cooperative_groups;

// Make JIT helpers visible without full namespace qualification
using SRDatalog::GPU::JIT::intersect_handles;

"""

JIT_FILE_FOOTER = """
// End of JIT batch file
"""


# -----------------------------------------------------------------------------
# Pipeline walks (handle assignment + counting)
# -----------------------------------------------------------------------------


@overload
def _assign_handle_positions_rec(node: m.ColumnSource, offset_box: list[int]) -> m.ColumnSource: ...
@overload
def _assign_handle_positions_rec(node: m.MirNode, offset_box: list[int]) -> m.MirNode: ...
def _assign_handle_positions_rec(node: m.MirNode, offset_box: list[int]) -> m.MirNode:
  '''Return a copy of `node` with `handle_start` filled in (and any
  source-bearing descendants rebuilt likewise). Pure: input `node` is
  not mutated.

  `offset_box` is a one-element list used as a mutable counter
  (Python closures can't reassign captured ints). The walk order
  preserves the legacy left-to-right pipeline traversal so the handle
  counter advances identically — joins claim a slot first, then
  recurse into their children.
  '''
  if isinstance(node, m.ColumnSource | m.Scan | m.Aggregate | m.Negation):
    new_node = replace(node, handle_start=offset_box[0])
    offset_box[0] += 1
    return new_node
  if isinstance(node, m.ColumnJoin | m.CartesianJoin):
    handle = offset_box[0]
    new_sources = [_assign_handle_positions_rec(src, offset_box) for src in node.sources]
    return replace(node, handle_start=handle, sources=new_sources)
  if isinstance(node, m.BalancedScan):
    handle = offset_box[0]
    new_source1 = _assign_handle_positions_rec(node.source1, offset_box)
    new_source2 = _assign_handle_positions_rec(node.source2, offset_box)
    return replace(node, handle_start=handle, source1=new_source1, source2=new_source2)
  if isinstance(node, m.PositionedExtract):
    new_sources = [_assign_handle_positions_rec(src, offset_box) for src in node.sources]
    return replace(node, sources=new_sources)
  # C5: `mir.TiledCartesian` wraps an eligible `CartesianJoin` (the
  # `TiledCartesian` pragma's materialized form). The wrap is opaque
  # to the handle walk; recurse into `inner` so the wrapped Cart's
  # handles get assigned identically to the unwrapped flow.
  if isinstance(node, m.TiledCartesian):
    new_inner = _assign_handle_positions_rec(node.inner, offset_box)
    assert isinstance(new_inner, m.CartesianJoin)
    return replace(node, inner=new_inner)
  return node


def assign_handle_positions(ops: list[m.MirNode]) -> list[m.MirNode]:
  '''Return a NEW pipeline list where every source-bearing node has its
  `handle_start` filled in. The handle counter starts at 0 and bumps
  in pipeline-order, left-to-right — joins claim a slot before their
  children recurse, matching the legacy in-place walk so byte-equiv
  is preserved.

  Caller rebinds: `pipeline = assign_handle_positions(pipeline)`. For
  ExecutePipeline contexts where `source_specs` must agree with the
  pipeline, prefer `assign_handle_positions_in_ep` instead.
  '''
  offset_box = [0]
  return [_assign_handle_positions_rec(op, offset_box) for op in ops]


def assign_handle_positions_in_ep(ep: m.ExecutePipeline) -> m.ExecutePipeline:
  '''Return a NEW `ExecutePipeline` with `handle_start` filled in on
  both the pipeline ops AND the parallel `source_specs` references.

  `source_specs` holds flattened references to the same source ops
  embedded inside the pipeline (joins flatten into their leaf
  sources). When the pipeline is rebuilt with fresh handle_start
  values, source_specs must be regenerated from the new pipeline so
  downstream consumers (e.g. `compute_view_slot_offsets` in
  `codegen.cuda.view_slots`) see matching handles, not stale -1s.

  Mirrors `_extract_pipeline_sources` in `srdatalog.ir.hir.lower`.
  '''
  from srdatalog.ir.hir.lower import _extract_pipeline_sources

  new_pipeline = assign_handle_positions(list(ep.pipeline))
  new_source_specs: list[m.ColumnSource | m.Scan | m.Negation | m.Aggregate] = []
  for op in new_pipeline:
    _extract_pipeline_sources(op, new_source_specs)
  return replace(ep, pipeline=new_pipeline, source_specs=new_source_specs)


def count_handles(ops: list[m.MirNode]) -> int:
  '''Number of `views[]` slots needed by the kernel — `max(handle_start) + 1`.'''
  result = 0
  for op in ops:
    # C5: `TiledCartesian` wraps a CartesianJoin; unwrap before
    # accounting so the wrapped Cart's sources count toward the
    # handle slot total identically to the unwrapped flow.
    actual = op.inner if isinstance(op, m.TiledCartesian) else op
    if isinstance(actual, m.ColumnJoin | m.CartesianJoin):
      for src in actual.sources:
        h = getattr(src, 'handle_start', -1)
        result = max(result, h + 1)
    elif isinstance(actual, m.Scan | m.Negation | m.Aggregate):
      result = max(result, getattr(actual, 'handle_start', -1) + 1)
  return result


def first_dest_arity(ops: list[m.MirNode]) -> int:
  '''Arity of the first InsertInto's column set. Sizes the DedupTable's
  hash function (one v0..vN-1 column per parameter).'''
  for op in ops:
    if isinstance(op, m.InsertInto):
      return len(op.vars)
  return 0


# -----------------------------------------------------------------------------
# View declarations
# -----------------------------------------------------------------------------


@dataclass
class ViewSpec:
  '''(rel_name, index, version, handle_idx) — handle_idx is the handle
  position of the FIRST op that referenced this view spec.'''

  rel_name: str
  index: list[int]
  version: str
  handle_idx: int


def _spec_key(rel_name: str, index: list[int], version: str = '') -> str:
  '''`Rel_<cols>_<VER>` — version-suffixed so DELTA / FULL share-key
  doesn't collapse them.'''
  base = rel_name + '_' + '_'.join(str(c) for c in index)
  return base + '_' + version if version else base


def _record_spec(
  specs: list[ViewSpec],
  seen: set[str],
  rel: str,
  idx: list[int],
  ver: str,
  handle: int,
) -> None:
  k = _spec_key(rel, idx, ver)
  if k in seen:
    return
  seen.add(k)
  specs.append(ViewSpec(rel_name=rel, index=list(idx), version=ver, handle_idx=handle))


def collect_unique_view_specs(ops: list[m.MirNode]) -> list[ViewSpec]:
  '''Walk the pipeline and collect a deduplicated list of `ViewSpec`s in
  first-occurrence order. Covers every op that references a view
  (ColumnJoin, CartesianJoin, Scan, Negation, Aggregate, BalancedScan,
  PositionedExtract).

  C5: `mir.TiledCartesian` wraps a `CartesianJoin`; the wrap is
  opaque to this walk so we unwrap before dispatching to the same
  source-collection logic used for the unwrapped Cart.
  '''
  specs: list[ViewSpec] = []
  seen: set[str] = set()
  for op in ops:
    if isinstance(op, m.TiledCartesian):
      op = op.inner
    if isinstance(op, m.ColumnJoin | m.CartesianJoin):
      for src in op.sources:
        if isinstance(src, m.ColumnSource):
          _record_spec(
            specs,
            seen,
            src.rel_name,
            list(src.index),
            src.version.code,
            src.handle_start,
          )
    elif isinstance(op, m.Scan | m.Negation | m.Aggregate):
      _record_spec(
        specs,
        seen,
        op.rel_name,
        list(op.index),
        op.version.code,
        op.handle_start,
      )
    elif isinstance(op, m.BalancedScan):
      s1, s2 = op.source1, op.source2
      _record_spec(
        specs,
        seen,
        s1.rel_name,
        list(s1.index),
        s1.version.code,
        s1.handle_start,
      )
      _record_spec(
        specs,
        seen,
        s2.rel_name,
        list(s2.index),
        s2.version.code,
        s2.handle_start,
      )
    elif isinstance(op, m.PositionedExtract):
      for src in op.sources:
        if isinstance(src, m.ColumnSource):
          _record_spec(
            specs,
            seen,
            src.rel_name,
            list(src.index),
            src.version.code,
            src.handle_start,
          )
  return specs


def emit_view_declarations(
  specs: list[ViewSpec],
  pipeline: list[m.MirNode],
  *,
  indent_level: int = 4,
  debug: bool = True,
  slot_mode: str = 'handle_idx',
  view_counts: list[int] | None = None,
) -> tuple[str, dict[str, str]]:
  '''Emit the top-of-kernel `auto view_X = views[i];` block.

  Returns `(decls_string, view_vars)`. `view_vars` maps both:
    - spec key (`<rel>_<cols>_<VER>`) → view variable name
    - `str(handle_idx)` → view variable name (so handle-bearing ops can
      look up "which view does this handle name reference?")

  `slot_mode` controls how the index into `views[]` is chosen:
    - `'handle_idx'`: use `sp.handle_idx` directly (matches the
      `jit_batch.<rule>.cpp` standalone-kernel goldens, which don't
      apply slot-offset compaction).
    - `'positional'`: use cumulative-sum-of-view_counts slot per spec
      (matches the `jit_runner.<rule>.cpp` production goldens via
      `compute_view_slot_offsets`).

  `view_counts` (per-spec, parallel to `specs`) is the number of
  physical view slots each spec consumes. Default = all 1s (DSAI).
  D2L FULL_VER specs consume 2 slots (HEAD + FULL); the dialect's
  view decl emits the BASE view at the first slot only (the second
  slot is referenced by BG histogram via `views[base+seg]`).
  '''
  view_vars: dict[str, str] = {}
  if not specs:
    return '', view_vars

  if view_counts is None:
    view_counts = [1] * len(specs)
  elif len(view_counts) != len(specs):
    raise ValueError(
      f'emit_view_declarations: view_counts length {len(view_counts)} '
      f'does not match specs length {len(specs)}'
    )

  indent = '  ' * indent_level
  code = ''
  code += indent + 'using ViewType = std::remove_cvref_t<decltype(views[0])>;\n'
  code += indent + 'using HandleType = ViewType::NodeHandle;\n\n'

  if debug:
    code += indent + f'// View declarations (deduplicated by spec, {len(specs)} unique views)\n'

  spec_to_view_var: list[tuple[str, str]] = []
  spec_to_base_slot: dict[str, int] = {}
  positional_cursor = 0
  for sp, vc in zip(specs, view_counts, strict=True):
    if slot_mode == 'positional':
      slot = positional_cursor
      positional_cursor += vc
    else:
      slot = sp.handle_idx
    key = _spec_key(sp.rel_name, sp.index, sp.version)
    idx_str = '_'.join(str(v) for v in sp.index)
    view_var = f'view_{sp.rel_name}_{idx_str}' + (f'_{sp.version}' if sp.version else '')
    code += indent + f'auto {view_var} = views[{slot}];\n'
    spec_to_view_var.append((key, view_var))
    spec_to_base_slot[key] = slot
    view_vars[key] = view_var

  for op in pipeline:
    # C5: `TiledCartesian` wraps a `CartesianJoin` — unwrap so the
    # handle-index map ('1' -> 'view_R_0_FULL_VER', etc.) gets
    # populated identically to the unwrapped flow.
    if isinstance(op, m.TiledCartesian):
      op = op.inner
    if isinstance(op, m.ColumnJoin | m.CartesianJoin):
      for src in op.sources:
        k = _spec_key(src.rel_name, list(src.index), src.version.code)
        for kv_key, view_var in spec_to_view_var:
          if kv_key == k:
            view_vars[str(src.handle_start)] = view_var
            view_vars[f'__base__{src.handle_start}'] = str(spec_to_base_slot[k])
            break
    elif isinstance(op, m.Scan | m.Negation | m.Aggregate):
      k = _spec_key(op.rel_name, list(op.index), op.version.code)
      for kv_key, view_var in spec_to_view_var:
        if kv_key == k:
          view_vars[str(op.handle_start)] = view_var
          view_vars[f'__base__{op.handle_start}'] = str(spec_to_base_slot[k])
          break

  code += '\n'
  return code, view_vars


# -----------------------------------------------------------------------------
# Functor envelope
# -----------------------------------------------------------------------------


def emit_dedup_table_struct(arity: int) -> str:
  '''Emit the DedupTable struct nested inside a rule's kernel scope.

  GPU hash table over `arity` 32-bit columns: `try_insert` (atomicCAS
  during count phase) and `check_winner` (read-only during materialize
  phase). Linear probing over up to 128 slots from an FNV-1a hash; the
  capacity is power-of-2 (host-side runner zero-initializes hash_slots
  between phases).
  '''
  code = (
    '  // GPU dedup hash table: full 64-bit hash + separate thread_id array\n'
    '  struct DedupTable {\n'
    '    unsigned long long* hash_slots; // full 64-bit hash per slot\n'
    '    uint32_t* tid_slots;            // winner thread_id per slot\n'
    '    uint32_t capacity;              // must be power of 2\n\n'
  )
  code += '    __device__ __forceinline__ unsigned long long compute_hash(\n'
  for col in range(arity):
    sep = ',' if col < arity - 1 else ')'
    code += f'        uint32_t v{col}{sep}\n'
  code += '    {\n'
  code += '      uint64_t h = 14695981039346656037ULL;\n'
  for col in range(arity):
    code += f'      h ^= (uint64_t)v{col}; h *= 1099511628211ULL;\n'
  code += '      return h | 1ULL; // ensure non-zero\n'
  code += '    }\n\n'

  code += '    __device__ __forceinline__ bool try_insert(\n'
  code += '        uint32_t thread_id,\n'
  for col in range(arity):
    sep = ',' if col < arity - 1 else ')'
    code += f'        uint32_t v{col}{sep}\n'
  code += '    {\n'
  code += '      unsigned long long h = compute_hash('
  for col in range(arity):
    code += f'v{col}' + (', ' if col < arity - 1 else ');\n')
  code += '      uint32_t base = (uint32_t)(h ^ (h >> 32)) & (capacity - 1);\n'
  code += '      for (uint32_t p = 0; p < 128; p++) {\n'
  code += '        uint32_t s = (base + p) & (capacity - 1);\n'
  code += '        unsigned long long old = atomicCAS(&hash_slots[s], 0ULL, h);\n'
  code += '        if (old == 0ULL) { tid_slots[s] = thread_id; return true; } // claimed\n'
  code += '        if (old == h) return false; // same hash = duplicate\n'
  code += '        // old != h: collision with different tuple -> probe next\n'
  code += '      }\n'
  code += '      return true; // probe overflow -> emit (safe)\n'
  code += '    }\n\n'

  code += '    __device__ __forceinline__ bool check_winner(\n'
  code += '        uint32_t thread_id,\n'
  for col in range(arity):
    sep = ',' if col < arity - 1 else ')'
    code += f'        uint32_t v{col}{sep}\n'
  code += '    {\n'
  code += '      unsigned long long h = compute_hash('
  for col in range(arity):
    code += f'v{col}' + (', ' if col < arity - 1 else ');\n')
  code += '      uint32_t base = (uint32_t)(h ^ (h >> 32)) & (capacity - 1);\n'
  code += '      for (uint32_t p = 0; p < 128; p++) {\n'
  code += '        uint32_t s = (base + p) & (capacity - 1);\n'
  code += '        unsigned long long stored = hash_slots[s];\n'
  code += '        if (stored == h) return tid_slots[s] == thread_id; // found: am I winner?\n'
  code += '        if (stored == 0ULL) return true; // not found -> probe overflow, emit\n'
  code += '        // different hash -> probe next (collision resolution)\n'
  code += '      }\n'
  code += '      return true; // probe overflow -> emit\n'
  code += '    }\n'
  code += '  };\n\n'
  return code


def emit_functor_start(
  rule_name: str,
  *,
  scalar_mode: bool = False,
  dedup_hash: bool = False,
) -> str:
  '''Open `struct Kernel_<rule> { ... operator()(...) const {`.

  When `dedup_hash=True`, operator() takes an additional
  `DedupTable dedup_table` parameter.
  '''
  group_size = 1 if scalar_mode else 32
  mode_comment = (
    '// SCALAR MODE: 1 thread per row, sequential search'
    if scalar_mode
    else '// WARP MODE: 32 threads share 1 row, cooperative search'
  )
  dedup_param = '      DedupTable dedup_table,\n' if dedup_hash else ''
  return (
    mode_comment + '\n'
    f'struct Kernel_{rule_name} {{\n'
    '  static constexpr int kBlockSize = 256;\n'
    f'  static constexpr int kGroupSize = {group_size};\n\n'
    '  template <typename Tile, typename Views, typename ValueType, typename Output>\n'
    '  __device__ void operator()(\n'
    '      Tile& tile,\n'
    '      const Views* views,\n'
    '      const ValueType* __restrict__ root_unique_values,\n'
    '      uint32_t num_unique_root_keys,\n'
    '      uint32_t num_root_keys,\n'
    '      uint32_t warp_id,\n'
    '      uint32_t num_warps,\n'
    f'{dedup_param}'
    '      Output& output\n'
    '  ) const {\n'
  )


def emit_functor_end() -> str:
  return '  }\n};\n'


# -----------------------------------------------------------------------------
# Top-level: wrap a body in the file envelope
# -----------------------------------------------------------------------------


def emit_full_file(
  ep: m.ExecutePipeline,
  body: str,
  *,
  scalar_mode: bool = False,
) -> str:
  '''Wrap a dialect-emitted operator() body in the standard file envelope.

  `body` must be everything between `operator() {` and the closing `}`,
  i.e. the view declarations followed by the dialect-emitted kernel
  logic. Caller composes `view_decls + emit(iir, emit_ctx)`.
  '''
  pipeline = list(ep.pipeline)
  num_handles = count_handles(pipeline)

  banner = (
    '// =============================================================\n'
    f'// JIT-Generated Kernel Functor: {ep.rule_name}\n'
    f'// Handles: {num_handles}\n'
    '// =============================================================\n\n'
  )

  header = banner + emit_functor_start(
    ep.rule_name,
    scalar_mode=scalar_mode,
    dedup_hash=ep.dedup_hash,
  )

  # Dedup table goes between the constexpr decls and the operator()
  # signature, matching the legacy line-for-line placement.
  if ep.dedup_hash:
    arity = first_dest_arity(pipeline)
    marker = 'static constexpr int kGroupSize = '
    marker_pos = header.find(marker)
    if marker_pos != -1:
      newline_after = header.find('\n\n', marker_pos)
      if newline_after != -1:
        insert_at = newline_after + 2
        header = header[:insert_at] + emit_dedup_table_struct(arity) + header[insert_at:]

  full_kernel = header + body + emit_functor_end()
  return JIT_FILE_PRELUDE + full_kernel + '\n' + JIT_FILE_FOOTER


__all__ = [
  'JIT_FILE_FOOTER',
  'JIT_FILE_PRELUDE',
  'ViewSpec',
  'assign_handle_positions',
  'assign_handle_positions_in_ep',
  'collect_unique_view_specs',
  'count_handles',
  'emit_dedup_table_struct',
  'emit_full_file',
  'emit_functor_end',
  'emit_functor_start',
  'emit_view_declarations',
  'first_dest_arity',
]
