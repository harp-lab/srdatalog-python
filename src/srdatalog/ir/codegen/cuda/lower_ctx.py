'''CudaRenderCtx — CUDA-target-private render scratch for MIR -> IIR lowering.

Per `docs/phase_decomposition_redesign.md` § 3.2.1.

The dialect-level `LoweringCtx`
(`src/srdatalog/ir/dialects/relation/sorted_array/lowerings/__init__.py`)
historically carried both:

  - structural / lowering-time state (e.g. `name_counter`, `bound_vars`,
    `handle_vars`, `inside_cartesian`) — target-agnostic;
  - CUDA C++ identifier scratch (e.g. `view_var_names`, `output_var`,
    `tiled_cartesian_valid_var`) — CUDA render-private.

PR-1d extracts the second bucket into `CudaRenderCtx`, the
target-private bag. `LoweringCtx` retains the structural fields and
exposes the render-private fields via forwarding properties for back-
compat. Per the spec, pragma scratch booleans
(`is_counting`, `dedup_hash`, `tiled_cartesian`, `ws_enabled`,
`bg_enabled`) are NOT moved here — they remain on `LoweringCtx` for
now and are scheduled for removal in PR-1e.

Design notes:
  - Plain `@dataclass` (NOT `frozen=True`, NOT `slots=True`). The
    spec shows `frozen=True, slots=True` but block-group test paths
    rely on dynamic attribute writes; the dataclass must remain
    mutable + non-slotted.
  - The companion framework-level `LowerCtx`
    (`src/srdatalog/ir/core/lower_ctx.py`) is OFF LIMITS per D10.
    `CudaRenderCtx` lives entirely under `codegen/cuda/`; it never
    appears in the framework-level type signatures.

This module also hosts `NegPreNarrowInfo`. It used to live in
`lowerings/__init__.py`; PR-1d moves it here because it is purely
CUDA-render shape (C++ var names, view-var names, etc.) and the
`neg_pre_narrow` field type-references it. `lowerings/__init__.py`
re-exports the name for any existing imports.
'''

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NegPreNarrowInfo:
  '''Pre-narrowed handle info for a Negation that follows a Cartesian.

  When a Negation's prefix vars are all (or partly) bound *before*
  the Cartesian, those vars don't change inside the Cartesian loop —
  so we can apply them once cooperatively before the loop, then
  cheaply check `valid()` per iteration. The remaining (in-Cartesian)
  vars are applied per-thread inside the loop via `prefix_seq`.

  Every field carries CUDA C++ identifier shape (var names emitted
  into rendered kernel source), so this lives under `codegen/cuda/`.

  Mirrors the legacy `NegPreNarrowInfo` in
  ir/codegen/cuda/context.py.
  '''

  var_name: str
  pre_vars: list[str]
  in_cartesian_vars: list[str]
  pre_consts: list[tuple[int, int]]
  view_var: str
  rel_name: str


@dataclass
class CudaRenderCtx:
  '''CUDA-target-private render scratch threaded through MIR -> IIR.

  All fields hold CUDA C++ identifier scratch or render-time scratch
  read by CUDA-shaped IIR-op emission. The renderer (and the IIR-op
  lowerings that emit CUDA-shaped ops) read these via the forwarding
  properties on `LoweringCtx`; the structural lowering logic does
  not touch them.

  Field tally:
    1. view_var_names              — handle_idx -> C++ view var name
    2. output_var                  — current C++ OutputContext var
    3. output_var_overrides        — rel_name -> override OutputContext
    4. view_slot_bases             — handle_idx -> base slot in views[]
    5. rel_index_types             — rel_name -> C++ index template
    6. tiled_cartesian_valid_var   — tiled-Cart ballot validity var
    7. ws_cartesian_valid_var      — WS Cart ballot validity var
    8. neg_pre_narrow              — handle_idx -> pre-narrowed handle info
    9. debug                       — emit `printf` debug breadcrumbs
   10. tile_var                    — CUB cooperative_groups tile var name

  Total: 10 fields. The spec table (§ 2.2.2) listed `dedup_hash_vars`
  and `ws_cartesian_bound_vars` too — but those live on the legacy
  `CodeGenContext` (`ir/codegen/cuda/context.py`), not on the current
  `LoweringCtx`. Per PR-1d's discipline (trust the monolith), they
  are NOT extracted here. They will land on `CudaRenderCtx` once the
  dialect-level `LoweringCtx` grows them (planned alongside the
  pragma-scratch-flag removal in PR-1e+).
  '''

  view_var_names: dict[str, str] = field(default_factory=dict)
  output_var: str = 'output'
  output_var_overrides: dict[str, str] = field(default_factory=dict)
  view_slot_bases: dict[str, int] = field(default_factory=dict)
  rel_index_types: dict[str, str] = field(default_factory=dict)
  tiled_cartesian_valid_var: str = ''
  ws_cartesian_valid_var: str = ''
  neg_pre_narrow: dict[int, NegPreNarrowInfo] = field(default_factory=dict)
  debug: bool = True
  tile_var: str = 'tile'


__all__ = [
  'CudaRenderCtx',
  'NegPreNarrowInfo',
]
