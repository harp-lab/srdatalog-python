'''CudaRenderCtx — CUDA-render-side identifier scratch state.

Spec: `docs/phase_decomposition_redesign.md` § 3.2.1 (Wave T1).

Today's dialect-level `LoweringCtx` (in
`src/srdatalog/ir/dialects/relation/sorted_array/lowerings/__init__.py`)
mixes generic lowering-walk scope state (bound vars, indent, counter)
with CUDA C++ identifier scratch (view variable names, output variable
name, per-handle pre-narrow info, …). The spec calls out the second
category as CUDA-render-specific — strings that the CUDA renderer's
own helpers emit and consume. Those fields belong here, under
`codegen/cuda/`, not on a target-agnostic `LoweringCtx`.

Per `docs/code_discipline.md` D10, the framework `LowerCtx` (in
`core/lower_ctx.py`) is pinned at 5 fields. This file ships the
CUDA-render-side companion: a target-private dataclass aggregating
every CUDA identifier the lowering walk produces. When future targets
(CPU/TBB, SYCL, …) come online, each ships its own analogous
`<target>RenderCtx`; the lowering walk threads the right one
based on the active target.

The PR-1 transition (per `docs/phase_decomposition_redesign.md` § 6.1)
keeps the dialect `LoweringCtx` as the surface every lowering helper
sees — the CUDA-render fields are still accessible via
`ctx.<field>` for byte-equivalence (the existing helpers don't change
their reads), AND via the canonical `ctx.render_ctx.<field>` form.
Future PRs migrate every read to the canonical form; PR-1 establishes
the shape.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CudaRenderCtx:
  '''CUDA-render-side identifier scratch.

  Every field is a CUDA C++ identifier or a per-handle CUDA-render
  side-table. Nothing here is consumed by a non-CUDA target.

  Fields:
    view_var_names         — handle_idx (str) -> view variable name
                              (`view_<rel>_<handle_idx>`).
    output_var             — name of the single-output OutputContext
                              variable.
    output_var_overrides   — per-relation OutputContext name map for
                              multi-head rules.
    view_slot_bases        — handle_idx (str) -> base slot in `views[]`.
                              Populated from `emit_view_declarations`
                              so D2L segment-loop emission can
                              reference the right HEAD/FULL pair.
    rel_index_types        — rel_name -> custom index type code (e.g.,
                              `Device2LevelIndex`). Empty string /
                              missing entry = plain DSAI.
    tiled_cartesian_valid_var
                           — name of the C++ ballot validity var
                              threaded by `_lower_nested_cart` when
                              rendering the tiled-mode body.
    ws_cartesian_valid_var — name of the C++ WS ballot validity var.
    ws_cartesian_bound_vars
                           — names of C++ WS-loop-bound vars.
    neg_pre_narrow         — handle_idx -> NegPreNarrowInfo (pre-
                              narrowed handle info for Negations
                              following a Cartesian).
    dedup_hash_vars        — names of C++ vars feeding the dedup hash
                              key. (Carried for parity with the legacy
                              CodeGenContext field; today populated by
                              the runner-side plumbing, not by the
                              lowering walk itself.)
    debug                  — emit `Comment` nodes describing the
                              source MIR shape for each emitted block.
  '''

  view_var_names: dict[str, str] = field(default_factory=dict)
  output_var: str = 'output'
  output_var_overrides: dict[str, str] = field(default_factory=dict)
  view_slot_bases: dict[str, int] = field(default_factory=dict)
  rel_index_types: dict[str, str] = field(default_factory=dict)
  tiled_cartesian_valid_var: str = ''
  ws_cartesian_valid_var: str = ''
  ws_cartesian_bound_vars: list[str] = field(default_factory=list)
  # `neg_pre_narrow` is typed loosely (`dict[int, Any]`) because the
  # value class — `NegPreNarrowInfo` — exists in two parallel forms
  # today: the legacy CUDA-side `codegen.cuda.context.NegPreNarrowInfo`
  # and the dialect-side `relation.sorted_array.lowerings.NegPreNarrowInfo`.
  # Both carry the same payload; a future PR collapses the duplicate.
  # Until then, `Any` lets both forms flow through without type-check
  # noise. The runtime behavior is identical.
  neg_pre_narrow: dict[int, Any] = field(default_factory=dict)
  dedup_hash_vars: list[str] = field(default_factory=list)
  debug: bool = True


__all__ = ['CudaRenderCtx']
