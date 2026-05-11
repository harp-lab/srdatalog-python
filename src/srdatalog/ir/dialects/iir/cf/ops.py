'''iir.cf op definitions — control flow.

All ops here are pure data (D1), frozen+slots dataclasses (D2, D3),
@final to lock the closed sum (D11). Names mirror the spec where
possible. Fields stay primitives or tuples of Op (no lists) so
strategy combinators traverse cleanly.

Naming convention for binders:

  Bind(name, expr)  — declare `auto <name> = <expr>;` in the
                       enclosing Block. The name is used verbatim
                       in generated code; the lowering chooses
                       names to match the legacy emitter's bump
                       order for byte-equivalence.

  VarRef(name)      — refer to a previously-bound name.

This explicit string-name approach is the M1 pragmatic choice. The
spec calls for lexical scoping (D8); a future refactor can replace
the string-keyed lookup with proper de Bruijn / Let scopes once the
byte-equivalence gate has been validated end-to-end.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from srdatalog.ir.core import Op


@final
@dataclass(frozen=True, slots=True)
class Block(Op):
  '''A sequence of statements emitted in order.'''

  stmts: tuple[Op, ...]


@final
@dataclass(frozen=True, slots=True)
class IndentBlock(Op):
  '''Render contained statements at +`extra` indent levels.

  Used to model the legacy emitter's mixed-indent quirks where some
  children of a scope are at a different indent than others. The
  most common case: in a root Scan, the var-bind statements are at
  the loop's inner indent while the InsertInto body is at the outer
  indent (because the body was rendered before `inc_indent`).
  '''

  extra: int
  stmts: tuple[Op, ...]


@final
@dataclass(frozen=True, slots=True)
class BlankLine(Op):
  '''Emit a single empty line. Used to match legacy emission where
  whitespace has structural meaning (e.g. between the degree fetch
  and the loop preamble).'''


@final
@dataclass(frozen=True, slots=True)
class Bind(Op):
  '''Declare `auto <name> = <expr>;` (or `<type> <name> = <expr>;`).

  `expr` is an expression-shaped Op; the target lowering renders it
  via emit_expr().
  '''

  name: str
  expr: Op
  type_decl: str = 'auto'


@final
@dataclass(frozen=True, slots=True)
class VarRef(Op):
  '''Refer to a previously-bound name. Renders as the bare name.'''

  name: str


@final
@dataclass(frozen=True, slots=True)
class IfReturnIfNot(Op):
  '''`if (!<cond>) return;` — the validity guard pattern.'''

  cond: Op


@final
@dataclass(frozen=True, slots=True)
class IfContinueIfNot(Op):
  '''`if (!<cond>) continue;` — the inner-loop validity guard.

  Used inside grid-stride loops over root_unique_values: a failed
  prefix narrowing on any source means this root_val has no
  intersection, so skip to the next iteration.
  '''

  cond: Op


@final
@dataclass(frozen=True, slots=True)
class IfReturn(Op):
  '''`if (<cond>) return;` — non-inverted early-exit guard.

  Mirror of `IfReturnIfNot` for cases where the condition is
  naturally positive (e.g. "if any source is invalid, return";
  the condition itself is the disjunction of invalidity checks).
  '''

  cond: Op


@final
@dataclass(frozen=True, slots=True)
class IfContinue(Op):
  '''`if (<cond>) continue;` — non-inverted inner-loop early-skip.

  Mirror of `IfContinueIfNot` for cases where the condition is
  naturally positive.
  '''

  cond: Op


@final
@dataclass(frozen=True, slots=True)
class CartesianFlatLoop(Op):
  '''Flat for-loop over the Cartesian product, partitioned by lane.

  Lowers (target.cuda) to:
      for (uint32_t <idx_var> = <lane_var>;
           <idx_var> < <bound_var>;
           <idx_var> += <group_size_var>) { <body> }

  Used by nested CartesianJoin: each thread in the tile takes a
  share of the Cartesian product based on its `lane_var =
  tile.thread_rank()` and stride `group_size_var = tile.size()`.
  '''

  idx_var: str
  bound_var: str
  lane_var: str
  group_size_var: str
  body: Op


@final
@dataclass(frozen=True, slots=True)
class Cartesian2DDecompose(Op):
  '''Adaptive 2-source flat-index decomposition.

  Lowers (target.cuda) to:
      const bool <major_var> = (<deg1_var> >= <deg0_var>);
      uint32_t <idx0_var>, <idx1_var>;
      if (<major_var>) {
        <idx0_var> = <flat_idx_var> / <deg1_var>;
        <idx1_var> = <flat_idx_var> % <deg1_var>;
      } else {
        <idx1_var> = <flat_idx_var> / <deg0_var>;
        <idx0_var> = <flat_idx_var> % <deg0_var>;
      }

  Picking which source is the divisor based on relative size keeps
  the modulus on the smaller dimension — matches the legacy
  `_nested_column_join_multi`'s adaptive shape.
  '''

  major_var: str
  idx0_var: str
  idx1_var: str
  flat_idx_var: str
  deg0_var: str
  deg1_var: str


@final
@dataclass(frozen=True, slots=True)
class CartesianNDecompose(Op):
  '''Countdown-remainder decomposition for an N-source flat index
  (N >= 3).

  Lowers (target.cuda) to:
      uint32_t remaining = <flat_idx>;
      uint32_t <idx_{N-1}> = remaining % <deg_{N-1}>;
      remaining /= <deg_{N-1}>;
      uint32_t <idx_{N-2}> = remaining % <deg_{N-2}>;
      remaining /= <deg_{N-2}>;
      ...
      uint32_t <idx_0> = remaining % <deg_0>;     (no final div)

  The 2-source case has its own adaptive `Cartesian2DDecompose`
  with `major_is_1` runtime flag — N>=3 doesn't bother with the
  adaptive branch.
  '''

  flat_idx_var: str
  idx_vars: tuple[str, ...]
  deg_vars: tuple[str, ...]


@final
@dataclass(frozen=True, slots=True)
class IntersectIter(Op):
  '''Intersect-and-iterate over multiple narrowed handles.

  Lowers (target.cuda) to:

      auto <intersect_var> = intersect_handles(tile, <iter_exprs...>);
      for (auto <iter_var> = <intersect_var>.begin();
           <iter_var>.valid(); <iter_var>.next()) {
        auto <value_var> = <iter_var>.value();
        auto positions = <iter_var>.positions();
        <body>
      }

  `iterator_exprs` are expression-shaped ops (typically SaIterators)
  that produce the per-source iterator pairs handed to
  intersect_handles. The literal name `positions` is part of the
  legacy convention; child_range calls inside the body reference it.

  Indent quirk under D2L segment loops: the `value`/`positions`
  lines and the body anchor against the OUTER indent
  (`ctx.indent_level - ctx.segment_depth`), not against the
  IntersectIter's own indent. This mirrors the legacy
  `_nested_column_join_multi` where `seg_indent` is a string-only
  offset and `ind(ctx)` (the structural indent) is unaffected by
  segment loops. The emit takes care of this via EmitCtx.segment_depth.
  '''

  intersect_var: str
  iter_var: str
  iterator_exprs: tuple[Op, ...]
  value_var: str
  body: Op


@final
@dataclass(frozen=True, slots=True)
class If(Op):
  '''`if (<cond>) { <body> }` — body emitted at the SAME indent as
  the wrapping `if` (matches the legacy emitter's no-inc-indent
  quirk for filter chains, where the body was rendered before the
  wrap was applied).

  Use IndentBlock inside `body` if some inner statements need to
  go deeper than the outer indent.
  '''

  cond: Op
  body: Op


@final
@dataclass(frozen=True, slots=True)
class GridStrideLoop(Op):
  '''Warp-strided grid-stride for-loop with body.

  Lowers to:
      for (uint32_t <idx_name> = warp_id;
           <idx_name> < <bound>;
           <idx_name> += num_warps) {
        <body>
      }
  '''

  idx_name: str
  bound: Op
  body: Op


@final
@dataclass(frozen=True, slots=True)
class ParallelFor(Op):
  '''Parallel-execution scaffold. The body is run by N workers
  according to the strategy. M1 supports only `warp_strided` (GPU
  warp-strided grid-stride).

  Strategy is a string for now; later milestones promote it to a
  proper sub-dialect (par.data.warp_strided, par.data.tbb_for, …).
  '''

  strategy: str
  body: Op


@final
@dataclass(frozen=True, slots=True)
class Phase(Op):
  '''Counting (mode='C') or materialize (mode='M') scope. The same
  body emits differently inside each phase via the surrounding
  OutputContext template; the IR carries the intent but the legacy
  emitter currently only emits the unified body.'''

  mode: str
  body: Op


@final
@dataclass(frozen=True, slots=True)
class LaneZeroGuard(Op):
  '''`if (tile.thread_rank() == 0) <body>` — single-thread guard
  applied around output writes when not inside a Cartesian (so 32
  cooperating threads don't all emit the same row).'''

  body: Op


@final
@dataclass(frozen=True, slots=True)
class WriteOutput(Op):
  '''Emit a row to the output context.

  Lowers to `<output_var>.emit_direct(<values>)` in materialize phase
  or `<output_var>.emit_direct()` in count phase (the polymorphic
  OutputContext template handles the dispatch at C++ level).
  '''

  output_var: str
  values: tuple[Op, ...]


@final
@dataclass(frozen=True, slots=True)
class AddCount(Op):
  '''Bump the count counter directly. Used by the count-as-product
  short-circuit (R1) and by counting-only paths.

  Lowers to `<output_var>.add_count(<delta>);`.
  '''

  output_var: str
  delta: Op


@final
@dataclass(frozen=True, slots=True)
class Comment(Op):
  '''Emit a `// ...` comment. Pass-through to the C++ source. The
  legacy emitter sprinkles these for debugging; the dialect carries
  them as IR so byte-equivalence preserves them.'''

  text: str


@final
@dataclass(frozen=True, slots=True)
class RawString(Op):
  '''Escape hatch for emission templates we haven't dialectified yet.
  Carries a literal string into the C++ output. The byte-equivalence
  port uses RawString sparingly to bridge gaps as it ports each MIR
  op kind. Each use is a candidate for replacement by a proper IR op
  in a later milestone.'''

  text: str


@final
@dataclass(frozen=True, slots=True)
class TiledBallotBlock(Op):
  '''Multi-output ballot-coalesced write block used inside tiled-
  Cartesian materialize emission.

  Lowers (target.cuda) to:

      {
        uint32_t _tc_ballot = tile.ballot(<valid_var>);
        uint32_t _tc_active = __popc(_tc_ballot);
        if (_tc_active > 0) {
          uint32_t _tc_mask = (1u << tile.thread_rank()) - 1u;
          uint32_t _tc_off = __popc(_tc_ballot & _tc_mask);
          for each output (dest_idx, values):
            if (<valid_var>) {
              uint32_t _tc_pos_<dest_idx> = old_size_<dest_idx>
                + warp_write_base + warp_local_count + _tc_off;
              output_data_<dest_idx>[col * static_cast<uint32_t>(
                output_stride_<dest_idx>) + _tc_pos_<dest_idx>] = vN;
              ...
            }
          warp_local_count += _tc_active;
        }
      }

  `outputs` is a tuple of `(dest_idx, sanitized_values, debug_text)`.
  Multi-head pipelines emit several entries; the ballot setup +
  `_tc_active` increment happen once around all of them. Replaces the
  legacy `tiled_cartesian_ballot_done` flag on `CodeGenContext`.
  '''

  valid_var: str
  outputs: tuple[tuple[int, tuple[str, ...], str], ...]


@final
@dataclass(frozen=True, slots=True)
class OuterAnchor(Op):
  '''Render `body` at the surrounding scope's indent (`ctx.indent_level
  - ctx.segment_depth`), regardless of how deep the wrapping
  D2lSegmentLoops have nested.

  Used to embed a CJ-multi body_op INSIDE a root-CJ D2lSegmentLoop's
  body (so the segment loop's brace closes AFTER the body) while
  keeping the body's first-line indent at the outer kernel level —
  matches the legacy `_root_cj_multi` pattern of pre-rendering body
  at the outer indent and letting the segment loops wrap textually
  around it.

  Resets `segment_depth` to 0 inside `body` so any further nested
  IntersectIter / D2lSegmentLoop in body anchors against the new
  (outer) base.
  '''

  body: Op
