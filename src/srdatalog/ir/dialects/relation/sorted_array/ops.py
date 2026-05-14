'''Sorted-array dialect ops — M1 subset.

Each op references its operands by *name* (string) for M1 pragmatism.
The legacy emitter passes view/handle variable names through string
keys; matching that lets the byte-equivalence gate compare directly.
A later milestone replaces this with lexical binding (D8) once the
gate has been validated end-to-end.

Operand naming convention:
  view_name   — name of a previously-declared view variable
                (`auto view_<rel>_<cols>_<ver> = views[<slot>];`).
  handle_name — name of a previously-bound handle variable (declared
                via iir.cf.Bind).
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import final

from srdatalog.ir.core import Op


@final
@dataclass(frozen=True, slots=True)
class SaRoot(Op):
  '''Root handle into the sorted-array's full row-range.

  Lowers (target.cuda) to: `HandleType(0, <view_name>.num_rows_, 0)`.
  '''

  view_name: str


@final
@dataclass(frozen=True, slots=True)
class SaValid(Op):
  '''Whether a handle is non-empty / non-degenerate.

  Lowers (target.cuda) to: `<handle_name>.valid()`.
  '''

  handle_name: str


@final
@dataclass(frozen=True, slots=True)
class SaDegree(Op):
  '''Branching factor / row count at a handle position.

  Lowers (target.cuda) to: `<handle_name>.degree()`.
  '''

  handle_name: str


@final
@dataclass(frozen=True, slots=True)
class SaGetVal(Op):
  '''Get the value at column `col`, row `idx_var_name`, in the view.

  Used inside a root scan: each var binding fetches its column.

  Lowers (target.cuda) to: `<view_name>.get_value(<col>, <idx_var_name>)`.
  '''

  view_name: str
  col: int
  idx_var_name: str


@final
@dataclass(frozen=True, slots=True)
class SaGetValAt(Op):
  '''Get the value at column position `col` of a narrowed handle's
  child at slot `idx_var_name`. Used in nested ColumnJoin paths;
  M1 declares the op for completeness but doesn't yet lower it.

  Lowers (target.cuda) to:
      `<view_name>.get_value_at(<handle_name>.begin(), <idx_var_name>)`.
  '''

  handle_name: str
  view_name: str
  idx_var_name: str


@final
@dataclass(frozen=True, slots=True)
class SaHint(Op):
  '''Range-narrowed root handle constructor (expression-shaped).

  Lowers (target.cuda) to:
      `HandleType(<lo_var>, <hi_var>, <depth>)`.

  Typically composed with SaPrefCoop:
      `HandleType(lo, hi, 0).prefix(root_val, tile, view)`.
  '''

  lo_var: str
  hi_var: str
  depth: int = 0


@final
@dataclass(frozen=True, slots=True)
class SaPrefCoop(Op):
  '''Cooperative prefix-narrowing on a parent handle expression.

  Lowers (target.cuda) to:
      `<parent>.prefix(<key>, tile, <view>)`.

  Used in multi-source root CJ where 32 threads cooperatively
  binary-search the parent handle for `key`.
  '''

  parent: Op
  key_var: str
  view_name: str


@final
@dataclass(frozen=True, slots=True)
class SaPrefSeq(Op):
  '''Sequential (per-thread) prefix-narrowing on a parent handle.

  Lowers (target.cuda) to:
      `<parent>.prefix_seq(<key>, <view>)`.

  Used inside a Cartesian loop where each thread already has its
  own (idx0, idx1, …) decomposition and runs the prefix narrowing
  independently — the cooperative form would be wrong because the
  threads don't agree on `key`.
  '''

  parent: Op
  key_var: str
  view_name: str


@final
@dataclass(frozen=True, slots=True)
class SaIterators(Op):
  '''Iterator pair for a handle, suitable to hand to
  `intersect_handles`.

  Lowers (target.cuda) to: `<handle>.iterators(<view>)`.
  '''

  handle_name: str
  view_name: str


@final
@dataclass(frozen=True, slots=True)
class SaChildRange(Op):
  '''Narrowed child range from a handle.

  Lowers (target.cuda) to:
      `<handle>.child_range(<pos_expr>, <key_var>, tile, <view>)`.

  Used inside an IntersectIter body to produce per-source child
  handles for the next nesting level.
  '''

  handle_name: str
  pos_expr: str
  key_var: str
  view_name: str


@final
@dataclass(frozen=True, slots=True)
class SaGetValAtPos(Op):
  '''Get the value at a column for a row inside a narrowed handle's
  begin-offset.

  Lowers (target.cuda) to:
      `<view>.get_value(<col>, <handle>.begin() + <idx_var>)`.

  Used by the nested CartesianJoin to bind per-source vars from
  the flat-decomposed indices.
  '''

  view_name: str
  col: int
  handle_name: str
  idx_var_name: str


@final
@dataclass(frozen=True, slots=True)
class SaTiledCartesian2D(Op):
  '''2-source nested CartesianJoin with tiled smem pre-load + ballot writes.

  Lowers to the legacy `_emit_tiled_cartesian` shape:

      if (<total_var> > 32) {
        // Tiled Cartesian: smem pre-load reads, standard emit_direct writes
        for (uint32_t <t0_base> = 0; <t0_base> < <degree_var0>; <t0_base> += kCartTileSize) {
          uint32_t <t0_len> = min(<t0_base> + (uint32_t)kCartTileSize, <degree_var0>) - <t0_base>;
          for (uint32_t _ti = <lane_var>; _ti < <t0_len>; _ti += <group_size_var>)
            s_cart[warp_in_block][0][_ti] = <view_var0>.get_value(<col0>, <handle_var0>.begin() + <t0_base> + _ti);
          for (uint32_t <t1_base> = 0; <t1_base> < <degree_var1>; <t1_base> += kCartTileSize) {
            uint32_t <t1_len> = min(<t1_base> + (uint32_t)kCartTileSize, <degree_var1>) - <t1_base>;
            for (uint32_t _ti = <lane_var>; _ti < <t1_len>; _ti += <group_size_var>)
              s_cart[warp_in_block][1][_ti] = <view_var1>.get_value(<col1>, <handle_var1>.begin() + <t1_base> + _ti);
            <tile_var>.sync();
            uint32_t <tile_total> = <t0_len> * <t1_len>;
            for (uint32_t <batch_var> = 0; <batch_var> < <tile_total>; <batch_var> += <group_size_var>) {
              uint32_t <flat_idx_var> = <batch_var> + <lane_var>;
              bool <valid_var> = <flat_idx_var> < <tile_total>;
              auto <var_name0> = <valid_var> ? s_cart[warp_in_block][0][<flat_idx_var> / <t1_len>] : ValueType{0};
              auto <var_name1> = <valid_var> ? s_cart[warp_in_block][1][<flat_idx_var> % <t1_len>] : ValueType{0};
              <tiled_body>
            }
            <tile_var>.sync();
          }
        }
      } else {
        for (uint32_t <fb_batch_var> = 0; <fb_batch_var> < <total_var>; <fb_batch_var> += <group_size_var>) {
          uint32_t <flat_idx_var> = <fb_batch_var> + <lane_var>;
          bool <valid_var> = <flat_idx_var> < <total_var>;
          const bool <major_var> = (<degree_var1> >= <degree_var0>);
          uint32_t <idx0_var>, <idx1_var>;
          if (<major_var>) { <idx0_var> = <flat_idx_var> / <degree_var1>; <idx1_var> = <flat_idx_var> % <degree_var1>; }
          else { <idx1_var> = <flat_idx_var> / <degree_var0>; <idx0_var> = <flat_idx_var> % <degree_var0>; }
          auto <var_name0> = <view_var0>.get_value(<col0>, <handle_var0>.begin() + <idx0_var>);
          auto <var_name1> = <view_var1>.get_value(<col1>, <handle_var1>.begin() + <idx1_var>);
          <fallback_body>
        }
      }

  `body` is the trailing-InsertInto run rendered as a TiledBallotBlock
  (the `ctx.tiled_cartesian_valid_var`-driven InsertInto variant). The
  legacy `_emit_tiled_cartesian` emits `tiled_body` in both branches
  when set — so the dialect carries one body and uses it twice. Body
  emits at the surrounding scope's indent (legacy quirk where body
  was rendered before the tiled wrap textually surrounded it).
  '''

  view_var0: str
  view_var1: str
  handle_var0: str
  handle_var1: str
  col0: int
  col1: int
  var_name0: str
  var_name1: str
  lane_var: str
  group_size_var: str
  total_var: str
  degree_var0: str
  degree_var1: str
  flat_idx_var: str
  t0_base: str
  t1_base: str
  t0_len: str
  t1_len: str
  tile_total: str
  batch_var: str
  valid_var: str
  fb_batch_var: str
  major_var: str
  idx0_var: str
  idx1_var: str
  body: Op


@final
@dataclass(frozen=True, slots=True)
class DedupTryInsert(Op):
  '''COMPOUND op: dedup-hash gate around an emission body.

  This is the *first* COMPOUND op in the codebase — per
  `docs/ir_dialect_contract.md`, COMPOUND ops have no direct renderer
  but carry a registered `@rewrite` that decomposes them into LEAF
  ops before codegen sees them. The rewrite expands to:

      {
        bool _p = dedup_table.try_insert(thread_id, <args...>);
        if (_p) {
          <then_body>
        }
      }

  i.e. `BracedBlock` + `Bind` + `If` from `iir.cf`, with
  `MemberCall` from `iir.expr`. After
  `PassDriver.apply_rewrites_to_fixpoint`, no `DedupTryInsert` remains
  in the tree — only the LEAF decomposition.

  `args` is the tuple of expression-shaped Ops to pass to
  `try_insert` after the leading `thread_id` (the row-keying value
  set that dedup is supposed to suppress duplicates of).

  `then_body` is the statement-shaped Op to execute when the
  try_insert returns true (i.e. this thread "won" the slot and the
  emission should proceed). Typically a `Block` or `LaneZeroGuard`.

  The `dedup_table` and `thread_id` operand names are fixed
  identifiers in the kernel scaffold; they don't appear as fields
  here because they're scaffold-global, not per-call.
  '''

  args: tuple[Op, ...]
  then_body: Op
