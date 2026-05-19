'''MIR -> IIR lowering for the sorted_array dialect.

Each milestone extends `lower_scan_pipeline` to handle more MIR op
kinds. The supported predicate `_supported_pipeline` documents which
shapes the dialect can faithfully reproduce against the legacy
emitter.

  M1: [Scan, InsertInto]
  M2: [Scan, (Filter | ConstantBind)*, InsertInto]
  M3: [CJ_multi (Filter | ConstantBind | CJ_multi)*, InsertInto]

Counter management mirrors the legacy `pipeline.py` save/restore
pattern: the body of a root op is lowered with a fresh counter
trajectory, then the root op's own scaffold takes the counter from
the same starting point. The numeric suffixes baked into IIR names
match what `gen_unique_name` would have produced in legacy.
'''

from __future__ import annotations

from dataclasses import dataclass, field

import srdatalog.ir.mir.types as mir
from srdatalog.ir.core import Op
from srdatalog.ir.dialects.iir.cf import (
  AddCount,
  Bind,
  BlankLine,
  Block,
  Cartesian2DDecompose,
  CartesianFlatLoop,
  CartesianNDecompose,
  Comment,
  GridStrideLoop,
  If,
  IfContinueIfNot,
  IfReturnIfNot,
  IndentBlock,
  IntersectIter,
  LaneZeroGuard,
  OuterAnchor,
  ParallelFor,
  RawString,
  TiledBallotBlock,
  VarRef,
)
from srdatalog.ir.dialects.parallel.data.block_group import (
  BgRootCjMulti,
  BgSourceSpec,
)
from srdatalog.ir.dialects.relation.d2l import D2lSegmentLoop, view_count
from srdatalog.ir.dialects.relation.sorted_array.ops import (
  SaChildRange,
  SaDegree,
  SaGetVal,
  SaGetValAt,
  SaGetValAtPos,
  SaHint,
  SaIterators,
  SaPrefCoop,
  SaPrefSeq,
  SaRoot,
  SaTiledCartesian2D,
  SaValid,
)
from srdatalog.ir.hir.types import Version


@dataclass
class NegPreNarrowInfo:
  '''Pre-narrowed handle info for a Negation that follows a Cartesian.

  When a Negation's prefix vars are all (or partly) bound *before*
  the Cartesian, those vars don't change inside the Cartesian loop —
  so we can apply them once cooperatively before the loop, then
  cheaply check `valid()` per iteration. The remaining (in-Cartesian)
  vars are applied per-thread inside the loop via `prefix_seq`.

  Mirrors the legacy `NegPreNarrowInfo` in
  ir/dialects/target/cuda/context.py.
  '''

  var_name: str
  pre_vars: list[str]
  in_cartesian_vars: list[str]
  pre_consts: list[tuple[int, int]]
  view_var: str
  rel_name: str


@dataclass
class LoweringCtx:
  '''Mutable state during MIR -> IIR walk.

  Mirrors the legacy `CodeGenContext` for the fields that matter to
  the dialect's emission decisions today. Other legacy fields
  (tiled_cartesian state, ws state, etc.) aren't needed yet —
  milestones add them as they cover those paths.
  '''

  name_counter: int = 0
  view_var_names: dict[str, str] = field(default_factory=dict)
  is_counting: bool = False
  inside_cartesian: bool = False
  output_var: str = 'output'
  tile_var: str = 'tile'
  debug: bool = True
  output_var_overrides: dict[str, str] = field(default_factory=dict)
  bound_vars: list[str] = field(default_factory=list)
  # rel_name -> custom index type code (e.g. 'Device2LevelIndex').
  # Empty string / missing entry = plain DSAI single-segment.
  rel_index_types: dict[str, str] = field(default_factory=dict)
  # handle_idx (str) -> base slot in views[]. Populated by
  # `compile_kernel_body` from `emit_view_declarations` so D2L
  # segment-loop emission can reference the right HEAD/FULL pair.
  view_slot_bases: dict[str, int] = field(default_factory=dict)
  # When True, InsertInto emit wraps the output write in a
  # `{ bool _p = dedup_table.try_insert(thread_id, ...); if (_p) {
  # ... } }` gate, and the materialize-phase write goes through
  # `atomicAdd(atomic_write_pos, 1u)` + `out_data_0[...]` instead of
  # `output.emit_direct(...)`. Threaded from `ep.dedup_hash`.
  dedup_hash: bool = False
  # When True, eligible 2-source / 1-var-per-source nested Cartesians
  # in materialize phase emit the `if (total > 32) { tiled smem path }
  # else { fallback path }` dispatch. Bodies inside both branches use
  # the `tiled_cartesian_valid_var` ballot-write variant of InsertInto.
  # Threaded from `_dialect_safe_kernel`'s `tiled_cartesian_eligible`
  # gate via `compile_kernel_body`.
  tiled_cartesian: bool = False
  # Empty (default) — InsertInto emits the standard
  # `output.emit_direct(...)` write. Non-empty — emits the tiled-
  # Cartesian ballot-write variant guarded by this var. Set by
  # `_lower_nested_cart` when rendering the tiled-mode body.
  tiled_cartesian_valid_var: str = ''
  # Work-stealing flag (mirrors legacy `ctx.ws_enabled`). In count
  # phase, InsertInto emits `<output_var>++` instead of
  # `<output_var>.emit_direct()` — the WS count uses a per-thread
  # local counter that the runner aggregates. The legacy emitter has
  # only this kernel-functor-level WS support; the runner-side WS
  # scaffolding (WCOJTask queue) was never finished.
  ws_enabled: bool = False
  # Work-stealing batched-Cartesian valid var. When set, Filter /
  # Negation fold their guard into `<v> = <v> && (<cond>);` and
  # InsertInto materialize emits `<output_var>.emit_warp_coalesced(
  # tile, <v>, <args>)` — a cooperative warp write instead of a
  # lane-zero-guarded emit_direct. Mirrors legacy
  # `ctx.ws_cartesian_valid_var`.
  ws_cartesian_valid_var: str = ''
  # Block-group flag (mirrors legacy `ctx.bg_enabled`). When True,
  # the root multi-source ColumnJoin emits via `BgRootCjMulti`
  # (block-group work-balanced partition + binary-search key loop)
  # instead of the standard grid-stride root_unique_values loop.
  # Cleared when descending into the body (the BG root's narrowed
  # handle already restricts work to this warp's slice).
  bg_enabled: bool = False
  # State-key -> handle var name. Lets nested CJ find the parent
  # handle to alias by the same (rel, cols, prefix_vars, ver) key
  # that the outer CJ used to register it.
  handle_vars: dict[str, str] = field(default_factory=dict)
  # Cartesian-bound var names. Used to decide which Negation prefix
  # vars are pre-Cartesian (= bound by an outer scope) vs in-Cartesian
  # (= bound by the current Cart and per-thread).
  cartesian_bound_vars: list[str] = field(default_factory=list)
  # handle_idx -> NegPreNarrowInfo. Populated by `_lower_nested_cart`
  # before its body renders so that the body's Negation handler can
  # pick up the pre-allocated handle.
  neg_pre_narrow: dict[int, NegPreNarrowInfo] = field(default_factory=dict)

  def fresh(self, prefix: str) -> str:
    self.name_counter += 1
    return f'{prefix}_{self.name_counter}'


# -----------------------------------------------------------------------------
# Public entry point + supported-shape predicate
# -----------------------------------------------------------------------------


def _trailing_inserts(rest: list[mir.MirNode]) -> list[mir.InsertInto]:
  '''Return the trailing run of InsertIntos at the end of `rest`.

  Multi-head rules emit several InsertIntos in sequence at the end
  of the pipeline. The legacy emitter walks them all in order.
  '''
  out: list[mir.InsertInto] = []
  for op in rest:
    if isinstance(op, mir.InsertInto):
      out.append(op)
    elif out:
      # Non-InsertInto after InsertInto: the trailing run is just
      # the contiguous tail. Stop here and let the caller decide.
      break
  return out


def _middle_ops(rest: list[mir.MirNode]) -> list[mir.MirNode]:
  '''Return `rest` with the trailing InsertIntos stripped off.'''
  trailing_count = len(_trailing_inserts(rest))
  return rest[: len(rest) - trailing_count] if trailing_count else list(rest)


def _supported_pipeline(ops: list[mir.MirNode]) -> bool:
  '''True iff the dialect can lower this pipeline shape today.

  The pipeline must end in one or more InsertIntos (multi-head rules
  emit several outputs from the same body). The middle ops between
  the head op and the first InsertInto are constrained per the
  milestone (Scan/CJ/Cart/Filter/ConstantBind/Negation).
  '''
  if len(ops) < 2:
    return False
  # Find the start of the trailing InsertInto sequence.
  insert_start = None
  for i, op in enumerate(ops):
    if isinstance(op, mir.InsertInto):
      insert_start = i
      break
  if insert_start is None or insert_start == 0:
    return False
  # All ops from insert_start onward must be InsertInto.
  if not all(isinstance(op, mir.InsertInto) for op in ops[insert_start:]):
    return False
  head = ops[0]
  middle = ops[1:insert_start]

  if isinstance(head, mir.Scan):
    # M1+M2+M5 + Scan+Cart (R8) shapes. CartesianJoin in the middle
    # dispatches to `_lower_nested_cart` via `_lower_inner_chain`,
    # which already handles 1+ source forms (with prefix narrowing).
    # Hit by ddisasm StackLiveVarBlockEnd1_D0_splitB.
    #
    # C5: `mir.TiledCartesian` (the wrap op the `TiledCartesian`
    # pragma materializes around an eligible Cartesian) is accepted
    # in the same slot — `_lower_inner_chain` dispatches it through
    # `lower_tiled_cartesian_in_chain`.
    #
    # B-CJ-single: single-source ColumnJoin (`len(sources) == 1`)
    # is accepted in the middle slot via `_lower_nested_cj_single` —
    # see `_lower_inner_chain` for the dispatch + the new path under
    # `USE_DECLARATIVE`.
    for op in middle:
      if isinstance(op, (mir.Filter, mir.ConstantBind)):
        continue
      if isinstance(op, mir.Negation):
        continue
      if isinstance(op, mir.CartesianJoin):
        continue
      if isinstance(op, mir.TiledCartesian):
        continue
      if isinstance(op, mir.ColumnJoin) and len(op.sources) == 1:
        continue
      return False
    return True

  if isinstance(head, mir.ColumnJoin) and len(head.sources) >= 2:
    # M3+M5+M7+M5.x shape: multi-source root CJ; middle can hold
    # nested CJs / Filter / ConstantBind / Negation / Cartesian.
    # C5: same `mir.TiledCartesian` allowance as the Scan-rooted case.
    # B-CJ-single: same single-source ColumnJoin allowance as above.
    for op in middle:
      if isinstance(op, (mir.Filter, mir.ConstantBind)):
        continue
      if isinstance(op, mir.ColumnJoin) and len(op.sources) >= 2:
        continue
      if isinstance(op, mir.ColumnJoin) and len(op.sources) == 1:
        continue
      if isinstance(op, mir.CartesianJoin):
        continue
      if isinstance(op, mir.TiledCartesian):
        continue
      if isinstance(op, mir.Negation):
        continue
      return False
    return True

  if isinstance(head, mir.CartesianJoin):
    # M7.x: root CartesianJoin followed by trailing InsertIntos.
    # No middle ops yet (no fixture uses Cart-then-Filter-then-Insert).
    return len(middle) == 0

  return False


def lower_scan_pipeline(
  ops: list[mir.MirNode],
  ctx: LoweringCtx,
) -> Op:
  '''Lower a supported pipeline shape to IIR.

  The function name is historical (M1 only handled Scan-rooted
  pipelines); it now dispatches on the head op. Raises ValueError
  if the shape isn't supported.

  C3 (per docs/phase_c_pragma_materialization.md §4.1): when the
  head op is a `BlockGroupRoot` wrap (inserted by the `BlockGroup`
  pragma handler at MirPragmaPass time), unwrap it and re-dispatch
  through `_lower_root_cj_bg` with `ctx.bg_enabled=True` for the
  duration. The unwrap happens here (not in the wrap op's
  registered `@lowering`) because the wrap op only carries the
  root; the trailing `rest = ops[1:]` lives on the enclosing
  pipeline and is naturally accessible at this dispatch site.
  '''
  head = ops[0] if ops else None
  if isinstance(head, mir.BlockGroupRoot):
    inner = head.inner
    rest = ops[1:]
    if not isinstance(inner, mir.ColumnJoin) or len(inner.sources) < 2:
      raise ValueError(
        f'lower_scan_pipeline: BlockGroupRoot wraps unsupported inner op '
        f'{type(inner).__name__!r}; expected multi-source ColumnJoin'
      )
    # Construct the unwrapped pipeline for the supported-shape check —
    # the BG path is structurally identical to non-BG root CJ multi,
    # so reusing `_supported_pipeline` here catches malformed rest.
    if not _supported_pipeline([inner, *rest]):
      raise ValueError(
        f'lower_scan_pipeline: unsupported pipeline shape '
        f'{[type(o).__name__ for o in [inner, *rest]]} under BlockGroupRoot'
      )
    prev_bg = ctx.bg_enabled
    try:
      ctx.bg_enabled = True
      return _lower_root_cj_bg(inner, rest, ctx)
    finally:
      ctx.bg_enabled = prev_bg

  if not _supported_pipeline(ops):
    raise ValueError(
      f'lower_scan_pipeline: unsupported pipeline shape {[type(o).__name__ for o in ops]}'
    )

  head = ops[0]
  rest = ops[1:]

  # Phase B / Wave 2A (per docs/phase_b_lowering_dispatcher.md §5):
  # ROOT-position MIR op types registered in `USE_DECLARATIVE` route
  # through their per-op `@lowering` rules instead of the legacy
  # `if isinstance` branches below. Mirrors the `_lower_inner_chain`
  # dispatch block for middle-of-chain ops (Filter / ConstantBind).
  # Scan is the first root op migrated (B-Scan); future root-op
  # migrations (B-CJ-multi, B-Cart) extend this block.
  # Layer 3 cleanup deletes both branches AND `USE_DECLARATIVE` once
  # every MIR op is migrated. Imports are function-local to keep the
  # module import graph linear (the sibling per-op modules import
  # back from this file).
  #
  # B-CJ-multi (per docs/phase_b_lowering_dispatcher.md §4 row
  # B-CJ-multi, order 6, difficulty `hard`): root-position multi-
  # source `mir.ColumnJoin` routes through `lower_mir_cj_multi_root`,
  # which delegates to `_lower_root_cj_multi`. Single-source CJ is
  # not a fixture at root today (`_supported_pipeline` requires
  # `len(sources) >= 2` for root CJ); the shape split here mirrors
  # the chain dispatcher's split in `_lower_inner_chain`.
  #
  # BlockGroup coexistence: the `BlockGroupRoot` unwrap above this
  # block (entered BEFORE `_should_use_declarative` is consulted)
  # handles the typed-pragma path; the legacy dual-write path
  # (`ep.block_group=True` + bare `ColumnJoin` head) reaches this
  # branch with `ctx.bg_enabled` already True, and the
  # `_lower_root_cj_multi` helper's own first-statement guard
  # (`if ctx.bg_enabled: return _lower_root_cj_bg(...)`) re-routes
  # to the BG variant. Both BG paths remain byte-equivalent.
  if _should_use_declarative(head):
    if isinstance(head, mir.Scan):
      from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_scan import (
        lower_mir_scan_in_chain,
      )

      return lower_mir_scan_in_chain(head, rest, ctx)
    if isinstance(head, mir.CartesianJoin):
      # B-Cart root entry: `_should_use_declarative` returns True for
      # all `mir.CartesianJoin` instances (no source-count guard,
      # unlike B-CJ-single). The C5 `mir.TiledCartesian` wrap op
      # only wraps NESTED Cartesians (`_wrap_eligible_carts` slices
      # `pipeline[1:]`), so root position never sees a wrap — only
      # bare `mir.CartesianJoin`. See `lowerings/lower_mir_cart.py`
      # module docstring for the full coexistence rationale.
      from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_cart import (
        lower_mir_cart_root,
      )

      return lower_mir_cart_root(head, rest, ctx)
    if isinstance(head, mir.ColumnJoin) and len(head.sources) >= 2:
      from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_cj_multi import (
        lower_mir_cj_multi_root,
      )

      return lower_mir_cj_multi_root(head, rest, ctx)
    raise AssertionError(
      f'lower_scan_pipeline: USE_DECLARATIVE contains {type(head).__name__!r} '
      f'(and `_should_use_declarative` returned True) but no root-'
      f'dispatch wiring exists for it. Add the `lower_mir_<op>_in_chain` '
      f'import + call above.'
    )

  if isinstance(head, mir.Scan):
    return _lower_root_scan(head, rest, ctx)
  if isinstance(head, mir.ColumnJoin):
    return _lower_root_cj_multi(head, rest, ctx)
  if isinstance(head, mir.CartesianJoin):
    return _lower_root_cart(head, rest, ctx)

  raise AssertionError('unreachable')


# -----------------------------------------------------------------------------
# Root Scan (M1+M2)
# -----------------------------------------------------------------------------


def _lower_root_scan(
  scan_op: mir.Scan,
  rest: list[mir.MirNode],
  ctx: LoweringCtx,
) -> Op:
  handle_idx = scan_op.handle_start
  view_var = ctx.view_var_names.get(str(handle_idx), '')
  if not view_var:
    raise ValueError(f'_lower_root_scan: no view var for handle_idx {handle_idx}')

  middle = _middle_ops(rest)
  trailing = _trailing_inserts(rest)

  # Step 1: render body BEFORE allocating own counter-bumped names.
  # Mirrors the legacy `pipeline.py` save/restore around body
  # rendering — body sees counter starting at the saved value, our
  # own outer scaffold restarts from the same saved value. Without
  # this, body ops that bump counter (Negation, nested CJ) get
  # higher counter values than legacy.
  pushed_var_count = 0
  for var_name in scan_op.vars:
    ctx.bound_vars.append(var_name)
    pushed_var_count += 1
  saved_counter = ctx.name_counter
  body_op = _lower_inner_chain(rest, ctx)
  ctx.name_counter = saved_counter
  for _ in range(pushed_var_count):
    ctx.bound_vars.pop()

  # Step 2: allocate own scaffold names.
  outer_stmts: list[Op] = []

  if ctx.debug:
    outer_stmts.append(
      Comment(text=f'Root Scan: {scan_op.rel_name} binding {", ".join(scan_op.vars)}')
    )
    outer_stmts.append(
      Comment(
        text=f'MIR: (scan :rel {scan_op.rel_name} '
        f':vars ({" ".join(scan_op.vars)}) :handle {handle_idx})'
      )
    )

  handle_var = ctx.fresh('root_handle')
  degree_var = ctx.fresh('degree')
  idx_var = ctx.fresh('idx')

  # Count-phase var elision: in count phase, mirror Nim's
  # `varName notin body` substring check that says "if this var
  # doesn't show up anywhere in the body, don't bother binding it."
  # The structural `_scan_var_used` predicate isn't enough — it
  # returns True for any var in InsertInto.vars even when the body
  # never emits those vars (R1 / cartesian_as_product short-circuits
  # the InsertInto entirely).
  #
  # S3A.2 (lowering ↔ render separation): use Print_i (the canonical
  # IIR s-expr form) instead of calling the C++ render. Both forms
  # contain every var name that appears in any IIR string field, so
  # substring semantics are preserved. The lowering no longer depends
  # on codegen.cuda.emit — IIR exists as data, only Print walks it.
  body_text_for_elision = ''
  if ctx.is_counting:
    from srdatalog.ir.print_iir import print_iir

    body_text_for_elision = print_iir(body_op)

  var_bind_stmts: list[Op] = []
  for col, var_name in enumerate(scan_op.vars):
    if ctx.is_counting:
      if var_name not in body_text_for_elision:
        continue
    var_bind_stmts.append(
      Bind(
        name=_sanitize_var_name(var_name),
        expr=SaGetVal(view_name=view_var, col=col, idx_var_name=idx_var),
      )
    )

  inner_stmts: list[Op] = []
  if var_bind_stmts:
    inner_stmts.append(IndentBlock(extra=1, stmts=tuple(var_bind_stmts)))

  inner_stmts.append(body_op)

  loop = GridStrideLoop(
    idx_name=idx_var,
    bound=VarRef(name=degree_var),
    body=Block(stmts=tuple(inner_stmts)),
  )

  # Single-view root scan: handle bind + validity check (return) +
  # degree + parallel-for. Matches Nim's `jitRootScan`
  # (codegen/target_jit/jit_root.nim:61-126), which does NOT
  # segment-wrap for D2L FULL_VER — see docs/milestones.md
  # "Nim-reference audit" for the gap (Nim-also-broken on FULL_VER's
  # HEAD/FULL split).
  outer_stmts.extend(
    [
      Bind(name=handle_var, expr=SaRoot(view_name=view_var)),
      IfReturnIfNot(cond=SaValid(handle_name=handle_var)),
      Bind(
        name=degree_var,
        expr=SaDegree(handle_name=handle_var),
        type_decl='uint32_t',
      ),
      BlankLine(),
      ParallelFor(strategy='warp_strided', body=loop),
    ]
  )

  return Block(stmts=tuple(outer_stmts))


# -----------------------------------------------------------------------------
# Root CartesianJoin (M7.x)
# -----------------------------------------------------------------------------


def _lower_root_cart(
  cart_op: mir.CartesianJoin,
  rest: list[mir.MirNode],
  ctx: LoweringCtx,
) -> Op:
  '''Lower a root CartesianJoin.

  Mirrors `jit_root_cartesian_join` in `ir/dialects/target/cuda/root.py`. Differs
  from nested Cart in several places:
    - No `lane`/`group_size`; uses `warp_id`/`num_warps` directly via
      a standard GridStrideLoop.
    - Per-source handles built from gen_root_handle (no aliases).
    - `return` (not `continue`) for validity / total-zero checks.
    - 2-source case: plain row-major decomposition (idx0 = flat/d1,
      idx1 = flat%d1) — no adaptive `major_is_1`.
    - Var binds use `<handle>.get_value_at(<view>, <idx>)` (SaGetValAt).

  Counter trajectory mirrors legacy: body is rendered first (with
  inside_cartesian=True), then scaffold names allocated in the order
  handle, degree per source, then total, flat_idx, idx vars.
  '''
  num_sources = len(cart_op.sources)
  assert num_sources >= 1

  # Step 1: render body with inside_cartesian=True so InsertInto
  # drops the lane-0 guard.
  saved_inside = ctx.inside_cartesian
  ctx.inside_cartesian = True
  pushed_var_count = 0
  for vars_from_src in cart_op.var_from_source:
    for v in vars_from_src:
      ctx.bound_vars.append(v)
      pushed_var_count += 1
  body_op = _lower_inner_chain(rest, ctx)
  for _ in range(pushed_var_count):
    ctx.bound_vars.pop()
  ctx.inside_cartesian = saved_inside

  # Step 2: allocate scaffold names. Order matches legacy: per
  # source (handle, degree), then total, flat_idx, idx vars.
  outer_stmts: list[Op] = []
  if ctx.debug:
    outer_stmts.append(
      Comment(
        text=f'Root CartesianJoin: bind {", ".join(cart_op.vars)} from {num_sources} source(s)'
      )
    )
    src_debug = ' '.join(f'({s.rel_name} :handle {s.handle_start})' for s in cart_op.sources)
    outer_stmts.append(
      Comment(
        text=f'MIR: (cartesian-join :vars ({" ".join(cart_op.vars)}) :sources ({src_debug} ))'
      )
    )

  handle_var_names: list[str] = []
  view_var_names: list[str] = []
  degree_var_names: list[str] = []

  for src in cart_op.sources:
    assert isinstance(src, mir.ColumnSource)
    handle_var = ctx.fresh(f'h_{src.rel_name}_{src.handle_start}')
    deg_var = ctx.fresh('degree')
    src_view = ctx.view_var_names.get(str(src.handle_start), '')
    if not src_view:
      raise ValueError(f'_lower_root_cart: no view var for source handle_idx {src.handle_start}')
    handle_var_names.append(handle_var)
    view_var_names.append(src_view)
    degree_var_names.append(deg_var)
    outer_stmts.append(Bind(name=handle_var, expr=SaRoot(view_name=src_view)))
  outer_stmts.append(BlankLine())

  # Combined validity check uses `return` (root level), not `continue`.
  validity_parts = ' || '.join(f'!{h}.valid()' for h in handle_var_names)
  outer_stmts.append(RawString(text=f'if ({validity_parts}) return;'))
  outer_stmts.append(BlankLine())

  for i in range(num_sources):
    outer_stmts.append(
      Bind(
        name=degree_var_names[i],
        expr=SaDegree(handle_name=handle_var_names[i]),
        type_decl='uint32_t',
      )
    )

  total_var = ctx.fresh('total')
  outer_stmts.append(
    Bind(
      name=total_var,
      expr=RawString(text=' * '.join(degree_var_names)),
      type_decl='uint32_t',
    )
  )
  outer_stmts.append(RawString(text=f'if ({total_var} == 0) return;'))
  outer_stmts.append(BlankLine())

  flat_idx_var = ctx.fresh('flat_idx')

  # idx_vars allocation order matches legacy.
  idx_vars: list[str] = []
  if num_sources == 1:
    idx_vars = [ctx.fresh('idx0')]
  else:
    idx_vars = [ctx.fresh(f'idx{s}') for s in range(num_sources)]

  # Inner loop body: idx decompose at +1 indent + var-binds at +1 indent
  # + body at outer indent (legacy quirk).
  inner_decompose_stmts: list[Op] = []
  if num_sources == 1:
    inner_decompose_stmts.append(
      Bind(
        name=idx_vars[0],
        expr=VarRef(name=flat_idx_var),
        type_decl='uint32_t',
      )
    )
  elif num_sources == 2:
    # Plain row-major (NO adaptive major_is_1 at root).
    inner_decompose_stmts.append(
      Bind(
        name=idx_vars[0],
        expr=RawString(text=f'{flat_idx_var} / {degree_var_names[1]}'),
        type_decl='uint32_t',
      )
    )
    inner_decompose_stmts.append(
      Bind(
        name=idx_vars[1],
        expr=RawString(text=f'{flat_idx_var} % {degree_var_names[1]}'),
        type_decl='uint32_t',
      )
    )
  else:
    inner_decompose_stmts.append(
      CartesianNDecompose(
        flat_idx_var=flat_idx_var,
        idx_vars=tuple(idx_vars),
        deg_vars=tuple(degree_var_names),
      )
    )

  inner_decompose_stmts.append(BlankLine())

  # Var binds via SaGetValAt: `<handle>.get_value_at(<view>, <idx>)`.
  for i, src in enumerate(cart_op.sources):
    assert isinstance(src, mir.ColumnSource)
    if i >= len(cart_op.var_from_source):
      continue
    for var_name in cart_op.var_from_source[i]:
      if ctx.is_counting and not _cart_var_used(var_name, [], _trailing_inserts(rest)):
        continue
      inner_decompose_stmts.append(
        Bind(
          name=_sanitize_var_name(var_name),
          expr=SaGetValAt(
            handle_name=handle_var_names[i],
            view_name=view_var_names[i],
            idx_var_name=idx_vars[i],
          ),
        )
      )

  inner_decompose_stmts.append(BlankLine())

  loop_body = Block(
    stmts=(
      IndentBlock(extra=1, stmts=tuple(inner_decompose_stmts)),
      body_op,
    )
  )

  loop = GridStrideLoop(
    idx_name=flat_idx_var,
    bound=VarRef(name=total_var),
    body=loop_body,
  )
  outer_stmts.append(ParallelFor(strategy='warp_strided', body=loop))

  return Block(stmts=tuple(outer_stmts))


# -----------------------------------------------------------------------------
# Root multi-source ColumnJoin (M3)
# -----------------------------------------------------------------------------


def _lower_root_cj_multi(
  cj_op: mir.ColumnJoin,
  rest: list[mir.MirNode],
  ctx: LoweringCtx,
) -> Op:
  '''Lower a root multi-source ColumnJoin.

  Mirrors `_root_cj_multi` in `ir/dialects/target/cuda/root.py`. Counter
  trajectory matches legacy: body is rendered with its own counter
  trajectory starting from saved=0; then outer names are allocated
  starting from saved=0 again. Body and outer have overlapping
  counter ranges but different prefixes — the legacy convention.

  When `ctx.bg_enabled` is set, dispatches to `_lower_root_cj_bg`
  (block-group work-balanced variant — N4.1).

  DEAD CODE NOTE (C3): the `if ctx.bg_enabled:` branch below is
  superseded by the `BlockGroupRoot` wrap op + the registered
  `@lowering(target=iir.cf, source=mir.BlockGroupRoot)` rule (in
  `parallel.block_group.pragmas.block_group`). The new lowering
  delegates back into `_lower_root_cj_bg` with `ctx.bg_enabled`
  forced True, so the branch remains functionally LIVE during the
  C3 dual-write transition (the `with_pragma(BlockGroup())` DSL
  dual-writes both the bool field AND the typed pragma; see
  `pragmas/block_group.py: materialize_block_group`). The branch
  will be removable once Phase A3 drops the `block_group: bool`
  field on `mir.ExecutePipeline` and Phase B migrates the host
  `_lower_root_cj_multi` lowering out of this monolith — until then
  it is the byte-equivalence anchor for the dual-write transition.
  '''
  if ctx.bg_enabled:
    return _lower_root_cj_bg(cj_op, rest, ctx)

  num_sources = len(cj_op.sources)
  assert num_sources >= 2

  # Step 1: register state keys + bind join var so the body's nested
  # CJ can find the outer handles by state key. Names of outer
  # handles are deterministic `h_<rel>_<src>_root`.
  source_handle_names: list[str] = []
  source_view_names: list[str] = []
  registered_state_keys: list[str] = []

  for src in cj_op.sources:
    assert isinstance(src, mir.ColumnSource)
    handle_var = f'h_{src.rel_name}_{src.handle_start}_root'
    source_handle_names.append(handle_var)

    src_view = ctx.view_var_names.get(str(src.handle_start), '')
    if not src_view:
      raise ValueError(
        f'_lower_root_cj_multi: no view var for source handle_idx {src.handle_start}'
      )
    source_view_names.append(src_view)

    state_key = _state_key(src.rel_name, list(src.index), [cj_op.var_name], src.version)
    ctx.handle_vars[state_key] = handle_var
    registered_state_keys.append(state_key)

  ctx.bound_vars.append(cj_op.var_name)

  # Step 2: render body BEFORE allocating our own counter-bumped
  # names. Body's counter trajectory starts at the current value
  # (typically 0 at the top of pipeline lowering) and bumps freely.
  saved_counter = ctx.name_counter
  body_op = _lower_inner_chain(rest, ctx)
  # Restore counter so our outer-scope allocations restart from the
  # same value the body started at. Body's bumps are persisted in
  # the IIR's pre-baked names; the counter just gets rewound.
  ctx.name_counter = saved_counter

  # Cleanup body-scoped state.
  ctx.bound_vars.pop()
  for k in registered_state_keys:
    ctx.handle_vars.pop(k, None)

  # Step 3: now allocate our outer-scope names.
  outer_stmts: list[Op] = []

  if ctx.debug:
    outer_stmts.append(
      Comment(
        text=f'Root ColumnJoin (multi-source intersection): '
        f'bind \'{cj_op.var_name}\' from {num_sources} sources'
      )
    )
    outer_stmts.append(Comment(text='Uses root_unique_values + prefix() pattern (like TMP)'))
    src_debug = ' '.join(f'({s.rel_name} :handle {s.handle_start})' for s in cj_op.sources)
    outer_stmts.append(
      Comment(text=f'MIR: (column-join :var {cj_op.var_name} :sources ({src_debug} ))')
    )

  y_idx_var = ctx.fresh('y_idx')
  root_val_var = ctx.fresh('root_val')

  loop_inner_stmts: list[Op] = [
    Bind(
      name=root_val_var,
      expr=RawString(text=f'root_unique_values[{y_idx_var}]'),
    ),
    BlankLine(),
  ]

  # Multi-view non-first sources defer their handle bind to a
  # segment-loop emission phase below — match legacy `_root_cj_multi`
  # phase split.
  segment_loop_sources: list[tuple[int, mir.ColumnSource, int, int]] = []
  for i, src in enumerate(cj_op.sources):
    assert isinstance(src, mir.ColumnSource)
    if i == 0:
      continue
    idx_type = ctx.rel_index_types.get(src.rel_name, '')
    vc = view_count(src.version.code, idx_type)
    if vc <= 1:
      continue
    base_slot = ctx.view_slot_bases.get(str(src.handle_start), src.handle_start)
    segment_loop_sources.append((i, src, vc, base_slot))

  segment_loop_idxs = {sl[0] for sl in segment_loop_sources}

  for i, src in enumerate(cj_op.sources):
    assert isinstance(src, mir.ColumnSource)
    handle_var = source_handle_names[i]
    src_view = source_view_names[i]

    if i in segment_loop_idxs:
      # Multi-view non-first source: skip phase-1 emission; the
      # segment loop will own its handle bind + validity check.
      continue

    if i == 0:
      hint_lo = ctx.fresh('hint_lo')
      hint_hi = ctx.fresh('hint_hi')
      loop_inner_stmts.append(Bind(name=hint_lo, expr=VarRef(name=y_idx_var), type_decl='uint32_t'))
      loop_inner_stmts.append(
        Bind(
          name=hint_hi,
          expr=RawString(text=f'{src_view}.num_rows_ - (num_unique_root_keys - {y_idx_var} - 1)'),
          type_decl='uint32_t',
        )
      )
      loop_inner_stmts.append(
        RawString(
          text=f'{hint_hi} = ({hint_hi} <= {src_view}.num_rows_) ? '
          f'{hint_hi} : {src_view}.num_rows_;'
        )
      )
      loop_inner_stmts.append(
        RawString(text=f'{hint_hi} = ({hint_hi} > {hint_lo}) ? {hint_hi} : {src_view}.num_rows_;')
      )
      loop_inner_stmts.append(
        Bind(
          name=handle_var,
          expr=SaPrefCoop(
            parent=SaHint(lo_var=hint_lo, hi_var=hint_hi, depth=0),
            key_var=root_val_var,
            view_name=src_view,
          ),
        )
      )
    else:
      loop_inner_stmts.append(
        Bind(
          name=handle_var,
          expr=SaPrefCoop(
            parent=SaRoot(view_name=src_view),
            key_var=root_val_var,
            view_name=src_view,
          ),
        )
      )
    loop_inner_stmts.append(IfContinueIfNot(cond=SaValid(handle_name=handle_var)))

  var_bind_op: Op = Bind(
    name=_sanitize_var_name(cj_op.var_name),
    expr=VarRef(name=root_val_var),
  )

  if segment_loop_sources:
    # Phase 2: open a D2lSegmentLoop per multi-view non-first source,
    # innermost-first. Each loop's body contains: handle bind +
    # validity check (using the LOCAL per-segment view var) + the
    # next inner level. Deepest level holds the var bind followed by
    # the body_op anchored to the outer indent — so the segment
    # loop's brace closes AFTER the entire join body, but body_op's
    # text starts at the surrounding kernel-body indent (legacy
    # quirk where body was pre-rendered before the segment loop wrap).
    inner_op: Op = Block(stmts=(var_bind_op, OuterAnchor(body=body_op)))
    for sl_idx, src, vc, base_slot in reversed(segment_loop_sources):
      handle_var = source_handle_names[sl_idx]
      fixed_view = source_view_names[sl_idx]
      local_view = f'view_{src.rel_name}_{src.handle_start}'

      seg_body_stmts: list[Op] = [
        Bind(
          name=handle_var,
          expr=SaPrefCoop(
            parent=SaRoot(view_name=local_view),
            key_var=root_val_var,
            view_name=local_view,
          ),
        ),
        IfContinueIfNot(cond=SaValid(handle_name=handle_var)),
        inner_op,
      ]
      seg_loop_op: Op = D2lSegmentLoop(
        seg_var=f'_seg_{sl_idx}',
        view_var=fixed_view,
        base_slot=base_slot,
        view_count=vc,
        declare=False,
        local_view_var=local_view,
        body=Block(stmts=tuple(seg_body_stmts)),
      )
      if ctx.debug:
        seg_comment = Comment(
          text=f'Segment loop: {src.rel_name} {src.version.code} has {vc} segments (FULL + HEAD)'
        )
        inner_op = Block(stmts=(seg_comment, seg_loop_op))
      else:
        inner_op = seg_loop_op

    loop_inner_stmts.append(inner_op)
    # body_op is INSIDE the segment loop chain; the outer Block
    # holds only the IndentBlock — no trailing body_op here.
    loop_body = Block(stmts=(IndentBlock(extra=1, stmts=tuple(loop_inner_stmts)),))
  else:
    loop_inner_stmts.append(var_bind_op)
    loop_body = Block(
      stmts=(
        IndentBlock(extra=1, stmts=tuple(loop_inner_stmts)),
        body_op,
      )
    )

  loop = GridStrideLoop(
    idx_name=y_idx_var,
    bound=RawString(text='num_unique_root_keys'),
    body=loop_body,
  )
  outer_stmts.append(ParallelFor(strategy='warp_strided', body=loop))

  return Block(stmts=tuple(outer_stmts))


# -----------------------------------------------------------------------------
# Root multi-source ColumnJoin — block-group variant (N4.1)
# -----------------------------------------------------------------------------


def _lower_root_cj_bg(
  cj_op: mir.ColumnJoin,
  rest: list[mir.MirNode],
  ctx: LoweringCtx,
) -> Op:
  '''Lower a root multi-source ColumnJoin in BG (block-group) mode.

  Mirrors legacy `jit_root_column_join_block_group` in
  `ir/dialects/target/cuda/root.py`. Body renders with `ctx.bg_enabled = False`
  (the BG root's narrowed handle already restricts work to this
  warp's slice, so nested ops use the standard parallel emission).

  The whole BG scaffold (work assignment preamble, binary search,
  per-key loop, per-source narrowing, warp-row redistribution,
  optional D2L segment loops) bundles into a single `BgRootCjMulti`
  op so the dialect's emit can reproduce the legacy structure
  byte-for-byte.

  Counter trajectory matches legacy: register state keys + bind var
  before body render so deeper ops can find handles by state key,
  render body (which bumps counter freely), restore counter to the
  saved point, then allocate our own counter-bumped scaffold names.
  '''
  num_sources = len(cj_op.sources)
  assert num_sources >= 2

  # Step 1: register state keys + bind join var so the body's nested
  # CJ can find the outer handles by state key. Names of outer
  # handles are deterministic `h_<rel>_<src>_root`.
  source_handle_names: list[str] = []
  source_view_names: list[str] = []
  source_view_counts: list[int] = []
  source_base_slots: list[int] = []
  source_index_types: list[str] = []
  registered_state_keys: list[str] = []

  for src in cj_op.sources:
    assert isinstance(src, mir.ColumnSource)
    handle_var = f'h_{src.rel_name}_{src.handle_start}_root'
    source_handle_names.append(handle_var)

    src_view = ctx.view_var_names.get(str(src.handle_start), '')
    if not src_view:
      raise ValueError(f'_lower_root_cj_bg: no view var for source handle_idx {src.handle_start}')
    source_view_names.append(src_view)
    idx_type = ctx.rel_index_types.get(src.rel_name, '')
    source_index_types.append(idx_type)
    source_view_counts.append(view_count(src.version.code, idx_type))
    source_base_slots.append(ctx.view_slot_bases.get(str(src.handle_start), src.handle_start))

    state_key = _state_key(src.rel_name, list(src.index), [cj_op.var_name], src.version)
    ctx.handle_vars[state_key] = handle_var
    registered_state_keys.append(state_key)

  ctx.bound_vars.append(cj_op.var_name)

  # Step 2: render body with bg_enabled cleared. The narrowed first-
  # source handle already restricts work, so nested ops use standard
  # warp-strided dispatch. Save/restore counter to mirror legacy.
  saved_counter = ctx.name_counter
  saved_bg_enabled = ctx.bg_enabled
  ctx.bg_enabled = False
  try:
    body_op = _lower_inner_chain(rest, ctx)
  finally:
    ctx.bg_enabled = saved_bg_enabled
  ctx.name_counter = saved_counter

  ctx.bound_vars.pop()
  for k in registered_state_keys:
    ctx.handle_vars.pop(k, None)

  # Step 3: allocate our outer-scope names. Order matches legacy
  # `jit_root_column_join_block_group` (key_idx, root_val, then
  # hint_lo/hint_hi for the first source).
  key_idx_var = ctx.fresh('bg_key_idx')
  root_val_var = ctx.fresh('root_val')
  hint_lo = ctx.fresh('hint_lo')
  hint_hi = ctx.fresh('hint_hi')

  source_specs = tuple(
    BgSourceSpec(
      rel_name=src.rel_name,
      view_var=source_view_names[i],
      handle_var=source_handle_names[i],
      view_count=source_view_counts[i],
      base_slot=source_base_slots[i],
      index_type=source_index_types[i],
    )
    for i, src in enumerate(cj_op.sources)
    if isinstance(src, mir.ColumnSource)
  )

  outer_stmts: list[Op] = []
  if ctx.debug:
    outer_stmts.append(
      Comment(
        text=f'Root ColumnJoin (BLOCK-GROUP): bind \'{cj_op.var_name}\' from {num_sources} sources'
      )
    )
    outer_stmts.append(
      Comment(text='Block-group work-balanced partitioning with inner redistribution')
    )

  outer_stmts.append(
    BgRootCjMulti(
      var_name=_sanitize_var_name(cj_op.var_name),
      is_counting=ctx.is_counting,
      key_idx_var=key_idx_var,
      root_val_var=root_val_var,
      hint_lo=hint_lo,
      hint_hi=hint_hi,
      sources=source_specs,
      body=body_op,
    )
  )

  return Block(stmts=tuple(outer_stmts))


# -----------------------------------------------------------------------------
# Inner-chain lowering: nested CJ / Filter / ConstantBind / InsertInto
# -----------------------------------------------------------------------------


def _should_use_declarative(head: mir.MirNode) -> bool:
  '''Gate: does the head op route through the new per-op `@lowering`
  path (instead of the legacy `if isinstance` branches below)?

  For every MIR op type covered by Wave 2A this is just
  `type(head) in USE_DECLARATIVE`. The helper exists as a single
  decision point so neither `_lower_inner_chain` nor
  `lower_scan_pipeline` ends up with two isinstance-cascades on the
  same op type.

  Historical note (B-CJ-single, PR #59): this helper used to gate
  `mir.ColumnJoin` on `len(sources) == 1` — the single-source case
  routed through the new path while multi-source fell through to
  the legacy `_lower_nested_cj_multi` / `_lower_root_cj_multi`
  branches. B-CJ-multi (this PR) drops that guard: ALL ColumnJoin
  variants route through the new per-op path. The chain-aware /
  root-aware shape split lives in the per-op module
  (`lower_mir_cj_single*` vs `lower_mir_cj_multi*`); the chain
  dispatcher / root dispatcher pick the right entry from
  `len(head.sources)`.

  Per docs/phase_b_lowering_dispatcher.md §4 — the "type AND
  structural shape" gate keeps the dispatch atomic. Today the
  structural shape is encoded in the per-op modules' own dispatch
  (not here); the helper is kept as a stable extension point if
  future B-PRs need a per-op gate again.
  '''
  from srdatalog.ir.dialects.relation.sorted_array import USE_DECLARATIVE

  return type(head) in USE_DECLARATIVE


def _lower_inner_chain(
  rest: list[mir.MirNode],
  ctx: LoweringCtx,
) -> Op:
  '''Lower the chain of post-root ops.

  `rest` may end in one or more InsertIntos (multi-head rules).
  When the head op is itself an InsertInto, this function emits all
  trailing InsertIntos in sequence as the terminal body.
  '''
  if not rest:
    raise ValueError('_lower_inner_chain: empty rest')

  head = rest[0]
  tail = rest[1:]

  # Phase B / Wave 2A (per docs/phase_b_lowering_dispatcher.md §5):
  # MIR op types registered in `USE_DECLARATIVE` route through their
  # per-op `@lowering` rules instead of the legacy `if isinstance`
  # branches below. The legacy branches stay LIVE in this file as
  # the safety-net implementation — the chain-aware variants in
  # `lowerings/lower_mir_<op>.py` delegate back into helpers from
  # this module (`_filter_expr`, `_sanitize_var_name`,
  # `_lower_insert_into`, `_lower_tiled_ballot_block`) so
  # byte-equivalence holds by construction. Layer 3 cleanup deletes
  # both the legacy branches AND `USE_DECLARATIVE` once every MIR op
  # is migrated. Imports are function-local to keep the module
  # import graph linear (the sibling per-op modules import back
  # from this file).
  #
  # Note: this dispatch sits ABOVE the `mir.InsertInto` head branch
  # so the B-InsertInto migration can intercept the terminal
  # InsertInto-run case via `lower_mir_insert_into_in_chain`. The
  # legacy `mir.InsertInto` head branch below remains the
  # safety-net implementation (entered when InsertInto is dropped
  # from `USE_DECLARATIVE`, e.g. during the byte-equivalence test).
  #
  # B-CJ-single (per docs/phase_b_lowering_dispatcher.md §4 row
  # B-CJ-single): `mir.ColumnJoin` is in `USE_DECLARATIVE` but only
  # the single-source case (`len(sources) == 1`) routes through the
  # new path — multi-source ColumnJoin stays on the legacy
  # `_lower_nested_cj_multi` branch below, owned by the next PR
  # (B-CJ-multi). The `_should_use_declarative` helper encapsulates
  # this "type AND structural shape" gate so the dispatch stays a
  # single decision; B-CJ-multi extends the helper to drop the
  # source-count guard once it lands.
  if _should_use_declarative(head):
    if isinstance(head, mir.Filter):
      from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_filter import (
        lower_mir_filter_in_chain,
      )

      return lower_mir_filter_in_chain(head, tail, ctx)
    if isinstance(head, mir.ConstantBind):
      from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_constant_bind import (
        lower_mir_constant_bind_in_chain,
      )

      return lower_mir_constant_bind_in_chain(head, tail, ctx)
    if isinstance(head, mir.InsertInto):
      from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_insert_into import (
        lower_mir_insert_into_in_chain,
      )

      return lower_mir_insert_into_in_chain(head, tail, ctx)
    if isinstance(head, mir.ColumnJoin):
      # B-CJ-single + B-CJ-multi: shape-aware dispatch by source count.
      # `_should_use_declarative` no longer gates by source count for
      # `mir.ColumnJoin` (it was a single `return True` from B-CJ-multi
      # onward) — both single-source and multi-source CJ route through
      # their respective per-op entry points here. The single dialect-
      # level `@lowering(mir.ColumnJoin, ...)` registration covers
      # both shapes for discipline ownership.
      if len(head.sources) == 1:
        from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_cj_single import (
          lower_mir_cj_single_in_chain,
        )

        return lower_mir_cj_single_in_chain(head, tail, ctx)
      from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_cj_multi import (
        lower_mir_cj_multi_in_chain,
      )

      return lower_mir_cj_multi_in_chain(head, tail, ctx)
    if isinstance(head, mir.Negation):
      from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_negation import (
        lower_mir_negation_in_chain,
      )

      return lower_mir_negation_in_chain(head, tail, ctx)
    if isinstance(head, mir.Aggregate):
      from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_aggregate import (
        lower_mir_aggregate_in_chain,
      )

      return lower_mir_aggregate_in_chain(head, tail, ctx)
    if isinstance(head, mir.CartesianJoin):
      # B-Cart mid-chain entry: `_should_use_declarative` returns
      # True for all `mir.CartesianJoin` instances. The C5
      # `mir.TiledCartesian` wrap op is a DIFFERENT MIR op type and
      # never reaches this branch (handled by the
      # `isinstance(head, mir.TiledCartesian)` branch below the
      # USE_DECLARATIVE block — `TiledCartesian` is NOT in the set
      # so it falls through to the legacy dispatch).
      #
      # The delegated `_lower_nested_cart` still performs its own
      # `_tiled_cart_eligible(...)` check and forwards to
      # `_lower_nested_cart_tiled` when the legacy `ctx.tiled_cartesian`
      # bool path is active — preserving the C5 dual-write transition.
      # See `lowerings/lower_mir_cart.py` module docstring for the
      # coexistence rationale.
      from srdatalog.ir.dialects.relation.sorted_array.lowerings.lower_mir_cart import (
        lower_mir_cart_in_chain,
      )

      return lower_mir_cart_in_chain(head, tail, ctx)
    raise AssertionError(
      f'_lower_inner_chain: USE_DECLARATIVE contains {type(head).__name__!r} '
      f'but no chain-dispatch wiring exists for it. Add the '
      f'`lower_mir_<op>_in_chain` import + call above.'
    )

  if isinstance(head, mir.InsertInto):
    # Terminal: emit all trailing InsertIntos in order.
    inserts = _trailing_inserts(rest)
    if len(inserts) != len(rest):
      raise ValueError(
        f'_lower_inner_chain: expected pure InsertInto tail at this '
        f'point, got {[type(o).__name__ for o in rest]}'
      )
    # Tiled-Cartesian ballot variant: render the trailing InsertInto
    # run as a single TiledBallotBlock that owns the ballot setup +
    # per-output write at once. Replaces the legacy stateful
    # `tiled_cartesian_ballot_done` flag.
    if ctx.tiled_cartesian_valid_var:
      return _lower_tiled_ballot_block(inserts, ctx)
    stmts: list[Op] = []
    for ins in inserts:
      stmts.extend(_lower_insert_into(ins, ctx))
    return Block(stmts=tuple(stmts))

  if isinstance(head, mir.Filter):
    cond_expr = _filter_expr(head.code)
    body_op = _lower_inner_chain(tail, ctx)
    # Tiled-Cartesian ballot path OR WS batched-valid path: fold the
    # condition into the active valid flag instead of emitting an
    # `if (cond) {...}` block. Avoids divergence around the
    # cooperative warp write. Mirrors legacy `jit_filter`
    # ws_cartesian_valid_var / tiled_cartesian_valid_var branches.
    fold_var = ctx.ws_cartesian_valid_var or ctx.tiled_cartesian_valid_var
    if fold_var:
      return Block(
        stmts=(
          RawString(text=f'{fold_var} = {fold_var} && ({cond_expr});'),
          body_op,
        )
      )
    return If(cond=RawString(text=cond_expr), body=body_op)

  if isinstance(head, mir.ConstantBind):
    var = _sanitize_var_name(head.var_name)
    bind_stmt = Bind(name=var, expr=RawString(text=head.code))
    rest_op = _lower_inner_chain(tail, ctx)
    if isinstance(rest_op, Block):
      return Block(stmts=(bind_stmt, *rest_op.stmts))
    return Block(stmts=(bind_stmt, rest_op))

  if isinstance(head, mir.ColumnJoin) and len(head.sources) == 1:
    # B-CJ-single safety-net branch: same delegation pattern as the
    # B-Filter / B-ConstantBind / B-InsertInto rows above. The new
    # path (when `mir.ColumnJoin` is in `USE_DECLARATIVE`) routes
    # through `lower_mir_cj_single_in_chain` which calls back into
    # this helper, so both paths share the same emission and the
    # byte-equivalence fixture can swap between them.
    return _lower_nested_cj_single(head, tail, ctx)

  if isinstance(head, mir.ColumnJoin) and len(head.sources) >= 2:
    return _lower_nested_cj_multi(head, tail, ctx)

  if isinstance(head, mir.CartesianJoin):
    return _lower_nested_cart(head, tail, ctx)

  # C5 (per docs/phase_c_pragma_materialization.md §4.3): the
  # `TiledCartesian` MIR wrap op is the materialized form of the
  # `TiledCartesian` Pragma. Dispatch here because the tiled emission
  # helper (`_lower_nested_cart_tiled`) needs both the wrapped
  # Cartesian AND the trailing chain (`tail`) — InsertIntos plus any
  # Filter / ConstantBind. The standalone @lowering entry registered
  # via the dialect package is a structural contract that asserts on
  # direct invocation; real emission flows through this branch.
  if isinstance(head, mir.TiledCartesian):
    from srdatalog.ir.dialects.relation.sorted_array.pragmas.tiled_cartesian import (
      lower_tiled_cartesian_in_chain,
    )

    return lower_tiled_cartesian_in_chain(head, tail, ctx)

  if isinstance(head, mir.Negation):
    return _lower_negation(head, tail, ctx)

  raise ValueError(f'unsupported inner op: {type(head).__name__}')


def _tiled_cart_eligible(
  cart_op: mir.CartesianJoin,
  ctx: LoweringCtx,
) -> bool:
  '''Mirror legacy `tiled_eligible` from pipeline.py: 2-source / 1-var-
  per-source / materialize phase / ctx.tiled_cartesian set. The `rest`
  shape is unconstrained (matches legacy) — Filters and other ops
  between Cart and the trailing InsertIntos pass through naturally
  via `_lower_inner_chain` with `ctx.tiled_cartesian_valid_var` set.

  DEAD CODE NOTE (C5, per docs/phase_c_pragma_materialization.md §5.1):
  the `ctx.tiled_cartesian` gate is the legacy `if ctx.tiled_cartesian:`
  branch that the C5 PR migrates to the `TiledCartesian` Pragma +
  `mir.TiledCartesian` wrap op. The predicate stays LIVE during the
  C5 dual-write transition because `complete_runner.py` still
  threads `LoweringCtx.tiled_cartesian` from
  `has_tiled_cartesian_eligible(pipeline)` and the wrap op dispatch
  in `_lower_inner_chain` flips the same flag for the body render.
  Phase A3 deletes the `ctx.tiled_cartesian` field and this branch
  collapses to the shape-only checks (which then move to
  `pragmas/tiled_cartesian.py:_is_eligible_cart`).
  '''
  if not ctx.tiled_cartesian or ctx.is_counting:
    return False
  if len(cart_op.sources) != 2:
    return False
  if len(cart_op.var_from_source) != 2:
    return False
  return len(cart_op.var_from_source[0]) == 1 and len(cart_op.var_from_source[1]) == 1


def _lower_nested_cart_tiled(
  cart_op: mir.CartesianJoin,
  rest: list[mir.MirNode],
  ctx: LoweringCtx,
) -> Op:
  '''Lower a tiled-Cartesian-eligible nested CartesianJoin (N7).

  The body is rendered ONCE with `ctx.tiled_cartesian_valid_var` set
  so the trailing InsertInto run becomes a `TiledBallotBlock`. Counter
  is then advanced past the `tc_valid_<n>` name slot and the standard
  per-Cart scaffold (lane, group_size, per-source degree+handle,
  total, flat_idx) allocates as in the non-tiled flow. Tiled-specific
  names (t0_base, t1_base, t0_len, t1_len, tile_total, batch_var,
  fb_batch_var, major_var, idx0, idx1) follow.

  Mirrors legacy pipeline.py `tiled_eligible` dual-body trick +
  jit_nested_cartesian_join + `_emit_tiled_cartesian` byte-for-byte.
  '''
  num_sources = 2
  src0, src1 = cart_op.sources[0], cart_op.sources[1]
  assert isinstance(src0, mir.ColumnSource)
  assert isinstance(src1, mir.ColumnSource)

  saved_counter = ctx.name_counter
  tc_valid_name = f'tc_valid_{saved_counter + 1}'

  # Body render with valid_var set — trailing InsertIntos turn into a
  # TiledBallotBlock via `_lower_inner_chain` -> `_lower_tiled_ballot_block`.
  saved_tcvv = ctx.tiled_cartesian_valid_var
  saved_inside = ctx.inside_cartesian
  ctx.tiled_cartesian_valid_var = tc_valid_name
  ctx.inside_cartesian = True
  pushed = 0
  for vfs in cart_op.var_from_source:
    for v in vfs:
      ctx.bound_vars.append(v)
      pushed += 1
  try:
    body_op = _lower_inner_chain(rest, ctx)
  finally:
    for _ in range(pushed):
      ctx.bound_vars.pop()
    ctx.inside_cartesian = saved_inside
    ctx.tiled_cartesian_valid_var = saved_tcvv

  # Reset counter to "post-dual-body" state (matches legacy
  # pipeline.py setting `ctx.name_counter = saved_name_counter + 1`).
  ctx.name_counter = saved_counter + 1

  # Allocate per-Cart scaffold names (matches legacy
  # `jit_nested_cartesian_join` lines 533+).
  lane_var = ctx.fresh('lane')
  group_size_var = ctx.fresh('group_size')

  handle_var_names: list[str] = []
  view_var_names: list[str] = []
  degree_var_names: list[str] = []
  alias_targets: list[str | None] = []

  for src in cart_op.sources:
    assert isinstance(src, mir.ColumnSource)
    degree_var_names.append(ctx.fresh('degree'))
    parent_state_key = _state_key(src.rel_name, list(src.index), src.prefix_vars, src.version)
    parent_handle = ctx.handle_vars.get(parent_state_key, '')
    if parent_handle:
      alias_targets.append(parent_handle)
    elif not src.prefix_vars:
      alias_targets.append(None)
    else:
      raise NotImplementedError(
        f'_lower_nested_cart_tiled: source {src.rel_name} has prefix '
        f'{src.prefix_vars} but no full-state-key match'
      )
    handle_var_names.append(ctx.fresh(f'h_{src.rel_name}_{src.handle_start}'))
    src_view = ctx.view_var_names.get(str(src.handle_start), '')
    if not src_view:
      raise ValueError(
        f'_lower_nested_cart_tiled: no view var for source handle_idx {src.handle_start}'
      )
    view_var_names.append(src_view)

  total_var = ctx.fresh('total')
  flat_idx_var = ctx.fresh('flat_idx')

  # Tiled-specific names. Order matches legacy `_emit_tiled_cartesian`.
  t0_base = ctx.fresh('t0_base')
  t1_base = ctx.fresh('t1_base')
  t0_len = ctx.fresh('t0_len')
  t1_len = ctx.fresh('t1_len')
  tile_total = ctx.fresh('tile_total')
  batch_var = ctx.fresh('tc_batch')
  # valid_var = tc_valid_name allocated above; no fresh allocation here.
  fb_batch_var = ctx.fresh('fb_batch')
  major_var = ctx.fresh('major_is_1')
  idx0_var = ctx.fresh('idx0')
  idx1_var = ctx.fresh('idx1')

  # Build IIR — debug comments + lane/group_size + handle binds +
  # validity + degrees + total/zero-check + SaTiledCartesian2D.
  stmts: list[Op] = []
  if ctx.debug:
    vars_bound_str = ', '.join(cart_op.vars)
    stmts.append(
      Comment(text=f'Nested CartesianJoin: bind {vars_bound_str} from {num_sources} source(s)')
    )
    src_debug = ' '.join(
      f'({s.rel_name} :handle {s.handle_start} :prefix ({" ".join(s.prefix_vars)}))'
      for s in cart_op.sources
    )
    stmts.append(
      Comment(
        text=f'MIR: (cartesian-join :vars ({" ".join(cart_op.vars)}) :sources ({src_debug} ))'
      )
    )

  stmts.append(
    Bind(
      name=lane_var,
      expr=RawString(text=f'{ctx.tile_var}.thread_rank()'),
      type_decl='uint32_t',
    )
  )
  stmts.append(
    Bind(
      name=group_size_var,
      expr=RawString(text=f'{ctx.tile_var}.size()'),
      type_decl='uint32_t',
    )
  )
  stmts.append(BlankLine())

  for i in range(num_sources):
    if alias_targets[i] is not None:
      stmts.append(
        RawString(
          text=f'auto {handle_var_names[i]} = {alias_targets[i]};  // reusing narrowed handle'
        )
      )
    else:
      stmts.append(
        Bind(
          name=handle_var_names[i],
          expr=SaRoot(view_name=view_var_names[i]),
        )
      )
  stmts.append(BlankLine())

  validity_parts = ' || '.join(f'!{h}.valid()' for h in handle_var_names)
  stmts.append(RawString(text=f'if ({validity_parts}) continue;'))
  stmts.append(BlankLine())

  for i in range(num_sources):
    stmts.append(
      Bind(
        name=degree_var_names[i],
        expr=SaDegree(handle_name=handle_var_names[i]),
        type_decl='uint32_t',
      )
    )

  stmts.append(
    Bind(
      name=total_var,
      expr=RawString(text=' * '.join(degree_var_names)),
      type_decl='uint32_t',
    )
  )
  stmts.append(RawString(text=f'if ({total_var} == 0) continue;'))
  stmts.append(BlankLine())

  stmts.append(
    SaTiledCartesian2D(
      view_var0=view_var_names[0],
      view_var1=view_var_names[1],
      handle_var0=handle_var_names[0],
      handle_var1=handle_var_names[1],
      col0=len(src0.prefix_vars),
      col1=len(src1.prefix_vars),
      var_name0=_sanitize_var_name(cart_op.var_from_source[0][0]),
      var_name1=_sanitize_var_name(cart_op.var_from_source[1][0]),
      lane_var=lane_var,
      group_size_var=group_size_var,
      total_var=total_var,
      degree_var0=degree_var_names[0],
      degree_var1=degree_var_names[1],
      flat_idx_var=flat_idx_var,
      t0_base=t0_base,
      t1_base=t1_base,
      t0_len=t0_len,
      t1_len=t1_len,
      tile_total=tile_total,
      batch_var=batch_var,
      valid_var=tc_valid_name,
      fb_batch_var=fb_batch_var,
      major_var=major_var,
      idx0_var=idx0_var,
      idx1_var=idx1_var,
      body=body_op,
    )
  )

  return Block(stmts=tuple(stmts))


def _lower_tiled_ballot_block(
  inserts: list[mir.InsertInto],
  ctx: LoweringCtx,
) -> Op:
  '''Lower a trailing run of InsertIntos as a single TiledBallotBlock
  (ballot-coalesced writes used by `SaTiledCartesian2D` body).

  Each InsertInto contributes one entry: (dest_idx, sanitized values,
  debug text). The runner's per-output-context naming
  (`output_ctx_<j>`) maps to dest_idx via the same convention as the
  legacy `_root_cj_multi` setup.
  '''
  outputs: list[tuple[int, tuple[str, ...], str]] = []
  for node in inserts:
    out_var = ctx.output_var_overrides.get(node.rel_name, ctx.output_var)
    if out_var.startswith('output_ctx_'):
      dest_idx = int(out_var[len('output_ctx_') :])
    else:
      dest_idx = 0
    values = tuple(_sanitize_var_name(v) for v in node.vars)
    debug = f'Emit: {node.rel_name}({", ".join(node.vars)})' if ctx.debug else ''
    outputs.append((dest_idx, values, debug))
  return TiledBallotBlock(
    valid_var=ctx.tiled_cartesian_valid_var,
    outputs=tuple(outputs),
  )


def _lower_negation(
  neg_op: mir.Negation,
  rest: list[mir.MirNode],
  ctx: LoweringCtx,
) -> Op:
  '''Lower an anti-join: body fires only when the narrowed handle is
  invalid (i.e. the prefix doesn't exist in the negated relation).

  Two paths:
    Standard (M5): not preceded by a Cartesian. Build a fresh
      chained-prefix handle from root and check it.
    Pre-narrow (M5.x): preceded by a Cartesian whose pre-Cartesian
      vars are bound earlier. The pre-narrowed handle is declared
      outside the Cart loop (by `_lower_nested_cart`). The Negation
      reuses it directly; if there are in-Cartesian vars left over,
      it applies them via prefix_seq inside the Cart loop.

  Counter trajectory mirrors the legacy: the body is rendered FIRST
  with its own bumps; our handle name (if any) is allocated AFTER
  body.
  '''
  src_idx = neg_op.handle_start
  rel_name = neg_op.rel_name

  view_var = ctx.view_var_names.get(str(src_idx), '')
  if not view_var:
    raise ValueError(f'_lower_negation: no view var for handle_idx {src_idx}')

  # N5.4 guard: Negation over a D2L FULL_VER source needs a segment
  # loop wrapping the validity check (each segment may invalidate
  # the prefix independently). The pre-narrow path (Negation inside
  # a Cartesian) is fine — the surrounding Cart's outer narrowing
  # already pinned the segment. Standard path is not.
  if src_idx not in ctx.neg_pre_narrow:
    idx_type = ctx.rel_index_types.get(rel_name, '')
    if view_count(neg_op.version.code, idx_type) > 1:
      raise NotImplementedError(
        f'_lower_negation: standard-path Negation over D2L FULL_VER '
        f'source ({rel_name}) needs a segment-loop wrap (N5.4); not '
        f'yet implemented in the dialect. Add a fixture exercising '
        f'this shape and implement the wrap if you hit this.'
      )

  # Pre-narrow path: a previous Cart already constructed the
  # pre-narrowed handle. Use it directly (or apply remaining
  # in-Cartesian vars via prefix_seq).
  if src_idx in ctx.neg_pre_narrow:
    info = ctx.neg_pre_narrow[src_idx]
    # Body rendered first (legacy counter trajectory).
    body_op = _lower_inner_chain(rest, ctx)

    # If there are in-Cartesian vars, allocate a new handle name
    # and apply them via SaPrefSeq chain from info.var_name.
    if info.in_cartesian_vars:
      narrowed_var = ctx.fresh(f'h_{rel_name}_neg_{src_idx}')
      narrowed_expr: Op = VarRef(name=info.var_name)
      for v in info.in_cartesian_vars:
        narrowed_expr = SaPrefSeq(
          parent=narrowed_expr,
          key_var=_sanitize_var_name(v),
          view_name=info.view_var,
        )
    else:
      narrowed_var = ''
      narrowed_expr = VarRef(name=info.var_name)

    stmts: list[Op] = []
    if ctx.debug:
      stmts.append(Comment(text=f'Negation: NOT EXISTS in {rel_name}'))
      stmts.append(
        Comment(
          text=f'MIR: (negation :rel {rel_name} :prefix '
          f'({" ".join(neg_op.prefix_vars)}) :handle {src_idx})'
        )
      )
      stmts.append(
        Comment(text=f'Using pre-narrowed handle (pre-Cartesian vars: {", ".join(info.pre_vars)})')
      )

    if narrowed_var:
      stmts.append(Bind(name=narrowed_var, expr=narrowed_expr))
      check_var = narrowed_var
    else:
      check_var = info.var_name

    fold_var = ctx.ws_cartesian_valid_var or ctx.tiled_cartesian_valid_var
    if fold_var:
      stmts.append(RawString(text=f'{fold_var} = {fold_var} && (!{check_var}.valid());'))
      stmts.append(body_op)
    else:
      stmts.append(
        If(
          cond=RawString(text=f'!{check_var}.valid()'),
          body=body_op,
        )
      )
    return Block(stmts=tuple(stmts))

  # Standard path: const_args path not yet lowered here (only the
  # pre-narrow path above handles const_args).
  if neg_op.const_args:
    raise NotImplementedError(
      f'_lower_negation: standard path with const_args not yet lowered; got {neg_op.const_args}'
    )

  # Standard path. Step 1: render body BEFORE allocating own
  # counter-bumped name.
  body_op = _lower_inner_chain(rest, ctx)

  # Step 2: allocate the negation handle name.
  neg_handle_var = ctx.fresh(f'h_{rel_name}_neg_{src_idx}')

  # Step 3: build the chained prefix expression.
  parent_handle_name = ctx.handle_vars.get(str(src_idx), '')
  parent_expr: Op
  if parent_handle_name:
    parent_expr = VarRef(name=parent_handle_name)
  else:
    parent_expr = SaRoot(view_name=view_var)

  # Cooperative prefix outside Cart, sequential prefix_seq inside.
  for var_name in neg_op.prefix_vars:
    if ctx.inside_cartesian:
      parent_expr = SaPrefSeq(
        parent=parent_expr,
        key_var=_sanitize_var_name(var_name),
        view_name=view_var,
      )
    else:
      parent_expr = SaPrefCoop(
        parent=parent_expr,
        key_var=_sanitize_var_name(var_name),
        view_name=view_var,
      )

  # Step 4: build the IIR.
  stmts = []
  if ctx.debug:
    stmts.append(Comment(text=f'Negation: NOT EXISTS in {rel_name}'))
    stmts.append(
      Comment(
        text=f'MIR: (negation :rel {rel_name} :prefix '
        f'({" ".join(neg_op.prefix_vars)}) :handle {src_idx})'
      )
    )

  stmts.append(Bind(name=neg_handle_var, expr=parent_expr))
  fold_var = ctx.ws_cartesian_valid_var or ctx.tiled_cartesian_valid_var
  if fold_var:
    stmts.append(RawString(text=f'{fold_var} = {fold_var} && (!{neg_handle_var}.valid());'))
    stmts.append(body_op)
  else:
    stmts.append(
      If(
        cond=RawString(text=f'!{neg_handle_var}.valid()'),
        body=body_op,
      )
    )
  return Block(stmts=tuple(stmts))


def _register_neg_pre_narrow(
  cart_op: mir.CartesianJoin,
  rest: list[mir.MirNode],
  ctx: LoweringCtx,
) -> list[int]:
  '''Pre-allocate the `info.var_name` for any Negation in `rest`
  whose prefix vars contain at least one bound BEFORE the Cart.

  Mirrors `_register_negation_pre_narrow` in `pipeline.py`. Returns
  the list of handle_idx values registered (so the caller can
  cleanup `ctx.neg_pre_narrow` after body rendering).
  '''
  cart_bound_set: set[str] = set()
  for vfs in cart_op.var_from_source:
    cart_bound_set.update(vfs)

  registered: list[int] = []
  for neg_op in rest:
    if not isinstance(neg_op, mir.Negation):
      continue

    pre_vars: list[str] = []
    in_vars: list[str] = []
    contiguous = True
    for v in neg_op.prefix_vars:
      if contiguous and v not in cart_bound_set:
        pre_vars.append(v)
      else:
        contiguous = False
        in_vars.append(v)

    if not (pre_vars or neg_op.const_args):
      continue  # no pre-narrow needed

    view_var = ctx.view_var_names.get(str(neg_op.handle_start), '')
    if not view_var:
      raise ValueError(
        f'_register_neg_pre_narrow: no view var for negation handle_idx {neg_op.handle_start}'
      )

    pre_narrow_var = ctx.fresh(f'h_{neg_op.rel_name}_neg_pre')
    ctx.neg_pre_narrow[neg_op.handle_start] = NegPreNarrowInfo(
      var_name=pre_narrow_var,
      pre_vars=pre_vars,
      in_cartesian_vars=in_vars,
      pre_consts=list(neg_op.const_args),
      view_var=view_var,
      rel_name=neg_op.rel_name,
    )
    registered.append(neg_op.handle_start)
  return registered


def _lower_nested_cart(
  cart_op: mir.CartesianJoin,
  rest: list[mir.MirNode],
  ctx: LoweringCtx,
) -> Op:
  '''Lower a nested CartesianJoin.

  Coverage:
    - 1, 2, or N>=3 sources (3+ uses countdown remainder via
      `CartesianNDecompose`).
    - Full state-key handle reuse for prefix-bearing sources;
      fresh root for prefix-empty.
    - `neg_pre_narrow` registration for following Negations (M5.x).
    - Tiled-Cartesian dispatch (N7): when ctx.tiled_cartesian is set
      and shape eligible, dispatch to `_lower_nested_cart_tiled`.

  Mirrors `jit_nested_cartesian_join` in `ir/dialects/target/cuda/instructions.py`.
  Counter trajectory matches legacy: pre-narrow var_names allocated
  during `_register_neg_pre_narrow`, then body rendered, then own
  scaffold names.
  '''
  num_sources = len(cart_op.sources)
  if num_sources < 1:
    raise ValueError('_lower_nested_cart: must have at least 1 source')

  # DEAD CODE NOTE (C5, per docs/phase_c_pragma_materialization.md §5.1):
  # this `if _tiled_cart_eligible(...)` dispatch is the legacy
  # `if ctx.tiled_cartesian:` branch that the C5 PR migrates to the
  # `mir.TiledCartesian` wrap op dispatched from `_lower_inner_chain`.
  # The branch stays LIVE during the C5 dual-write transition because
  # the legacy runner-driven path (`complete_runner.py` ->
  # `LoweringCtx.tiled_cartesian`) still sets `ctx.tiled_cartesian=True`
  # for shape-eligible pipelines, independent of any rule pragma. The
  # wrap op fires only when `ep.tiled_cartesian is False` (synthesized
  # EPs / post-A3 state); the dual-write `materialize_tiled_cartesian`
  # handler short-circuits when the bool is set so the two paths
  # never both fire on the same Cartesian. Phase A3 deletes the
  # `ctx.tiled_cartesian` field, the bool short-circuit, and this
  # dispatch — leaving the wrap-op path as the sole emission driver.
  if _tiled_cart_eligible(cart_op, ctx):
    return _lower_nested_cart_tiled(cart_op, rest, ctx)

  # Step 1: register neg_pre_narrow info for any Negations in rest.
  # This bumps counter for each pre-narrow `var_name`, matching
  # legacy `_register_negation_pre_narrow` which allocates BEFORE
  # body rendering.
  saved_cart_bound = list(ctx.cartesian_bound_vars)
  for vfs in cart_op.var_from_source:
    ctx.cartesian_bound_vars.extend(vfs)
  registered_neg_idxs = _register_neg_pre_narrow(cart_op, rest, ctx)

  # Step 2: render body BEFORE allocating own counter-bumped names.
  # inside_cartesian flips so InsertInto inside drops the lane-0
  # guard (matches legacy `need_lane0_guard = not ctx.inside_cartesian`).
  saved_inside = ctx.inside_cartesian
  ctx.inside_cartesian = True
  pushed_var_count = 0
  for vars_from_src in cart_op.var_from_source:
    for v in vars_from_src:
      ctx.bound_vars.append(v)
      pushed_var_count += 1
  body_op = _lower_inner_chain(rest, ctx)
  for _ in range(pushed_var_count):
    ctx.bound_vars.pop()
  ctx.inside_cartesian = saved_inside
  ctx.cartesian_bound_vars = saved_cart_bound

  # Snapshot pre-narrow info we registered, then clear from ctx so
  # nested scopes don't pick up stale entries.
  pre_narrow_infos: list[NegPreNarrowInfo] = []
  for h_idx in registered_neg_idxs:
    pre_narrow_infos.append(ctx.neg_pre_narrow[h_idx])
    del ctx.neg_pre_narrow[h_idx]

  # Step 2: allocate scaffold names AFTER body. Order matches legacy:
  # lane, group_size, then per-source (degree, handle), then total.
  lane_var = ctx.fresh('lane')
  group_size_var = ctx.fresh('group_size')

  handle_var_names: list[str] = []
  view_var_names: list[str] = []
  degree_var_names: list[str] = []
  # Per source, either an alias-source string (reusing narrowed
  # handle, comment-suffixed) or None (meaning "fresh root").
  alias_targets: list[str | None] = []

  for src in cart_op.sources:
    assert isinstance(src, mir.ColumnSource)

    # Per legacy: degree var allocated first for each source.
    degree_var_names.append(ctx.fresh('degree'))

    # Look up parent handle by full state-key match. Three cases
    # mirror the legacy `_nested_cartesian_join`:
    #   1. exact match -> alias with comment.
    #   2. no match, no prefix_vars -> fresh root.
    #   3. no match, has prefix_vars -> fresh root + chained `.prefix(...)`
    #      (used by Scan + CartesianJoin where the prefix vars are bound
    #      by the Scan, not by an enclosing CJ — handle_vars is empty
    #      because Scan doesn't register state keys).
    parent_state_key = _state_key(src.rel_name, list(src.index), src.prefix_vars, src.version)
    parent_handle = ctx.handle_vars.get(parent_state_key, '')

    if parent_handle:
      alias_targets.append(parent_handle)
    else:
      # No parent → fresh root, with optional chained prefix narrowing.
      # `None` means "construct fresh below; prefix application handled
      # in the bind-emission loop further down via src.prefix_vars".
      alias_targets.append(None)

    handle_var_names.append(ctx.fresh(f'h_{src.rel_name}_{src.handle_start}'))

    src_view = ctx.view_var_names.get(str(src.handle_start), '')
    if not src_view:
      raise ValueError(f'_lower_nested_cart: no view var for source handle_idx {src.handle_start}')
    view_var_names.append(src_view)

  total_var = ctx.fresh('total')

  # R1 — count-as-product short-circuit. When counting and the rest is
  # pure-InsertInto with no following negation, the inner Cartesian
  # loop is replaced by a closed-form `add_count(lane_share)`. Skips
  # `flat_idx`, `idx_vars`, `major_is_1` allocations — matching the
  # legacy `cartesian_as_product` branch in
  # `instructions.py:jit_nested_cartesian_join`.
  #
  # Disabled for dedup_hash: each tuple needs an in-kernel
  # `dedup_table.try_insert(...)` test, so the body must run per-tuple
  # (Nim emits the full Cart loop + var binds + dedup test in count
  # phase, not the closed-form add_count).
  #
  # DEAD CODE NOTE (C2): the `not ctx.dedup_hash` clause is the
  # counterpart of the dedup branch in `_lower_insert_into`. Both
  # are superseded by the `DedupGate` lowering registered by
  # `pragmas/dedup_hash.py`; both remain LIVE during the C2 dual-
  # write transition (see `_lower_insert_into`'s docstring).
  cartesian_as_product = (
    ctx.is_counting
    and not ctx.dedup_hash
    and not pre_narrow_infos
    and all(isinstance(op, mir.InsertInto) for op in rest)
  )

  # Allocate const-prefix vars for pre-narrow emissions BEFORE
  # flat_idx — this matches the legacy counter trajectory where
  # const_var allocations happen during the pre-narrow emission
  # block which is between total/0 and flat_idx.
  pre_narrow_emissions: list[tuple[NegPreNarrowInfo, list[str]]] = []
  for info in pre_narrow_infos:
    const_var_names: list[str] = []
    for _ in info.pre_consts:
      const_var_names.append(ctx.fresh(f'h_{info.rel_name}_neg_pre_const'))
    pre_narrow_emissions.append((info, const_var_names))

  flat_idx_var = '' if cartesian_as_product else ctx.fresh('flat_idx')

  # idx-var allocation:
  #   1 source -> [idx0]
  #   2 sources -> [idx0, idx1] + major_is_1 (adaptive shape)
  #   N>=3 -> [idx0, idx1, ..., idx{N-1}] (countdown remainder)
  # R1 elides these — the closed-form add_count needs no per-thread idx.
  idx_vars: list[str] = []
  major_var = ''
  if cartesian_as_product:
    pass
  elif num_sources == 1:
    idx_vars = [ctx.fresh('idx0')]
  elif num_sources == 2:
    idx_vars = [ctx.fresh('idx0'), ctx.fresh('idx1')]
    major_var = ctx.fresh('major_is_1')
  else:
    idx_vars = [ctx.fresh(f'idx{s}') for s in range(num_sources)]

  # Step 3: build IIR.
  stmts: list[Op] = []

  if ctx.debug:
    vars_bound_str = ', '.join(cart_op.vars)
    stmts.append(
      Comment(text=f'Nested CartesianJoin: bind {vars_bound_str} from {num_sources} source(s)')
    )
    src_debug = ' '.join(
      f'({s.rel_name} :handle {s.handle_start} :prefix ({" ".join(s.prefix_vars)}))'
      for s in cart_op.sources
    )
    stmts.append(
      Comment(
        text=f'MIR: (cartesian-join :vars ({" ".join(cart_op.vars)}) :sources ({src_debug} ))'
      )
    )

  stmts.append(
    Bind(
      name=lane_var,
      expr=RawString(text=f'{ctx.tile_var}.thread_rank()'),
      type_decl='uint32_t',
    )
  )
  stmts.append(
    Bind(
      name=group_size_var,
      expr=RawString(text=f'{ctx.tile_var}.size()'),
      type_decl='uint32_t',
    )
  )
  stmts.append(BlankLine())

  for i in range(num_sources):
    src = cart_op.sources[i]
    assert isinstance(src, mir.ColumnSource)
    if alias_targets[i] is not None:
      # Reuse narrowed handle from parent (with legacy comment).
      stmts.append(
        RawString(
          text=f'auto {handle_var_names[i]} = {alias_targets[i]};  // reusing narrowed handle'
        )
      )
    elif src.prefix_vars:
      # Fresh root + chained `.prefix(<key>)` per prefix var. Used when
      # the prefix vars are bound by a non-CJ outer scope (e.g. Scan)
      # so handle_vars has no parent. Mirrors Nim's
      # `genChainedPrefixCalls(genRootHandle(view), prefix_vars, view)`
      # in jit_instructions.nim.
      expr: Op = SaRoot(view_name=view_var_names[i])
      for v in src.prefix_vars:
        expr = SaPrefCoop(
          parent=expr,
          key_var=_sanitize_var_name(v),
          view_name=view_var_names[i],
        )
      stmts.append(Bind(name=handle_var_names[i], expr=expr))
    else:
      # Fresh root construction; no comment.
      stmts.append(
        Bind(
          name=handle_var_names[i],
          expr=SaRoot(view_name=view_var_names[i]),
        )
      )
  stmts.append(BlankLine())

  validity_parts = ' || '.join(f'!{h}.valid()' for h in handle_var_names)
  stmts.append(RawString(text=f'if ({validity_parts}) continue;'))
  stmts.append(BlankLine())

  for i in range(num_sources):
    stmts.append(
      Bind(
        name=degree_var_names[i],
        expr=SaDegree(handle_name=handle_var_names[i]),
        type_decl='uint32_t',
      )
    )

  stmts.append(
    Bind(
      name=total_var,
      expr=RawString(text=' * '.join(degree_var_names)),
      type_decl='uint32_t',
    )
  )
  stmts.append(RawString(text=f'if ({total_var} == 0) continue;'))
  stmts.append(BlankLine())

  # Pre-narrow handle bindings — between total/0 check and flat
  # loop. Each info emits a comment + zero or more const_prefix
  # binds + the final var_name bind.
  #
  # Emission order: Nim's `Table[int, ...]` hash-bucket order over
  # the negation handle indices. Counter-suffix names (`_neg_pre_<n>`)
  # were already allocated at registration time (forward), so reordering
  # only the EMIT order keeps the names stable while matching Nim
  # byte-for-byte. F2 fix; see ddisasm StackLiveVarPriorUsed
  # (handles 8, 9 → Nim emits handle 9 first because hashWangYi1(9)&63
  # = 3 < hashWangYi1(8)&63 = 54).
  emit_order_indices = _nim_table_iter_order(registered_neg_idxs)
  for emit_idx in emit_order_indices:
    info, const_var_names = pre_narrow_emissions[emit_idx]
    if ctx.debug:
      stmts.append(
        Comment(
          text=f'Pre-narrow negation handle for {info.rel_name} '
          f'(pre-Cartesian vars: {", ".join(info.pre_vars)})'
        )
      )
    current_expr: Op = SaRoot(view_name=info.view_var)
    for k, (_col_idx, const_val) in enumerate(info.pre_consts):
      const_var = const_var_names[k]
      stmts.append(
        Bind(
          name=const_var,
          expr=SaPrefCoop(
            parent=current_expr,
            key_var=str(const_val),
            view_name=info.view_var,
          ),
        )
      )
      current_expr = VarRef(name=const_var)
    if info.pre_vars:
      for v in info.pre_vars:
        current_expr = SaPrefCoop(
          parent=current_expr,
          key_var=_sanitize_var_name(v),
          view_name=info.view_var,
        )
      stmts.append(Bind(name=info.var_name, expr=current_expr))
    else:
      stmts.append(Bind(name=info.var_name, expr=current_expr))
    stmts.append(BlankLine())

  # R1 short-circuit: emit per-lane add_count, no inner loop.
  if cartesian_as_product:
    total_expr = ' * (uint64_t)'.join(degree_var_names)
    short_circuit_stmts: list[Op] = []
    if ctx.debug:
      short_circuit_stmts.append(
        Comment(text='Count-as-product: per-lane share without inner loop')
      )
    short_circuit_stmts.append(RawString(text='{'))
    short_circuit_stmts.append(
      IndentBlock(
        extra=1,
        stmts=(
          Bind(
            name='cap_total',
            expr=RawString(text=f'(uint64_t){total_expr}'),
            type_decl='uint64_t',
          ),
          Bind(
            name='lane_total',
            expr=RawString(text='static_cast<uint32_t>(cap_total)'),
            type_decl='uint32_t',
          ),
          Bind(
            name='lane_share',
            expr=RawString(
              text=f'({lane_var} < lane_total) ? '
              f'((lane_total - {lane_var} + {group_size_var} - 1) / '
              f'{group_size_var}) : 0'
            ),
            type_decl='uint32_t',
          ),
          AddCount(output_var=ctx.output_var, delta=VarRef(name='lane_share')),
        ),
      )
    )
    short_circuit_stmts.append(RawString(text='}'))
    stmts.extend(short_circuit_stmts)
    return Block(stmts=tuple(stmts))

  # Cartesian flat loop body: indent +1 stmts (decompose, var-binds)
  # then body at outer (loop) indent (legacy quirk).
  inner_decompose_stmts: list[Op] = []
  if num_sources == 1:
    inner_decompose_stmts.append(
      Bind(
        name=idx_vars[0],
        expr=VarRef(name=flat_idx_var),
        type_decl='uint32_t',
      )
    )
  elif num_sources == 2:
    inner_decompose_stmts.append(
      Cartesian2DDecompose(
        major_var=major_var,
        idx0_var=idx_vars[0],
        idx1_var=idx_vars[1],
        flat_idx_var=flat_idx_var,
        deg0_var=degree_var_names[0],
        deg1_var=degree_var_names[1],
      )
    )
  else:
    inner_decompose_stmts.append(
      CartesianNDecompose(
        flat_idx_var=flat_idx_var,
        idx_vars=tuple(idx_vars),
        deg_vars=tuple(degree_var_names),
      )
    )

  inner_decompose_stmts.append(BlankLine())

  for i, src in enumerate(cart_op.sources):
    assert isinstance(src, mir.ColumnSource)
    if i >= len(cart_op.var_from_source):
      continue
    prefix_len = len(src.prefix_vars)
    for v_idx, var_name in enumerate(cart_op.var_from_source[i]):
      if ctx.is_counting and not _cart_var_used(var_name, rest, _trailing_inserts(rest)):
        continue
      col_idx = prefix_len + v_idx
      inner_decompose_stmts.append(
        Bind(
          name=_sanitize_var_name(var_name),
          expr=SaGetValAtPos(
            view_name=view_var_names[i],
            col=col_idx,
            handle_name=handle_var_names[i],
            idx_var_name=idx_vars[i],
          ),
        )
      )

  inner_decompose_stmts.append(BlankLine())

  loop_body = Block(
    stmts=(
      IndentBlock(extra=1, stmts=tuple(inner_decompose_stmts)),
      body_op,
    )
  )

  stmts.append(
    CartesianFlatLoop(
      idx_var=flat_idx_var,
      bound_var=total_var,
      lane_var=lane_var,
      group_size_var=group_size_var,
      body=loop_body,
    )
  )
  return Block(stmts=tuple(stmts))


def _cart_var_used(
  var_name: str,
  rest: list[mir.MirNode],
  inserts: list[mir.InsertInto],
) -> bool:
  '''Counting-phase optimization gate for Cartesian var-binds.

  Multi-head: any of the trailing InsertIntos referencing `var_name`
  counts as a usage.
  '''
  return any(_var_used_in_op(var_name, op) for op in (*rest, *inserts))


def _lower_nested_cj_multi(
  cj_op: mir.ColumnJoin,
  rest: list[mir.MirNode],
  ctx: LoweringCtx,
) -> Op:
  '''Lower a nested multi-source ColumnJoin.

  Mirrors `_nested_column_join_multi` in `ir/dialects/target/cuda/instructions.py`.
  Counter trajectory: body is rendered FIRST (its counter bumps
  persist), then our outer scaffold names are allocated. The IIR
  carries pre-baked names that match the legacy emitter's order.

  Source handling per src.prefix_vars:
    - non-empty: alias the parent handle (looked up by state key
      registered by the surrounding CJ).
    - empty (fresh): construct a fresh root via `HandleType(0,
      view.num_rows_, 0)`. No alias.
  '''
  num_sources = len(cj_op.sources)
  assert num_sources >= 2

  inner_var_sanitized = _sanitize_var_name(cj_op.var_name)

  # Step 1: pre-register the deterministic ch_<rel>_<src>_<var>
  # names so any deeper nested CJ in body can find them by state key.
  registered_state_keys: list[str] = []
  for src in cj_op.sources:
    assert isinstance(src, mir.ColumnSource)
    ch_name = f'ch_{src.rel_name}_{src.handle_start}_{inner_var_sanitized}'
    new_state_key = _state_key(
      src.rel_name,
      list(src.index),
      [*src.prefix_vars, cj_op.var_name],
      src.version,
    )
    ctx.handle_vars[new_state_key] = ch_name
    registered_state_keys.append(new_state_key)

  ctx.bound_vars.append(cj_op.var_name)

  # Step 2: render body before allocating our own counter-bumped
  # names. Body's bumps persist (legacy semantics for nested
  # contexts: no save/restore at this level).
  body_op = _lower_inner_chain(rest, ctx)

  ctx.bound_vars.pop()
  for k in registered_state_keys:
    ctx.handle_vars.pop(k, None)

  # Step 3: allocate our scaffold names — aliases (or fresh roots
  # for prefix-empty sources), intersect, iter.
  source_alias_names: list[str] = []
  source_view_names: list[str] = []
  alias_bind_stmts: list[Op] = []

  for src in cj_op.sources:
    assert isinstance(src, mir.ColumnSource)

    src_view = ctx.view_var_names.get(str(src.handle_start), '')
    if not src_view:
      raise ValueError(
        f'_lower_nested_cj_multi: no view var for source handle_idx {src.handle_start}'
      )
    source_view_names.append(src_view)

    alias_var = ctx.fresh(f'h_{src.rel_name}_{src.handle_start}')
    source_alias_names.append(alias_var)

    if src.prefix_vars:
      # Aliased from a parent handle in the enclosing scope.
      parent_state_key = _state_key(src.rel_name, list(src.index), src.prefix_vars, src.version)
      parent_handle = ctx.handle_vars.get(parent_state_key, '')
      if not parent_handle:
        raise ValueError(
          f'_lower_nested_cj_multi: no parent handle for state key {parent_state_key!r}'
        )
      alias_bind_stmts.append(Bind(name=alias_var, expr=VarRef(name=parent_handle)))
    else:
      # Fresh source: brand-new root handle, no narrowing.
      alias_bind_stmts.append(Bind(name=alias_var, expr=SaRoot(view_name=src_view)))

  intersect_var = ctx.fresh('intersect')
  iter_var = ctx.fresh('it')

  iterator_exprs = tuple(
    SaIterators(handle_name=hn, view_name=vn)
    for hn, vn in zip(source_alias_names, source_view_names)
  )

  # Detect multi-view fresh-root sources (D2L FULL_VER with no
  # prefix_vars) — each one needs a wrapping `for (_nseg_<i>)` segment
  # loop that rebinds its view variable to HEAD then FULL.
  segment_loops: list[tuple[int, mir.ColumnSource, int, int]] = []
  for i, src in enumerate(cj_op.sources):
    assert isinstance(src, mir.ColumnSource)
    if src.prefix_vars:
      continue
    idx_type = ctx.rel_index_types.get(src.rel_name, '')
    vc = view_count(src.version.code, idx_type)
    if vc <= 1:
      continue
    base_slot = ctx.view_slot_bases.get(str(src.handle_start), src.handle_start)
    segment_loops.append((i, src, vc, base_slot))

  has_segments = bool(segment_loops)

  # child_range bindings live INSIDE the for-loop body. Without
  # segment loops, the legacy quirk places them at +1 indent (via
  # IndentBlock(+1)). With segment loops, the loop_body is itself
  # nested inside a +1-indented Block, which already accounts for
  # the depth — child_binds emit at the same indent as the surrounding
  # for-iter, matching the legacy `_nested_column_join_multi`'s
  # ind(ctx) trick.
  child_bind_stmts: list[Op] = []
  for i, src in enumerate(cj_op.sources):
    assert isinstance(src, mir.ColumnSource)
    ch_name = f'ch_{src.rel_name}_{src.handle_start}_{inner_var_sanitized}'
    child_bind_stmts.append(
      Bind(
        name=ch_name,
        expr=SaChildRange(
          handle_name=source_alias_names[i],
          pos_expr=f'positions[{i}]',
          key_var=inner_var_sanitized,
          view_name=source_view_names[i],
        ),
      )
    )

  iter_loop_body = Block(
    stmts=(IndentBlock(extra=1, stmts=tuple(child_bind_stmts)), body_op),
  )

  stmts: list[Op] = []
  if ctx.debug:
    stmts.append(
      Comment(
        text=f'Nested ColumnJoin (intersection): '
        f'bind \'{cj_op.var_name}\' from {num_sources} sources'
      )
    )
    src_debug = ' '.join(
      f'({s.rel_name} :handle {s.handle_start} :prefix ({" ".join(s.prefix_vars)}))'
      for s in cj_op.sources
    )
    stmts.append(Comment(text=f'MIR: (column-join :var {cj_op.var_name} :sources ({src_debug} ))'))

  intersect_iter_op: Op = IntersectIter(
    intersect_var=intersect_var,
    iter_var=iter_var,
    iterator_exprs=iterator_exprs,
    value_var=inner_var_sanitized,
    body=iter_loop_body,
  )

  if has_segments:
    # D2lSegmentLoop bumps ctx.indent_level by 1 (so alias_binds and
    # IntersectIter emit one level deeper). It also bumps
    # ctx.segment_depth so IntersectIter knows to anchor its body
    # lines / body_op back to the surrounding (outer) indent — see
    # the IntersectIter emit case.
    wrapped: Op = Block(stmts=(*alias_bind_stmts, intersect_iter_op))
    # Open segment loops outermost-first (sources[0] outer, sources[-1]
    # inner) — matches legacy `_nested_column_join_multi`.
    for sl_idx, src, vc, base_slot in reversed(segment_loops):
      wrapped = D2lSegmentLoop(
        seg_var=f'_nseg_{sl_idx}',
        view_var=ctx.view_var_names[str(src.handle_start)],
        base_slot=base_slot,
        view_count=vc,
        declare=False,
        local_view_var='',
        body=wrapped,
      )
    stmts.append(wrapped)
  else:
    stmts.extend(alias_bind_stmts)
    stmts.append(intersect_iter_op)

  return Block(stmts=tuple(stmts))


# -----------------------------------------------------------------------------
# Nested single-source ColumnJoin (B-CJ-single)
# -----------------------------------------------------------------------------


def _lower_nested_cj_single(
  cj_op: mir.ColumnJoin,
  rest: list[mir.MirNode],
  ctx: LoweringCtx,
) -> Op:
  '''Lower a nested single-source ColumnJoin.

  Mirrors the runtime `JoinExecutor::execute_single_source` semantics
  in `runtime/.../instructions/instruction_column_join.h`: a sequential
  per-source iteration that binds `cj_op.var_name` from one
  ColumnSource, no intersection needed. Each thread takes a
  lane-strided share of the source's degree:

      auto <h> = <parent_handle or root>;
      uint32_t <degree> = <h>.degree();
      uint32_t <lane> = tile.thread_rank();
      uint32_t <group_size> = tile.size();
      for (uint32_t <idx> = <lane>; <idx> < <degree>; <idx> += <group_size>) {
        auto <var_name> = <h>.get_value_at(<view>, <idx>);
        auto <ch> = <h>.child_range(<idx>, <var_name>, tile, <view>);
        <body>
      }

  Counter trajectory: body is rendered FIRST (its counter bumps
  persist), then our outer scaffold names are allocated. This matches
  `_lower_nested_cj_multi`'s ordering so the byte-equivalence test
  fixture can swap between the legacy branch and the new path
  without seeing counter drift.

  Source handling per src.prefix_vars (same convention as
  `_lower_nested_cj_multi`):
    - non-empty: alias the parent handle (looked up by state key
      registered by the surrounding CJ).
    - empty (fresh): construct a fresh root via SaRoot. No alias.

  Per `docs/phase_b_lowering_dispatcher.md` §4 row B-CJ-single
  (difficulty `medium`): this helper is the single-source half of
  the ColumnJoin migration. The multi-source half (`>=2` sources)
  stays in `_lower_nested_cj_multi` and is migrated by B-CJ-multi
  (the next PR).
  '''
  num_sources = len(cj_op.sources)
  assert num_sources == 1
  src = cj_op.sources[0]
  assert isinstance(src, mir.ColumnSource)

  inner_var_sanitized = _sanitize_var_name(cj_op.var_name)

  # Step 1: pre-register the deterministic ch_<rel>_<src>_<var> name
  # so any deeper nested CJ in body can find the narrowed child by
  # state key. Same convention as `_lower_nested_cj_multi`.
  ch_name = f'ch_{src.rel_name}_{src.handle_start}_{inner_var_sanitized}'
  new_state_key = _state_key(
    src.rel_name,
    list(src.index),
    [*src.prefix_vars, cj_op.var_name],
    src.version,
  )
  ctx.handle_vars[new_state_key] = ch_name
  ctx.bound_vars.append(cj_op.var_name)

  # Step 2: render body before allocating our own counter-bumped
  # names. Body's bumps persist (legacy semantics for nested
  # contexts: no save/restore at this level — same as
  # `_lower_nested_cj_multi`).
  body_op = _lower_inner_chain(rest, ctx)

  ctx.bound_vars.pop()
  ctx.handle_vars.pop(new_state_key, None)

  # Step 3: allocate our own scaffold names.
  src_view = ctx.view_var_names.get(str(src.handle_start), '')
  if not src_view:
    raise ValueError(
      f'_lower_nested_cj_single: no view var for source handle_idx {src.handle_start}'
    )

  alias_var = ctx.fresh(f'h_{src.rel_name}_{src.handle_start}')
  lane_var = ctx.fresh('lane')
  group_size_var = ctx.fresh('group_size')
  degree_var = ctx.fresh('degree')
  idx_var = ctx.fresh('idx0')

  if src.prefix_vars:
    # Aliased from a parent handle in the enclosing scope.
    parent_state_key = _state_key(src.rel_name, list(src.index), src.prefix_vars, src.version)
    parent_handle = ctx.handle_vars.get(parent_state_key, '')
    if not parent_handle:
      raise ValueError(
        f'_lower_nested_cj_single: no parent handle for state key {parent_state_key!r}'
      )
    alias_bind: Op = Bind(name=alias_var, expr=VarRef(name=parent_handle))
  else:
    # Fresh source: brand-new root handle, no narrowing.
    alias_bind = Bind(name=alias_var, expr=SaRoot(view_name=src_view))

  stmts: list[Op] = []
  if ctx.debug:
    stmts.append(
      Comment(
        text=f'Nested ColumnJoin (single source): bind \'{cj_op.var_name}\' from {src.rel_name}'
      )
    )
    src_debug = f'({src.rel_name} :handle {src.handle_start} :prefix ({" ".join(src.prefix_vars)}))'
    stmts.append(Comment(text=f'MIR: (column-join :var {cj_op.var_name} :sources ({src_debug} ))'))

  stmts.append(alias_bind)
  stmts.append(IfContinueIfNot(cond=SaValid(handle_name=alias_var)))
  stmts.append(
    Bind(
      name=lane_var,
      expr=RawString(text=f'{ctx.tile_var}.thread_rank()'),
      type_decl='uint32_t',
    )
  )
  stmts.append(
    Bind(
      name=group_size_var,
      expr=RawString(text=f'{ctx.tile_var}.size()'),
      type_decl='uint32_t',
    )
  )
  stmts.append(
    Bind(
      name=degree_var,
      expr=SaDegree(handle_name=alias_var),
      type_decl='uint32_t',
    )
  )

  # Loop body: bind the value, register the child range, then run
  # the recursive body. The child_bind matches the legacy multi-
  # source name convention (`ch_<rel>_<src>_<var>`) so deeper
  # CJs can look it up by state key.
  val_bind = Bind(
    name=inner_var_sanitized,
    expr=SaGetValAt(
      handle_name=alias_var,
      view_name=src_view,
      idx_var_name=idx_var,
    ),
  )
  child_bind = Bind(
    name=ch_name,
    expr=SaChildRange(
      handle_name=alias_var,
      pos_expr=idx_var,
      key_var=inner_var_sanitized,
      view_name=src_view,
    ),
  )
  loop_body = Block(stmts=(val_bind, child_bind, body_op))

  stmts.append(
    CartesianFlatLoop(
      idx_var=idx_var,
      bound_var=degree_var,
      lane_var=lane_var,
      group_size_var=group_size_var,
      body=loop_body,
    )
  )

  return Block(stmts=tuple(stmts))


def _lower_insert_into(node: mir.InsertInto, ctx: LoweringCtx) -> list[Op]:
  '''Lower an InsertInto under the M1-M3 narrow-flag assumptions.

  When `ctx.dedup_hash` is set, the entire emit is wrapped in a
    `{ bool _p = dedup_table.try_insert(thread_id, v0, ...);
       if (_p) { <write> } }`
  gate. In materialize phase the write goes through
    `[if (lane==0)] { uint32_t pos = atomicAdd(atomic_write_pos, 1u);
       out_data_0[(pos + out_base_0) + col * out_stride_0] = vN; ... }`
  instead of `output.emit_direct(...)`. The `out_data_0` /
  `atomic_write_pos` names are kernel parameters injected by the
  runner-side dedup_hash plumbing — the dialect treats them as
  free variables.

  DEAD CODE NOTE (C2): the `if ctx.dedup_hash:` branches below
  (search for `use_dedup`) are superseded by the `@lowering(
  target=iir.cf, source=mir.DedupGate)` rule registered by
  `pragmas/dedup_hash.py`. The new lowering delegates back into
  this helper with `ctx.dedup_hash` forced True, so the branches
  remain functionally LIVE during the C2 transition (the
  `with_pragma(DedupHash())` DSL dual-writes both the bool field
  AND the typed pragma; see `pragmas/dedup_hash.py:
  materialize_dedup_hash`). The branches will be removable once
  Phase A3 drops the `dedup_hash: bool` field on
  `mir.ExecutePipeline` and Phase B migrates the host `InsertInto`
  lowering out of this monolith — until then they are the
  byte-equivalence anchor for the dual-write transition.

  DEAD CODE NOTE (C4): the `if ctx.ws_enabled:` branch below
  (count-phase WS, search for `ws_enabled`) is superseded by the
  `@lowering(target=iir.cf, source=mir.WSScope)` rule registered by
  `srdatalog.ir.dialects.parallel.atomic_ws.pragmas.work_stealing`.
  Same dual-write contract as C2: the new lowering delegates back
  into this helper with `ctx.ws_enabled` forced True, so the branch
  remains functionally LIVE during the C4 transition (the
  `with_pragma(WorkStealing())` DSL dual-writes both the bool field
  AND the typed pragma; see `pragmas/work_stealing.py:
  materialize_work_stealing`). The branch will be removable once
  Phase A3 drops the `work_stealing: bool` field on
  `mir.ExecutePipeline`. The `ws_cartesian_valid_var` branch
  further below is unrelated to the C4 migration — it's a separate
  legacy plumbing path (batched-Cartesian WS materialize) not
  driven by `ctx.ws_enabled` and not in C4 scope.

  DEAD CODE NOTE (C6 - count): the `if ctx.is_counting:` branches
  in this function (and elsewhere in this monolith) are superseded
  by the `@lowering(target=iir.cf, source=mir.CountPhase)` rule
  registered by `iir/cf/pragmas/count.py`. Per
  `docs/phase_c_pragma_materialization.md` §4.3, `count` is the
  legacy phase flag that becomes `iir.cf.Phase(C, body)` at the
  IIR layer. The new lowering's wrap op is only inserted when
  `ep.count is False` (the dual-write short-circuit in
  `materialize_count`); when `ep.count is True` the legacy
  `is_counting`-driven path here continues to emit, keeping the
  count-phase byte-equivalence anchor intact. The branches will
  be removable once Phase A3 drops the `count: bool` field on
  `mir.ExecutePipeline` and Phase B migrates count-phase emission
  out of this monolith — until then they are LIVE and
  load-bearing for every runner-goldens kernel (every kernel has
  a count phase).

  DEAD CODE NOTE (C6 - fanout): the `ep.use_fan_out` consumer
  paths live in `complete_runner.py` (out of scope for this PR
  per the C6 task constraint), not in this monolith. The C6
  `@pragma_handler(FanOut, on=ExecutePipeline)` short-circuits
  when `ep.use_fan_out is True`, so the legacy runner path stays
  the byte-equivalence anchor for fan-out rules. The `@lowering(
  target=iir.cf, source=mir.FanOut)` rule in `pragmas/fanout.py`
  is reachable only when only the typed pragma is set (the
  post-A3 pure-typed state, or test fixtures).
  '''
  out_var = ctx.output_var_overrides.get(node.rel_name, ctx.output_var)
  vars_list = list(node.vars)

  stmts: list[Op] = []
  if ctx.debug:
    stmts.append(Comment(text=f'Emit: {node.rel_name}({", ".join(vars_list)})'))

  # Multi-head count phase: secondary outputs are flagged
  # `__skip_counting__` so the runner doesn't double-count rows the
  # primary already accounted for. Emit a comment in place of the
  # increment, matching legacy `jit_insert_into`.
  if ctx.is_counting and out_var == '__skip_counting__':
    if ctx.debug:
      stmts.append(Comment(text=f'Skip counting for secondary output {node.rel_name}'))
    return stmts

  sanitized_list = [_sanitize_var_name(v) for v in vars_list]
  use_dedup = ctx.dedup_hash and bool(vars_list)

  if use_dedup:
    args_str = ', '.join(sanitized_list)
    stmts.append(RawString(text=f'{{ bool _p = dedup_table.try_insert(thread_id, {args_str});'))
    stmts.append(RawString(text='  if (_p) {'))

  if ctx.is_counting:
    if ctx.ws_enabled:
      # WS count: per-thread `local_count` (uint32_t) — `<out>++`
      # instead of `<out>.emit_direct()`. Lane-0 guard outside Cart.
      body: Op = RawString(text=f'{out_var}++;')
    else:
      body = RawString(text=f'{out_var}.emit_direct();')
    if not ctx.inside_cartesian:
      stmts.append(LaneZeroGuard(body=body))
    else:
      stmts.append(body)
  elif use_dedup:
    # Materialize + dedup: atomic-add write into out_data_0.
    # Inner block at SAME indent as the surrounding lines (the
    # `if (lane==0) {` opener and matching `}` on their own
    # ctx.ind() lines; per-write rows carry an embedded 2-space
    # prefix in the RawString text — mirrors legacy emit_helpers).
    if not ctx.inside_cartesian:
      stmts.append(RawString(text=f'if ({ctx.tile_var}.thread_rank() == 0) {{'))
    else:
      stmts.append(RawString(text='{'))
    stmts.append(RawString(text='  uint32_t pos = atomicAdd(atomic_write_pos, 1u);'))
    for col, name in enumerate(sanitized_list):
      stmts.append(
        RawString(text=f'  out_data_0[(pos + out_base_0) + {col} * out_stride_0] = {name};')
      )
    stmts.append(RawString(text='}'))
  elif ctx.ws_cartesian_valid_var:
    # WS Cartesian batched-valid materialize: emit_warp_coalesced(
    # tile, valid, args). All threads call cooperatively — no lane-0
    # guard, valid bit gates the actual write.
    sanitized = ', '.join(sanitized_list)
    stmts.append(
      RawString(
        text=f'{out_var}.emit_warp_coalesced({ctx.tile_var}, '
        f'{ctx.ws_cartesian_valid_var}, {sanitized});'
      )
    )
  else:
    sanitized = ', '.join(sanitized_list)
    body = RawString(text=f'{out_var}.emit_direct({sanitized});')
    if not ctx.inside_cartesian:
      stmts.append(LaneZeroGuard(body=body))
    else:
      stmts.append(body)

  if use_dedup:
    stmts.append(RawString(text='} }'))

  return stmts


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _filter_expr(code: str) -> str:
  expr = code.strip()
  if expr.startswith('return '):
    expr = expr[len('return ') :]
  if expr.endswith(';'):
    expr = expr[:-1]
  return expr


def _var_used_in_op(var_name: str, op: mir.MirNode) -> bool:
  '''True iff `var_name` is referenced by `op` — covers every MIR op
  that can introduce or consume a variable name. Mirrors the legacy
  substring-on-rendered-body check by enumerating every structural
  position a var name can appear in.'''
  if isinstance(op, mir.Filter):
    return var_name in op.vars
  if isinstance(op, mir.ConstantBind):
    return var_name in op.code
  if isinstance(op, mir.ColumnJoin):
    if var_name == op.var_name:
      return True
    return any(var_name in src.prefix_vars for src in op.sources)
  if isinstance(op, mir.CartesianJoin):
    for vfs in op.var_from_source:
      if var_name in vfs:
        return True
    return any(var_name in src.prefix_vars for src in op.sources)
  if isinstance(op, mir.Negation | mir.Aggregate):
    return var_name in op.prefix_vars
  if isinstance(op, mir.InsertInto):
    return var_name in op.vars
  if isinstance(op, mir.PositionedExtract):
    return any(var_name in src.prefix_vars for src in op.sources)
  return False


def _scan_var_used(
  var_name: str,
  middle: list[mir.MirNode],
  inserts: list[mir.InsertInto],
) -> bool:
  '''Counting-phase optimization gate. Returns True iff `var_name`
  is referenced anywhere downstream — in middle ops or any of the
  trailing InsertIntos (multi-head).

  Mirrors Nim's `varName notin body` substring check on the rendered
  body. Note: Nim's check is conservatively True even for vars only
  appearing in the `// Emit: Edge(x, y)` comment text, so this
  predicate must include InsertInto.vars even in count phase to match
  byte-for-byte.
  '''
  return any(_var_used_in_op(var_name, op) for op in (*middle, *inserts))


def _nim_int_hash(x: int) -> int:
  '''Port of Nim 2.x `hash(x: int)` = `hashWangYi1(uint64(x))`.

  Nim's default `Table[int, V]` uses this hash for the slot calc
  `slot = hash(key) & (cap - 1)`. We reproduce it bit-exact so the
  pre-narrow Negation emit order matches the Nim reference.
  '''
  P0 = 0xA0761D6478BD642F
  P1 = 0xE7037ED1A0B428DB
  P58 = 0xEB44ACCAB455D165 ^ 8
  mask64 = (1 << 64) - 1

  def hi_xor_lo(a: int, b: int) -> int:
    prod = (a * b) & ((1 << 128) - 1)
    return ((prod >> 64) ^ (prod & mask64)) & mask64

  return hi_xor_lo(hi_xor_lo(P0, (x & mask64) ^ P1), P58)


def _nim_table_iter_order(keys: list[int], cap: int = 64) -> list[int]:
  '''Indices into `keys` in Nim `Table[int, ...]` iteration order.

  Reproduces Nim's `Table` insert + iterate behavior:
    - slot = `hashWangYi1(key) & (cap - 1)`
    - linear-probe forward on collision (`(h + 1) & (cap - 1)`)
    - iterate slots in ascending order

  `cap=64` matches Nim's `defaultInitialSize` for Table (rehashing
  doesn't fire below the load-factor threshold for any pre-narrow
  count we hit in practice).

  Returns a permutation of `range(len(keys))` — the order in which
  the caller should walk its parallel `keys`-indexed list.
  '''
  if not keys:
    return []
  slots: list[int] = [-1] * cap
  for i, k in enumerate(keys):
    h = _nim_int_hash(k) & (cap - 1)
    while slots[h] != -1:
      h = (h + 1) & (cap - 1)
    slots[h] = i
  return [s for s in slots if s != -1]


def _state_key(
  rel_name: str,
  index: list[int],
  prefix_vars: list[str],
  version: Version,
) -> str:
  '''Mirror gen_handle_state_key from legacy.

  Format: `<rel>_<col0>_<col1>_..._<version>` (or with prefix_vars
  appended).
  '''
  ver_str = version.code
  base = rel_name + '_' + '_'.join(str(c) for c in index)
  if ver_str:
    base = base + '_' + ver_str
  if prefix_vars:
    base = base + '_' + '_'.join(prefix_vars)
  return base


def _sanitize_var_name(name: str) -> str:
  '''Mirror the legacy `sanitize_var_name`. C++ keywords get a `_val`
  suffix; everything else passes through.'''
  cpp_keywords = {
    'class',
    'struct',
    'union',
    'enum',
    'typedef',
    'template',
    'using',
    'namespace',
    'public',
    'private',
    'protected',
    'const',
    'volatile',
    'mutable',
    'static',
    'extern',
    'inline',
    'virtual',
    'override',
    'final',
    'explicit',
    'friend',
    'new',
    'delete',
    'this',
    'typeid',
    'sizeof',
    'alignof',
    'true',
    'false',
    'nullptr',
    'auto',
    'register',
    'thread_local',
    'if',
    'else',
    'switch',
    'case',
    'default',
    'while',
    'do',
    'for',
    'break',
    'continue',
    'return',
    'goto',
    'try',
    'catch',
    'throw',
  }
  if name in cpp_keywords:
    return f'{name}_val'
  return name
