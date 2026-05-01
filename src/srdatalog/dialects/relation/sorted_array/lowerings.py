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

import srdatalog.mir.types as mir
from srdatalog.dialects.iir.cf import (
  Bind,
  BlankLine,
  Block,
  Cartesian2DDecompose,
  CartesianFlatLoop,
  Comment,
  GridStrideLoop,
  If,
  IfContinueIfNot,
  IfReturnIfNot,
  IndentBlock,
  IntersectIter,
  LaneZeroGuard,
  ParallelFor,
  RawString,
  VarRef,
)
from srdatalog.dialects.relation.sorted_array.ops import (
  SaChildRange,
  SaDegree,
  SaGetVal,
  SaGetValAtPos,
  SaHint,
  SaIterators,
  SaPrefCoop,
  SaRoot,
  SaValid,
)
from srdatalog.hir.types import Version
from srdatalog.ir_core import Op


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
  # State-key -> handle var name. Lets nested CJ find the parent
  # handle to alias by the same (rel, cols, prefix_vars, ver) key
  # that the outer CJ used to register it.
  handle_vars: dict[str, str] = field(default_factory=dict)

  def fresh(self, prefix: str) -> str:
    self.name_counter += 1
    return f'{prefix}_{self.name_counter}'


# -----------------------------------------------------------------------------
# Public entry point + supported-shape predicate
# -----------------------------------------------------------------------------


def _supported_pipeline(ops: list[mir.MirNode]) -> bool:
  '''True iff the dialect can lower this pipeline shape today.'''
  if len(ops) < 2:
    return False
  if not isinstance(ops[-1], mir.InsertInto):
    return False
  head = ops[0]
  middle = ops[1:-1]

  if isinstance(head, mir.Scan):
    # M1+M2+M5 shapes
    for op in middle:
      if isinstance(op, (mir.Filter, mir.ConstantBind)):
        continue
      if isinstance(op, mir.Negation):
        continue
      return False
    return True

  if isinstance(head, mir.ColumnJoin) and len(head.sources) >= 2:
    # M3+M5+M7 shape: multi-source root CJ; middle can hold
    # nested CJs / Filter / ConstantBind / Negation / Cartesian.
    # However Negation immediately following a Cartesian uses the
    # legacy `neg_pre_narrow` optimization which we don't yet
    # lower — reject those for now.
    has_cart = False
    for op in middle:
      if isinstance(op, (mir.Filter, mir.ConstantBind)):
        continue
      if isinstance(op, mir.ColumnJoin) and len(op.sources) >= 2:
        continue
      if isinstance(op, mir.CartesianJoin) and 1 <= len(op.sources) <= 2:
        has_cart = True
        continue
      if isinstance(op, mir.Negation):
        # Defer Cart-then-Neg pre-narrow handling (M5.x).
        if has_cart:
          return False
        # Const_args: not lowered yet.
        if op.const_args:
          return False
        continue
      return False
    return True

  return False


def lower_scan_pipeline(
  ops: list[mir.MirNode],
  ctx: LoweringCtx,
) -> Op:
  '''Lower a supported pipeline shape to IIR.

  The function name is historical (M1 only handled Scan-rooted
  pipelines); it now dispatches on the head op. Raises ValueError
  if the shape isn't supported.
  '''
  if not _supported_pipeline(ops):
    raise ValueError(
      f'lower_scan_pipeline: unsupported pipeline shape '
      f'{[type(o).__name__ for o in ops]}'
    )

  head = ops[0]
  rest = ops[1:]

  if isinstance(head, mir.Scan):
    return _lower_root_scan(head, rest, ctx)
  if isinstance(head, mir.ColumnJoin):
    return _lower_root_cj_multi(head, rest, ctx)

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
    raise ValueError(
      f'_lower_root_scan: no view var for handle_idx {handle_idx}'
    )

  middle = list(rest[:-1])
  insert_op = rest[-1]
  assert isinstance(insert_op, mir.InsertInto)

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
  body_op = _lower_inner_chain(middle, insert_op, ctx)
  ctx.name_counter = saved_counter
  for _ in range(pushed_var_count):
    ctx.bound_vars.pop()

  # Step 2: allocate own scaffold names.
  outer_stmts: list[Op] = []

  if ctx.debug:
    outer_stmts.append(
      Comment(
        text=f'Root Scan: {scan_op.rel_name} binding {", ".join(scan_op.vars)}'
      )
    )
    outer_stmts.append(
      Comment(
        text=f'MIR: (scan :rel {scan_op.rel_name} '
        f':vars ({" ".join(scan_op.vars)}) :handle {handle_idx})'
      )
    )

  handle_var = ctx.fresh('root_handle')
  outer_stmts.append(Bind(name=handle_var, expr=SaRoot(view_name=view_var)))
  outer_stmts.append(IfReturnIfNot(cond=SaValid(handle_name=handle_var)))

  degree_var = ctx.fresh('degree')
  outer_stmts.append(
    Bind(name=degree_var, expr=SaDegree(handle_name=handle_var), type_decl='uint32_t')
  )
  outer_stmts.append(BlankLine())

  idx_var = ctx.fresh('idx')

  var_bind_stmts: list[Op] = []
  for col, var_name in enumerate(scan_op.vars):
    if ctx.is_counting and not _scan_var_used(var_name, middle, insert_op):
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

  Mirrors `_root_cj_multi` in `codegen/jit/root.py`. Counter
  trajectory matches legacy: body is rendered with its own counter
  trajectory starting from saved=0; then outer names are allocated
  starting from saved=0 again. Body and outer have overlapping
  counter ranges but different prefixes — the legacy convention.
  '''
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
        f'_lower_root_cj_multi: no view var for source handle_idx '
        f'{src.handle_start}'
      )
    source_view_names.append(src_view)

    state_key = _state_key(
      src.rel_name, list(src.index), [cj_op.var_name], src.version
    )
    ctx.handle_vars[state_key] = handle_var
    registered_state_keys.append(state_key)

  ctx.bound_vars.append(cj_op.var_name)

  # Step 2: render body BEFORE allocating our own counter-bumped
  # names. Body's counter trajectory starts at the current value
  # (typically 0 at the top of pipeline lowering) and bumps freely.
  saved_counter = ctx.name_counter
  insert_op = rest[-1]
  assert isinstance(insert_op, mir.InsertInto)
  body_op = _lower_inner_chain(list(rest[:-1]), insert_op, ctx)
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
    outer_stmts.append(
      Comment(text='Uses root_unique_values + prefix() pattern (like TMP)')
    )
    src_debug = ' '.join(
      f'({s.rel_name} :handle {s.handle_start})' for s in cj_op.sources
    )
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

  for i, src in enumerate(cj_op.sources):
    assert isinstance(src, mir.ColumnSource)
    handle_var = source_handle_names[i]
    src_view = source_view_names[i]

    if i == 0:
      hint_lo = ctx.fresh('hint_lo')
      hint_hi = ctx.fresh('hint_hi')
      loop_inner_stmts.append(
        Bind(name=hint_lo, expr=VarRef(name=y_idx_var), type_decl='uint32_t')
      )
      loop_inner_stmts.append(
        Bind(
          name=hint_hi,
          expr=RawString(
            text=f'{src_view}.num_rows_ - '
            f'(num_unique_root_keys - {y_idx_var} - 1)'
          ),
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
        RawString(
          text=f'{hint_hi} = ({hint_hi} > {hint_lo}) ? '
          f'{hint_hi} : {src_view}.num_rows_;'
        )
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
    loop_inner_stmts.append(
      IfContinueIfNot(cond=SaValid(handle_name=handle_var))
    )

  loop_inner_stmts.append(
    Bind(
      name=_sanitize_var_name(cj_op.var_name),
      expr=VarRef(name=root_val_var),
    )
  )

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
# Inner-chain lowering: nested CJ / Filter / ConstantBind / InsertInto
# -----------------------------------------------------------------------------


def _lower_inner_chain(
  middle: list[mir.MirNode],
  insert: mir.InsertInto,
  ctx: LoweringCtx,
) -> Op:
  '''Lower the chain of post-root ops ending in InsertInto.'''
  if not middle:
    return Block(stmts=tuple(_lower_insert_into(insert, ctx)))

  head = middle[0]
  rest = middle[1:]

  if isinstance(head, mir.Filter):
    cond_expr = _filter_expr(head.code)
    return If(
      cond=RawString(text=cond_expr),
      body=_lower_inner_chain(rest, insert, ctx),
    )

  if isinstance(head, mir.ConstantBind):
    var = _sanitize_var_name(head.var_name)
    bind_stmt = Bind(name=var, expr=RawString(text=head.code))
    rest_op = _lower_inner_chain(rest, insert, ctx)
    if isinstance(rest_op, Block):
      return Block(stmts=(bind_stmt, *rest_op.stmts))
    return Block(stmts=(bind_stmt, rest_op))

  if isinstance(head, mir.ColumnJoin) and len(head.sources) >= 2:
    return _lower_nested_cj_multi(head, rest, insert, ctx)

  if isinstance(head, mir.CartesianJoin):
    return _lower_nested_cart(head, rest, insert, ctx)

  if isinstance(head, mir.Negation):
    return _lower_negation(head, rest, insert, ctx)

  raise ValueError(f'unsupported inner op: {type(head).__name__}')


def _lower_negation(
  neg_op: mir.Negation,
  rest: list[mir.MirNode],
  insert: mir.InsertInto,
  ctx: LoweringCtx,
) -> Op:
  '''Lower an anti-join: body fires only when the narrowed handle is
  invalid (i.e. the prefix doesn't exist in the negated relation).

  M5 baseline (no flags, no pre-narrow, no const_args):
    - Build chained-prefix handle: SaPrefCoop nesting from SaRoot.
    - Wrap body in `If(!handle.valid(), body)`.

  Mirrors `jit_negation` in `codegen/jit/scan_negation.py` for the
  default (non-pre-narrow, non-WS, non-tiled, non-const-args) path.

  Counter trajectory: like the legacy, the body is rendered FIRST
  with its own bumps; our handle name is allocated AFTER body.
  '''
  if neg_op.const_args:
    raise NotImplementedError(
      f'_lower_negation: const_args not yet lowered; got {neg_op.const_args}'
    )

  src_idx = neg_op.handle_start
  rel_name = neg_op.rel_name

  view_var = ctx.view_var_names.get(str(src_idx), '')
  if not view_var:
    raise ValueError(
      f'_lower_negation: no view var for handle_idx {src_idx}'
    )

  # Step 1: render body BEFORE allocating own counter-bumped name.
  body_op = _lower_inner_chain(rest, insert, ctx)

  # Step 2: allocate the negation handle name.
  # Legacy: `gen_unique_name(ctx, f"h_{rel_name}_neg_{src_idx}")`
  # which gives `h_<rel>_neg_<src_idx>_<n>`. Our `fresh()` matches.
  neg_handle_var = ctx.fresh(f'h_{rel_name}_neg_{src_idx}')

  # Step 3: build the chained prefix expression.
  # Start from a parent handle (if registered) or a fresh root.
  parent_handle_name = ctx.handle_vars.get(str(src_idx), '')
  parent_expr: Op
  if parent_handle_name:
    parent_expr = VarRef(name=parent_handle_name)
  else:
    parent_expr = SaRoot(view_name=view_var)

  # Apply each prefix variable as a chained SaPrefCoop. M5 only
  # handles the cooperative path (not inside_cartesian).
  if ctx.inside_cartesian:
    raise NotImplementedError(
      '_lower_negation: inside_cartesian path (prefix_seq) not yet lowered'
    )

  for var_name in neg_op.prefix_vars:
    parent_expr = SaPrefCoop(
      parent=parent_expr,
      key_var=_sanitize_var_name(var_name),
      view_name=view_var,
    )

  # Step 4: build the IIR.
  stmts: list[Op] = []
  if ctx.debug:
    stmts.append(Comment(text=f'Negation: NOT EXISTS in {rel_name}'))
    stmts.append(
      Comment(
        text=f'MIR: (negation :rel {rel_name} :prefix '
        f'({" ".join(neg_op.prefix_vars)}) :handle {src_idx})'
      )
    )

  stmts.append(Bind(name=neg_handle_var, expr=parent_expr))
  stmts.append(
    If(
      cond=RawString(text=f'!{neg_handle_var}.valid()'),
      body=body_op,
    )
  )
  return Block(stmts=tuple(stmts))


def _lower_nested_cart(
  cart_op: mir.CartesianJoin,
  rest: list[mir.MirNode],
  insert: mir.InsertInto,
  ctx: LoweringCtx,
) -> Op:
  '''Lower a nested CartesianJoin under M7's narrow assumptions:
    - 1 or 2 sources (no N>=3 yet — countdown decomposition is M7.x).
    - All sources have full state-key handle reuse (the common case
      where the outer CJ already narrowed to the same prefix).
    - No tiled-Cartesian (M8), no count-as-product (separate rule),
      no negation pre-narrow (M5).

  Mirrors `jit_nested_cartesian_join` in `codegen/jit/instructions.py`.
  Counter trajectory matches legacy: body is rendered first; outer
  scaffold names are allocated after.

  Sets `ctx.inside_cartesian = True` while body is rendered so the
  InsertInto inside skips the lane-0 guard.
  '''
  num_sources = len(cart_op.sources)
  if num_sources < 1 or num_sources > 2:
    raise NotImplementedError(
      f'_lower_nested_cart: M7 supports 1-2 sources; got {num_sources}'
    )

  # Step 1: render body BEFORE allocating own counter-bumped names.
  # inside_cartesian flips so InsertInto inside drops the lane-0
  # guard (matches legacy `need_lane0_guard = not ctx.inside_cartesian`).
  saved_inside = ctx.inside_cartesian
  ctx.inside_cartesian = True
  pushed_var_count = 0
  for vars_from_src in cart_op.var_from_source:
    for v in vars_from_src:
      ctx.bound_vars.append(v)
      pushed_var_count += 1
  body_op = _lower_inner_chain(rest, insert, ctx)
  for _ in range(pushed_var_count):
    ctx.bound_vars.pop()
  ctx.inside_cartesian = saved_inside

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
    #   3. no match, has prefix_vars -> chained prefix (M7.x, not yet).
    parent_state_key = _state_key(
      src.rel_name, list(src.index), src.prefix_vars, src.version
    )
    parent_handle = ctx.handle_vars.get(parent_state_key, '')

    if parent_handle:
      alias_targets.append(parent_handle)
    elif not src.prefix_vars:
      alias_targets.append(None)  # fresh root
    else:
      raise NotImplementedError(
        f'_lower_nested_cart: source {src.rel_name} has prefix '
        f'{src.prefix_vars} but no full-state-key match; the '
        f'parent-prefix chained-prefix path isn\'t lowered yet'
      )

    handle_var_names.append(
      ctx.fresh(f'h_{src.rel_name}_{src.handle_start}')
    )

    src_view = ctx.view_var_names.get(str(src.handle_start), '')
    if not src_view:
      raise ValueError(
        f'_lower_nested_cart: no view var for source handle_idx '
        f'{src.handle_start}'
      )
    view_var_names.append(src_view)

  total_var = ctx.fresh('total')
  flat_idx_var = ctx.fresh('flat_idx')

  # 2-source decomposition allocates idx0, idx1, then major.
  idx_vars: list[str] = []
  major_var = ''
  if num_sources == 1:
    idx_vars = [ctx.fresh('idx0')]
  elif num_sources == 2:
    idx_vars = [ctx.fresh('idx0'), ctx.fresh('idx1')]
    major_var = ctx.fresh('major_is_1')

  # Step 3: build IIR.
  stmts: list[Op] = []

  if ctx.debug:
    vars_bound_str = ', '.join(cart_op.vars)
    stmts.append(
      Comment(
        text=f'Nested CartesianJoin: bind {vars_bound_str} from {num_sources} source(s)'
      )
    )
    src_debug = ' '.join(
      f'({s.rel_name} :handle {s.handle_start} '
      f':prefix ({" ".join(s.prefix_vars)}))'
      for s in cart_op.sources
    )
    stmts.append(
      Comment(
        text=f'MIR: (cartesian-join :vars ({" ".join(cart_op.vars)}) '
        f':sources ({src_debug} ))'
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
      # Reuse narrowed handle from parent (with legacy comment).
      stmts.append(
        RawString(
          text=f'auto {handle_var_names[i]} = {alias_targets[i]};  '
          f'// reusing narrowed handle'
        )
      )
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

  inner_decompose_stmts.append(BlankLine())

  for i, src in enumerate(cart_op.sources):
    assert isinstance(src, mir.ColumnSource)
    if i >= len(cart_op.var_from_source):
      continue
    prefix_len = len(src.prefix_vars)
    for v_idx, var_name in enumerate(cart_op.var_from_source[i]):
      if ctx.is_counting and not _cart_var_used(var_name, rest, insert):
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
  insert: mir.InsertInto,
) -> bool:
  '''Counting-phase optimization gate for Cartesian var-binds.'''
  if var_name in insert.vars:
    return True
  for op in rest:
    if isinstance(op, mir.Filter) and var_name in op.vars:
      return True
    if isinstance(op, mir.ConstantBind) and var_name in op.code:
      return True
  return False


def _lower_nested_cj_multi(
  cj_op: mir.ColumnJoin,
  rest: list[mir.MirNode],
  insert: mir.InsertInto,
  ctx: LoweringCtx,
) -> Op:
  '''Lower a nested multi-source ColumnJoin.

  Mirrors `_nested_column_join_multi` in `codegen/jit/instructions.py`.
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
  body_op = _lower_inner_chain(rest, insert, ctx)

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
        f'_lower_nested_cj_multi: no view var for source handle_idx '
        f'{src.handle_start}'
      )
    source_view_names.append(src_view)

    alias_var = ctx.fresh(f'h_{src.rel_name}_{src.handle_start}')
    source_alias_names.append(alias_var)

    if src.prefix_vars:
      # Aliased from a parent handle in the enclosing scope.
      parent_state_key = _state_key(
        src.rel_name, list(src.index), src.prefix_vars, src.version
      )
      parent_handle = ctx.handle_vars.get(parent_state_key, '')
      if not parent_handle:
        raise ValueError(
          f'_lower_nested_cj_multi: no parent handle for state key '
          f'{parent_state_key!r}'
        )
      alias_bind_stmts.append(
        Bind(name=alias_var, expr=VarRef(name=parent_handle))
      )
    else:
      # Fresh source: brand-new root handle, no narrowing.
      alias_bind_stmts.append(
        Bind(name=alias_var, expr=SaRoot(view_name=src_view))
      )

  intersect_var = ctx.fresh('intersect')
  iter_var = ctx.fresh('it')

  iterator_exprs = tuple(
    SaIterators(handle_name=hn, view_name=vn)
    for hn, vn in zip(source_alias_names, source_view_names)
  )

  # child_range bindings live INSIDE the for-loop body, at +1 indent.
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

  loop_body = Block(
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
      f'({s.rel_name} :handle {s.handle_start} '
      f':prefix ({" ".join(s.prefix_vars)}))'
      for s in cj_op.sources
    )
    stmts.append(
      Comment(text=f'MIR: (column-join :var {cj_op.var_name} :sources ({src_debug} ))')
    )

  stmts.extend(alias_bind_stmts)
  stmts.append(
    IntersectIter(
      intersect_var=intersect_var,
      iter_var=iter_var,
      iterator_exprs=iterator_exprs,
      value_var=inner_var_sanitized,
      body=loop_body,
    )
  )
  return Block(stmts=tuple(stmts))


def _lower_insert_into(node: mir.InsertInto, ctx: LoweringCtx) -> list[Op]:
  '''Lower an InsertInto under the M1-M3 narrow-flag assumptions.'''
  out_var = ctx.output_var_overrides.get(node.rel_name, ctx.output_var)
  vars_list = list(node.vars)

  stmts: list[Op] = []
  if ctx.debug:
    stmts.append(Comment(text=f'Emit: {node.rel_name}({", ".join(vars_list)})'))

  if ctx.is_counting:
    body: Op = RawString(text=f'{out_var}.emit_direct();')
  else:
    sanitized = ', '.join(_sanitize_var_name(v) for v in vars_list)
    body = RawString(text=f'{out_var}.emit_direct({sanitized});')

  if not ctx.inside_cartesian:
    stmts.append(LaneZeroGuard(body=body))
  else:
    stmts.append(body)

  return stmts


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _filter_expr(code: str) -> str:
  expr = code.strip()
  if expr.startswith('return '):
    expr = expr[len('return '):]
  if expr.endswith(';'):
    expr = expr[:-1]
  return expr


def _scan_var_used(
  var_name: str,
  middle: list[mir.MirNode],
  insert: mir.InsertInto,
) -> bool:
  '''Counting-phase optimization gate.'''
  if var_name in insert.vars:
    return True
  for op in middle:
    if isinstance(op, mir.Filter) and var_name in op.vars:
      return True
    if isinstance(op, mir.ConstantBind) and var_name in op.code:
      return True
    if isinstance(op, mir.ColumnJoin):
      for src in op.sources:
        if var_name in src.prefix_vars:
          return True
      if var_name == op.var_name:
        return True
  return False


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
    'class', 'struct', 'union', 'enum', 'typedef', 'template',
    'using', 'namespace', 'public', 'private', 'protected',
    'const', 'volatile', 'mutable', 'static', 'extern', 'inline',
    'virtual', 'override', 'final', 'explicit', 'friend',
    'new', 'delete', 'this', 'typeid', 'sizeof', 'alignof',
    'true', 'false', 'nullptr', 'auto', 'register', 'thread_local',
    'if', 'else', 'switch', 'case', 'default', 'while', 'do', 'for',
    'break', 'continue', 'return', 'goto', 'try', 'catch', 'throw',
  }
  if name in cpp_keywords:
    return f'{name}_val'
  return name
