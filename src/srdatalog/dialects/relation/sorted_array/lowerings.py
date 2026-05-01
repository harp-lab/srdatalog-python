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
    # M1+M2 shapes
    return all(
      isinstance(op, (mir.Filter, mir.ConstantBind)) for op in middle
    )

  if isinstance(head, mir.ColumnJoin) and len(head.sources) >= 2:
    # M3 shape: multi-source root CJ; middle can hold more
    # multi-source CJs, Filter, or ConstantBind.
    for op in middle:
      if isinstance(op, (mir.Filter, mir.ConstantBind)):
        continue
      if isinstance(op, mir.ColumnJoin) and len(op.sources) >= 2:
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
    ctx.bound_vars.append(var_name)

  inner_stmts: list[Op] = []
  if var_bind_stmts:
    inner_stmts.append(IndentBlock(extra=1, stmts=tuple(var_bind_stmts)))

  inner_stmts.append(_lower_inner_chain(middle, insert_op, ctx))

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

  raise ValueError(f'unsupported inner op: {type(head).__name__}')


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
