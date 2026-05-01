'''MIR -> IIR lowering for the sorted_array dialect.

Each milestone (M1, M2, …) extends `lower_pipeline` to handle more
MIR op kinds. The supported predicate `_supported_pipeline` documents
which shapes the dialect can faithfully reproduce against the legacy
emitter.

  M1: [Scan, InsertInto]
  M2: [Scan, (Filter | ConstantBind)*, InsertInto]

The lowering threads a `LoweringCtx` whose `name_counter` mirrors the
legacy `gen_unique_name` bump order, so the names baked into IIR
match what `jit_pipeline()` would have allocated. The target.cuda
emit then renders the IIR verbatim.
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
  IfReturnIfNot,
  IndentBlock,
  LaneZeroGuard,
  ParallelFor,
  RawString,
  VarRef,
)
from srdatalog.dialects.relation.sorted_array.ops import (
  SaDegree,
  SaGetVal,
  SaRoot,
  SaValid,
)
from srdatalog.ir_core import Op


@dataclass
class LoweringCtx:
  '''Mutable state during MIR -> IIR walk.

  Mirrors the legacy `CodeGenContext` for the fields that matter to
  the dialect's emission decisions. Other legacy fields (handle_vars
  string dicts, tiled_cartesian state, ws state, etc.) aren't needed
  yet — milestones add them as they cover those paths.
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

  def fresh(self, prefix: str) -> str:
    self.name_counter += 1
    return f'{prefix}_{self.name_counter}'


# -----------------------------------------------------------------------------
# Public entry point + supported-shape predicate
# -----------------------------------------------------------------------------


def _supported_pipeline(ops: list[mir.MirNode]) -> bool:
  '''True iff the dialect can lower this pipeline shape today.

  M1: exactly [Scan, InsertInto].
  M2: [Scan, (Filter | ConstantBind)*, InsertInto].
  '''
  if len(ops) < 2:
    return False
  if not isinstance(ops[0], mir.Scan):
    return False
  if not isinstance(ops[-1], mir.InsertInto):
    return False
  for op in ops[1:-1]:
    if not isinstance(op, (mir.Filter, mir.ConstantBind)):
      return False
  return True


def lower_scan_pipeline(
  ops: list[mir.MirNode],
  ctx: LoweringCtx,
) -> Op:
  '''Lower [Scan, (Filter|ConstantBind)*, InsertInto] -> IIR Block.

  Raises ValueError if the pipeline shape isn't supported.

  The IIR shape mirrors §10.1, §10.6, §10.7, §10.10 of the spec:

    Block([
      <debug comments>,
      Bind(handle_var = SaRoot(view)),
      IfReturnIfNot(SaValid(handle_var)),
      Bind(degree_var = SaDegree(handle_var)),
      ParallelFor("warp_strided", body=GridStrideLoop(idx_var, degree_var,
        body=Block([
          IndentBlock(extra=1, stmts=(   # var-binds at +1 indent
            Bind(x = SaGetVal(view, 0, idx_var)),
            ...
          )),
          # Filter / ConstantBind / InsertInto chain — at outer indent,
          # mirroring the legacy emitter's "body rendered before
          # inc_indent" quirk.
          <inner stmts>
        ])
      )),
    ])
  '''
  if not _supported_pipeline(ops):
    raise ValueError(
      f'lower_scan_pipeline: supports only [Scan, '
      f'(Filter|ConstantBind)*, InsertInto]; '
      f'got {[type(o).__name__ for o in ops]}'
    )

  scan_op = ops[0]
  middle = list(ops[1:-1])
  insert_op = ops[-1]
  assert isinstance(scan_op, mir.Scan)
  assert isinstance(insert_op, mir.InsertInto)

  handle_idx = scan_op.handle_start
  view_var = ctx.view_var_names.get(str(handle_idx), '')
  if not view_var:
    raise ValueError(
      f'lower_scan_pipeline: no view var registered for handle_idx '
      f'{handle_idx}; view_management should run first'
    )

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
  outer_stmts.append(Bind(name=handle_var, expr=SaRoot(view_name=view_var)))
  outer_stmts.append(IfReturnIfNot(cond=SaValid(handle_name=handle_var)))

  degree_var = ctx.fresh('degree')
  outer_stmts.append(
    Bind(name=degree_var, expr=SaDegree(handle_name=handle_var), type_decl='uint32_t')
  )
  outer_stmts.append(BlankLine())

  idx_var = ctx.fresh('idx')

  # Var-bind statements at +1 indent (legacy uses inc_indent before
  # emitting them). Wrapped in IndentBlock so the body chain below
  # stays at the loop's outer indent.
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
# Inner-chain lowering: Filter / ConstantBind / InsertInto
# -----------------------------------------------------------------------------


def _lower_inner_chain(
  middle: list[mir.MirNode],
  insert: mir.InsertInto,
  ctx: LoweringCtx,
) -> Op:
  '''Lower the Filter/ConstantBind chain ending in an InsertInto into
  a single IIR op. Each Filter wraps the rest in `If(cond, body)`;
  each ConstantBind prefixes the rest with `Bind(name, expr)`. The
  recursion preserves order.
  '''
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
    # ConstantBind precedes the rest as a sequence — wrap in Block.
    if isinstance(rest_op, Block):
      return Block(stmts=(bind_stmt, *rest_op.stmts))
    return Block(stmts=(bind_stmt, rest_op))

  raise ValueError(f'unsupported inner op: {type(head).__name__}')


def _lower_insert_into(node: mir.InsertInto, ctx: LoweringCtx) -> list[Op]:
  '''Lower a single InsertInto under the narrow no-flag assumption:
    - not inside_cartesian (so lane-0 guard applies)
    - no dedup_hash, no tiled_cartesian, no ws
    - is_counting toggles between emit_direct() and emit_direct(vars)
  '''
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


def _filter_expr(code: str) -> str:
  '''Strip Nim's `return <expr>;` envelope from a Filter's code.

  Filter.code from HIR is `"return <bool_expr>;"` (the function body
  of an inline filter); the legacy emitter pulls out the `<bool_expr>`
  for `if (<bool_expr>) { ... }`. Mirror that here.
  '''
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
  '''Counting-phase optimization gate: is `var_name` referenced
  anywhere downstream of the Scan?

  Refs come from: Filter's bound vars, ConstantBind's deps, and the
  InsertInto's var list. M2 conservatively checks all of these.
  '''
  if var_name in insert.vars:
    return True
  for op in middle:
    if isinstance(op, mir.Filter) and var_name in op.vars:
      return True
    if isinstance(op, mir.ConstantBind):
      # No `deps` field on the dataclass; rely on textual search of
      # `code` as a conservative approximation. Not perfect but matches
      # the legacy emitter's "if name appears in the rendered string"
      # heuristic.
      if var_name in op.code:
        return True
  return False


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
