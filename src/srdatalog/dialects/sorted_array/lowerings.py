'''MIR -> IIR lowering for the sorted_array dialect (M1 subset).

Covers the [Scan, InsertInto] pipeline shape with no feature flags
active (no inside_cartesian, no dedup_hash, no tiled_cartesian, no
ws, no bg, no multi-view, no scalar_mode). This is the smallest
fixture shape the audit identified.

For M1 byte-equivalence, the lowering threads a small `LoweringCtx`
that allocates names matching the legacy emitter's `gen_unique_name`
bump order. Emission is then a verbatim walk of the IIR — the names
in the C++ output line up with what `jit_pipeline()` would have
produced.

Later milestones (M2+) extend this to additional MIR ops; the
emission convention stays the same.
'''

from __future__ import annotations

from dataclasses import dataclass, field

import srdatalog.mir.types as mir
from srdatalog.dialects.iir_cf import (
  Bind,
  BlankLine,
  Block,
  Comment,
  GridStrideLoop,
  IfReturnIfNot,
  IndentBlock,
  LaneZeroGuard,
  ParallelFor,
  RawString,
  VarRef,
)
from srdatalog.dialects.sorted_array.ops import (
  SaDegree,
  SaGetVal,
  SaRoot,
  SaValid,
)
from srdatalog.ir_core import Op


@dataclass
class LoweringCtx:
  '''Mutable state during MIR -> IIR walk.

  - `name_counter`: starts at 0; `fresh(prefix)` increments first then
    returns `<prefix>_<counter>`. Mirrors `ctx.gen_unique_name` in the
    legacy emitter.
  - `view_var_names`: map from handle_idx string to the view variable
    that view_management.py would have emitted. We don't re-emit view
    declarations (those happen outside the inner pipeline), but we
    do need to know the names to plug into SaRoot, SaGetVal.
  - `is_counting`: matches legacy `ctx.is_counting`.
  - `inside_cartesian`: matches legacy `ctx.inside_cartesian`. M1
    pipelines never set this true.
  - `output_var`: matches legacy `ctx.output_var_name` lookup.
  - `tile_var`: name of the tile/thread-group variable. Default
    "tile" matches legacy.
  - `debug`: emit `// Root Scan: ...` and `// MIR: ...` comments.
  '''

  name_counter: int = 0
  view_var_names: dict[str, str] = field(default_factory=dict)
  is_counting: bool = False
  inside_cartesian: bool = False
  output_var: str = 'output'
  tile_var: str = 'tile'
  debug: bool = True
  output_var_overrides: dict[str, str] = field(default_factory=dict)

  def fresh(self, prefix: str) -> str:
    self.name_counter += 1
    return f'{prefix}_{self.name_counter}'


def _supported_pipeline(ops: list[mir.MirNode]) -> bool:
  '''M1 supports exactly the [Scan, InsertInto] shape.'''
  if len(ops) != 2:
    return False
  return isinstance(ops[0], mir.Scan) and isinstance(ops[1], mir.InsertInto)


def lower_scan_pipeline(
  ops: list[mir.MirNode],
  ctx: LoweringCtx,
) -> Op:
  '''Lower [Scan, InsertInto] -> IIR Block.

  Raises ValueError if the pipeline shape isn't M1-supported.

  The IIR shape mirrors §10.1 + §10.10 of the spec:

    Block([
      <debug comments>,
      Bind(handle_var = SaRoot(view)),
      IfReturnIfNot(SaValid(handle_var)),
      Bind(degree_var = SaDegree(handle_var)),
      ParallelFor("warp_strided", body=GridStrideLoop(idx_var, degree_var,
        body=Block([
          Bind(x = SaGetVal(view, 0, idx_var)),  -- per var
          Bind(y = SaGetVal(view, 1, idx_var)),
          <body: WriteOutput wrapped in LaneZeroGuard>,
        ])
      )),
    ])

  The ParallelFor wraps GridStrideLoop because the strategy carries
  the `warp_strided` policy as a separate concern; the loop itself
  is the inner kernel structure.

  Note: the `if (!valid) return;` and degree fetch happen *outside*
  ParallelFor in the legacy emitter (they come before the loop). For
  byte-equivalence we model that by sequencing them as Bind/IfReturn
  statements before the ParallelFor in the outer Block.
  '''
  if not _supported_pipeline(ops):
    raise ValueError(
      f'lower_scan_pipeline: M1 supports only [Scan, InsertInto]; '
      f'got {[type(o).__name__ for o in ops]}'
    )

  scan_op = ops[0]
  insert_op = ops[1]
  assert isinstance(scan_op, mir.Scan)
  assert isinstance(insert_op, mir.InsertInto)

  handle_idx = scan_op.handle_start
  view_var = ctx.view_var_names.get(str(handle_idx), '')
  if not view_var:
    raise ValueError(
      f'lower_scan_pipeline: no view var registered for handle_idx '
      f'{handle_idx}; view_management should run first'
    )

  stmts: list[Op] = []

  if ctx.debug:
    stmts.append(
      Comment(text=f'Root Scan: {scan_op.rel_name} binding {", ".join(scan_op.vars)}')
    )
    stmts.append(
      Comment(
        text=f'MIR: (scan :rel {scan_op.rel_name} '
        f':vars ({" ".join(scan_op.vars)}) :handle {handle_idx})'
      )
    )

  handle_var = ctx.fresh('root_handle')
  stmts.append(Bind(name=handle_var, expr=SaRoot(view_name=view_var)))
  stmts.append(IfReturnIfNot(cond=SaValid(handle_name=handle_var)))

  degree_var = ctx.fresh('degree')
  stmts.append(
    Bind(name=degree_var, expr=SaDegree(handle_name=handle_var), type_decl='uint32_t')
  )
  stmts.append(BlankLine())

  idx_var = ctx.fresh('idx')

  # Var-bind statements at +1 indent (legacy uses inc_indent before
  # emitting them). Wrapped in IndentBlock so the InsertInto body
  # below stays at the loop's outer indent — matches the legacy
  # emitter's mixed-indent quirk where the body was rendered before
  # `inc_indent`.
  var_bind_stmts: list[Op] = []
  for col, var_name in enumerate(scan_op.vars):
    if ctx.is_counting and not _var_used_in_body(var_name, insert_op):
      # Counting-phase optimization: skip unused-var fetches.
      continue
    var_bind_stmts.append(
      Bind(
        name=_sanitize_var_name(var_name),
        expr=SaGetVal(view_name=view_var, col=col, idx_var_name=idx_var),
      )
    )

  loop_body_stmts: list[Op] = []
  if var_bind_stmts:
    loop_body_stmts.append(IndentBlock(extra=1, stmts=tuple(var_bind_stmts)))
  loop_body_stmts.extend(_lower_insert_into(insert_op, ctx))

  loop = GridStrideLoop(
    idx_name=idx_var,
    bound=VarRef(name=degree_var),
    body=Block(stmts=tuple(loop_body_stmts)),
  )

  stmts.append(ParallelFor(strategy='warp_strided', body=loop))

  return Block(stmts=tuple(stmts))


def _lower_insert_into(node: mir.InsertInto, ctx: LoweringCtx) -> list[Op]:
  '''Lower a single InsertInto under M1's narrow assumptions:
    - not inside_cartesian (so lane-0 guard applies)
    - no dedup_hash, no tiled_cartesian, no ws — baseline path
    - is_counting toggles between emit_direct() and emit_direct(vars)
  '''
  out_var = ctx.output_var_overrides.get(node.rel_name, ctx.output_var)
  vars_list = list(node.vars)

  stmts: list[Op] = []
  if ctx.debug:
    stmts.append(Comment(text=f'Emit: {node.rel_name}({", ".join(vars_list)})'))

  if ctx.is_counting:
    # Counting: emit_direct() with no args. The legacy form is
    # `if (tile.thread_rank() == 0) <out_var>.emit_direct();` — render
    # via RawString for now (M1 covers materialize as the primary path).
    body = RawString(text=f'{out_var}.emit_direct();')
  else:
    sanitized = ', '.join(_sanitize_var_name(v) for v in vars_list)
    body = RawString(text=f'{out_var}.emit_direct({sanitized});')

  if not ctx.inside_cartesian:
    stmts.append(LaneZeroGuard(body=body))
  else:
    stmts.append(body)

  return stmts


def _var_used_in_body(var_name: str, body_op: mir.InsertInto) -> bool:
  '''Counting-phase optimization gate: is `var_name` referenced by the
  insert? For the M1 [Scan, InsertInto] shape the body is exactly one
  InsertInto, so this reduces to membership in its var list.'''
  return var_name in body_op.vars


def _sanitize_var_name(name: str) -> str:
  '''Mirror the legacy `sanitize_var_name`. C++ keywords get a `_val`
  suffix; everything else passes through. The full keyword list is
  in codegen/jit/context.py; M1 inlines a minimal subset (the ones
  that actually appear in fixtures). Falling back to the legacy
  helper for the full set is fine — kept simple here.
  '''
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
