'''CUDA renderer registry.

Per docs/stage3a_execution_plan.md §7 task S3A.3, this package replaces
codegen/cuda/emit.py's hardcoded 41-case match with a per-op-class
registry that each dialect's render module opts into. Adding a new IIR
dialect now requires adding a `codegen/cuda/render/<dialect>.py` with
`@register_render` decorators — no edits to this file or to any
existing per-dialect render module. P1 fix.

Public API (preserved for external callers — was previously in
codegen/cuda/emit.py):

    EmitCtx          mutable per-emission state (indent, tile var, ...)
    emit(op, ctx)    statement-shaped op → C++ source line(s)
    emit_expr(op, ctx) expression-shaped op → C++ source fragment

Each IIR-contributing dialect imports from this module:

    from srdatalog.ir.codegen.cuda.render import (
        EmitCtx, register_render, emit, emit_expr,
    )

    @register_render(Block, mode='stmt')
    def _render_block(op: Block, ctx: EmitCtx) -> str:
        return ''.join(emit(s, ctx) for s in op.stmts)

The `mode` argument distinguishes statement vs expression handlers.
A single op class CAN appear in both registries (e.g. `RawString`
is renderable in both modes with different output forms).

Dispatch (`emit` / `emit_expr`) raises `KeyError` with a useful
message on an unregistered op — that's the "did this codegen forget
to register a handler" signal. Per S3A.8 (per-Codegen registry +
verify_completeness), startup will eventually catch this earlier.
'''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from srdatalog.ir.core import Op

RenderFn = Callable[[Op, 'EmitCtx'], str]


@dataclass
class EmitCtx:
  '''Mutable emission state during the C++ walk.

  - `indent_level`: counts of 2-space units. Legacy emitter starts
    at indent=2 (operator() body); M1 callers pass that in to match.
  - `tile_var`: name of the tile/thread-group variable. "tile" by default.
  - `segment_depth`: how many `D2lSegmentLoop`s wrap the current
    point. Lets `IntersectIter` reproduce the legacy
    `_nested_column_join_multi` indent quirk where segment loops
    bump the structural indent (alias_binds, intersect at +segs)
    but the for-iter body lines (auto value, positions, child_binds,
    body_op) anchor against the *outer* indent (+1, +1, +1, +0
    respectively).
  '''

  indent_level: int = 2
  tile_var: str = 'tile'
  segment_depth: int = 0

  def ind(self) -> str:
    return '  ' * self.indent_level


# Module-level registries. S3A.8 will move these onto a per-`Codegen`
# instance with `verify_completeness()` at `Compiler.register_codegen`
# construction time. For now (S3A.3), module-globals keep the diff
# focused on the registry-vs-match split.
_STMT_HANDLERS: dict[type[Op], RenderFn] = {}
_EXPR_HANDLERS: dict[type[Op], RenderFn] = {}


def register_render(op_class: type[Op], *, mode: str = 'stmt') -> Callable[[RenderFn], RenderFn]:
  '''Decorator: register `fn` as the renderer for `op_class` in the
  given mode ('stmt' or 'expr').'''
  if mode not in ('stmt', 'expr'):
    raise ValueError(f"register_render mode must be 'stmt' or 'expr', got {mode!r}")
  registry = _STMT_HANDLERS if mode == 'stmt' else _EXPR_HANDLERS

  def _wrap(fn: RenderFn) -> RenderFn:
    if op_class in registry:
      raise ValueError(f"register_render: {op_class.__name__} already registered in mode={mode!r}")
    registry[op_class] = fn
    return fn

  return _wrap


def emit(op: Op, ctx: EmitCtx) -> str:
  '''Statement-mode dispatch. Returns C++ source text ending in `\\n`.'''
  handler = _STMT_HANDLERS.get(type(op))
  if handler is None:
    raise KeyError(
      f'codegen.cuda: no statement-mode renderer registered for '
      f'{type(op).__name__} (module={type(op).__module__!r}). '
      'Did the dialect ship its codegen/cuda/render/<dialect>.py?'
    )
  return handler(op, ctx)


def emit_expr(op: Op, ctx: EmitCtx) -> str:
  '''Expression-mode dispatch. Returns C++ source fragment with no
  leading indent or trailing newline.'''
  handler = _EXPR_HANDLERS.get(type(op))
  if handler is None:
    raise KeyError(
      f'codegen.cuda: no expression-mode renderer registered for '
      f'{type(op).__name__} (module={type(op).__module__!r}). '
      'Did the dialect ship its codegen/cuda/render/<dialect>.py?'
    )
  return handler(op, ctx)


def _eager_register_all() -> None:
  '''Force-import each dialect's render module so its
  @register_render decorators run. Called at first use of `emit` /
  `emit_expr` to avoid a load-time cycle (the dialect render modules
  import EmitCtx + register_render from this module).

  Idempotent — register_render itself rejects duplicate registration.
  '''
  global _registered_all
  if _registered_all:
    return
  _registered_all = True
  # Import for side-effect (the @register_render decorators).
  from srdatalog.ir.codegen.cuda.render import (
    d2l,
    iir_cf,
    iir_expr,
    parallel_data,
    sorted_array,
  )


_registered_all = False


# Ensure handlers are registered the first time someone imports from
# this package. This preserves the property "import codegen.cuda.render
# and emit(op, ctx) just works" — no need for callers to know about
# the per-dialect registration step.
_eager_register_all()


__all__ = ['EmitCtx', 'RenderFn', 'register_render', 'emit', 'emit_expr']
