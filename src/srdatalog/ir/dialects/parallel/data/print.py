'''Print_i for the parallel.data dialect.

One Op (`BgRootCjMulti`) plus a helper dataclass (`BgSourceSpec`).
The helper is not an `Op` subclass, so it has its own internal
print_source_spec() that BgRootCjMulti's printer calls directly.
'''

from __future__ import annotations

from srdatalog.ir.core import Op
from srdatalog.ir.dialects.parallel.data.block_group import BgRootCjMulti, BgSourceSpec
from srdatalog.ir.print_iir import _bool, _ind, print_iir

OPS: tuple[type, ...] = (BgRootCjMulti,)


def _print_source_spec(spec: BgSourceSpec, indent: int) -> str:
  p = _ind(indent)
  return (
    p
    + f'(bg-source-spec #:rel-name {spec.rel_name} '
    + f'#:view-var {spec.view_var} '
    + f'#:handle-var {spec.handle_var} '
    + f'#:view-count {spec.view_count} '
    + f'#:base-slot {spec.base_slot} '
    + f'#:index-type {spec.index_type})'
  )


def print_op(op: Op, indent: int = 0) -> str:
  p = _ind(indent)

  if isinstance(op, BgRootCjMulti):
    sources = '\n'.join(_print_source_spec(s, indent + 2) for s in op.sources)
    body = print_iir(op.body, indent + 1)
    return (
      p
      + f'(bg-root-cj-multi #:var-name {op.var_name} '
      + f'#:is-counting {_bool(op.is_counting)} '
      + f'#:key-idx-var {op.key_idx_var} '
      + f'#:root-val-var {op.root_val_var} '
      + f'#:hint-lo {op.hint_lo} #:hint-hi {op.hint_hi}\n'
      + p
      + '  #:sources (\n'
      + sources
      + ')\n'
      + p
      + '  #:body\n'
      + body
      + ')'
    )

  raise TypeError(f'parallel.data print_op: unknown op {type(op).__name__}')


__all__ = ['OPS', 'print_op']
