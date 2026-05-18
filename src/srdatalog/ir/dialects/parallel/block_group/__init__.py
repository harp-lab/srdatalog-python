'''parallel.block_group dialect — typed-pragma materialization for
block-group work-balanced parallelism.

Per `docs/phase_c_pragma_materialization.md` §4.1, the `block_group`
pragma owns its own sub-dialect (this package) because:

  - The materialized form has multiple ops downstream
    (`BgRootCjMulti`, `BgSourceSpec` — already defined in the sibling
    `parallel.data.block_group` module that owns the legacy strategy
    op vocabulary).
  - Codegen scaffolding is distinct (per-warp work-balance setup
    outside the kernel body — visible at runner-level emit).
  - External re-use is plausible (a future runner library could
    consume these ops for scheduling decisions).

The pragma module (`pragmas/block_group.py`) is co-located with the
dialect so importing the dialect runs the `@pragma_handler` +
`@lowering` registrations as side effects, matching the C2 sorted_array
template (`relation.sorted_array/pragmas/dedup_hash.py`).

What this dialect contributes today:
  - `BlockGroup` Pragma subclass (in `pragmas/block_group.py`).
  - `@pragma_handler(BlockGroup, on=mir.ExecutePipeline)` that wraps
    the root multi-source `ColumnJoin` in `mir.BlockGroupRoot`.
  - `@lowering(target=iir.cf, source=mir.BlockGroupRoot)` that
    delegates back to the sorted_array dialect's `_lower_root_cj_bg`
    helper with `ctx.bg_enabled` forced True for the call — same
    IIR as the legacy `if ctx.bg_enabled:` branch by construction.

The actual BG strategy ops (`BgRootCjMulti`, `BgSourceSpec`) stay
in `parallel.data.block_group` (the existing strategy-vocabulary
package) — this sub-dialect is the pragma materialization surface,
not a re-home of the strategy ops. Future refactors may unify them
once the legacy bool field on `mir.ExecutePipeline` goes away (A3).

Spec: `docs/phase_c_pragma_materialization.md` §4.1, §5 (C3 PR row),
§5.1 (per-PR acceptance gate).
'''

from __future__ import annotations

from srdatalog.ir.core import Dialect

DIALECT = Dialect(
  name='parallel.block_group',
  ops=[],
)


def _register_passes() -> None:
  '''Wire the BlockGroup pragma handler + BlockGroupRoot lowering.

  Importing `pragmas.block_group` runs the `@pragma_handler(
  BlockGroup, on=ExecutePipeline)` registration as a side effect; we
  then register the `@lowering(target=iir.cf, source=mir.
  BlockGroupRoot)` rule on this dialect so PassDriver discovery
  matches the C2 sorted_array template (`relation/sorted_array/
  __init__.py:_register_passes`).
  '''
  from typing import Any

  import srdatalog.ir.mir.types as mir
  from srdatalog.ir.core.passes import lowering, verifier
  from srdatalog.ir.dialects.parallel.block_group.pragmas.block_group import (
    lower_block_group_root,
  )

  @lowering(
    DIALECT,
    mir.BlockGroupRoot,
    consumes=('mir',),
    produces=('iir.cf', 'relation.sorted_array', 'parallel.data'),
  )
  def lower_mir_block_group_root(op: Any, ctx: Any) -> Any:
    return lower_block_group_root(op, ctx)

  # Verifier scaffolding — BG-specific invariants (the wrapped inner
  # op MUST be a multi-source ColumnJoin, since
  # `jit_root_column_join_block_group` only fires there) land
  # incrementally as we encode them.
  @verifier(DIALECT)
  def _verify(_prog):
    return []


_register_passes()


__all__ = ['DIALECT']
