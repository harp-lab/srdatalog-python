'''Data-parallelism strategy dialects.

Each strategy decides how `ParallelFor` distributes work across
threads/warps/blocks. A pipeline picks one at lowering time
based on its workload characteristics (uniform vs skewed,
fixed-shape vs dynamic).

Currently registered ops:
  - BgRootCjMulti — block-group root multi-source ColumnJoin
'''

from __future__ import annotations

from srdatalog.ir.core import Dialect
from srdatalog.ir.dialects.parallel.data.block_group import BgRootCjMulti

DIALECT = Dialect(
  name='parallel.data',
  ops=[BgRootCjMulti],
)


# Verifier scaffolding — block-group / parallel-strategy invariants
# land incrementally as we encode them.
def _register_passes() -> None:
  from srdatalog.ir.core.passes import verifier

  @verifier(DIALECT)
  def _verify(_prog):
    return []


_register_passes()


__all__ = ['DIALECT', 'BgRootCjMulti']
