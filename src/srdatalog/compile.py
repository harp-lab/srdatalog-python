'''Public CUDA compile entry point. Thin re-export shim.

The real implementation lives at
[`srdatalog.ir.dialects.target.cuda.api`][1]. This module exists at
the top level so that `from srdatalog.compile import …` keeps working
for downstream users while the implementation sits with the CUDA
target where it belongs.

[1]: ./ir/dialects/target/cuda/api.py
'''

from srdatalog.ir.dialects.target.cuda.api import (
  Target,
  compile_kernel_body,
  compile_pipeline,
  compile_runner,
)

__all__ = ['Target', 'compile_kernel_body', 'compile_pipeline', 'compile_runner']
