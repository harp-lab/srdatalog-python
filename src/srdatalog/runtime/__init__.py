'''Bundled C++ runtime headers.

The Python-emitted JIT code includes `srdatalog.h` + friends from the
`generalized_datalog/` subtree shipped here as package data. Callers
point the compiler at this directory via:

    from srdatalog.runtime import runtime_include_path
    from srdatalog import CompilerConfig

    cfg = CompilerConfig(
      include_paths=[runtime_include_path()],
    )

Works identically whether `srdatalog` is installed via `uv pip install`
or imported from a source checkout.
'''
from __future__ import annotations

from pathlib import Path


def runtime_include_path() -> str:
  '''Absolute path to the bundled `generalized_datalog/` headers.

  Use in `CompilerConfig.include_paths` so the compiler finds
  `srdatalog.h`, `gpu/runtime/query.h`, etc.
  '''
  here = Path(__file__).resolve().parent
  return str(here / "generalized_datalog")
