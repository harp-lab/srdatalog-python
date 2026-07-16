'''D2L dialect ↔ CUDA target plugin.

How the D2L (Device2LevelIndex) dialect emits as CUDA: registers an
`IndexPlugin` with `codegen.cuda.plugin`'s registry that supplies the
per-version view_count (FULL=2 for HEAD+FULL, DELTA/NEW=1) and the
host-side view-setup code emitted around CUDA kernels. Importing
this module — typically via `import srdatalog.ir.dialects.relation.d2l`
— is enough to register the plugin (the D2L package's `__init__`
re-imports this side-effectfully).

D2L design recap:
  - HEAD + FULL sorted arrays per relation.
  - Merge: DELTA → HEAD, O(D+H) instead of O(D+F).
  - Compaction: HEAD → FULL only when HEAD exceeds ratio% of FULL,
    or at fixpoint exit.
  - Join (DELTA read): 1 sorted array (same as DSAI).
  - Join (FULL read): 2 sorted arrays — kernel iterates both via the
    `D2lSegmentLoop` IIR op.

Usage in schema:
  Path {. index: "SRDatalog::GPU::Device2LevelIndex" .}: Relation[int, int]
'''

from srdatalog.ir.codegen.cuda.plugin import (
  IndexPlugin,
  new_default_plugin,
  register_index_plugin,
)

# -----------------------------------------------------------------------------
# 2-Level view-level hook overrides
# -----------------------------------------------------------------------------


def _two_level_view_count(version: str) -> int:
  '''DELTA / NEW: 1 sorted array. FULL: 2 sorted arrays (HEAD + FULL).'''
  return 2 if version == "FULL_VER" else 1


def _two_level_gen_host_view_setup(idx_expr: str, version: str) -> list[str]:
  '''For FULL reads, push both HEAD and FULL segment views onto the
  kernel's views[] buffer. For DELTA / NEW, single view — same as DSAI.'''
  if version == "FULL_VER":
    return [f"{idx_expr}.full_view()", f"{idx_expr}.head_view()"]
  return [f"{idx_expr}.view()"]


def new_two_level_plugin() -> IndexPlugin:
  '''Build a Device2LevelIndex plugin by starting from the default DSAI
  plugin and overriding the view-level hooks. Expression-level hooks
  are unchanged — once the kernel has a view, prefix/degree/valid/get
  all work identically.
  '''
  plugin = new_default_plugin()
  plugin.name = "Device2LevelIndex"
  plugin.cpp_type = "SRDatalog::GPU::Device2LevelIndex"
  plugin.cpp_headers = ["gpu/device_2level_index.h"]
  plugin.view_count = _two_level_view_count
  plugin.gen_host_view_setup = _two_level_gen_host_view_setup
  return plugin


# Module-level instance + registration (compile-time in Nim; import-time here).
two_level_plugin = new_two_level_plugin()
register_index_plugin(two_level_plugin)
