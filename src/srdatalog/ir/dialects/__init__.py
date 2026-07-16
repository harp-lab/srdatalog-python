'''Registered dialects, organized by category.

Folder layout mirrors the dotted dialect-name convention used in
docs/ir_lowering_semantics.md. Each category subdirectory groups
sibling dialects of the same kind:

  relation/       — data-structure dialects (sorted_array, lsm, uf, …)
  iir/            — index-aware IR cross-cutting (cf, future: mem)
  target/         — codegen backends (cuda, future: hip, cpp_tbb, …)
  par/            — parallelism strategies (future: data.warp_strided,
                    data.tbb_for, task, simd, scalar)

Adding a new dialect: pick the right category, drop a subpackage
in. No edits to other categories or to ir_core needed (Property P1
in design_principles.md).
'''
