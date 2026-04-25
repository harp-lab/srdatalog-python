'''High-level project builder.

`build_project(program, project_name, ...)` runs the full pipeline in
one call:

  Program → HIR → MIR
         → emit per-rule complete runner structs
         → compute schema definitions + DB blueprint alias
         → write the .cpp tree to ~/.cache/srdatalog/jit/<project>_<hash>/

Returns the same dict as `cache.write_jit_project`.

The resulting `jit_batch_N.cpp` files are byte-identical to what Nim's
codegen writes to its own JIT cache — they self-contain the schema +
DB type alias, so the same compile flags / external deps that work
for Nim's output work here too.

This module is a thin wrapper over `srdatalog.pipeline.compile_program`
plus the file-emitting layer (`cache.write_jit_project` + optional
shard/main file emission). The compile phase lives in `pipeline.py`
so viz / other consumers can share it without touching disk.
'''

from __future__ import annotations

from typing import TYPE_CHECKING

from srdatalog.codegen.jit.cache import write_jit_project
from srdatalog.codegen.jit.main_file import (
  gen_extern_c_shim,
  gen_main_file_content,
  gen_run_dispatcher_file,
  gen_step_shard_file,
  gen_unity_main_file_content,
)
from srdatalog.pipeline import compile_program

if TYPE_CHECKING:
  from srdatalog.dsl import Program


def build_project(
  program: Program,
  project_name: str,
  *,
  cache_base: str | None = None,
  emit_main_file: bool = True,
  shard_step_bodies: bool = False,
  unity: bool = False,
) -> dict[str, object]:
  '''Compile `program` end-to-end and write the .cpp tree.

  Args:
    program: a `srdatalog.Program` (DSL output).
    project_name: human-readable name used in C++ identifiers + cache
      dir. The C++ side derives `<project>_DB`, `<project>_DB_Blueprint`,
      `<project>_DB_DeviceDB`, and the cache lives at
      `~/.cache/srdatalog/jit/<project>_DB_<hash>/`.
    cache_base: override `~/.cache/srdatalog` (e.g., a tmpdir for
      tests, or `./build` for an in-tree project layout).
    emit_main_file: if True, also write a `main.cpp` containing the
      `_Runner` struct + relation typedefs. Set False if you only want
      the batch files (e.g., when integrating with an existing host
      binary that defines its own `<Project>_Runner`).
    shard_step_bodies: if True (default), emit each `step_N` and the
      `run()` dispatcher as its own compilable .cpp shard. This moves
      the heavy template instantiation out of main.cpp so the shards
      compile in parallel with the batch files. Set False to get the
      old layout (step bodies inline in main.cpp as template methods)
      — useful for byte-match tests against the Nim fragment fixture.
    unity: if True (default), emit ONE big .cpp containing the
      preamble + all JitRunner structs + _Runner + extern "C" shim.
      Parses `srdatalog.h` once per build instead of N times — the
      dominant cost when PCH isn't available. On doop this cuts cold
      compile from ~100s to ~20s. Set False for the traditional
      main + batch layout (better for byte-match testing against the
      Nim reference or for partial recompiles once PCH works).

  Returns the dict from `cache.write_jit_project`:
    { "dir", "main", "batches": [...], "schema_header", "kernel_header" }
  '''
  cr = compile_program(program, project_name)

  main_cpp = ""
  if emit_main_file:
    if unity:
      main_cpp = gen_unity_main_file_content(
        project_name,
        cr.hir.relation_decls,
        cr.mir,
        cr.step_bodies,
        cr.runner_decls,
        cr.per_rule_runners,
        extra_index_headers=cr.extra_headers,
        canonical_indices=cr.canonical_indices,
      )
    else:
      main_cpp = gen_main_file_content(
        project_name,
        cr.hir.relation_decls,
        cr.mir,
        cr.step_bodies,
        cr.runner_decls,
        cache_dir_hint="<cache>",
        jit_batch_count=1,
        emit_preamble=True,  # standalone TU — add #include "srdatalog.h" + namespaces
        extra_index_headers=cr.extra_headers,
        decl_only_runner=shard_step_bodies,
        canonical_indices=cr.canonical_indices,
      )
    main_cpp += "\n" + gen_extern_c_shim(project_name, cr.hir.relation_decls)

  # In unity mode we want no jit_batch_*.cpp files — they'd be
  # redundant (every JitRunner is already inlined in main.cpp).
  result = write_jit_project(
    cr.ext_db,
    main_file_content=main_cpp,
    per_rule_runners=[] if unity else cr.per_rule_runners,
    schema_definitions=cr.schema_defs,
    db_type_alias=cr.db_alias,
    extra_headers=cr.extra_headers,
    cache_base=cache_base,
  )

  # Sharded mode: emit one .cpp per step_N + one for run(), writing them
  # into the cache dir and appending their paths to the batch list so
  # compile_jit_project picks them up on its parallel compile queue.
  if shard_step_bodies and emit_main_file:
    import os

    for i in range(len(cr.mir.steps)):
      shard = gen_step_shard_file(
        project_name,
        cr.hir.relation_decls,
        cr.runner_decls,
        cr.mir,
        cr.step_bodies,
        i,
        extra_index_headers=cr.extra_headers,
      )
      path = os.path.join(str(result["dir"]), f"step_body_{i}.cpp")
      with open(path, "w") as f:
        f.write(shard)
      result["batches"].append(path)

    run_cpp = gen_run_dispatcher_file(
      project_name,
      cr.hir.relation_decls,
      cr.runner_decls,
      cr.mir,
      extra_index_headers=cr.extra_headers,
    )
    path = os.path.join(str(result["dir"]), "runner_dispatcher.cpp")
    with open(path, "w") as f:
      f.write(run_cpp)
    result["batches"].append(path)

  return result
