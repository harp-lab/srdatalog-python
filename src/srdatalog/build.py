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
'''

from __future__ import annotations

from typing import TYPE_CHECKING

from srdatalog.codegen.batchfile import _collect_pipelines
from srdatalog.codegen.jit.cache import write_jit_project
from srdatalog.codegen.jit.complete_runner import gen_complete_runner
from srdatalog.codegen.jit.main_file import (
  gen_db_type_alias_for_batch,
  gen_extern_c_shim,
  gen_main_file_content,
  gen_run_dispatcher_file,
  gen_schema_definitions_for_batch,
  gen_step_shard_file,
  gen_unity_main_file_content,
)
from srdatalog.codegen.jit.orchestrator_jit import gen_step_body
from srdatalog.hir import compile_to_hir, compile_to_mir

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
  hir = compile_to_hir(program)
  mir = compile_to_mir(program)

  # The C++ side calls the device-DB type `<project>_DB_DeviceDB`.
  # See main_file.gen_db_type_alias_for_batch — the `ruleset_name`
  # arg there is the ext_db name (`<project>_DB`).
  ext_db = f"{project_name}_DB"
  device_db = f"{ext_db}_DeviceDB"

  # Per-relation index type map (default-LSM for unset relations). The
  # DSL exposes this via `Relation(..., index_type="...")`; HIR copies
  # it into RelationDecl.index_type; complete_runner uses it to pick
  # the C++ index template (Device2LevelIndex, DeviceLSMIndex, etc).
  rel_index_types = {d.rel_name: d.index_type for d in hir.relation_decls if d.index_type}

  # Map each non-default index type → the header that defines it.
  # Kept here (not in runtime/__init__.py) because the registration
  # mirrors Nim's "plugin" registry and is an emit-time concern.
  _INDEX_HEADER = {
    "SRDatalog::GPU::Device2LevelIndex": "gpu/device_2level_index.h",
    "SRDatalog::GPU::DeviceTvjoinIndex": "gpu/device_tvjoin_index.h",
  }
  seen_idx_types = set(rel_index_types.values())
  extra_headers = [_INDEX_HEADER[t] for t in seen_idx_types if t in _INDEX_HEADER]

  # Per-step bodies (orchestrator) + per-rule complete runner (kernel
  # struct, phase methods, execute()).
  step_bodies = [
    gen_step_body(step, device_db, is_rec, i) for i, (step, is_rec) in enumerate(mir.steps)
  ]
  per_rule: list[tuple[str, str]] = []
  runner_decls: dict[str, str] = {}
  for ep in _collect_pipelines(mir):
    decl, full = gen_complete_runner(ep, device_db, rel_index_types=rel_index_types)
    per_rule.append((ep.rule_name, full))
    runner_decls[ep.rule_name] = decl

  # Compute the inlined schema + DB strings — these get prepended to
  # every batch file by JitBatchManager so the batches compile in
  # isolation.
  schema_defs = gen_schema_definitions_for_batch(hir.relation_decls)
  db_alias = gen_db_type_alias_for_batch(ext_db, hir.relation_decls)

  # Optional standalone main file (Python-only convenience — Nim's
  # main is the user's `.nim` file, not generated by codegen). Includes
  # the extern "C" ctypes shim so Python can dlopen + call into it.
  #
  # Collect per-relation canonical indices so the final print_size block
  # queries the actual planned index (not the natural `{0,...,arity-1}`).
  # Mirrors Nim's compileToMirWithDecls canonicalIndices merge (hir.nim:310-314):
  # walk all strata, later strata override earlier entries.
  canonical_indices: dict[str, list[int]] = {}
  for s in hir.strata:
    for rel, cols in s.canonical_index.items():
      canonical_indices[rel] = list(cols)

  main_cpp = ""
  if emit_main_file:
    if unity:
      # Single-TU path: srdatalog.h parses once for the whole project.
      main_cpp = gen_unity_main_file_content(
        project_name,
        hir.relation_decls,
        mir,
        step_bodies,
        runner_decls,
        per_rule,
        extra_index_headers=extra_headers,
        canonical_indices=canonical_indices,
      )
    else:
      main_cpp = gen_main_file_content(
        project_name,
        hir.relation_decls,
        mir,
        step_bodies,
        runner_decls,
        cache_dir_hint="<cache>",
        jit_batch_count=1,
        emit_preamble=True,  # standalone TU — add #include "srdatalog.h" + namespaces
        extra_index_headers=extra_headers,
        decl_only_runner=shard_step_bodies,
        canonical_indices=canonical_indices,
      )
    main_cpp += "\n" + gen_extern_c_shim(project_name, hir.relation_decls)

  # In unity mode we want no jit_batch_*.cpp files — they'd be
  # redundant (every JitRunner is already inlined in main.cpp).
  result = write_jit_project(
    ext_db,
    main_file_content=main_cpp,
    per_rule_runners=[] if unity else per_rule,
    schema_definitions=schema_defs,
    db_type_alias=db_alias,
    extra_headers=extra_headers,
    cache_base=cache_base,
  )

  # Sharded mode: emit one .cpp per step_N + one for run(), writing them
  # into the cache dir and appending their paths to the batch list so
  # compile_jit_project picks them up on its parallel compile queue.
  if shard_step_bodies and emit_main_file:
    import os

    for i in range(len(mir.steps)):
      shard = gen_step_shard_file(
        project_name,
        hir.relation_decls,
        runner_decls,
        mir,
        step_bodies,
        i,
        extra_index_headers=extra_headers,
      )
      path = os.path.join(str(result["dir"]), f"step_body_{i}.cpp")
      with open(path, "w") as f:
        f.write(shard)
      result["batches"].append(path)

    run_cpp = gen_run_dispatcher_file(
      project_name,
      hir.relation_decls,
      runner_decls,
      mir,
      extra_index_headers=extra_headers,
    )
    path = os.path.join(str(result["dir"]), "runner_dispatcher.cpp")
    with open(path, "w") as f:
      f.write(run_cpp)
    result["batches"].append(path)

  return result
