'''Pure compile pipeline — Program → compiled artifacts, no disk I/O.

`build_project` (build.py) and `srdatalog.viz` both want the same
intermediate artifacts: HIR, MIR, per-rule runner code, schema defs,
DB alias. The difference is what they do NEXT — `build_project`
writes a .cpp tree to disk; viz renders in a webview.

Splitting the pipeline at the compile/write boundary lets both
callers share the work. This module is that shared core.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from srdatalog.codegen.batchfile import _collect_pipelines
from srdatalog.codegen.jit.complete_runner import gen_complete_runner
from srdatalog.codegen.jit.main_file import (
  gen_db_type_alias_for_batch,
  gen_schema_definitions_for_batch,
)
from srdatalog.codegen.jit.orchestrator_jit import gen_step_body
from srdatalog.hir import compile_to_hir, compile_to_mir

if TYPE_CHECKING:
  from srdatalog.dsl import Program
  from srdatalog.hir.types import HirProgram
  from srdatalog.mir.nodes import Program as MirProgram


# Per-relation index type → header file. Registered at emit time
# because it mirrors Nim's "plugin" registry (an emit-time concern,
# not a runtime one).
_INDEX_HEADER = {
  "SRDatalog::GPU::Device2LevelIndex": "gpu/device_2level_index.h",
  "SRDatalog::GPU::DeviceTvjoinIndex": "gpu/device_tvjoin_index.h",
}


@dataclass(frozen=True)
class CompileResult:
  '''Everything a downstream consumer (build_project, viz) needs from
  the compile pipeline, in memory.'''

  hir: HirProgram
  mir: MirProgram
  # The <project>_DB and <project>_DB_DeviceDB C++ type names the
  # codegen uses. Stored here so file-emitting callers don't have to
  # rebuild the string.
  ext_db: str
  device_db: str
  # Per-orchestrator-step body (one per MIR step, in step order).
  step_bodies: list[str]
  # Per-rule complete runner: (rule_name, full_cpp_code). This is the
  # struct + kernels + phase methods + execute() for one rule. Goes
  # into a jit_batch_N.cpp when writing to disk.
  per_rule_runners: list[tuple[str, str]]
  # Per-rule runner forward declarations, keyed by rule name. main.cpp
  # references these; jit_batch files provide the definitions.
  runner_decls: dict[str, str]
  # Schema definitions string — inlined into every batch file so the
  # batch compiles in isolation.
  schema_defs: str
  # `using <project>_DB_Blueprint = Database<...>;` — inlined into
  # every batch file.
  db_alias: str
  # Per-relation canonical index, merged across strata (later strata
  # override earlier). Used by the print_size emit block so the
  # readback query hits the actual planned index.
  canonical_indices: dict[str, list[int]] = field(default_factory=dict)
  # Extra #include headers needed by non-default index plugins
  # (Device2LevelIndex, etc).
  extra_headers: list[str] = field(default_factory=list)
  # Per-relation index type string (C++ template name), for relations
  # that override the default. Empty-string entries are dropped.
  rel_index_types: dict[str, str] = field(default_factory=dict)


def compile_program(program: Program, project_name: str) -> CompileResult:
  '''Run the full compile pipeline — HIR → MIR → all emitted strings.

  Stops before any file I/O. The resulting `CompileResult` is the
  point both `build_project` (writes it to disk) and the viz module
  (renders it in a webview) branch from.
  '''
  hir = compile_to_hir(program)
  mir = compile_to_mir(program)

  ext_db = f"{project_name}_DB"
  device_db = f"{ext_db}_DeviceDB"

  rel_index_types = {d.rel_name: d.index_type for d in hir.relation_decls if d.index_type}
  seen_idx_types = set(rel_index_types.values())
  extra_headers = [_INDEX_HEADER[t] for t in seen_idx_types if t in _INDEX_HEADER]

  step_bodies = [
    gen_step_body(step, device_db, is_rec, i) for i, (step, is_rec) in enumerate(mir.steps)
  ]
  per_rule_runners: list[tuple[str, str]] = []
  runner_decls: dict[str, str] = {}
  for ep in _collect_pipelines(mir):
    decl, full = gen_complete_runner(ep, device_db, rel_index_types=rel_index_types)
    per_rule_runners.append((ep.rule_name, full))
    runner_decls[ep.rule_name] = decl

  schema_defs = gen_schema_definitions_for_batch(hir.relation_decls)
  db_alias = gen_db_type_alias_for_batch(ext_db, hir.relation_decls)

  # Merge canonical indices across strata; later strata override earlier
  # entries. Mirrors Nim's compileToMirWithDecls canonicalIndices merge
  # (hir.nim:310-314).
  canonical_indices: dict[str, list[int]] = {}
  for s in hir.strata:
    for rel, cols in s.canonical_index.items():
      canonical_indices[rel] = list(cols)

  return CompileResult(
    hir=hir,
    mir=mir,
    ext_db=ext_db,
    device_db=device_db,
    step_bodies=step_bodies,
    per_rule_runners=per_rule_runners,
    runner_decls=runner_decls,
    schema_defs=schema_defs,
    db_alias=db_alias,
    canonical_indices=canonical_indices,
    extra_headers=extra_headers,
    rel_index_types=rel_index_types,
  )
