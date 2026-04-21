'''JIT cache directory management + batch file writer.

Port of `src/srdatalog/codegen/target_jit/jit_file.nim`:
  - `getJitCacheDir` / `ensureJitCacheDir`
  - `JitBatchManager` with `addKernel` / `writeBatchFiles` /
    `writeSchemaHeader` / `writeKernelDeclHeader`

Python uses `~/.cache/srdatalog/jit/<project>_<hash>/` (vs Nim's
`~/.cache/nim/jit/...`) so the two toolchains don't clobber each
other's caches. Callers that need Nim-compatible output can pass an
explicit cache dir via the `cache_dir` arg.

The one-shot entry point `write_jit_project()` glues everything
together — given the string outputs from `main_file.py` + per-rule
complete runner emissions, it lays out the full .cpp tree on disk.
Set `SRDATALOG_SKIP_JIT_REGEN=1` to reuse existing files (debugging
mode — matches Nim's behavior).
'''
from __future__ import annotations

import hashlib
import os
from pathlib import Path

# Match Nim's RULES_PER_BATCH default (jit_file.nim:30). Override via
# env `SRDATALOG_RULES_PER_BATCH=N`.
_DEFAULT_RULES_PER_BATCH = int(os.environ.get("SRDATALOG_RULES_PER_BATCH", "8"))
MAX_BATCH_FILES = 16


# -----------------------------------------------------------------------------
# Cache directory
# -----------------------------------------------------------------------------

def _project_hash(project_name: str) -> str:
  '''4-hex-digit project hash — same shape as Nim's `(hash(name) and 0xFFFF).toHex(4)`.
  We use sha256 (first 4 hex digits) so the mapping is reproducible across
  Python versions (`hash()` is randomized).'''
  h = hashlib.sha256(project_name.encode()).hexdigest()
  return h[:4].upper()


def get_jit_cache_dir(project_name: str, base: str | None = None) -> str:
  '''`~/.cache/srdatalog/jit/<project>_<hash4>/`. `base` overrides
  `~/.cache/srdatalog` — e.g. tests pass a tmpdir.'''
  if base is None:
    home = os.environ.get("HOME", "/tmp")
    base = os.path.join(home, ".cache", "srdatalog")
  return os.path.join(base, "jit", f"{project_name}_{_project_hash(project_name)}")


def ensure_jit_cache_dir(project_name: str, base: str | None = None) -> str:
  '''Create the cache dir if needed; return the path.'''
  d = get_jit_cache_dir(project_name, base)
  os.makedirs(d, exist_ok=True)
  return d


def get_batch_file_name(batch_index: int) -> str:
  return f"jit_batch_{batch_index}.cpp"


# -----------------------------------------------------------------------------
# JIT batch common header / footer (matches jit_file.nim constants)
# -----------------------------------------------------------------------------

JIT_COMMON_INCLUDES = """\
// JIT-Generated Rule Kernel Batch
// This file is auto-generated - do not edit
#define SRDATALOG_JIT_BATCH  // Guard: exclude host-side helpers from JIT compilation

// Main project header - includes all necessary boost/hana, etc.
#include "srdatalog.h"

#include <cstdint>
#include <cooperative_groups.h>

// JIT-specific headers (relative to generalized_datalog/)
#include "gpu/device_sorted_array_index.h"
#include "gpu/runtime/output_context.h"
#include "gpu/runtime/jit/intersect_handles.h"
#include "gpu/runtime/jit/jit_executor.h"
#include "gpu/runtime/jit/materialized_join.h"
#include "gpu/runtime/jit/ws_infrastructure.h"  // WCOJTask, WCOJTaskQueue, ChunkedOutputContext
#include "gpu/runtime/query.h"  // For DeviceRelationType

namespace cg = cooperative_groups;

// Make JIT helpers visible without full namespace qualification
using SRDatalog::GPU::JIT::intersect_handles;

"""

JIT_FILE_FOOTER = """
// End of JIT batch file
"""


# -----------------------------------------------------------------------------
# JitBatchManager
# -----------------------------------------------------------------------------

class JitBatchManager:
  '''Shards per-rule runner code across fixed-size batch files,
  then writes them + the schema/kernel headers to the cache dir.

  Mirrors Nim's `JitBatchManager` in `jit_file.nim:100-270`.
  '''
  def __init__(
    self, project_name: str, rules_per_batch: int = _DEFAULT_RULES_PER_BATCH,
    cache_base: str | None = None,
  ):
    self.project_name = project_name
    self.rules_per_batch = rules_per_batch
    self.cache_base = cache_base
    self.batches: dict[int, list[str]] = {}
    self.rule_names: list[str] = []
    self.rule_count = 0
    self.schema_definitions = ""
    self.db_type_alias = ""
    self.kernel_declarations = ""

  # --- public accumulators ---

  def set_schema_definitions(self, schema_defs: str) -> None:
    self.schema_definitions = schema_defs

  def set_db_type_alias(self, db_type: str) -> None:
    self.db_type_alias = db_type

  def add_kernel_declaration(self, decl_code: str) -> None:
    self.kernel_declarations += decl_code

  def add_kernel(self, kernel_code: str, rule_name: str | None = None) -> None:
    '''Add one `JitRunner_<rule>` struct (complete with __global__
    kernels) to the next batch slot.'''
    batch_idx = self.rule_count // self.rules_per_batch
    self.batches.setdefault(batch_idx, []).append(kernel_code)
    if rule_name is not None:
      self.rule_names.append(rule_name)
    self.rule_count += 1

  def batch_count(self) -> int:
    return len(self.batches)

  # --- content generation (no I/O) ---

  def generate_batch_file(
    self, batch_idx: int, extra_headers: list[str] | None = None,
  ) -> str:
    if batch_idx not in self.batches:
      return ""
    extra = extra_headers or []

    code = JIT_COMMON_INCLUDES
    if extra:
      code += "// Extra index headers (from registered plugins)\n"
      for h in extra:
        code += f'#include "{h}"\n'
      code += "\n"
    if self.schema_definitions:
      code += "// Project-specific schema definitions (inlined)\n"
      code += self.schema_definitions
      code += "\n\n"
    if self.db_type_alias:
      code += "// DB type alias for JitRunner type derivation\n"
      code += self.db_type_alias
      code += "\n\n"
    code += (
      f"// Batch {batch_idx} - {len(self.batches[batch_idx])} rules\n\n"
    )
    for kernel_code in self.batches[batch_idx]:
      code += kernel_code
      code += "\n"
    code += JIT_FILE_FOOTER
    return code

  def generate_schema_header(self) -> str:
    out = f"// Auto-generated schema definitions for {self.project_name}\n"
    out += "// This file is auto-generated - do not edit\n\n"
    out += "#pragma once\n"
    out += '#include "srdatalog.h"\n\n'
    out += self.schema_definitions
    out += "\n"
    return out

  def generate_kernel_decl_header(self) -> str:
    out = f"// Auto-generated kernel declarations for {self.project_name}\n"
    out += "// This file is auto-generated - do not edit\n\n"
    out += "#pragma once\n\n"
    out += '#include "gpu/runtime/jit/ws_infrastructure.h"\n\n'
    out += self.kernel_declarations
    out += "\n"
    return out

  # --- I/O ---

  def _skip_regen(self) -> bool:
    return os.environ.get("SRDATALOG_SKIP_JIT_REGEN", "") == "1"

  def _write_if_changed(self, path: str, content: str) -> bool:
    '''Write content to `path` only if it differs from what's already
    there. Returns True if the file was written. Avoids touching
    mtimes when the contents haven't changed — builds that depend on
    timestamp-based rebuild (ninja, make) won't re-run.'''
    try:
      with open(path, "r") as f:
        if f.read() == content:
          return False
    except FileNotFoundError:
      pass
    with open(path, "w") as f:
      f.write(content)
    return True

  def write_schema_header(self) -> str:
    if not self.schema_definitions:
      return ""
    cache_dir = ensure_jit_cache_dir(self.project_name, self.cache_base)
    path = os.path.join(cache_dir, f"{self.project_name}_schemas.h")
    self._write_if_changed(path, self.generate_schema_header())
    return path

  def write_kernel_decl_header(self) -> str:
    if not self.kernel_declarations:
      return ""
    cache_dir = ensure_jit_cache_dir(self.project_name, self.cache_base)
    path = os.path.join(cache_dir, f"{self.project_name}_kernels.h")
    self._write_if_changed(path, self.generate_kernel_decl_header())
    return path

  def write_batch_files(
    self, extra_headers: list[str] | None = None,
  ) -> list[str]:
    '''Write all shards + headers to the cache dir. Returns the list
    of batch file paths (headers are written but not returned — they
    aren't compiled directly).'''
    cache_dir = ensure_jit_cache_dir(self.project_name, self.cache_base)
    skip = self._skip_regen()

    self.write_schema_header()
    self.write_kernel_decl_header()

    written: list[str] = []
    for batch_idx in sorted(self.batches.keys()):
      path = os.path.join(cache_dir, get_batch_file_name(batch_idx))
      if skip and os.path.exists(path):
        written.append(path)
        continue
      self._write_if_changed(path, self.generate_batch_file(batch_idx, extra_headers))
      written.append(path)
    return written


# -----------------------------------------------------------------------------
# One-shot project writer
# -----------------------------------------------------------------------------

def write_jit_project(
  project_name: str,
  main_file_content: str,
  per_rule_runners: list[tuple[str, str]],
  *,
  schema_definitions: str = "",
  db_type_alias: str = "",
  extra_headers: list[str] | None = None,
  cache_base: str | None = None,
  main_file_name: str = "main.cpp",
) -> dict[str, object]:
  '''Lay out the full .cpp tree for a project.

  Args:
    project_name: cache dir name (e.g. "TrianglePlan_DB").
    main_file_content: output of `main_file.gen_main_file_content`.
    per_rule_runners: list of `(rule_name, full_runner_cpp)` tuples —
      typically the `full` returned by `complete_runner.gen_complete_runner`
      for each non-materialized `ExecutePipeline`. Gets sharded across
      jit_batch_N.cpp files.
    schema_definitions: optional project schema header content.
    db_type_alias: optional DB type alias string (inlined into each
      batch file for template derivation).
    extra_headers: per-rule plugin headers (e.g.
      "gpu/device_2level_index.h") #include'd into every batch file.
    cache_base: override `~/.cache/srdatalog` (tests pass a tmpdir).
    main_file_name: output name for the top-level main file.

  Returns: dict with keys `dir`, `main`, `batches` (list[str]),
  `schema_header`, `kernel_header` (possibly "") — every path absolute.
  '''
  mgr = JitBatchManager(project_name, cache_base=cache_base)
  if schema_definitions:
    mgr.set_schema_definitions(schema_definitions)
  if db_type_alias:
    mgr.set_db_type_alias(db_type_alias)
  for rule_name, runner_code in per_rule_runners:
    mgr.add_kernel(runner_code, rule_name)

  batch_paths = mgr.write_batch_files(extra_headers)
  schema_path = mgr.write_schema_header()
  kernel_path = mgr.write_kernel_decl_header()

  cache_dir = ensure_jit_cache_dir(project_name, cache_base)
  main_path = os.path.join(cache_dir, main_file_name)
  mgr._write_if_changed(main_path, main_file_content)

  return {
    "dir": cache_dir,
    "main": main_path,
    "batches": batch_paths,
    "schema_header": schema_path,
    "kernel_header": kernel_path,
  }
