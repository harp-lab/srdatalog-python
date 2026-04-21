'''JIT batch file assembly.

Port of src/srdatalog/codegen/target_jit/jit_file.nim (the pure emit
entry points — we skip the filesystem-writing procs since our callers
return the string directly).

Key entry point:
  gen_jit_file_content(rule_name, pipeline, source_specs=(), dest_specs=())
    = JIT_COMMON_INCLUDES + jit_full_kernel(rule_name, pipeline) + "\\n"
      + optional explicit JitExecutor instantiation
      + JIT_FILE_FOOTER

That string is what Nim's `genJitFileContentFromExecutePipeline` returns
and what our fixture-dumping tool captures as `jit_batch.<rule>.cpp`.

Other procs ported:
  - JIT_COMMON_INCLUDES, JIT_FILE_FOOTER constants
  - gen_jit_file_content_from_execute_pipeline(node) — convenience
'''

from __future__ import annotations

from collections.abc import Sequence

import srdatalog.mir.types as m
from srdatalog.codegen.jit.context import new_code_gen_context
from srdatalog.codegen.jit.emit_helpers import assign_handle_positions
from srdatalog.codegen.jit.kernel_functor import jit_full_kernel

# -----------------------------------------------------------------------------
# File preamble and footer (byte-identical to jit_file.nim)
# -----------------------------------------------------------------------------

JIT_COMMON_INCLUDES = """\
// JIT-Generated Rule Kernel Batch
// This file is auto-generated - do not edit
#define SRDATALOG_JIT_BATCH  // Guard: exclude host-side helpers from JIT compilation

// Main project header - includes all necessary boost/hana, etc.
#include "srdatalog.h"

#include <cooperative_groups.h>
#include <cstdint>

// JIT-specific headers (relative to generalized_datalog/)
#include "gpu/device_sorted_array_index.h"
#include "gpu/runtime/jit/intersect_handles.h"
#include "gpu/runtime/jit/jit_executor.h"
#include "gpu/runtime/jit/materialized_join.h"
#include "gpu/runtime/jit/ws_infrastructure.h"  // WCOJTask, WCOJTaskQueue, ChunkedOutputContext
#include "gpu/runtime/output_context.h"
#include "gpu/runtime/query.h"  // For DeviceRelationType

namespace cg = cooperative_groups;

// Make JIT helpers visible without full namespace qualification
using SRDatalog::GPU::JIT::intersect_handles;

"""

JIT_FILE_FOOTER = """
// End of JIT batch file
"""


# -----------------------------------------------------------------------------
# File content generation
# -----------------------------------------------------------------------------


def gen_jit_file_content(
  rule_name: str,
  pipeline: list[m.MirNode],
  source_specs: Sequence[str] = (),
  dest_specs: Sequence[str] = (),
) -> str:
  '''Complete .cu file content for one rule.

  Equivalent to Nim's `genJitFileContent`. `source_specs` / `dest_specs`
  are optional C++ IndexSpec type strings — when both are provided, an
  `template struct JitExecutor<...>` explicit instantiation is appended.
  When omitted (the default), only the kernel functor itself is emitted
  (this matches Nim's `genJitFileContentFromExecutePipeline` which calls
  through with empty specs).
  '''
  result = JIT_COMMON_INCLUDES

  # Assign handle positions before codegen (Nim mutates a copy here).
  mutable_pipe = list(pipeline)
  assign_handle_positions(mutable_pipe)

  ctx = new_code_gen_context()
  result += jit_full_kernel(rule_name, mutable_pipe, ctx)
  result += "\n"

  if source_specs and dest_specs:
    result += "// Explicit template instantiation\n"
    sources_tuple = "std::tuple<" + ", ".join(source_specs) + ">"
    dests_tuple = "std::tuple<" + ", ".join(dest_specs) + ">"
    result += f"template struct JitExecutor<Kernel_{rule_name}, {sources_tuple}, {dests_tuple}>;\n"
    result += "\n"

  result += JIT_FILE_FOOTER
  return result


def gen_jit_file_content_from_execute_pipeline(node: m.ExecutePipeline) -> str:
  '''Convenience wrapper matching Nim's `genJitFileContentFromExecutePipeline`.
  The fixture-dumping tool (srdatalog_plan.nim) uses this entry point
  to dump `jit_batch.<rule>.cpp` files, so our Python output is
  byte-comparable to those fixtures once the kernel functor emit is
  complete.
  '''
  assert isinstance(node, m.ExecutePipeline)
  return gen_jit_file_content(node.rule_name, list(node.pipeline))


def gen_jit_file_name(rule_name: str) -> str:
  '''Filename convention for per-rule .cu files.'''
  return f"jit_rule_{rule_name.lower()}.cu"
