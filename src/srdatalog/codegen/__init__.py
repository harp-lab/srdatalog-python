'''C++ codegen backend.

The live path is JIT: HIR→MIR→`codegen.jit.*` emits per-rule kernels,
orchestrator steps, runner struct, and a main.cpp driver. Non-JIT
submodules at this level are small shared utilities used by that path.

Submodules:
  schema    — FactDefinition / Pragma / SchemaDefinition (C++ prelude emission)
  helpers   — view-spec collection + shared C++ string helpers
  batchfile — pipeline-collection utilities (used by build.py + JIT tests)
  jit/      — the Nim-faithful JIT codegen backend (orchestrator_jit,
              complete_runner, main_file, root, instructions, ...)
'''
