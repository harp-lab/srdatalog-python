'''srdatalog — Python frontend with JIT C++/CUDA codegen.

Public API surface. Import what you need:

    from srdatalog import Var, Relation, Program, compile_to_mir
    from srdatalog import build_and_load, CompilerConfig, EntryPoint
    from srdatalog.runtime import runtime_include_path

See README.md for an end-to-end example.
'''

# DSL
from srdatalog.dsl import Var, Relation, Program

# Compilation pipeline
from srdatalog.hir import compile_to_hir, compile_to_mir

# Codegen + build
from srdatalog.codegen.jit.main_file import gen_main_file_content
from srdatalog.codegen.jit.complete_runner import gen_complete_runner
from srdatalog.codegen.jit.orchestrator_jit import gen_step_body
from srdatalog.codegen.jit.cache import write_jit_project
from srdatalog.codegen.jit.compiler import (
  CompilerConfig, BuildResult, CompileResult,
  compile_cpp, link_shared, compile_jit_project,
)
from srdatalog.codegen.jit.loader import (
  EntryPoint, JitRuntime, build_and_load, gen_runtime_shim_template,
)

__version__ = "0.1.0"

__all__ = [
  # DSL
  "Var", "Relation", "Program",
  # Compile
  "compile_to_hir", "compile_to_mir",
  # Codegen
  "gen_main_file_content", "gen_complete_runner", "gen_step_body",
  "write_jit_project",
  # Build
  "CompilerConfig", "BuildResult", "CompileResult",
  "compile_cpp", "link_shared", "compile_jit_project",
  # Load
  "EntryPoint", "JitRuntime", "build_and_load", "gen_runtime_shim_template",
]
