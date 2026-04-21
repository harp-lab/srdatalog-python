'''srdatalog — Python frontend with JIT C++/CUDA codegen.

Drives the full pipeline in Python — DSL → MIR → emit C++ → compile
→ dlopen — without depending on Nim or xmake. The bundled runtime
headers + the bundled vendor deps + auto-detected CUDA are everything
the compile needs.

Public API. Typical use:

    from srdatalog import (
      Var, Relation, Program, build_project,
      CompilerConfig, build_and_load, EntryPoint,
    )
    from srdatalog.runtime import (
      runtime_include_paths, cuda_include_paths, runtime_defines,
    )

    prog = Program(relations=[...], rules=[...])
    project = build_project(prog, project_name="MyPlan", cache_base="./build")

    cfg = CompilerConfig(
      include_paths=runtime_include_paths() + cuda_include_paths(),
      defines=runtime_defines(),
    )
    runtime = build_and_load(project, entry_points=[
      EntryPoint("srdatalog_init"),
      EntryPoint("srdatalog_run"),
    ], compiler_config=cfg)

    runtime.srdatalog_init()
    runtime.srdatalog_run(b"/path/to/data")

See README + examples/triangle.py.
'''

# DSL
# One-shot end-to-end builder (the recommended entry point)
from srdatalog.build import build_project
from srdatalog.codegen.jit.cache import write_jit_project

# Compile + link via the bundled compiler wrapper
from srdatalog.codegen.jit.compiler import (
  BuildResult,
  CompilerConfig,
  CompileResult,
  compile_cpp,
  compile_jit_project,
  link_shared,
)
from srdatalog.codegen.jit.complete_runner import gen_complete_runner

# Load + call via ctypes
from srdatalog.codegen.jit.loader import (
  EntryPoint,
  JitRuntime,
  build_and_load,
  gen_runtime_shim_template,
)

# Codegen — emit .cpp tree
from srdatalog.codegen.jit.main_file import (
  gen_db_type_alias_for_batch,
  gen_main_file_content,
  gen_schema_definitions_for_batch,
)
from srdatalog.codegen.jit.orchestrator_jit import gen_step_body
from srdatalog.dsl import Program, Relation, Var

# Compilation pipeline
from srdatalog.hir import compile_to_hir, compile_to_mir

__version__ = "0.1.0"

__all__ = [
  # DSL
  "Var",
  "Relation",
  "Program",
  # Compile
  "compile_to_hir",
  "compile_to_mir",
  # Codegen building blocks
  "gen_main_file_content",
  "gen_schema_definitions_for_batch",
  "gen_db_type_alias_for_batch",
  "gen_complete_runner",
  "gen_step_body",
  "write_jit_project",
  # End-to-end emit
  "build_project",
  # Build (compile + link)
  "CompilerConfig",
  "BuildResult",
  "CompileResult",
  "compile_cpp",
  "link_shared",
  "compile_jit_project",
  # Load + call
  "EntryPoint",
  "JitRuntime",
  "build_and_load",
  "gen_runtime_shim_template",
]
