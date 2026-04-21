'''Command-line entry point.

Installed by pyproject.toml as `srdatalog`:

    srdatalog emit    <program.py> --project P                    # emit .cpp tree
    srdatalog compile <program.py> --project P [--out libP.so]    # emit + compile
    srdatalog info                                                # paths + versions

The `<program.py>` must define `program()` returning a `srdatalog.Program`
instance. The tool imports it and runs the full pipeline.
'''
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_program(path: str, project_name: str):
  '''Import `path` and return its `program()` result.'''
  spec = importlib.util.spec_from_file_location("user_program", path)
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  if not hasattr(mod, "program"):
    sys.exit(f"{path}: module must export a `program()` function")
  prog = mod.program()
  from srdatalog import Program
  if not isinstance(prog, Program):
    sys.exit(f"{path}: program() must return a srdatalog.Program")
  return prog


def _emit_project(prog, project_name: str, cache_base: str | None):
  '''Compile Program → write jit_runner/*.cpp + main.cpp tree to disk.
  Returns the `write_jit_project` result dict.'''
  from srdatalog import (
    compile_to_hir, compile_to_mir, gen_step_body, gen_complete_runner,
    gen_main_file_content, write_jit_project,
  )
  from srdatalog.codegen.batchfile import _collect_pipelines

  hir = compile_to_hir(prog)
  mir = compile_to_mir(prog)
  db_type = f"{project_name}_DB_DeviceDB"

  step_bodies = [
    gen_step_body(step, db_type, is_rec, i)
    for i, (step, is_rec) in enumerate(mir.steps)
  ]
  per_rule: list[tuple[str, str]] = []
  runner_decls: dict[str, str] = {}
  for ep in _collect_pipelines(mir):
    decl, full = gen_complete_runner(ep, db_type)
    per_rule.append((ep.rule_name, full))
    runner_decls[ep.rule_name] = decl

  main_cpp = gen_main_file_content(
    project_name, hir.relation_decls, mir, step_bodies, runner_decls,
    cache_dir_hint="<cache>", jit_batch_count=1,
  )
  return write_jit_project(
    f"{project_name}_DB",
    main_file_content=main_cpp,
    per_rule_runners=per_rule,
    cache_base=cache_base,
  )


def cmd_emit(args: argparse.Namespace) -> int:
  prog = _load_program(args.program, args.project)
  result = _emit_project(prog, args.project, args.cache_base)
  print(f"main   : {result['main']}")
  for b in result["batches"]:
    print(f"batch  : {b}")
  if result["schema_header"]:
    print(f"schema : {result['schema_header']}")
  if result["kernel_header"]:
    print(f"kernels: {result['kernel_header']}")
  print(f"dir    : {result['dir']}")
  return 0


def cmd_compile(args: argparse.Namespace) -> int:
  from srdatalog import CompilerConfig, compile_jit_project
  from srdatalog.runtime import runtime_include_path

  prog = _load_program(args.program, args.project)
  result = _emit_project(prog, args.project, args.cache_base)

  cfg = CompilerConfig(
    cxx=args.cxx or "",
    include_paths=[runtime_include_path()] + (args.include or []),
    defines=args.define or [],
    cxx_flags=args.cxx_flag or [],
    jobs=args.jobs,
  )
  build = compile_jit_project(result, cfg)
  if not build.ok():
    for r in build.compile_results:
      if r.returncode != 0:
        print(f"[FAIL compile {r.output}]\n{r.stderr}", file=sys.stderr)
    if build.link_result and build.link_result.returncode != 0:
      print(f"[FAIL link]\n{build.link_result.stderr}", file=sys.stderr)
    return 1

  out = args.out or build.artifact
  if args.out and args.out != build.artifact:
    import shutil as _sh
    _sh.copy(build.artifact, args.out)
  print(f"built {out} ({build.elapsed_sec:.2f}s)")
  return 0


def cmd_info(args: argparse.Namespace) -> int:
  import srdatalog
  from srdatalog.runtime import runtime_include_path
  print(f"srdatalog  : {srdatalog.__version__}")
  print(f"python     : {sys.version.split()[0]}")
  print(f"runtime    : {runtime_include_path()}")
  print(f"package    : {Path(srdatalog.__file__).parent}")
  return 0


def main(argv: list[str] | None = None) -> int:
  p = argparse.ArgumentParser(
    prog="srdatalog",
    description="Compile / emit Datalog programs via the srdatalog Python frontend.",
  )
  sub = p.add_subparsers(dest="cmd", required=True)

  pe = sub.add_parser("emit", help="Emit the .cpp tree (no compile).")
  pe.add_argument("program", help="Path to Python file exporting `program()` -> srdatalog.Program")
  pe.add_argument("--project", required=True, help="Project name (used for cache dir + runner types)")
  pe.add_argument("--cache-base", default=None,
                  help="Override ~/.cache/srdatalog (e.g., ./build)")
  pe.set_defaults(func=cmd_emit)

  pc = sub.add_parser("compile", help="Emit + compile to a shared library.")
  pc.add_argument("program")
  pc.add_argument("--project", required=True)
  pc.add_argument("--cache-base", default=None)
  pc.add_argument("--out", default=None, help="Copy produced .so to this path")
  pc.add_argument("--cxx", default=None, help="Override compiler (default: $CXX or clang++)")
  pc.add_argument("--include", action="append", help="Extra -I path (repeatable)")
  pc.add_argument("--define", action="append", help="Extra -D macro (repeatable)")
  pc.add_argument("--cxx-flag", action="append", help="Extra C++ flag (repeatable)")
  pc.add_argument("--jobs", type=int, default=0, help="Parallel compile jobs (0 = cpu_count)")
  pc.set_defaults(func=cmd_compile)

  pi = sub.add_parser("info", help="Show package paths / version.")
  pi.set_defaults(func=cmd_info)

  args = p.parse_args(argv)
  return args.func(args)


if __name__ == "__main__":
  raise SystemExit(main())
