# srdatalog

Python frontend for **Super-Reconfigurable Datalog** — a column-oriented Datalog engine
with JIT C++/CUDA code generation. The library takes a Python DSL program, compiles it
through HIR → MIR → GPU-targeted C++, writes the resulting `.cpp` tree to a cache, and
(optionally) builds a loadable shared library you can call from Python.

## Install

```bash
# Editable install while iterating on the library itself
uv pip install -e .

# Or: build a wheel
uv build
```

The wheel bundles the `generalized_datalog/` C++ runtime headers as package data, so
no external checkout is needed at install time. A C++ toolchain (`clang++` / `g++`, and
`nvcc` for GPU targets) is required at **compile time** — the library discovers it via
`$CXX` or by scanning `PATH`.

## Quickstart

```python
from srdatalog import (
  Var, Relation, Program,
  compile_to_hir, compile_to_mir,
  gen_step_body, gen_complete_runner, gen_main_file_content,
  write_jit_project,
  CompilerConfig, compile_jit_project,
  JitRuntime, EntryPoint, build_and_load,
)
from srdatalog.runtime import runtime_include_path
from srdatalog.codegen.batchfile import _collect_pipelines

# 1. Build a program
x, y, z = Var("x"), Var("y"), Var("z")
Edge = Relation("Edge", 2)
Path = Relation("Path", 2)
prog = Program(
  relations=[Edge, Path],
  rules=[
    (Path(x, y) <= Edge(x, y)).named("TCBase"),
    (Path(x, z) <= Path(x, y) & Edge(y, z)).named("TCRec"),
  ],
)

# 2. Compile to MIR
hir = compile_to_hir(prog)
mir = compile_to_mir(prog)

# 3. Emit the C++ tree
steps = [gen_step_body(s, "TC_DB_DeviceDB", r, i) for i, (s, r) in enumerate(mir.steps)]
per_rule = []
runner_decls = {}
for ep in _collect_pipelines(mir):
  decl, full = gen_complete_runner(ep, "TC_DB_DeviceDB")
  per_rule.append((ep.rule_name, full))
  runner_decls[ep.rule_name] = decl
main_cpp = gen_main_file_content(
  "TC", hir.relation_decls, mir, steps, runner_decls,
)
project = write_jit_project(
  "TC_DB", main_file_content=main_cpp, per_rule_runners=per_rule,
)

# 4. Compile + load
cfg = CompilerConfig(include_paths=[runtime_include_path()])
runtime = build_and_load(
  project,
  entry_points=[
    EntryPoint("srdatalog_init"),
    EntryPoint("srdatalog_run"),
  ],
  compiler_config=cfg,
)

# 5. Call
runtime.srdatalog_init()
runtime.srdatalog_run(b"/path/to/data")
```

Step 4 assumes you've added an `extern "C"` shim to the project — see
`gen_runtime_shim_template()`. The shim wraps your `<Project>_Runner::run()` into a
C-ABI entry point the Python loader can dlopen.

## CLI

```bash
# Emit the .cpp tree (no compile)
srdatalog emit examples/triangle.py --project TrianglePlan

# Emit + compile to a .so
srdatalog compile examples/triangle.py --project TrianglePlan --out /tmp/libtri.so

# Package info
srdatalog info
```

See `examples/triangle.py` for a runnable end-to-end demo.

## Architecture

| Stage | Module | What it does |
|---|---|---|
| DSL → HIR | `srdatalog.hir` | Parse rules, infer types, stratify, index planning |
| HIR → MIR | `srdatalog.hir_lower`, `srdatalog.mir_passes` | Lower to imperative IR + optimization passes |
| MIR → C++ | `srdatalog.codegen.jit.*` | Emit kernels, runner struct, main file |
| Write | `srdatalog.codegen.jit.cache` | Shard into `jit_batch_N.cpp`, write to cache dir |
| Compile | `srdatalog.codegen.jit.compiler` | Invoke `clang++` / `nvcc` / `g++` in parallel with hash cache |
| Load | `srdatalog.codegen.jit.loader` | dlopen via `ctypes`, bind `extern "C"` entries |

The JIT codegen layer produces byte-identical output to the upstream Nim reference for
**125/127** runner fixtures (the remaining 2 require a work-stealing runner variant still
pending a port).

## Development

```bash
git clone <repo>
cd srdatalog-python
uv sync --group dev
uv run pytest
```

## Status & roadmap

- ✅ Phase 1-4: JIT codegen byte-match (125/127 fixtures)
- ✅ Phase 6: `_Runner` struct + main-file emission
- ✅ Phase 7: on-disk cache manager
- ✅ Phase 8: compiler wrapper with parallel + hash cache
- ✅ Phase 9: ctypes loader
- ⬜ Work-stealing runner variant (Phase 5 — deferred)
- ⬜ DSL plumbing for `input_file` / `print_size` pragmas on `Relation(...)`
- ⬜ Pre-built runtime library distribution (currently: user builds via `xmake`)

## License

MIT. See `LICENSE`.
