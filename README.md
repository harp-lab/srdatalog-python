# srdatalog

Python frontend for **Super-Reconfigurable Datalog** — a column-oriented Datalog engine
with JIT C++/CUDA code generation. The library takes a Python DSL program, compiles it
through HIR → MIR → GPU-targeted C++, writes the resulting `.cpp` tree to a cache, and
(optionally) builds a loadable shared library you can call from Python.

## Install

```bash
# Contributors (editable):
uv sync --group dev
uv run python scripts/populate_vendor.py   # fetch boost/highway/RMM/spdlog (~3 min, one-time)
uv pip install -e .

# End users (from PyPI — vendor is already bundled in the wheel):
uv pip install srdatalog
```

Wheels bundle:
- `generalized_datalog/` — this project's C++ runtime headers (~1.5 MB)
- `vendor/{boost,highway,rmm,spdlog}/` — third-party headers, fetched at
  wheel-build time by `scripts/populate_vendor.py` (gitignored in source).

Source distributions do **not** ship vendored deps — contributors fetch them
locally (see above) so the git repo stays small.

A C++ toolchain (`clang++` / `g++`, and a CUDA toolkit for GPU targets) is
required at **compile time** — the library auto-detects it via `$CXX`,
`$CUDA_HOME`, or standard install paths.

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

## Docker / RunPod

A self-contained image (CUDA 12.9 + clang-20 + uv + Python 3.12 + RunPod
infra: SSH, Jupyter, nginx proxy) lives under [docker/](docker/).
Same image serves two purposes — a RunPod dev pod, and a sandbox for
verifying the published wheel is actually self-contained.

```bash
# Build (via docker buildx bake — reads docker-bake.hcl)
docker buildx bake

# Run as a RunPod-style dev pod
docker run --gpus all -p 22:22 -p 8888:8888 \
  -e PUBLIC_KEY="$(cat ~/.ssh/id_ed25519.pub)" \
  -e JUPYTER_PASSWORD=changeme \
  stargazermiao/srdatalog-python-runpod:latest

# Verify wheel is self-contained + publishable
docker run --gpus all --rm stargazermiao/srdatalog-python-runpod:latest \
  bash docker/test_wheel.sh
```

`test_wheel.sh` runs `uv build`, installs the produced wheel into a fresh
venv (source tree not on sys.path), and drives a clang++ JIT compile of a
triangle-count program against *only* the headers the wheel shipped. If
that succeeds, the wheel is publishable.

## Status & roadmap

- ✅ Phase 1-4: JIT codegen byte-match (125/127 fixtures)
- ✅ Phase 6: `_Runner` struct + main-file emission
- ✅ Phase 7: on-disk cache manager
- ✅ Phase 8: compiler wrapper with parallel + hash cache
- ✅ Phase 9: ctypes loader
- ⬜ Work-stealing runner variant (Phase 5 — deferred)
- ⬜ DSL plumbing for `input_file` / `print_size` pragmas on `Relation(...)`
- ⬜ Pre-built runtime library distribution

## License

MIT. See `LICENSE`.
