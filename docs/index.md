# srdatalog

Python frontend for **Super-Reconfigurable Datalog** — a column-oriented,
GPU-targeted Datalog engine. You write rules in a small Python DSL;
the library lowers them through HIR → MIR → C++/CUDA, compiles the
resulting tree via ninja + clang++, loads the shared library, and
hands you back a `ctypes` interface to call `init / load / run / size
/ shutdown`.

```python
from srdatalog import Var, Relation, Program, build_project, compile_jit_project, CompilerConfig
from srdatalog.runtime import (runtime_include_paths, cuda_include_paths,
                               cuda_compile_flags, cuda_link_flags,
                               cuda_libs, runtime_defines)
import ctypes

x, y, z = Var("x"), Var("y"), Var("z")
Edge = Relation("Edge", 2, input_file="Edge.csv")
Path = Relation("Path", 2, print_size=True)
prog = Program(
  rules=[
    (Path(x, y) <= Edge(x, y)).named("TCBase"),
    (Path(x, z) <= Path(x, y) & Edge(y, z)).named("TCRec"),
  ],
)

project = build_project(prog, project_name="TC", cache_base="./build")
cfg = CompilerConfig(
  include_paths=runtime_include_paths() + cuda_include_paths(),
  defines=runtime_defines(),
  cxx_flags=cuda_compile_flags() + ["-fPIC"],
  link_flags=cuda_link_flags(),
  libs=cuda_libs() + ["boost_container"],
  shared=True,
)
build = compile_jit_project(project, cfg)
lib = ctypes.CDLL(build.artifact, mode=ctypes.RTLD_GLOBAL)
lib.srdatalog_init()
lib.srdatalog_load_all(b"/path/to/data")
lib.srdatalog_run(0)
```

## Docs

```{toctree}
:maxdepth: 2
:caption: Guides

getting_started
architecture
benchmarks
compile_performance
cuda_pch_blocker
```

```{toctree}
:maxdepth: 1
:caption: API reference

api/index
```

## Project status

- **17 benchmarks** auto-migrated from the upstream Nim reference (doop,
  andersen, galen, polonius, ddisasm, reg_scc, tc, triangle, sg, cspa,
  crdt, and the 6 LSQB triangle variants). See [Benchmarks](benchmarks).
- **125 / 127** runner fixtures byte-match Nim's emitted C++.
- **Ninja + ccache** backend; cold compile of doop on batik is ~100s,
  warm is ~3s.
- **PCH is currently blocked** by a clang-20 ODR-check regression — see
  [CUDA PCH blocker](cuda_pch_blocker).

## Source

Hosted at <https://github.com/harp-lab/srdatalog-python>. Issues and
pull requests welcome.
