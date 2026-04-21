# Getting started

## Install

Contributors (editable checkout):

```bash
git clone https://github.com/harp-lab/srdatalog-python.git
cd srdatalog-python
uv sync --group dev
uv run python scripts/populate_vendor.py   # fetch boost / highway / RMM / spdlog (~3 min, one-time)
uv pip install -e .
```

End users (PyPI wheel, once published — vendor is already bundled):

```bash
uv pip install srdatalog
```

## Toolchain requirements at JIT compile time

The library emits C++23 + CUDA code, so the backend compiler
constraints are tighter than a vanilla C++ project:

| Requirement | Why |
|---|---|
| **clang++ ≥ 20** (mandatory) | `gcc` can't compile CUDA kernels (no `-x cuda`); `nvcc` doesn't support C++23; clang-18/19 reject some of our `constexpr` / literal-type usage. clang-20 is the first version that compiles the runtime cleanly. |
| **CUDA 12.x toolkit** | CUDA 13 dropped `thrust/optional.h`, which our pinned RMM still needs. The auto-detector in {py:func}`srdatalog.runtime.find_cuda_root` prefers a 12.x install when both are present. |
| **`libboost_container`** on the link path | The runtime's PMR allocators reference `boost::container::pmr::get_default_resource` — a *definition*, not header-only, and not shipped in the vendored boost subset. `apt install libboost-container-dev` or `brew install boost`. |
| `ccache` (optional) | Auto-detected; near-instant warm rebuilds. Opt out with `SRDATALOG_JIT_NO_CCACHE=1`. |
| `ninja` (optional) | Auto-installed via the `ninja` PyPI wheel, so no system install needed. |

The library auto-detects clang++ via `$CXX` or `$PATH`, and CUDA via
`$CUDA_HOME` or the usual install paths.

## The pipeline

Every program flows through five phases:

```
Python DSL (dsl.py)
  → HIR                     (srdatalog.hir)
  → MIR                     (srdatalog.mir)
  → C++/CUDA .cpp tree      (srdatalog.codegen.jit.*)
  → shared library via ninja + clang++   (srdatalog.codegen.jit.compiler_ninja)
  → dlopen + extern "C" shim             (user-side ctypes)
```

{py:func}`srdatalog.build.build_project` covers phases 1 – 3 in one call,
and {py:func}`srdatalog.compile_jit_project` covers phase 4. Phase 5 is
a stock `ctypes.CDLL`.

See [Architecture](architecture) for a deeper tour of what each phase
does.

## End-to-end example

The canonical "run any benchmark" script:

```bash
# Synthetic triangle count (no input CSVs needed)
python examples/run_benchmark.py triangle

# Transitive closure on a CSV directory
python examples/run_benchmark.py tc --data /path/to/edges

# Doop points-to on batik (defaults baked into doop_run.py)
python examples/doop_run.py

# Cap iterations for a quick sanity check
python examples/run_benchmark.py galen --data /path/to/data --max-iter 3
```

`run_benchmark.py` prints one line per phase (DSL build, emit, ninja
compile, load, run, size readback) with timings, so you can see where
time is going on your box.

## Writing your own benchmark

```python
from srdatalog import Var, Relation, Program

x, y, z = Var("x"), Var("y"), Var("z")
Edge = Relation("Edge", 2, input_file="Edge.csv")
TC   = Relation("TC", 2, print_size=True)

prog = Program(
  relations=[Edge, TC],
  rules=[
    (TC(x, y) <= Edge(x, y)).named("TC_Base"),
    (TC(x, z) <= TC(x, y) & Edge(y, z)).named("TC_Rec"),
  ],
)
```

Relation pragmas you'll actually use:
- `input_file="X.csv"` — included in the auto-generated `load_all` shim.
- `print_size=True` — runner emits a size readback after the fixpoint.
- `index_type="SRDatalog::GPU::Device2LevelIndex"` — override the
  default LSM index (needed for very fat relations like `VarPointsTo`).

Rule builders: `atom <= body1 & body2 & ~neg & Filter((v,), "...")`.
See the {py:mod}`srdatalog.dsl` API reference for every operator.

## Translating from Nim

The upstream Nim reference has a few dozen benchmark programs under
`integration_tests/examples/`. `tools/nim_to_dsl.py` auto-translates
them:

```bash
python tools/nim_to_dsl.py path/to/upstream.nim --out examples/upstream.py
```

It handles schema blocks, rules blocks, dataset_consts, all the
negation and filter shorthands, plan pragmas, and the `split` marker.
See the [Benchmarks](benchmarks) page for the full list.
