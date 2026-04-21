# Compile performance

The JIT compile path isn't free — srdatalog emits a C++/CUDA tree
that drags in boost/hana + RMM + spdlog, then hands it to clang++
twice (host + device passes) per translation unit. Cold compiles of
doop on batik take ~100 s, which is in the same ballpark as the
upstream Nim reference but still the dominant part of the dev loop.
This page documents what we've tried, what works, and where the
remaining wall time is going.

## Headline numbers (doop on batik, 14 batch files, 16-core box)

| Configuration | Cold | Warm (ccache hit) |
|---|---|---|
| Original ThreadPoolExecutor, no ccache | ~97 s | ~97 s |
| **Current: ninja + ccache** | ~100 s | **~3 s** |
| Ninja, no ccache, `--cuda-host-only` for main.cpp | ~100 s | — |
| Ninja, no ccache, no `--cuda-host-only` | ~154 s | — |
| Unity build (one huge TU) | ~116 s | — |
| Sharded step bodies (75 TUs) without PCH | ~323 s | — |

## What the compiler is actually doing

`clang -ftime-trace` on a typical `jit_batch_N.cpp` reports (per
compile pass, and there are *two* passes per TU — host and device):

| Phase | Time |
|---|---|
| `Source` (parse srdatalog.h + transitively ~500 KB of boost/hana + RMM + spdlog) | ~5 – 7 s |
| `InstantiateClass` + `InstantiateFunction` (template work) | ~3 s |
| `Backend` / codegen | ~1 s |

That's ~10 s of frontend per pass, so ~20 s wall-clock per batch file
in isolation. With 14 batches + main.cpp and 16-way ninja
parallelism, the critical path works out to roughly what we see:
~100 s cold.

## Optimizations we've shipped

### 1. Ninja + ccache orchestrator

{py:mod}`srdatalog.codegen.jit.compiler_ninja` emits a `build.ninja`
in the cache dir and shells out to the ninja binary from the `ninja`
PyPI wheel. If `ccache` is on `$PATH`, it's automatically prepended
to the `cxx` variable — warm rebuilds after `rm -rf build/jit/` drop
from ~100 s to ~3 s.

Opt out with `SRDATALOG_JIT_NO_CCACHE=1` (per-process) or
`use_ccache=False` on {py:func}`srdatalog.codegen.jit.compiler_ninja.emit_build_ninja`.
Opt out of ninja itself with `SRDATALOG_JIT_NO_NINJA=1` — this falls
back to the ThreadPoolExecutor path in
{py:mod}`srdatalog.codegen.jit.compiler`.

### 2. `--cuda-host-only` for non-kernel TUs

clang's CUDA mode runs two full compile passes per TU. Only the
`jit_batch_*.cpp` files actually contain `__global__` kernels — the
main.cpp / step-body shards merely *call* them. Adding
`--cuda-host-only` to those host-only TUs skips the redundant device
pass, saving ~50 % of their compile time.

The rule split is in the ninja emit: `cxx_host_only` vs `cxx`.

### 3. ccache is implicit

Every call to {py:func}`srdatalog.codegen.jit.compile_jit_project`
goes through ninja by default, which picks up ccache automatically.
No user action needed beyond `apt install ccache`.

## What we tried that didn't work

### Precompiled headers (PCH)

The obvious win — parse `srdatalog.h` once, reuse its AST across all
15 TUs — is blocked by a clang-20 bug. See
[CUDA PCH blocker](cuda_pch_blocker) for the gory details. The
scaffold stays in `compiler_ninja.py` behind `use_pch=True` and will
start working when either clang ships a fix, or the runtime headers
are restructured to avoid pulling in `cuda_wrappers/new`
transitively.

### Unity builds

Concatenating every batch into one TU got us down to 1 source file
but back up to ~116 s wall — the template instantiation went serial
within the one TU, and that was more expensive than 14-way parallel
with 14 preamble parses. Unity is a win only when you *also* have
PCH to amortize the preamble.

### Sharding step bodies into separate TUs

Going the other direction — 60 tiny step-body `.cpp` files
parallelized on 16 cores — made things *worse* (~323 s) because each
of the 60 TUs paid the full srdatalog.h preamble cost. Classic case
where parallelism fails to compensate for re-parsed shared headers.

## Remaining levers

In rough order of ROI:

1. **Trim `srdatalog.h`**. A large fraction of the 5 – 7 s `Source`
   parse is boost/hana / RMM / spdlog transitives we don't all use
   per TU. A slim `srdatalog_jit.h` with only what JIT batches need
   would directly cut the preamble. Estimated ~30 – 50 s savings
   cold. Medium effort.

2. **Restructure `gpu/search.h`** to split host / device. Would
   unblock host-only PCH (currently fails because host TUs still
   pull in device-only `__popc`/`__ffs` intrinsics). Once PCH lands,
   another ~40 – 60 s savings. High effort (touches the runtime).

3. **Wait for clang to fix the PCH ODR bug**. No cost to us but
   entirely outside our control.

## Measuring yourself

```bash
# Cold compile with ccache disabled:
SRDATALOG_JIT_NO_CCACHE=1 python examples/run_benchmark.py doop \
    --data /path/to/batik_interned \
    --meta /path/to/batik_meta.json

# Per-TU breakdown with -ftime-trace:
# (add to CompilerConfig.cxx_flags=["-ftime-trace"])
python -c "
import json
for f in sorted(...):
    d = json.load(open(f))
    ...
"
```

The per-phase timings in `run_benchmark.py`'s output tell you
quickly which phase is eating time:

- If **emit** is slow, the program is pathologically large for the
  codegen (not the compiler).
- If **compile** is slow, it's the story above — preamble parsing
  dominates.
- If **load** is slow, your CSV I/O is the bottleneck.
- If **run** is slow, the GPU kernels are doing the work — see the
  per-step timings in the `Step N` lines the runner prints.
