# CUDA PCH blocker

Why PCH is off by default, and what it would take to turn it on.

## The symptom

Attempting to build a precompiled header for `srdatalog.h` and
consume it in a normal compile:

```bash
# Build host-only PCH
clang++ -std=c++23 -fPIC -x cuda --cuda-host-only \
  --cuda-path=/opt/cuda-12.9 \
  -I... -D... -Xclang -emit-pch \
  -c _pch_stub.cu -o srdatalog.pch

# Consume it in a trivial TU
echo 'int main() {return 0;}' > t.cu
clang++ ... --cuda-host-only -include-pch srdatalog.pch -c t.cu -o t.o
```

fails with:

```
cuda_wrappers/new:95:51: error: 'operator new' has different
  definitions in different modules; defined here first difference
  is 1st parameter with type 'unsigned long'
  95 | __device__ inline void *operator new(__SIZE_TYPE__, void *__ptr)
/usr/include/c++/13/new:174:57: note: but in '' found 1st parameter
  with type 'std::size_t' (aka 'unsigned long')
 174 | _GLIBCXX_NODISCARD inline void* operator new(std::size_t, ...)
```

Both `__SIZE_TYPE__` and `std::size_t` resolve to `unsigned long` on
x86_64, but clang-20's PCH module-ODR check compares the textual
declarations and rejects the mismatch.

## Why none of the obvious workarounds apply

| Workaround | Blocker |
|---|---|
| `-stdlib=libc++` (avoid libstdc++'s `<new>`) | NVIDIA's CUDA SDK `host_defines.h:67` explicitly rejects libc++: `"libc++ is not supported on x86 system"`. |
| Older clang (≤ 18) | Doesn't compile our C++23 runtime (`EnhancedID` constexpr literal-type error, fixed only in clang-19/20). |
| `gcc` | No `-x cuda` — can't compile device kernels. |
| `nvcc` | No C++23 support (we use `std::format`, concepts, `requires`). |
| `-Xclang -fno-validate-pch` / `-fallow-pch-with-compiler-errors` | ODR check is semantic, not "validation" — these flags don't disable it. |
| Split host + device PCH (`--cuda-host-only` + `--cuda-device-only` in parallel) | The device-side `-Xclang -emit-pch` still runs `ptxas` over the PCH bytes and fails: `ptxas fatal: Unexpected non-ASCII character encountered on line 1`. |

## What would unblock it

One of:

1. **Clang fixes the ODR regression.** The check was tightened in
   clang-20 and now false-positives on `cuda_wrappers/new` vs
   libstdc++'s `<new>`. Worth filing upstream.

2. **Restructure `gpu/search.h`** (and a few other runtime headers)
   to isolate host-only vs device-only code so that the *consumer*
   TU doesn't transitively pull in `cuda_wrappers/new`. Multi-day
   structural refactor but would unlock a ~40 – 60 s compile-time
   win for doop and should make future toolchain upgrades easier.

3. **Swap out the GCC libstdc++ for a CUDA-compatible alternative.**
   Not a realistic workaround on Linux with NVIDIA CUDA 12.x — every
   distribution ships libstdc++ and NVIDIA's headers assume it.

## The scaffold is still there

{py:func}`srdatalog.codegen.jit.compiler_ninja.emit_build_ninja`
accepts a `use_pch=True` kwarg that emits the full split host/device
PCH rules. It's off by default because it doesn't work today, but
the code path is preserved so someone (you, or clang, or a runtime
refactor) can flip it on and immediately measure the savings.

```python
from srdatalog.codegen.jit.compiler_ninja import emit_build_ninja
emit_build_ninja(project_result, config, use_pch=True)
```

If you try this and it works on your setup, please open an issue
with your clang / CUDA / libstdc++ versions — it probably means the
upstream bug is fixed.
