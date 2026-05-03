'''Target-emission dialects.

Each subpackage is one codegen backend that lowers IIR ops to a
specific C++ flavor. Adding a new hardware target = adding a sibling
subpackage; existing dialects don't change.

Currently registered:
  - cuda: NVIDIA CUDA via cooperative_groups, atomicCAS, warp tiles.

Planned:
  - hip: AMD HIP (already partially supported via vendored shim).
  - cpp_tbb: CPU C++ with TBB parallel-for (Stage 3).
  - cpp_omp: CPU C++ with OpenMP.
  - metal: Apple Metal shading language.
  - sycl: SYCL kernels.
'''
