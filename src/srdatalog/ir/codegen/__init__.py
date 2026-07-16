'''Codegen backends — IIR Render registries.

Each subpackage is one codegen that consumes IIR ops and produces
target-specific text. Codegens are not dialects: they have no ops,
no lowerings, no rewrites. They are renderers + target configuration.

See docs/stage3a_execution_plan.md §3 for the file layout contract.

Currently registered:
  - cuda: NVIDIA CUDA C++ emission.

Planned (per docs/stage3a_execution_plan.md §10 deferrals):
  - cpp_tbb: CPU C++ with TBB parallel-for.
  - cpp_omp: CPU C++ with OpenMP.
  - hip:     AMD HIP.
  - metal:   Apple Metal shading language.
  - sycl:    SYCL kernels.
'''
