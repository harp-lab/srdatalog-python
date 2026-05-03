'''target.cuda.build — CUDA build / loader / cache subsystem.

Wraps NVCC + Ninja to compile generated `.cu` source into a shared
library, caches compiled artifacts by content-hash, and loads them
back via `ctypes` so the host Python process can call into the
generated kernels. Pure plumbing — no IR knowledge.
'''
