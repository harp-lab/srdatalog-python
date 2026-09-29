"""Export live two-level indexes without allocating their merged relation on the GPU."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_native_segmented_export_preserves_exact_logical_tuples(tmp_path):
  if os.environ.get("SRDATALOG_JIT_RUN_COMPILE_TESTS") != "1":
    pytest.skip("Opt in to native CUDA compilation with SRDATALOG_JIT_RUN_COMPILE_TESTS=1")

  from srdatalog import CompilerConfig
  from srdatalog.ir.codegen.cuda.build.compiler import compile_cpp, link_shared
  from srdatalog.ir.codegen.cuda.main_file import _gen_relation_export_helper
  from srdatalog.runtime import (
    cuda_compile_flags,
    cuda_include_paths,
    cuda_libs,
    cuda_link_flags,
    runtime_defines,
    runtime_include_paths,
  )

  (tmp_path / "generated_relation_export.h").write_text(_gen_relation_export_helper())
  config = CompilerConfig(
    include_paths=[str(tmp_path)] + runtime_include_paths() + cuda_include_paths(),
    defines=runtime_defines(),
    cxx_flags=cuda_compile_flags() + ["-fPIC"],
    link_flags=cuda_link_flags(),
    libs=cuda_libs() + ["boost_container"],
    shared=True,
  )
  source = Path(__file__).parent / "fixtures" / "device_segmented_export.cpp"
  obj, library = tmp_path / "export.o", tmp_path / "export.so"
  compiled = compile_cpp(str(source), str(obj), config)
  assert compiled.returncode == 0, compiled.stderr
  linked = link_shared([str(obj)], str(library), config)
  assert linked.returncode == 0, linked.stderr
  exported = tmp_path / "segmented.tsv"
  result = subprocess.run(
    [
      sys.executable,
      "-c",
      "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); "
      "fn=lib.run_device_segmented_export_regression; fn.argtypes=[ctypes.c_char_p]; "
      "sys.exit(fn(sys.argv[2].encode()))",
      str(library),
      str(exported),
    ],
    capture_output=True,
    text=True,
    timeout=120,
  )
  assert result.returncode == 0, result.stdout + result.stderr
  observed = [tuple(map(int, line.split("\t"))) for line in exported.read_text().splitlines()]
  ids = {2 * row for row in range(65539)} | {2 * row + 1 for row in range(65543)}
  expected = {(value - 70000, value % 7, 2 * value + 1) for value in ids}
  assert len(observed) == len(expected)
  assert set(observed) == expected
