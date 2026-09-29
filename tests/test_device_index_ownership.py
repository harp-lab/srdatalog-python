"""Native index builds preserve tuples/provenance and bound storage by owned data and root keys."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
  ("fixture", "entrypoint"),
  [
    ("device_index_ownership.cpp", "run_device_index_ownership_regression"),
    ("device_storage_release.cpp", "run_device_storage_release_regression"),
  ],
)
def test_device_index_consuming_build(tmp_path, fixture, entrypoint):
  if os.environ.get("SRDATALOG_JIT_RUN_COMPILE_TESTS") != "1":
    pytest.skip("Opt in to native CUDA compilation with SRDATALOG_JIT_RUN_COMPILE_TESTS=1")

  from srdatalog import CompilerConfig
  from srdatalog.ir.codegen.cuda.build.compiler import compile_cpp, link_shared
  from srdatalog.runtime import (
    cuda_compile_flags,
    cuda_include_paths,
    cuda_libs,
    cuda_link_flags,
    runtime_defines,
    runtime_include_paths,
  )

  config = CompilerConfig(
    include_paths=runtime_include_paths() + cuda_include_paths(),
    defines=runtime_defines(),
    cxx_flags=cuda_compile_flags() + ["-fPIC"],
    link_flags=cuda_link_flags(),
    libs=cuda_libs() + ["boost_container"],
    shared=True,
  )
  source = Path(__file__).parent / "fixtures" / fixture
  obj = tmp_path / "ownership.o"
  library = tmp_path / "ownership.so"
  compiled = compile_cpp(str(source), str(obj), config)
  assert compiled.returncode == 0, compiled.stderr
  linked = link_shared([str(obj)], str(library), config)
  assert linked.returncode == 0, linked.stderr
  result = subprocess.run(
    [
      sys.executable,
      "-c",
      "import ctypes,sys; lib=ctypes.CDLL(sys.argv[1]); "
      "sys.exit(getattr(lib, sys.argv[2])())",
      str(library),
      entrypoint,
    ],
    capture_output=True,
    text=True,
    timeout=120,
  )
  assert result.returncode == 0, result.stdout + result.stderr
