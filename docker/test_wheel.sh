#!/usr/bin/env bash
# Verify that `uv build` produces a self-contained, installable,
# JIT-functional wheel for srdatalog.
#
# Procedure:
#   1. uv build                         — produces sdist + wheel in dist/
#   2. Inspect wheel contents           — assert vendor/ + generalized_datalog/
#                                         are present (the force-include
#                                         declared in pyproject.toml).
#   3. Install wheel into a fresh venv  — no dev group, no source tree
#                                         on sys.path, no vendor on disk.
#   4. Import smoke test                — `import srdatalog` + `srdatalog info`
#   5. Emit smoke test                  — `srdatalog emit examples/triangle.py`
#                                         (header-only, no compiler needed).
#   6. JIT compile smoke test           — `srdatalog compile examples/triangle.py`
#                                         driving clang++ against the bundled
#                                         headers. This is the real
#                                         "self-contained" proof: the wheel
#                                         alone (no source checkout) can
#                                         round-trip Python → .so.
#
# Usage:
#   bash docker/test_wheel.sh                    # all of the above
#   bash docker/test_wheel.sh --skip-compile     # steps 1-5 only (no clang)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

SKIP_COMPILE=0
for arg in "$@"; do
  case "$arg" in
    --skip-compile) SKIP_COMPILE=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

say() { printf "\n\033[1;36m[test_wheel]\033[0m %s\n" "$*"; }
die() { printf "\n\033[1;31m[test_wheel FAIL]\033[0m %s\n" "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Build
# ---------------------------------------------------------------------------
say "step 1/6 — uv build"
rm -rf dist/
uv build

WHEEL="$(ls dist/*.whl | head -1)"
SDIST="$(ls dist/*.tar.gz | head -1)"
[[ -f "${WHEEL}" ]] || die "no wheel produced in dist/"
[[ -f "${SDIST}" ]] || die "no sdist produced in dist/"
say "  wheel: ${WHEEL} ($(du -h "${WHEEL}" | cut -f1))"
say "  sdist: ${SDIST} ($(du -h "${SDIST}" | cut -f1))"

# ---------------------------------------------------------------------------
# 2. Wheel contents
# ---------------------------------------------------------------------------
say "step 2/6 — inspect wheel contents"
WHEEL_LISTING="$(unzip -Z1 "${WHEEL}")"
check_in_wheel() {
  local pat="$1" desc="$2"
  if ! grep -q "${pat}" <<<"${WHEEL_LISTING}"; then
    die "wheel missing ${desc} (pattern: ${pat})"
  fi
  printf "  ok %-40s (%s files)\n" "${desc}" \
    "$(grep -c "${pat}" <<<"${WHEEL_LISTING}")"
}
check_in_wheel "srdatalog/runtime/generalized_datalog/" "runtime/generalized_datalog/"
check_in_wheel "srdatalog/runtime/vendor/boost/"        "vendor/boost/"
check_in_wheel "srdatalog/runtime/vendor/highway/"      "vendor/highway/"
check_in_wheel "srdatalog/runtime/vendor/rmm/"          "vendor/rmm/"
check_in_wheel "srdatalog/runtime/vendor/spdlog/"       "vendor/spdlog/"

# ---------------------------------------------------------------------------
# 3. Fresh venv install (no source tree, no dev deps)
# ---------------------------------------------------------------------------
say "step 3/6 — install into fresh venv"
VENV="$(mktemp -d)/venv"
uv venv "${VENV}" --python 3.12
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
uv pip install "${WHEEL}"

# Physically leave the source tree so nothing on sys.path shadows the install.
TESTDIR="$(mktemp -d)"
cp examples/triangle.py "${TESTDIR}/triangle.py"
cd "${TESTDIR}"

# ---------------------------------------------------------------------------
# 4. Import
# ---------------------------------------------------------------------------
say "step 4/6 — import + srdatalog info"
python -c "import srdatalog; print('srdatalog', srdatalog.__version__)"
srdatalog info

# ---------------------------------------------------------------------------
# 5. Emit (no compiler required)
# ---------------------------------------------------------------------------
say "step 5/6 — srdatalog emit (codegen only)"
srdatalog emit triangle.py --project TrianglePlan --cache-base ./jit
ls -la ./jit/*/ || true

# ---------------------------------------------------------------------------
# 6. JIT compile — the real self-containment test
#
# The CLI's `srdatalog compile` only wires `runtime_include_path()` (the
# generalized_datalog/ subdir). A real compile needs vendor paths + CUDA
# too, both of which the wheel must surface via `runtime_include_paths()`
# and `cuda_include_paths()`. We drive that programmatically so a failure
# here tells us exactly which piece the wheel is missing.
# ---------------------------------------------------------------------------
if [[ "${SKIP_COMPILE}" == "1" ]]; then
  say "step 6/6 — skipped (--skip-compile)"
else
  say "step 6/6 — programmatic JIT compile against bundled headers + CUDA"
  python - <<'PY'
import sys
from pathlib import Path

from srdatalog import (
    Program, Relation, Var,
    build_project,
    CompilerConfig, compile_jit_project,
)
from srdatalog.runtime import (
    runtime_include_paths, cuda_include_paths,
    cuda_compile_flags, runtime_defines, runtime_undefines,
    has_vendored_deps, find_cuda_root,
)

if not has_vendored_deps():
    sys.exit("FAIL: wheel does not ship vendor/ — populate hook didn't run at build")
if find_cuda_root() is None:
    sys.exit("FAIL: no CUDA toolkit visible from container — nvcc missing?")

x, y, z = Var("x"), Var("y"), Var("z")
h, f = Var("h"), Var("f")
R = Relation("RRel", 2, column_types=(int, int))
S = Relation("SRel", 3, column_types=(int, int, int))
T = Relation("TRel", 3, column_types=(int, int, int))
Z = Relation("ZRel", 3, column_types=(int, int, int))
prog = Program(
    relations=[R, S, T, Z],
    rules=[
        (Z(x, y, z) <= R(x, y) & S(y, z, h) & T(z, x, f)).named("Triangle"),
    ],
)
result = build_project(prog, project_name="TrianglePlan", cache_base="./jit")
print(f"emitted {len(result['batches'])} batch(es) in {result['dir']}")

cfg = CompilerConfig(
    cxx="clang++",
    include_paths=runtime_include_paths() + cuda_include_paths(),
    defines=runtime_defines(),
    cxx_flags=cuda_compile_flags(),
)
build = compile_jit_project(result, cfg)
if not build.ok():
    for r in build.compile_results:
        if r.returncode != 0:
            print(f"[FAIL compile {r.output}]\n{r.stderr[:2000]}", file=sys.stderr)
    if build.link_result and build.link_result.returncode != 0:
        print(f"[FAIL link]\n{build.link_result.stderr[:2000]}", file=sys.stderr)
    sys.exit(1)
art = Path(build.artifact)
print(f"built {art} ({art.stat().st_size // 1024} KB, {build.elapsed_sec:.1f}s)")
PY
fi

deactivate
cd "${REPO_ROOT}"

say "PASS — wheel is self-contained and installable."
