'''Shared helpers for integration-program byte-match tests.

Each test_integration_<program>.py module builds a Python Program, compiles
it through the HIR + MIR pipelines, and byte-diffs the output against the
Nim golden in python/tests/fixtures/integration/.

JIT codegen tests byte-diff C++ output against Nim fixtures in
python/tests/fixtures/jit/<stem>/. Normalization strips whitespace so the
fixtures survive the repo's clang-format pre-commit hook.
'''
import json
import re
import sys
from pathlib import Path


from srdatalog.dsl import Program
from srdatalog.hir import compile_to_hir, compile_to_mir
from srdatalog.hir_emit import hir_to_obj
from srdatalog.mir.emit import print_mir_sexpr


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "integration"
JIT_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "jit"


def diff_hir(prog: Program, fixture_stem: str) -> None:
  hir = compile_to_hir(prog)
  actual = hir_to_obj(hir)
  golden = json.loads((FIXTURES / f"{fixture_stem}.hir.json").read_text())
  golden.pop("hirSExpr", None)
  norm = lambda d: json.dumps(d, indent=2, ensure_ascii=False)
  if norm(actual) != norm(golden):
    import difflib
    d = "\n".join(
      difflib.unified_diff(
        norm(golden).splitlines(), norm(actual).splitlines(),
        fromfile="nim", tofile="python", lineterm="", n=3,
      )
    )
    raise AssertionError(f"{fixture_stem} HIR mismatch:\n" + d[:4000])


def diff_mir(prog: Program, fixture_stem: str) -> None:
  mir_prog = compile_to_mir(prog)
  actual = print_mir_sexpr(mir_prog)
  golden = (FIXTURES / f"{fixture_stem}.mir.sexpr").read_text().rstrip("\n")
  if actual != golden:
    import difflib
    d = "\n".join(
      difflib.unified_diff(
        golden.splitlines(), actual.splitlines(),
        fromfile="nim", tofile="python", lineterm="", n=3,
      )
    )
    raise AssertionError(f"{fixture_stem} MIR mismatch:\n" + d[:4000])


# -----------------------------------------------------------------------------
# JIT C++ codegen byte-diff helpers
# -----------------------------------------------------------------------------

def _cpp_norm(s: str) -> str:
  '''Byte-match normalization that survives clang-format reformatting.

  1. Strip // line-comments entirely. Clang-format line-wraps long
     comments like `// MIR: (column-join :var y :sources ((...)) )` into
     two lines, leaving a spurious `// ` mid-content after collapse.
     The comments are informational — the C++ structure is the real
     signal — so dropping them is cleaner than trying to reassemble.
  2. Collapse runs of whitespace to a single space.
  3. Strip whitespace adjacent to structural punctuation `(`, `)`, `,`,
     `;`, `{`, `}`. Leaves `<`/`>` alone (template vs comparison).
  '''
  # Strip // comments to end-of-line.
  s = re.sub(r"//[^\n]*", "", s)
  # Collapse whitespace.
  s = re.sub(r"\s+", " ", s).strip()
  # Strip around structural punctuation, including `<`/`>`. Stripping
  # around angle brackets could theoretically collapse `x < y` into
  # `x<y`, but that's still an equivalent comparison for our "is the
  # emitted code structurally identical" check — we're not parsing,
  # just comparing. Worth it because clang-format routinely line-wraps
  # template args with `<\n    `, which otherwise leaves a space inside
  # the template that we'd need to track.
  for p in (r"\(", r"\)", r",", r";", r"\{", r"\}", r"<", r">"):
    s = re.sub(rf"\s*{p}\s*", p.replace("\\", ""), s)
  return s


def _unified_cpp_diff(golden: str, actual: str, label: str, limit: int = 4000) -> str:
  import difflib
  return "\n".join(
    difflib.unified_diff(
      golden.splitlines(), actual.splitlines(),
      fromfile="nim", tofile="python", lineterm="", n=3,
    )
  )[:limit]


def diff_orchestrator(fixture_stem: str, actual_cpp: str) -> None:
  '''Byte-diff the emitted orchestrator step bodies against the Nim golden
  in `fixtures/jit/<stem>/orchestrator.cpp`. Normalization collapses
  whitespace so clang-format doesn't fight us; semantic drift still fails.
  '''
  golden_path = JIT_FIXTURES / fixture_stem / "orchestrator.cpp"
  golden = golden_path.read_text()
  if _cpp_norm(actual_cpp) != _cpp_norm(golden):
    d = _unified_cpp_diff(golden, actual_cpp, fixture_stem)
    raise AssertionError(f"{fixture_stem} orchestrator mismatch:\n" + d)


def diff_jit_batch(fixture_stem: str, rule_name: str, actual_cpp: str) -> None:
  '''Byte-diff a JIT batch file against the Nim golden in
  `fixtures/jit/<stem>/jit_batch.<rule>.cpp`.
  '''
  golden_path = JIT_FIXTURES / fixture_stem / f"jit_batch.{rule_name}.cpp"
  golden = golden_path.read_text()
  if _cpp_norm(actual_cpp) != _cpp_norm(golden):
    d = _unified_cpp_diff(golden, actual_cpp, f"{fixture_stem}::{rule_name}")
    raise AssertionError(
      f"{fixture_stem} jit_batch.{rule_name} mismatch:\n" + d
    )


def diff_jit_runner(fixture_stem: str, rule_name: str, actual_cpp: str) -> None:
  '''Byte-diff a complete JIT runner struct against the Nim golden in
  `fixtures/jit/<stem>/jit_runner.<rule>.cpp`. The fixture is the `full`
  output of `jitCompleteRunner(ep, dbTypeName, relIndexTypes)` — the full
  `struct JitRunner_<rule>` with kernel defs + phase methods + execute().
  '''
  golden_path = JIT_FIXTURES / fixture_stem / f"jit_runner.{rule_name}.cpp"
  golden = golden_path.read_text()
  if _cpp_norm(actual_cpp) != _cpp_norm(golden):
    d = _unified_cpp_diff(golden, actual_cpp, f"{fixture_stem}::{rule_name}")
    raise AssertionError(
      f"{fixture_stem} jit_runner.{rule_name} mismatch:\n" + d
    )
