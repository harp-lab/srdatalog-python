'''Integration helpers for HIR fixtures, MIR lifetimes, and JIT codegen.

MIR checks track live DELTA indexes and completed FULL indexes instead of
pinning the implementation's instruction ordering to Nim snapshots.
'''

import json
import re
from pathlib import Path

from srdatalog.dsl import Program
from srdatalog.ir.hir import compile_to_hir, compile_to_mir
from srdatalog.ir.hir.emit import hir_to_obj
from srdatalog.ir.hir.types import Version
from srdatalog.ir.mir import types as mir

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "integration"


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
        norm(golden).splitlines(),
        norm(actual).splitlines(),
        fromfile="nim",
        tofile="python",
        lineterm="",
        n=3,
      )
    )
    raise AssertionError(f"{fixture_stem} HIR mismatch:\n" + d[:4000])


def check_delta_lifetimes(
  instructions: list[mir.MirNode], *, recursive: bool
) -> tuple[set[tuple[str, tuple[int, ...]]], set[tuple[str, tuple[int, ...]]]]:
  '''Check DELTA dataflow, including every index read after ownership transfer.'''
  live: set[tuple[str, tuple[int, ...]]] = set()
  built: set[tuple[str, tuple[int, ...]]] = set()
  merged: set[tuple[str, tuple[int, ...]]] = set()
  finalized: set[str] = set()
  for op in instructions:
    if isinstance(op, mir.ComputeDeltaIndex):
      assert op.rel_name not in finalized, f"Repeated finalization of {op.rel_name}"
      finalized.add(op.rel_name)
      key = (op.rel_name, tuple(op.canonical_index))
      live.add(key)
      built.add(key)
    elif isinstance(op, mir.ClearRelation) and op.version is Version.DELTA:
      live = {key for key in live if key[0] != op.rel_name}
    elif isinstance(op, mir.RebuildIndexFromIndex) and op.version is Version.DELTA:
      source = (op.rel_name, tuple(op.source_index))
      assert source in live, f"Rebuilding from consumed DELTA {source}"
      target = (op.rel_name, tuple(op.target_index))
      live.add(target)
      built.add(target)
    elif isinstance(op, mir.MergeIndex):
      key = (op.rel_name, tuple(op.index))
      assert key in live, f"Merging consumed DELTA {key}"
      assert key not in merged, f"Repeated merge of {key}"
      assert not (recursive and op.consume_delta), f"Consuming recursive DELTA {key}"
      merged.add(key)
      if op.consume_delta:
        live.remove(key)
  if recursive:
    # These sources will be read on the next fixpoint iteration. The first
    # iteration dispatches DELTA through FULL, not the consumed seed DELTA.
    for op in instructions:
      pipelines = op.ops if isinstance(op, mir.ParallelGroup) else [op]
      for pipeline in pipelines:
        if isinstance(pipeline, mir.ExecutePipeline):
          for source in pipeline.source_specs:
            if source.version is Version.DELTA:
              key = (source.rel_name, tuple(source.index))
              assert key in live, f"Missing next-iteration DELTA {key}"
  else:
    assert not live, f"Nonrecursive DELTAs retained past last use: {live}"
    assert merged == built, f"Missing FULL indexes: {built - merged}"
  return live, merged


def check_mir_lifetimes(prog: Program) -> None:
  '''Compile real query plans and verify finalization and export lifetimes.'''
  hir = compile_to_hir(prog)
  strata = iter(hir.strata)
  full: set[tuple[str, tuple[int, ...]]] = set()
  for node, recursive in compile_to_mir(prog, hir=hir).steps:
    if isinstance(node, mir.FixpointPlan):
      stratum = next(strata)
      assert recursive == stratum.is_recursive
      live, merged = check_delta_lifetimes(node.instructions, recursive=recursive)
      modified = (
        stratum.scc_members
        if recursive
        else {head.rel for v in stratum.base_variants for head in v.original_rule.heads}
      )
      required = {
        (rel, tuple(index))
        for rel in modified
        for index in stratum.required_indices.get(rel, [])
      }
      if recursive:
        assert required <= live, f"Missing recursive DELTA indexes: {required - live}"
      else:
        assert required <= merged, f"Missing nonrecursive FULL indexes: {required - merged}"
      full.update(merged)
    elif isinstance(node, mir.PostStratumReconstructInternCols):
      key = (node.rel_name, tuple(node.canonical_index))
      assert key in full, f"Exporting missing canonical FULL {key}"
  assert next(strata, None) is None, "Missing stratum finalization"


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
      golden.splitlines(),
      actual.splitlines(),
      fromfile="nim",
      tofile="python",
      lineterm="",
      n=3,
    )
  )[:limit]
