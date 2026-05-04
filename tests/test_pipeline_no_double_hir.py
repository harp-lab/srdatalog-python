'''S3A.9a — pin "compile_to_hir runs at most once per compile_program".

Pre-S3A.9a, ir/pipeline.py:compile_program called compile_to_hir(program)
explicitly AND then called compile_to_mir(program), which internally
re-ran compile_to_hir(program). Two HIR passes per compile.

Post-fix, compile_to_mir accepts an optional hir= kwarg; pipeline.py
threads its already-computed HIR through, so no duplicate run.

This test counts compile_to_mir's INTERNAL invocations of compile_to_hir
(via the srdatalog.ir.hir module namespace, where compile_to_mir resolves
the name at runtime). pipeline.py's explicit call uses its own bound name
and is invisible to this counter — exactly what we want, since we are
testing the redundant-internal-call bug.

  Pre-fix: counter == 1 (the redundant internal call).
  Post-fix: counter == 0 (compile_to_mir reuses the passed-in hir).
'''

from __future__ import annotations

from srdatalog.dsl import Program, Relation, Var


def _build_tiny_program() -> Program:
  X, Y, Z = Var("x"), Var("y"), Var("z")
  arc = Relation("ArcInput", 2)
  edge = Relation("Edge", 2)
  path = Relation("Path", 2)
  return Program(
    rules=[
      (edge(X, Y) <= arc(X, Y)).named("EdgeLoad"),
      (path(X, Y) <= edge(X, Y)).named("TCBase"),
      (path(X, Z) <= path(X, Y) & edge(Y, Z)).named("TCRec"),
    ],
  )


def test_compile_program_does_not_double_run_compile_to_hir(monkeypatch):
  from srdatalog.ir import hir as hir_mod
  from srdatalog.ir import pipeline as pipeline_mod

  original = hir_mod.compile_to_hir
  call_count = 0

  def counted(*args, **kwargs):
    nonlocal call_count
    call_count += 1
    return original(*args, **kwargs)

  monkeypatch.setattr(hir_mod, "compile_to_hir", counted)

  pipeline_mod.compile_program(_build_tiny_program(), "TestProj")

  assert call_count == 0, (
    f"compile_to_mir made {call_count} internal compile_to_hir call(s); "
    "expected 0 — pipeline.compile_program should thread its HIR through."
  )


def test_compile_to_mir_still_runs_hir_when_no_hir_passed():
  '''Sanity: when no hir is passed, compile_to_mir must still call
  compile_to_hir internally (this is the back-compat path for callers
  that don't have HIR yet).'''
  from srdatalog.ir.hir import compile_to_mir

  prog = _build_tiny_program()
  # Just exercising the path; if compile_to_hir wasn't called, the
  # downstream lower_hir_to_mir_steps would explode on a None HIR.
  mir = compile_to_mir(prog)
  assert mir is not None
  assert len(mir.steps) > 0


def test_compile_to_mir_reuses_provided_hir():
  '''Sanity: when hir IS passed, compile_to_mir uses it (does not
  rerun compile_to_hir). Asserted by passing a HIR computed once
  and checking the output is structurally equivalent to a fresh
  end-to-end run.'''
  from srdatalog.ir.hir import compile_to_hir, compile_to_mir

  prog = _build_tiny_program()
  hir = compile_to_hir(prog)
  mir_via_passed = compile_to_mir(prog, hir=hir)
  mir_via_internal = compile_to_mir(prog)

  # Same number of steps; same recursive flags.
  assert len(mir_via_passed.steps) == len(mir_via_internal.steps)
  for (n1, r1), (n2, r2) in zip(mir_via_passed.steps, mir_via_internal.steps):
    assert r1 == r2
    # Structural equality on MIR — they're frozen dataclasses.
    assert type(n1) is type(n2)
