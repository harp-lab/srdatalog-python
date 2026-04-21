'''JIT codegen byte-match test on gen_jit_code — the simplest integration
program (1 rule: Triangle, non-recursive).

Goal-line test: this fails today because the Python JIT codegen is a
skeleton (`_generate_body` returns `"unimplemented\\n"`). It goes green
as the jit_instructions / jit_root / jit_complete_runner ports land.

Run directly:
  python3 tests/test_jit_codegen_gen_jit_code.py
'''

import sys

from integration_helpers import diff_jit_batch, diff_orchestrator
from test_integration_gen_jit_code import build_gen_jit_code

from srdatalog.codegen.batchfile import _collect_pipelines
from srdatalog.codegen.jit.file import gen_jit_file_content_from_execute_pipeline
from srdatalog.codegen.jit.orchestrator_jit import gen_step_body
from srdatalog.hir import compile_to_mir


def _python_orchestrator(prog, db_type_name: str = "TriangleRules_DB_DeviceDB") -> str:
  '''Emit concatenated step_N bodies — matches what Nim's genStepBody
  dumps into fixture `orchestrator.cpp`. Uses the new Nim-faithful JIT
  orchestrator (codegen.jit.orchestrator_jit), not mhk's legacy
  codegen.orchestrator.
  '''
  mir = compile_to_mir(prog)
  parts = [
    gen_step_body(node, db_type_name, is_rec, i) for i, (node, is_rec) in enumerate(mir.steps)
  ]
  return "".join(parts)


def _python_jit_batch(prog, rule_name: str) -> str:
  '''Emit the full JIT batch .cu file contents (preamble + kernel functor
  + footer) for one pipeline via the Nim-faithful jit_file path. Matches
  what Nim's genJitFileContentFromExecutePipeline writes to fixture.
  '''
  mir = compile_to_mir(prog)
  for ep in _collect_pipelines(mir):
    if ep.rule_name == rule_name:
      return gen_jit_file_content_from_execute_pipeline(ep)
  raise ValueError(f"no pipeline named {rule_name!r} in gen_jit_code")


def test_gen_jit_code_orchestrator_byte_match():
  '''Step-body C++ matches Nim's genStepBody output.'''
  actual = _python_orchestrator(build_gen_jit_code())
  diff_orchestrator("gen_jit_code", actual)


def test_gen_jit_code_jit_batch_triangle_byte_match():
  '''Complete jit_batch.Triangle.cpp byte-match: preamble + kernel functor
  + footer. Matches what Nim's genJitFileContentFromExecutePipeline
  writes to the fixture.'''
  actual = _python_jit_batch(build_gen_jit_code(), "Triangle")
  diff_jit_batch("gen_jit_code", "Triangle", actual)


if __name__ == "__main__":
  import inspect

  this = sys.modules[__name__]
  passed = 0
  failed = 0
  for name, fn in inspect.getmembers(this, inspect.isfunction):
    if not name.startswith("test_"):
      continue
    try:
      fn()
      print(f"OK  {name}")
      passed += 1
    except AssertionError as e:
      print(f"FAIL {name}")
      print(str(e)[:2000])
      failed += 1
  print(f"\n{passed} pass / {failed} fail")
