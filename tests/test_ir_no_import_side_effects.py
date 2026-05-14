'''Pre-flight discipline test for S3A.8 (kill A7).

Per docs/stage3a_execution_plan.md §6.1: importing a module under
`ir/dialects/` must not mutate state in any *other* module. The
historical violation: importing `srdatalog.ir.dialects.relation.d2l`
triggered a side-effect import of `d2l/cuda.py`, which called
`register_index_plugin(two_level_plugin)` and mutated the codegen
plugin registry. After S3A.8 the registration is explicit
(`codegen.cuda.register_default_plugins()`); importing the dialect
alone does nothing to the codegen registry.

Bundle C (PluginRegistry encapsulation) replaced the bare
`_PLUGIN_REGISTRY` dict with a `PluginRegistry` class — these tests
now read state via `get_default_registry().registered_types()` /
`in get_default_registry()` instead of poking the private dict.

The test runs in a subprocess because Python caches modules — a
second import of d2l in the same process is a no-op even if the
first import had side effects. Subprocess gives a fresh interpreter
state.
'''

from __future__ import annotations

import subprocess
import sys


def _run_in_subprocess(code: str) -> tuple[int, str, str]:
  r = subprocess.run(
    [sys.executable, '-c', code],
    capture_output=True,
    text=True,
  )
  return r.returncode, r.stdout, r.stderr


def test_d2l_import_does_not_mutate_codegen_plugin_registry():
  '''Pin A7: importing srdatalog.ir.dialects.relation.d2l must not
  mutate srdatalog.ir.codegen.cuda.plugin._PLUGIN_REGISTRY.

  Pre-S3A.8 this would fail (d2l/__init__.py had `from . import cuda
  as _cuda` which side-effect-imported d2l/cuda.py, which ran
  `register_index_plugin(two_level_plugin)` at module load).
  '''
  code = """
import sys

# Snapshot before any d2l-related imports.
from srdatalog.ir.codegen.cuda.plugin import get_default_registry
before = get_default_registry().registered_types()

# Import the dialect.
import srdatalog.ir.dialects.relation.d2l  # noqa: F401

# Snapshot after.
after = get_default_registry().registered_types()

if before != after:
    added = set(after) - set(before)
    removed = set(before) - set(after)
    print(f'FAIL: d2l import mutated registry. added={added}, removed={removed}',
          file=sys.stderr)
    sys.exit(1)
print('OK')
"""
  rc, out, err = _run_in_subprocess(code)
  assert rc == 0, f'subprocess failed:\nstdout: {out}\nstderr: {err}'
  assert 'OK' in out


def test_d2l_cuda_module_import_does_not_mutate_registry():
  '''Per S3A.8, even directly importing d2l.cuda must not mutate the
  codegen registry. Registration is exposed as register_d2l_cuda_plugin()
  and only happens when called explicitly.
  '''
  code = """
import sys

from srdatalog.ir.codegen.cuda.plugin import get_default_registry
before = get_default_registry().registered_types()

import srdatalog.ir.dialects.relation.d2l.cuda  # noqa: F401

after = get_default_registry().registered_types()

if before != after:
    print(f'FAIL: importing d2l.cuda mutated registry. diff={set(after) ^ set(before)}',
          file=sys.stderr)
    sys.exit(1)
print('OK')
"""
  rc, out, err = _run_in_subprocess(code)
  assert rc == 0, f'subprocess failed:\nstdout: {out}\nstderr: {err}'
  assert 'OK' in out


def test_register_default_plugins_does_register_d2l():
  '''Inverse: when called explicitly, register_default_plugins() does
  add the D2L plugin to the registry.'''
  code = """
import sys

from srdatalog.ir.codegen.cuda.plugin import get_default_registry
import srdatalog.ir.dialects.relation.d2l.cuda  # safe per the test above

before = 'SRDatalog::GPU::Device2LevelIndex' in get_default_registry()
if before:
    print(f'FAIL: d2l already registered before explicit call', file=sys.stderr)
    sys.exit(1)

from srdatalog.ir.codegen.cuda import register_default_plugins
register_default_plugins()

if 'SRDatalog::GPU::Device2LevelIndex' not in get_default_registry():
    print(f'FAIL: register_default_plugins did not register d2l', file=sys.stderr)
    sys.exit(1)
print('OK')
"""
  rc, out, err = _run_in_subprocess(code)
  assert rc == 0, f'subprocess failed:\nstdout: {out}\nstderr: {err}'
  assert 'OK' in out


def test_compile_pipeline_triggers_default_plugin_registration():
  '''End-to-end smoke: production compile entry points implicitly
  register defaults via the explicit call inside their function body.
  Running compile_pipeline on a real DSL program leaves the registry
  populated.'''
  code = """
import sys

# Build a tiny program.
from srdatalog.dsl import Program, Relation, Var

X, Y, Z = Var('x'), Var('y'), Var('z')
edge = Relation('Edge', 2)
path = Relation('Path', 2)
prog = Program(rules=[
    (path(X, Y) <= edge(X, Y)).named('Base'),
    (path(X, Z) <= edge(X, Y) & path(Y, Z)).named('Rec'),
])

from srdatalog.ir.codegen.cuda.plugin import get_default_registry
before = 'SRDatalog::GPU::Device2LevelIndex' in get_default_registry()
if before:
    print('FAIL: d2l already registered before compile_program', file=sys.stderr)
    sys.exit(1)

from srdatalog.ir.pipeline import compile_program
# pipeline.compile_program → gen_complete_runner → ... → eventually
# api.compile_kernel_body which calls register_default_plugins().
compile_program(prog, 'TestProj')

if 'SRDatalog::GPU::Device2LevelIndex' not in get_default_registry():
    print('FAIL: d2l not registered after compile_program', file=sys.stderr)
    sys.exit(1)
print('OK')
"""
  rc, out, err = _run_in_subprocess(code)
  assert rc == 0, f'subprocess failed:\nstdout: {out}\nstderr: {err}'
  assert 'OK' in out
