'''HIR + MIR dialect-catalog registration tests.

Verifies that `srdatalog.ir.hir` and `srdatalog.ir.mir` expose `DIALECT`
catalog objects matching the framework's `srdatalog.ir.core.Dialect`
contract, and that they register cleanly with a fresh `Compiler`.

This is the first step of the HIR/MIR standardization (issue tracked
in docs/milestones.md). The HIR/MIR types themselves are not yet
`Op`-subclassed or `frozen+slots` — that follow-up step requires
refactoring the ~15 mutation sites that currently mutate handle_start
/ concurrent_write / etc. on MIR nodes.

Today's contract: each dialect catalogs its types/ops by class
reference; pattern-matching/strategy/verifier hooks land later.
'''

from __future__ import annotations

import pytest

from srdatalog.ir.core import Compiler, Dialect


def test_hir_dialect_exists():
  from srdatalog.ir.hir import DIALECT

  assert isinstance(DIALECT, Dialect)
  assert DIALECT.name == 'hir'
  # HIR catalogs its core types (HirProgram, HirStratum, HirRuleVariant,
  # RelationDecl, AccessPattern, Version). These are not Op-subclassed
  # yet — they're catalogued under `types` for now.
  assert len(DIALECT.types) >= 6
  type_names = {t.__name__ for t in DIALECT.types}
  for expected in (
    'AccessPattern',
    'HirProgram',
    'HirRuleVariant',
    'HirStratum',
    'RelationDecl',
    'Version',
  ):
    assert expected in type_names, f'HIR DIALECT missing {expected}'


def test_mir_dialect_exists():
  from srdatalog.ir.mir import DIALECT

  assert isinstance(DIALECT, Dialect)
  assert DIALECT.name == 'mir'
  # MIR catalogs every op kind in `types.py`. Today these are mutable
  # @dataclass — the standardization is to bring them under the framework
  # registry. Frozen+slots+Op-subclass migration is a follow-up.
  assert len(DIALECT.ops) >= 25
  op_names = {o.__name__ for o in DIALECT.ops}
  # Spot-check the load-bearing op kinds that the dialect-emit relies on.
  for expected in (
    'CartesianJoin',
    'ColumnJoin',
    'ColumnSource',
    'ExecutePipeline',
    'FixpointPlan',
    'InsertInto',
    'Negation',
    'Program',
    'Scan',
  ):
    assert expected in op_names, f'MIR DIALECT missing {expected}'


def test_hir_and_mir_register_with_compiler():
  '''Both dialects must register cleanly with a fresh Compiler — same
  contract as `relation.sorted_array`, `iir.cf`, `target.cuda`.'''
  from srdatalog.ir.hir import DIALECT as HIR_DIALECT
  from srdatalog.ir.mir import DIALECT as MIR_DIALECT

  compiler = Compiler()
  compiler.register_dialect(HIR_DIALECT)
  compiler.register_dialect(MIR_DIALECT)

  names = {d.name for d in compiler.dialects}
  assert names == {'hir', 'mir'}

  # Idempotent registration is forbidden — re-registering raises.
  with pytest.raises(ValueError, match=r"already registered"):
    compiler.register_dialect(HIR_DIALECT)
