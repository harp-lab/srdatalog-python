'''Unsafe bitmap specializations must fail rather than discard rule semantics.'''

import pytest

from srdatalog.dsl import Filter, Program, Relation, Var
from srdatalog.ir.hir import compile_to_hir


def projection():
  join, value, destination = Var('join'), Var('value'), Var('destination')
  assign = Relation('Assign', 2)
  points = Relation('Points', 2)
  output = Relation('Output', 2)
  return join, value, destination, assign, points, output


@pytest.mark.parametrize('restriction', ['filter', 'negation', 'constant', 'retained_join'])
def test_bitmap_rejects_rules_that_are_not_unconstrained_projection(restriction):
  join, value, destination, assign, points, output = projection()
  if restriction == 'filter':
    rule = output(value, destination) <= assign(join, destination) & points(join, value) & Filter(
      ('value',), 'return value > 0;'
    )
  elif restriction == 'negation':
    excluded = Relation('Excluded', 2)
    rule = output(value, destination) <= (
      assign(join, destination) & points(join, value) & ~excluded(value, destination)
    )
  elif restriction == 'constant':
    rule = output(value, destination) <= assign(7, destination) & points(7, value)
  else:
    rule = output(join, destination) <= assign(join, destination) & points(join, value)
  with pytest.raises(ValueError):
    compile_to_hir(Program([rule.named('Projection').with_plan(dedup_bitmap=True)]))


def test_bitmap_rejects_provenance_instead_of_erasing_annotations():
  join, value, destination, assign, _, output = projection()
  points = Relation('AnnotatedPoints', 2, semiring='BooleanSR')
  rule = output(value, destination) <= assign(join, destination) & points(join, value)
  with pytest.raises(ValueError):
    compile_to_hir(Program([rule.with_plan(dedup_bitmap=True)]))


@pytest.mark.parametrize('mode', ['hash', 'count'])
def test_bitmap_rejects_conflicting_execution_modes(mode):
  join, value, destination, assign, points, output = projection()
  rule = output(value, destination) <= assign(join, destination) & points(join, value)
  if mode == 'hash':
    rule = rule.with_plan(dedup_bitmap=True, dedup_hash=True)
  else:
    rule = rule.with_count().with_plan(dedup_bitmap=True)
  with pytest.raises(ValueError):
    compile_to_hir(Program([rule]))


def test_bitmap_rejects_a_plan_that_matches_no_recursive_variant():
  join, value, destination, assign, _, output = projection()
  seed = Relation('Seed', 2)
  rule = output(value, destination) <= assign(join, destination) & output(value, join)
  with pytest.raises(ValueError):
    compile_to_hir(
      Program(
        [
          (output(value, destination) <= seed(value, destination)).named('Base'),
          rule.named('Step').with_plan(dedup_bitmap=True),
        ]
      )
    )
