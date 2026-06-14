import srdatalog.mir.types as m
from srdatalog.hir.types import Version
from srdatalog.mir.emit import print_mir_sexpr


def test_negation_without_consts_stays_legacy_four_field_form():
  node = m.Negation(
    rel_name="MethodImplemented",
    version=Version.FULL,
    index=[0, 1, 2, 3],
    prefix_vars=["simplename", "descriptor", "mtype"],
  )
  assert print_mir_sexpr(node) == (
    "(negation MethodImplemented full (0 1 2 3) (simplename descriptor mtype))"
  )


def test_negation_with_consts_emits_fifth_const_args_field():
  node = m.Negation(
    rel_name="Method_Modifier",
    version=Version.FULL,
    index=[0, 1],
    prefix_vars=["meth"],
    const_args=[(0, 2161502)],
  )
  assert print_mir_sexpr(node) == (
    "(negation Method_Modifier full (0 1) (meth) ((0 2161502)))"
  )
