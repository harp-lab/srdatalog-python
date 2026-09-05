import srdatalog.ir.mir.types as m
from srdatalog.ir.hir.types import Version
from srdatalog.ir.mir.print import print_mir_sexpr


def test_negation_without_consts_stays_canonical_form():
  node = m.Negation(
    rel_name="MethodImplemented",
    version=Version.FULL,
    index=[0, 1, 2, 3],
    prefix_vars=["simplename", "descriptor", "mtype"],
  )
  assert print_mir_sexpr(node) == (
    "(negation #:schema MethodImplemented #:ver FULL "
    "#:index (MethodImplemented 0 1 2 3) #:prefix (simplename descriptor mtype))"
  )


def test_negation_with_consts_emits_const_args_field():
  node = m.Negation(
    rel_name="Method_Modifier",
    version=Version.FULL,
    index=[0, 1],
    prefix_vars=["meth"],
    const_args=[(0, 2161502)],
  )
  assert print_mir_sexpr(node) == (
    "(negation #:schema Method_Modifier #:ver FULL "
    "#:index (Method_Modifier 0 1) #:prefix (meth) #:consts ((0 2161502)))"
  )
