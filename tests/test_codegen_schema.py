'''Schema emission for C++ codegen.

Verifies the `using` aliases emitted by `FactDefinition` / `SchemaDefinition`
match the non-template API's expected C++.
'''

from srdatalog.codegen.schema import FactDefinition, Pragma, SchemaDefinition


def test_fact_definition_default_semiring():
  fd = FactDefinition("PointsTo", [int, int])
  assert str(fd) == (
    'using PointsTo = AST::RelationSchema<decltype("PointsTo"_s), BooleanSR, std::tuple<int, int>>;'
  )


def test_fact_definition_custom_semiring():
  fd = FactDefinition("R", [int], pragmas={"semiring": "MaxSR"})
  assert str(fd) == ('using R = AST::RelationSchema<decltype("R"_s), MaxSR, std::tuple<int>>;')


def test_schema_definition_str():
  s = SchemaDefinition(
    facts=[
      FactDefinition("PointsTo", [int, int]),
      FactDefinition("AddressOf", [int, int]),
    ]
  )
  want = (
    'using PointsTo = AST::RelationSchema<decltype("PointsTo"_s), BooleanSR, '
    'std::tuple<int, int>>;\n'
    'using AddressOf = AST::RelationSchema<decltype("AddressOf"_s), BooleanSR, '
    'std::tuple<int, int>>;\n'
  )
  assert str(s) == want


def test_schema_batch_prelude_wires_db_typedefs():
  s = SchemaDefinition(
    facts=[
      FactDefinition("PointsTo", [int, int]),
      FactDefinition("AddressOf", [int, int]),
    ]
  )
  got = s.get_batch_prelude("Andersen")
  assert 'using PointsTo = SRDatalog::AST::RelationSchema<' in got
  assert (
    'using AndersenFixpoint_DB_Blueprint = SRDatalog::AST::Database<PointsTo, AddressOf>;' in got
  )
  assert (
    'using AndersenFixpoint_DB_DeviceDB = SRDatalog::AST::SemiNaiveDatabase<'
    'AndersenFixpoint_DB_Blueprint, SRDatalog::GPU::DeviceRelationType>;'
  ) in got


def test_pragma_enum_values():
  assert Pragma.INPUT.value == "input"
  assert Pragma.PRINT_SIZE.value == "print_size"
  assert Pragma.SEMIRING.value == "semiring"


if __name__ == "__main__":
  test_fact_definition_default_semiring()
  test_fact_definition_custom_semiring()
  test_schema_definition_str()
  test_schema_batch_prelude_wires_db_typedefs()
  test_pragma_enum_values()
  print("OK")
