import unittest

from srdatalog.mir.schema import FactDefinition, Pragma, SchemaDefinition

# Add the parent directory to the path so we can import mir_schema


def print_diff(expected, actual):
  # for debugging diff
  expected = expected.replace("<", "\n<")
  actual = actual.replace("<", "\n<")
  print("expected: \n\n\n" + expected)
  print("actual:\n\n\n" + actual)


class TestFactDefinition(unittest.TestCase):
  def test_andersen(self):
    fixture = FactDefinition("AddressOf", [int, int], {Pragma.INPUT: "addressOf.csv"})

    expected = '''using AddressOf = AST::RelationSchema<decltype("AddressOf"_s), BooleanSR, std::tuple<int, int>>;'''

    actual = str(fixture).replace(" ", "").replace("\n", "")  # remove whitespace formatting
    expected = expected.replace(" ", "").replace("\n", "")  # remove whitespace formatting

    self.assertEqual(expected, actual)


class TestSchemaDefinition(unittest.TestCase):
  """Test cases for SchemaDefinition class"""

  def test_andersen(self):
    fixture = SchemaDefinition(
      [
        FactDefinition("AddressOf", [int, int], {Pragma.INPUT: "addressOf.csv"}),
        FactDefinition("Assign", [int, int], {Pragma.INPUT: "assign.csv"}),
        FactDefinition("Load", [int, int], {Pragma.INPUT: "load.csv"}),
        FactDefinition("Store", [int, int], {Pragma.INPUT: "store.csv"}),
        FactDefinition("PointsTo", [int, int], {Pragma.PRINT_SIZE: True}),
      ]
    )

    expected = '''
        using AddressOf = AST::RelationSchema<decltype("AddressOf"_s), BooleanSR, std::tuple<int, int>>;
        using Assign = AST::RelationSchema<decltype("Assign"_s), BooleanSR, std::tuple<int, int>>;
        using Load = AST::RelationSchema<decltype("Load"_s), BooleanSR, std::tuple<int, int>>;
        using Store = AST::RelationSchema<decltype("Store"_s), BooleanSR, std::tuple<int, int>>;
        using PointsTo = AST::RelationSchema<decltype("PointsTo"_s), BooleanSR, std::tuple<int, int>>;'''

    actual = str(fixture).replace(" ", "").replace("\n", "")  # remove whitespace formatting
    expected = expected.replace(" ", "").replace("\n", "")  # remove whitespace formatting

    self.assertEqual(expected, actual)
