"""Semantic checks for the conservative canonical-Program CPU translation."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from srdatalog.dsl import SPLIT, Const, Filter, Program, Relation, Var

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from doop_suite.cpu import translate_program


def test_translation_preserves_recursive_multihead_filter_negation_and_split(tmp_path):
  souffle = shutil.which("souffle")
  if souffle is None:
    pytest.skip("Souffle is required to execute the translated logical program")
  x, y, z = Var("x"), Var("y"), Var("z")
  seed = Relation("Seed", 2, input_file="Seed.csv")
  blocked = Relation("Blocked", 2, input_file="Blocked.csv")
  forward = Relation("Forward", 2)
  reverse = Relation("Reverse", 2)
  from_one = Relation("FromOne", 1)
  program = Program(
    rules=[
      (
        (forward(x, y) | reverse(y, x))
        <= seed(x, y)
        & SPLIT
        & ~blocked(x, Var("_"))
        & Filter(("x", "y"), "return x != -1 && y != 8;")
      ).with_plan(var_order=["y", "x"]),
      forward(x, z) <= forward(x, y) & forward(y, z),
      from_one(y) <= forward(x, y) & Filter(("x",), "return x == 1;"),
    ]
  )
  text, _ = translate_program(program)
  source = tmp_path / "semantic.dl"
  source.write_text(text, encoding="utf-8")
  (tmp_path / "Seed.csv").write_text("1\t2\n2\t3\n3\t4\n5\t6\n1\t2\n7\t8\n-1\t0\n")
  (tmp_path / "Blocked.csv").write_text("3\t99\n5\t1\n")
  outputs = tmp_path / "outputs"
  outputs.mkdir()
  subprocess.run(
    [souffle, "-F", str(tmp_path), "-D", str(outputs), str(source)],
    check=True,
    capture_output=True,
    text=True,
    timeout=30,
  )

  def tuples(name):
    rows = (outputs / f"{name}.tsv").read_text().splitlines()
    parsed = [tuple(map(int, row.split("\t"))) for row in rows]
    assert len(parsed) == len(set(parsed)), "The oracle must export set, not bag, results"
    return set(parsed)

  assert tuples("Forward") == {(1, 2), (2, 3), (1, 3)}
  assert tuples("Reverse") == {(2, 1), (3, 2)}
  assert tuples("FromOne") == {(2,), (3,)}


@pytest.mark.parametrize(
  "code",
  [
    "return x != 1 || x != 2;",  # Disjunction must not be silently treated as conjunction.
    "return x == 010;",  # C++ octal is not a decimal identifier.
    "return x == 2147483648;",  # GPU's integer domain is signed int32.
  ],
)
def test_unsupported_filter_semantics_are_rejected(code):
  x = Var("x")
  seed = Relation("Seed", 1, input_file="Seed.csv")
  result = Relation("Result", 1)
  with pytest.raises(ValueError):
    translate_program(Program(rules=[result(x) <= seed(x) & Filter(("x",), code)]))


def test_constant_cpp_expression_cannot_override_metadata_literal():
  x = Var("x")
  seed = Relation("Seed", 1, input_file="Seed.csv")
  result = Relation("Result", 2)
  with pytest.raises(ValueError):
    translate_program(Program(rules=[result(x, Const(1, cpp_expr="2")) <= seed(x)]))
