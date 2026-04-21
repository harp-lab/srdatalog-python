from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Pragma(Enum):
  '''An enum of valid pragmas to pass to a FactDefinition.
  INPUT pragma specifies the filename to load data from for this fact, and must be present to use load_data() in the FFI wrapper.
  '''

  INPUT = "input"  # type str
  PRINT_SIZE = "print_size"  # type bool
  SEMIRING = "semiring"  # type bool


@dataclass
class FactDefinition:
  '''Defines a fact to be used in the program schema. Ensure that FactDefinition names match the names of
  variables used in the MirInstructions exactly.'''

  name: str
  params: list[type]  # e.g. [int, int] for a binary relation with two integer columns
  pragmas: dict[Pragma, Any] = field(default_factory=dict)

  def __str__(self):
    semiring = self.pragmas["semiring"] if "semiring" in self.pragmas else "BooleanSR"
    return f"using {self.name} = AST::RelationSchema<decltype(\"{self.name}\"_s), {semiring}, std::tuple<{', '.join(p.__name__ for p in self.params)}>>;"


@dataclass
class SchemaDefinition:
  '''Define all variables used in the program here.'''

  facts: list[FactDefinition]

  def __str__(self):
    code = ""
    for fact in self.facts:
      code += str(fact) + "\n"

    return code
