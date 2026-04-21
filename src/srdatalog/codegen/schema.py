'''Schema types for C++ codegen.

Describes the per-program relation declarations that drive the emitted
C++ `using` aliases (RelationSchema, Database, SemiNaiveDatabase). Ported
from mhk's `python-api-notemplate` branch `nt_schema.py`; independent of
MIR types, so nothing here couples back to `mir_types`.

Two emission formats exist:
  - `str(schema)`      — plain alias list, used by the orchestrator today.
  - `schema.get_batch_prelude(name)` — same aliases plus the Database /
    SemiNaiveDatabase typedefs, used at the top of each JIT batch file.
'''
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Pragma(Enum):
  '''Fact pragmas. INPUT carries the CSV filename to load from (required
  for `load_data()` in the FFI wrapper). PRINT_SIZE and SEMIRING are bool.
  '''
  INPUT = "input"
  PRINT_SIZE = "print_size"
  SEMIRING = "semiring"


@dataclass
class FactDefinition:
  '''A schema-declared relation. `name` must match the relation name used
  by MIR nodes (InsertInto.rel_name, ColumnSource.rel_name, ...).
  `params` is the column type tuple, e.g. `[int, int]` for an arity-2
  int-valued relation.
  '''
  name: str
  params: list[type]
  pragmas: dict[Pragma, Any] = field(default_factory=dict)

  def __str__(self) -> str:
    semiring = self.pragmas.get("semiring", "BooleanSR")
    params_str = ", ".join(p.__name__ for p in self.params)
    return (
      f'using {self.name} = AST::RelationSchema<decltype("{self.name}"_s), '
      f'{semiring}, std::tuple<{params_str}>>;'
    )


@dataclass
class SchemaDefinition:
  '''All relations used by a program. Order matters for the emitted
  `AST::Database<...>` template argument list.'''
  facts: list[FactDefinition]

  def __str__(self) -> str:
    return "".join(str(f) + "\n" for f in self.facts)

  def get_batch_prelude(self, name: str) -> str:
    '''Header block emitted at the top of a JIT batch file: per-relation
    aliases plus the blueprint / device-DB typedefs keyed on `name`.
    '''
    parts: list[str] = []
    for fact in self.facts:
      semiring = fact.pragmas.get("semiring", "BooleanSR")
      params_str = ", ".join(p.__name__ for p in fact.params)
      parts.append(
        f'using {fact.name} = SRDatalog::AST::RelationSchema<'
        f'decltype("{fact.name}"_s), {semiring}, std::tuple<{params_str}>>;'
      )
    fact_names = ", ".join(f.name for f in self.facts)
    parts.append(
      f'using {name}Fixpoint_DB_Blueprint = SRDatalog::AST::Database<{fact_names}>;'
    )
    parts.append(
      f'using {name}Fixpoint_DB_DeviceDB = SRDatalog::AST::SemiNaiveDatabase<'
      f'{name}Fixpoint_DB_Blueprint, SRDatalog::GPU::DeviceRelationType>;'
    )
    return "".join(parts)
