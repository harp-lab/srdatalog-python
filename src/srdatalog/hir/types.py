'''Python mirror of src/srdatalog/hir/hir_types.nim.

These are the high-level IR types that HIR passes (stratification, planning,
index selection, lowering) read and write. The Python pipeline must produce
HIR that serializes byte-identically to the Nim golden for differential testing.

Keep field names in sync with the Nim side; the canonical emitter (hir_emit.py, TBD)
will translate between snake_case Python and the camelCase keys used in
json_printer.nim's output.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from srdatalog.dsl import Rule


class Version(Enum):
  '''Mirrors Nim hir_types.Version.

  `.value` stays the plain string (`"FULL"`, `"DELTA"`, `"NEW"`) so the MIR
  S-expr emitter's byte-match against Nim continues to work. For C++
  codegen the extra properties below surface the forms the backend needs:
    - `.method` — method-suffix used in C++ helpers (`full`, `delta`, `newt`)
    - `.number` — numeric index Nim's template backend uses (`"0"`, `"1"`, `"2"`)
    - `.code`   — `*_VER` macro the non-template codegen emits into types
  '''

  FULL = "FULL"
  DELTA = "DELTA"
  NEW = "NEW"

  @property
  def method(self) -> str:
    return _CPP_METHOD[self]

  @property
  def number(self) -> str:
    return _CPP_NUMBER[self]

  @property
  def code(self) -> str:
    return _CPP_CODE[self]


_CPP_METHOD = {
  Version.FULL: "full",
  Version.DELTA: "delta",
  Version.NEW: "newt",
}
_CPP_NUMBER = {Version.FULL: "0", Version.DELTA: "1", Version.NEW: "2"}
_CPP_CODE = {
  Version.FULL: "FULL_VER",
  Version.DELTA: "DELTA_VER",
  Version.NEW: "NEW_VER",
}


@dataclass
class AccessPattern:
  '''Access pattern for a single body clause.

  Mirrors hir_types.nim AccessPattern. All fields default-initialized for ease
  of construction inside passes; the planner fills them in.
  '''

  rel_name: str = ""
  version: Version = Version.FULL
  access_order: list[str] = field(default_factory=list)  # Variables in access order
  index_cols: list[int] = field(default_factory=list)  # Column indices in access order
  prefix_len: int = 0  # # bound vars at start of access_order
  clause_idx: int = -1  # Original body clause index
  const_args: list[tuple[int, int]] = field(default_factory=list)  # (col, const) pairs


@dataclass
class HirRuleVariant:
  '''One delta variant of a rule (mirrors HirRuleVariant).

  Non-recursive rules have delta_idx == -1 and a single base variant. Recursive
  rules fan out into N variants (one delta per recursive body clause).
  '''

  original_rule: Rule
  delta_idx: int = -1
  clause_versions: list[Version] = field(default_factory=list)
  clause_order: list[int] = field(default_factory=list)
  var_order: list[str] = field(default_factory=list)
  join_vars: set[str] = field(default_factory=set)
  split_at: int = -1
  temp_vars: list[str] = field(default_factory=list)
  temp_rel_name: str = ""
  access_patterns: list[AccessPattern] = field(default_factory=list)
  negation_patterns: list[AccessPattern] = field(default_factory=list)
  balanced_root: list[str] = field(default_factory=list)
  balanced_sources: list[str] = field(default_factory=list)
  # Codegen hints (pragma-driven)
  fanout: bool = False
  work_stealing: bool = False
  block_group: bool = False
  dedup_hash: bool = False
  count: bool = False


@dataclass
class HirStratum:
  '''An SCC + its rule variants. Mirrors HirStratum.'''

  scc_members: set[str] = field(default_factory=set)
  is_recursive: bool = False
  is_generated: bool = False
  stratum_rules: list[Rule] = field(default_factory=list)
  base_variants: list[HirRuleVariant] = field(default_factory=list)
  recursive_variants: list[HirRuleVariant] = field(default_factory=list)
  # Filled by Pass 4 (index selection):
  required_indices: dict[str, list[list[int]]] = field(default_factory=dict)
  canonical_index: dict[str, list[int]] = field(default_factory=dict)
  # Hooks (C++ injection points):
  before_hook: str = ""
  after_hook: str = ""


@dataclass
class RelationDecl:
  '''Relation declaration; subset of syntax.nim RelationDecl.

  NimNode-typed fields in Nim (relname, types, semiring) are reduced to strings here.
  Python emitter must reproduce the same repr() the Nim emitter writes.
  '''

  rel_name: str
  types: list[str]  # e.g. ["int", "int"]; matches NimNode.repr() output
  semiring: str = "BooleanSR"
  input_file: str = ""
  print_size: bool = False
  count_only: bool = False
  output_file: str = ""
  index_type: str = ""
  is_generated: bool = False
  is_temp: bool = False


@dataclass
class HirProgram:
  '''The complete HIR program. Mirrors HirProgram.'''

  strata: list[HirStratum] = field(default_factory=list)
  relation_decls: list[RelationDecl] = field(default_factory=list)
  global_index_map: dict[str, list[list[int]]] = field(default_factory=dict)
