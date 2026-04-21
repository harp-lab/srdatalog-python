'''Defines the various enums used in representing MIR. These commands are used together to create the instructions portion of the SRDatalogProgram,
and are also used in the C++ generation of the program. Each command corresponds to a specific C++ function or set of functions.
Most commands are only designed to go in the "instructions" field of a Block or the "body" field of an ExecutePipeline, but a few are more
specialized and must go elsewhere.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

MISSING_HANDLE = (
  -99
)  # inserted when nodes are converted to C++ but no MirInstructions is associated
ATTRIBUTES_CONTAINING_CHILDREN = [
  'instructions',
  'body',
  'sources',
  'dests',
]  # for recursive programs

# TODO
# write tests for MultiHandleVHM, SingleVHM


# completion checklist
#     aggregate = "moAggregate"                 hold off on this
#     moConstantBind:                           hold off on this


## Functions


def generate_single_vhm(prefixes: tuple[str] = (), cursor: int = -1):
  '''Generates a var-handle-map between prefixes for any function which uses only a single cursor (Scan, Aggregate)'''
  if prefixes == ():  # no prefixes = easy
    return "decltype(boost::hana::make_map())"

  if cursor < 0:
    print("""Warning: trying to create a handle-map for a node without a program. 
            Cursors will be assigned relative to this function, but are not usable in a full compiled program.""")
    cursor = 0

  pairs = []
  for p in prefixes:
    pairs.append(
      f'boost::hana::make_pair(boost::hana::type_c<SRDatalog::AST::Var<decltype("{p}"_s)>>, std::integer_sequence<std::size_t, {cursor}>{{}})'
    )

  return "decltype(boost::hana::make_map(" + ", ".join(pairs) + "))"


def generate_multi_vhm(sources=[], cursor=-1):
  '''Generates a var-handle-map between prefixes for any function which uses multiple cursors (Join functions)'''

  if cursor < 0:
    print("""Warning: trying to create a handle-map for a node without a program. 
            Cursors will be assigned relative to this function, but are not usable in a full compiled program.""")
    cursor = 0

  var_to_cursors = {}
  offset = cursor
  for src in sources:  # each source has its own cursor
    if isinstance(src, ColumnSource):
      for p in src.prefix:
        if p not in var_to_cursors:
          var_to_cursors[p] = []
        var_to_cursors[p].append(offset)
      offset += 1

    else:
      print(f"Object of type {type(src)} discovered in vhm for join, was expecting ColumnSource")

  if len(var_to_cursors) == 0:
    return "decltype(boost::hana::make_map())"
  else:
    pairs = []
    for v, handles in var_to_cursors.items():
      handleSeq = "std::integer_sequence<std::size_t, " + ", ".join(str(h) for h in handles) + ">"
      pairs.append(
        f"boost::hana::make_pair(boost::hana::type_c<SRDatalog::AST::Var<decltype(\"{v}\"_s)>>, {handleSeq}{{}})"
      )
    return "decltype(boost::hana::make_map(" + ", ".join(pairs) + "))"


def collect_index_specs(node: MirNode, seen: set = None) -> list[IndexSpec]:
  """Creates IndexSpecs from each RebuildIndex, ColumnSource, or Aggregate node that is in the passed tree"""
  if seen is None:
    seen = set()

  result = []

  # Make an IndexSpec from this node if it's in RebuildIndex, ColumnSource, or Aggregate
  if isinstance(node, (RebuildIndex, ColumnSource, Aggregate)):
    spec_id = id(node)
    if spec_id not in seen:
      seen.add(spec_id)
      new_spec = IndexSpec(node.fact, node.version)
      result.append(new_spec)

  # Recursively collect from child nodes
  for attr_name in ATTRIBUTES_CONTAINING_CHILDREN:
    if hasattr(node, attr_name):
      children = getattr(node, attr_name, None)
      if children:
        for child in children:
          result.extend(collect_index_specs(child, seen))

  return result


## Basic node classes


@dataclass
class FactIndex:
  '''Represents an index (a relation name + index positions, eg [PointsTo, 0, 1])'''

  name: str
  args: list

  def __str__(self):
    '''e.g.: PointsTo, 0, 1'''
    return f"{self.name}, {', '.join(str(v) for v in self.args)}"

  def ftype(self):
    '''e.g.: SRDatalog::mir::dsl::index<PointsTo, 0, 1>'''
    return f"SRDatalog::mir::dsl::index<{self}>"


class Version(Enum):
  '''Enum for version. .name is "NEW", .method is "newt", .code is "NEW_VER"'''

  NEW = ("newt", "2", 'NEW_VER')
  DELTA = ("delta", "1", 'DELTA_VER')
  FULL = ("full", "0", "FULL_VER")

  @property
  def method(self) -> str:
    return self.value[0]

  @property
  def number(self) -> int:
    return self.value[1]

  @property
  def code(self) -> int:
    return self.value[2]


@dataclass
class MirNode:
  '''abstract base class for a MIR command -- only contains the structure, and cursor allocation data
  Note that the program field must be assigned in order to create correct C++, but it doesn't necessarily need to be
  assigned at construction time. This is handled by the _recursively_set_program in MirInstructions.
  '''

  program: MirInstructions | None = field(default=None, init=False)

  def _request_cursor(self, qty=1):
    if self.program:
      return self.program._request_cursor(qty)
    else:
      print(
        "cursor request could not be handled (this block of code is not part of a MirInstructions node)"
      )
      return MISSING_HANDLE


@dataclass
class FMirNode(MirNode):
  '''abstract factual mir node -- contains a fact and a version. Still useless on its own,
  this is just to reduce a little boilerplate'''

  fact: FactIndex
  version: Version


@dataclass
class MirInstructions:
  '''
  The wrapper used for generating C++ from MIR. Must be wrapped around a structure of MirNodes:
  these MirNodes denote the logical execution of the program.
  You should make a new MirInstructions for each program: reusing them will probably cause problems.
  '''

  structure: list[Block]

  def __init__(self, structure: list[MirNode]):
    self.cursors_allocated = 0
    for s in structure:
      self._recursively_set_program(s)
    self.structure = structure

  def _recursively_set_program(self, node: MirNode):
    '''Sets all nodes within self.structure to be owned by this program -- important for C++ gen'''
    node.program = self
    for attr in ATTRIBUTES_CONTAINING_CHILDREN:
      children = getattr(node, attr, None)
      if children:
        for child in children:
          self._recursively_set_program(child)

  def _request_cursor(self, qty):
    self.cursors_allocated += qty
    # print(f"{qty} cursor(s) requested in program {id(self)} ({self.cursors_allocated} assigned)")  #debug
    return self.cursors_allocated - qty

  def reset_cursors(self):
    '''Cursors are automatically assigned the first time program is put to string.
    If you want to regenerate cursors for some reason (e.g. manually adjusting cursor start, moving a subsection of nodes to
    a new program, etc) run this cmd. Note that this also resets cursors allocated to 0: change after running if you want to
    start from a different position.'''
    # todo write test
    self.cursors_allocated = 0
    self._reset_cursors_r(self.structure)

  def _reset_cursors_r(self, node):
    if hasattr(node, "cursor"):
      node.cursor = None
    for attr in ATTRIBUTES_CONTAINING_CHILDREN:
      children = getattr(node, attr, None)
      if children:
        for child in children:
          self._reset_cursors_r(child)

  def cpp(self):  # synonym for str()
    build = ""
    for s_in in range(len(self.structure)):
      build += f"constexpr auto step_{s_in} = {self.structure[s_in]!s} \nusing step_{s_in}_t = decltype(step_{s_in});"
    return build

  def __str__(self):
    return self.cpp()


## Utility Nodes


@dataclass
class IndexSpec(FMirNode):
  def __str__(self):
    index_str = ', '.join(str(v) for v in self.fact.args)
    inner_spec = f"{self.fact.name}, std::integer_sequence<int, {index_str}>, {self.version.number}"
    return f"SRDatalog::mir::IndexSpecT<{inner_spec}>"


@dataclass
class ColumnSource(FMirNode):
  prefix: tuple[str] = ()

  def __str__(self):
    variable_str = ""
    if self.prefix != ():
      variable_str = ", " + ", ".join(f'"{v}"_v' for v in self.prefix)

    return f'SRDatalog::mir::dsl::column_source({self.fact.ftype()}().{self.version.method}(){variable_str})'


## Instructions


@dataclass
class RebuildIndex(FMirNode):
  def __str__(self):
    return f"rebuild_index({self.fact.ftype()}().{self.version.method}())"


@dataclass
class MergeIndex(FMirNode):
  def __str__(self):
    return f"merge_index({self.fact.ftype()}().{self.version.method}())"


@dataclass
class CheckSize(MirNode):
  fact_name: str
  version: Version

  def __str__(self):
    return f"check_size<{self.fact_name}, {self.version.code}>()"


@dataclass
class ComputeDelta(FMirNode):
  def __str__(self):
    return f"compute_delta({self.fact.ftype()}().{self.version.method}())"


@dataclass
class ClearRelation(MirNode):
  fact_name: str
  version: Version

  def __str__(self):
    return f"clear_relation<{self.fact_name}, {self.version.code}>()"


@dataclass
class MergeRelation(MirNode):
  fact_name: str

  def __str__(self):
    return f"merge_relation<{self.fact_name}>()"


@dataclass
class InsertInto(FMirNode):
  # the dedup-index of the MIR corresponds to self.fact.args
  terms: tuple[str]

  def __str__(self):
    # make index string: if no columns are given, just use name and _new
    if self.fact.args == []:
      index_str = f"index<{self.fact.name}>().{self.version.method}()"
    else:
      index_str = f"{self.fact.ftype()}().{self.version.method}()"

    variable_str = ", ".join(f'"{v}"_v' for v in self.terms)

    return (
      f'insert_into<{self.fact.name}, {self.version.code}, decltype({index_str})>({variable_str})'
    )


@dataclass
class Scan(FMirNode):
  vars: tuple[str]
  prefix: tuple[str]
  cursor: int = MISSING_HANDLE

  def __str__(self):
    if self.cursor < 0:
      self.cursor = self._request_cursor()

    # Convert variables to the format "var"_v, "var"_v, ...
    variable_str = ", ".join(f'"{v}"_v' for v in self.vars)

    prefixes = ", ".join(str(x) for x in self.prefix) if self.prefix else ""

    var_to_handle_map = generate_single_vhm(prefixes=self.prefix, cursor=self.cursor)

    return f'''SRDatalog::mir::dsl::scan_h<{self.cursor}, {var_to_handle_map}>(SRDatalog::mir::dsl::vars({variable_str}),{self.fact.ftype()}().{self.version.method}(){', ' + prefixes if prefixes != "" else ''})'''


# Structural commands


@dataclass
class ExecutePipeline(MirNode):
  '''Represents a fixpoint execution pipeline'''

  sources: list[IndexSpec]
  dests: list[IndexSpec]
  body: list[MirNode]

  def __str__(self):
    source_str = "std::tuple<" + ", ".join(str(node) for node in self.sources) + ">"
    dest_str = "std::tuple<" + ", ".join(str(node) for node in self.dests) + ">"
    body_str = ",\n".join(str(node) for node in self.body)

    dedup_source_schemas = []
    for n in self.sources:
      if n.fact.name not in dedup_source_schemas and isinstance(n, IndexSpec):
        dedup_source_schemas.append(n.fact.name)
    list_source_schemas = [
      n for n in dedup_source_schemas
    ]  # should all be IndexSpecs, not 100% sure
    schema_str = "std::tuple<" + ", ".join(list_source_schemas) + ">"

    return f"execute(pipeline<\n{source_str},\n{dest_str},\n{schema_str}>\n(\n{body_str})\n)"


@dataclass
class Block(MirNode):
  dests: list[IndexSpec]
  instructions: list[MirNode]  # a list of primarily ExecutePipelines, as well as other instructions
  recursive: bool = True

  def __str__(self):
    retrieved_specs = collect_index_specs(self)

    # dedup specs
    final_specs = []
    final_specs_str = []
    for s in retrieved_specs:
      strver = "std::tuple<" + str(s) + ">"
      if strver not in final_specs_str:
        final_specs_str.append(strver)
        final_specs.append(s)

    sourceSpecs_str = ", ".join(str(spec) for spec in final_specs)

    instructions_str = ", \n".join(str(instr) for instr in self.instructions)

    outSpecs_name_str = ", ".join(spec.fact.name for spec in self.dests)

    outSpecs_str = ", ".join(str(spec) for spec in self.dests)

    return f"""fixpoint_plan<
      std::tuple<{outSpecs_name_str}>,
      std::tuple<{outSpecs_str}>,
      std::tuple<{sourceSpecs_str}>
      >(
        {instructions_str}
      );"""
    # last bit:      using step_0_t = decltype(step_0);


@dataclass
class ColumnJoin(MirNode):
  vars: tuple[str]
  sources: tuple[ColumnSource]
  cursor: int = MISSING_HANDLE

  def __str__(self):
    if self.cursor < 0:
      sourcecount = 0
      for src in self.sources:
        if isinstance(src, ColumnSource):
          sourcecount += 1
      self.cursor = self._request_cursor(qty=sourcecount)  # self.cursor is the start

    csource_str = ", \n".join(str(source) for source in self.sources)
    variable_str = ", ".join(f'"{v}"_v' for v in self.vars)
    var_to_handle_map = generate_multi_vhm(
      sources=self.sources, cursor=self.cursor
    )  # TODO, is this the correct arg LOL

    return f'''
      SRDatalog::mir::dsl::column_join_h
      <{self.cursor}, {var_to_handle_map}>
      ({variable_str}, 
        {csource_str}
      )
      '''


@dataclass
class CartesianJoin(MirNode):
  # TODO multiple handles needed maybe?

  vars: tuple[str]
  sources: tuple[ColumnSource]
  cursor: int = MISSING_HANDLE

  def __str__(self):
    if self.cursor < 0:
      sourcecount = 0
      for src in self.sources:
        if isinstance(src, ColumnSource):
          sourcecount += 1
      self.cursor = self._request_cursor(qty=sourcecount)  # self.cursor is the start

    var_to_handle_map = generate_multi_vhm(
      sources=self.sources, cursor=self.cursor
    )  # TODO, this changes if there are any prefixes
    variable_str = "SRDatalog::mir::dsl::vars(" + ", ".join(f'"{v}"_v' for v in self.vars) + ")"
    csource_str = ", \n".join(str(source) for source in self.sources)

    return f'''
      SRDatalog::mir::dsl::cartesian_join_h<
      {self.cursor}, 
      {var_to_handle_map}
      >
    (
      {variable_str}, 
      {csource_str}
    )'''


@dataclass
class CppHook(MirNode):
  code: str
  label: str = ""

  def __str__(self):
    display_label = f" /* {self.label} */ " if self.label != "" else ""

    return f"inject_cpp_hook([] __host__ (auto& db) {{ {display_label} \n{self.code} }})"


@dataclass
class Aggregate(FMirNode):
  pass


# of (
#   kind: moAggregate,
#   agResultVar: @resVar,
#   agAggFunc: @aggFunc,
#   agRelName: @rel,
#   agVersion: @ver,
#   agIndex: @idx,
#   agPrefixVars: @prefixVars,
#   agHandleStart: @hs,
# ):
#   # dsl: aggregate_h<HS, VHM, AggFunc>(var(result), index<Schema, Ver, cols...>, vars(prefix...))
#   let res = "SRDatalog::mir::dsl::var(" & genTString(resVar) & ")"
#   let idxStr = genIndexSpecWithCols(rel, ver, idx)
#   let prefixStr =
#     "SRDatalog::mir::dsl::vars(" & prefixVars.map(genTString).join(", ") & ")"
#   let vhmStr = genvar_to_cursorsMap(prefixVars, hs)
#   return
#     "SRDatalog::mir::dsl::aggregate_h<" & $hs & ", " & vhmStr & ", " & aggFunc & ">(" &
#     res & ", " & idxStr & ", " & prefixStr & ")"


@dataclass
class Filter(MirNode):
  vars: tuple[str]
  code: str  # potential TODO can this code be generated more automatically?

  def __str__(self):
    variable_str = ", ".join(f'"{v}"_v' for v in self.vars)
    arg_str = ", ".join(f'auto {v}' for v in self.vars)

    return f"filter(vars({variable_str}), []({arg_str}) {{ {self.code} }})"


@dataclass
class Negate(FMirNode):
  prefix: tuple[str]
  cursor: int = MISSING_HANDLE

  def __str__(self):
    if self.cursor < 0:
      self.cursor = self._request_cursor()

    vhmstr = "decltype(boost::hana::make_map())"  # Empty VHM (intentional)
    variable_str = "SRDatalog::mir::dsl::vars(" + ", ".join(f'"{v}"_v' for v in self.prefix) + ")"

    return f"SRDatalog::mir::dsl::negation_h<{self.cursor}, {vhmstr}>({self.fact.ftype()}().{self.version.method}(), {variable_str})"
