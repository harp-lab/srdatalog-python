"""Prepare symbolic DOOP facts for the canonical 39-input integer program.

This adapter deliberately supports the known DOOP raw schema, not arbitrary
Datalog imports. It does not run an engine. Complete projected relations are
materialized with ``LC_ALL=C sort -u``; set DOOP_SORT_TMPDIR to place sort scratch
on a different existing filesystem. Otherwise scratch is local to the output.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import ExitStack
from pathlib import Path

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
CONSTANTS = {
  "abstract": "abstract",
  "public": "public",
  "static": "static",
  "main": "main",
  "clinit": "<clinit>",
  "clinit_descriptor": "void()",
  "main_descriptor": "void(java.lang.String[])",
  "java_lang_Object": "java.lang.Object",
  "java_lang_Cloneable": "java.lang.Cloneable",
  "java_io_Serializable": "java.io.Serializable",
  "java_lang_String_type": "java.lang.String",
  "java_lang_Class_type": "java.lang.Class",
  "java_lang_Object_array": "java.lang.Object[]",
  # These historical metadata key names denote the three upstream main exclusions.
  "class_init_method": "<java.util.prefs.Base64: void main(java.lang.String[])>",
  "register_natives_method": "<sun.java2d.loops.GraphicsPrimitiveMgr: void main(java.lang.String[])>",
  "desiredAssertionStatus_method": "<sun.security.provider.PolicyParser: void main(java.lang.String[])>",
}
ALIASES = {
  "Method_Modifier": "Method-Modifier",
  "Var_Type": "Var-Type",
  "Var_DeclaringMethod": "Var-DeclaringMethod",
}
DIRECT = {
  "DirectSuperclass": 2,
  "DirectSuperinterface": 2,
  "MainClass": 1,
  "FormalParam": 3,
  "ComponentType": 2,
  "AssignReturnValue": 2,
  "ActualParam": 3,
  "Method_Modifier": 2,
  "Var_Type": 2,
  "ClassType": 1,
  "ArrayType": 1,
  "InterfaceType": 1,
  "Var_DeclaringMethod": 2,
  "ApplicationClass": 1,
  "ThisVar": 2,
}
# Raw arity, projected columns, enclosing-method column.
FAT = {
  "AssignHeapAllocation": (6, (2, 3, 4), 4),
  "AssignLocal": (5, (2, 3, 4), 4),
  "AssignCast": (6, (4, 2, 3, 5), 5),
  "LoadInstanceField": (6, (3, 4, 2, 5), 5),
  "StoreInstanceField": (6, (2, 3, 4, 5), 5),
  "LoadStaticField": (5, (3, 2, 4), 4),
  "StoreStaticField": (5, (2, 3, 4), 4),
  "LoadArrayIndex": (5, (3, 2, 4), 4),
  "StoreArrayIndex": (5, (2, 3, 4), 4),
  "Return": (4, (2, 3), 3),
  "StaticMethodInvocation": (4, (0, 2, 3), 3),
}
SCHEMA = {
  **DIRECT,
  "HeapAllocation_Type": 2,
  "Field_DeclaringType": 2,
  "Method_SimpleName": 2,
  "Method_DeclaringType": 2,
  "Method_Descriptor": 2,
  "Instruction_Method": 2,
  "MethodInvocation_Method": 2,
  "isStaticMethodInvocation_Insn": 1,
  "isVirtualMethodInvocation_Insn": 1,
  "SpecialMethodInvocation_Base": 2,
  "VirtualMethodInvocation_Base": 2,
  "VirtualMethodInvocation_SimpleName": 2,
  "VirtualMethodInvocation_Descriptor": 2,
  **{
    ("ReturnVar" if name == "Return" else name): len(columns)
    for name, (_, columns, _) in FAT.items()
  },
}


def _dump(path: Path, value: dict) -> None:
  with path.open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
    handle.write("\n")


def _file_info(path: Path, *, count_rows: bool = False) -> dict:
  digest = hashlib.sha256()
  size = count = 0
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
      size += len(chunk)
      if count_rows:
        count += chunk.count(b"\n")
  info = {"path": path.name, "bytes": size, "sha256": digest.hexdigest()}
  if count_rows:
    info["rows"] = count
  return info


def _program_contract(program: Path) -> str:
  content = program.read_bytes()
  found = {}
  for node in ast.walk(ast.parse(content, filename=str(program))):
    if not (
      isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Relation"
    ):
      continue
    keywords = {key.arg: key.value for key in node.keywords}
    if "input_file" not in keywords:
      continue
    name, arity = (ast.literal_eval(arg) for arg in node.args[:2])
    filename = ast.literal_eval(keywords["input_file"])
    if name in found or filename != f"{name}.csv" or type(arity) is not int:
      raise ValueError(f"Unsupported input declaration for {name!r} in {program}")
    found[name] = arity
  if found != SCHEMA:
    raise ValueError(f"{program} does not declare the canonical 39-input DOOP schema")
  return hashlib.sha256(content).hexdigest()


def _int32(value: str, location: str) -> int:
  if re.fullmatch(r"[+-]?[0-9]+", value) is None:
    raise ValueError(f"{location}: expected signed int32, found {value!r}")
  number = int(value)
  if not INT32_MIN <= number <= INT32_MAX:
    raise ValueError(f"{location}: value {value!r} is outside signed int32")
  return number


def _integer_rows(directory: Path, name: str):
  with (directory / f"{name}.csv").open(encoding="ascii") as handle:
    for line in handle:
      yield tuple(map(int, line.rstrip("\n").split("\t")))


def _entrypoints(directory: Path, meta: dict, symbols: dict) -> dict:
  roots = {row[0] for row in _integer_rows(directory, "MainClass")}
  candidates = {
    method for method, owner in _integer_rows(directory, "Method_DeclaringType") if owner in roots
  }
  for relation, value in (
    ("Method_SimpleName", meta["main"]),
    ("Method_Descriptor", meta["main_descriptor"]),
  ):
    candidates &= {method for method, attr in _integer_rows(directory, relation) if attr == value}
  public, static = set(), set()
  for modifier, method in _integer_rows(directory, "Method_Modifier"):
    if method in candidates:
      if modifier == meta["public"]:
        public.add(method)
      if modifier == meta["static"]:
        static.add(method)
  candidates &= public & static
  exclusions = {
    meta[key]
    for key in ("class_init_method", "register_natives_method", "desiredAssertionStatus_method")
  }
  wanted = roots | candidates
  names = {value: symbol for symbol, value in symbols.items() if value in wanted}

  def describe(values):
    return [{"id": value, "symbol": names[value]} for value in sorted(values)]

  return {
    "main_classes": describe(roots),
    "candidate_main_methods": describe(candidates),
    "selected_main_methods": describe(candidates - exclusions),
    "excluded_candidates": sorted(candidates & exclusions),
  }


class _Preparation:
  def __init__(self, source: Path, staging: Path):
    self.source = source
    self.staging = staging
    self.paths = {}
    self.source_relations = {}
    self.relations = {}
    self.symbols = {}
    self.handles = {}
    self.seen_instructions = {}

  def locate(self, name: str, *, required: bool = True) -> Path | None:
    if name in self.paths:
      return self.paths[name]
    stems = dict.fromkeys((ALIASES.get(name, name), name))
    paths = [
      self.source / f"{stem}.facts" for stem in stems if (self.source / f"{stem}.facts").is_file()
    ]
    if len(paths) > 1:
      raise ValueError(f"Ambiguous source relation {name}: {paths}")
    if not paths:
      if required:
        raise FileNotFoundError(f"Missing required source relation {name}.facts in {self.source}")
      return None
    self.paths[name] = paths[0]
    return paths[0]

  def intern(self, symbol: str) -> int:
    value = self.symbols.get(symbol)
    if value is None:
      value = len(self.symbols) + 1
      if value > INT32_MAX:
        raise ValueError("Global symbol IDs exceed the signed int32 oracle domain")
      self.symbols[symbol] = value
    return value

  def source_rows(self, name: str, arity: int, numeric: tuple[int, ...] = ()):
    path = self.locate(name)
    digest = hashlib.sha256()
    size = count = 0
    with path.open("rb") as handle:
      before = os.fstat(handle.fileno())
      for count, raw in enumerate(handle, 1):
        digest.update(raw)
        size += len(raw)
        line = raw.decode("utf-8")
        if line.endswith("\n"):
          line = line[:-1]
        if line.endswith("\r"):
          line = line[:-1]
        fields = line.split("\t")
        if len(fields) != arity:
          raise ValueError(f"{path}:{count}: expected {arity} columns, found {len(fields)}")
        for column in numeric:
          fields[column] = _int32(fields[column], f"{path}:{count}:column {column + 1}")
        yield tuple(fields)
      after = os.fstat(handle.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
      raise ValueError(f"Source relation changed during preparation: {path}")
    self.source_relations[name] = {
      "path": path.name,
      "arity": arity,
      "rows": count,
      "bytes": size,
      "sha256": digest.hexdigest(),
    }

  def begin(self, stack: ExitStack, name: str, sources: list[str]) -> None:
    for source in sources:
      self.locate(source)
    self.handles[name] = stack.enter_context(
      (self.staging / f"{name}.csv").open("x", encoding="ascii", newline="\n")
    )
    self.relations[name] = {
      "arity": SCHEMA[name],
      "rows_before_set_dedup": 0,
      "sources": sources,
      "operation": "projected_tuple_set",
    }

  def emit(self, name: str, values: tuple) -> None:
    if len(values) != SCHEMA[name]:
      raise ValueError(f"{name}: projection arity mismatch")
    encoded = (self.intern(value) if isinstance(value, str) else value for value in values)
    self.handles[name].write("\t".join(map(str, encoded)) + "\n")
    self.relations[name]["rows_before_set_dedup"] += 1

  def check_value(self, attribute: str, instruction: str, values: tuple) -> None:
    key = (attribute, self.intern(instruction))
    value = tuple(self.intern(item) for item in values)
    previous = self.seen_instructions.setdefault(key, value)
    if previous != value:
      raise ValueError(
        f"Nonfunctional {attribute} for {instruction!r}; relational normalization required"
      )

  def check_instruction(self, source: str, row: tuple, method_column: int) -> None:
    # Shared attributes must be functional across categories too. The same
    # instruction may validly occur in several categories (e.g. multiarray).
    self.check_value("Instruction_Method", row[0], (row[method_column],))
    payload = row[2:5] if source == "AssignHeapAllocation" else row[2:]
    self.check_value(source, row[0], payload)
    if source in ("AssignHeapAllocation", "AssignLocal", "AssignCast"):
      self.check_value("AssignInstruction_To", row[0], (row[3],))
    if source in ("LoadInstanceField", "StoreInstanceField", "LoadStaticField", "StoreStaticField"):
      self.check_value(
        "FieldInstruction_Signature", row[0], (row[4 if "Instance" in source else 3],)
      )
    if source.endswith("MethodInvocation"):
      self.check_value("MethodInvocation_Method", row[0], (row[2],))

  def project(self) -> None:
    with ExitStack() as stack:
      for name, arity in DIRECT.items():
        self.begin(stack, name, [name])
        numeric = (0,) if name in ("FormalParam", "ActualParam") else ()
        for row in self.source_rows(name, arity, numeric):
          self.emit(name, row)
      self.begin(stack, "Field_DeclaringType", ["Field"])
      for row in self.source_rows("Field", 4):
        self.emit("Field_DeclaringType", row[:2])
      methods = {}
      for name in ("Method_SimpleName", "Method_DeclaringType", "Method_Descriptor"):
        self.begin(stack, name, ["Method"])
      for method, simple, params, declaring, returns, _, _ in self.source_rows("Method", 7, (6,)):
        # Legacy inputs include parentheses; Chopin parameter lists do not.
        params = params if params.startswith("(") and params.endswith(")") else f"({params})"
        descriptor = returns + params
        attributes = (simple, params, declaring, returns)
        previous = methods.setdefault(method, attributes)
        if previous != attributes:
          raise ValueError(
            f"Nonfunctional Method attributes for {method!r}; relational normalization required"
          )
        self.emit("Method_SimpleName", (method, simple))
        self.emit("Method_DeclaringType", (method, declaring))
        self.emit("Method_Descriptor", (method, descriptor))
      if self.locate("HeapAllocation_Type", required=False) is not None:
        self.begin(stack, "HeapAllocation_Type", ["HeapAllocation_Type"])
        for row in self.source_rows("HeapAllocation_Type", 2):
          self.emit("HeapAllocation_Type", row)
      else:
        self.begin(stack, "HeapAllocation_Type", ["NormalHeap", "StringConstant"])
        for row in self.source_rows("NormalHeap", 2):
          self.emit("HeapAllocation_Type", row)
        for row in self.source_rows("StringConstant", 1):
          self.emit("HeapAllocation_Type", (row[0], "java.lang.String"))
      invocation_sources = [
        "StaticMethodInvocation",
        "SpecialMethodInvocation",
        "VirtualMethodInvocation",
      ]
      self.begin(stack, "Instruction_Method", list(FAT) + invocation_sources[1:])
      self.begin(stack, "MethodInvocation_Method", invocation_sources)
      self.begin(stack, "isStaticMethodInvocation_Insn", ["StaticMethodInvocation"])
      for source, (arity, columns, method_column) in FAT.items():
        target = "ReturnVar" if source == "Return" else source
        self.begin(stack, target, [source])
        numeric = (1, 5) if source == "AssignHeapAllocation" else (1,)
        for row in self.source_rows(source, arity, numeric):
          self.check_instruction(source, row, method_column)
          self.emit(target, tuple(row[column] for column in columns))
          self.emit("Instruction_Method", (row[0], row[method_column]))
          if source == "StaticMethodInvocation":
            self.emit("isStaticMethodInvocation_Insn", (row[0],))
            self.emit("MethodInvocation_Method", (row[0], row[2]))
      for kind in ("Special", "Virtual"):
        source = kind + "MethodInvocation"
        self.begin(stack, source + "_Base", [source])
        if kind == "Virtual":
          self.begin(stack, "isVirtualMethodInvocation_Insn", [source])
          for name in ("VirtualMethodInvocation_SimpleName", "VirtualMethodInvocation_Descriptor"):
            self.begin(stack, name, [source, "Method"])
        for row in self.source_rows(source, 5, (1,)):
          self.check_instruction(source, row, 4)
          invocation, _, method, base, enclosing = row
          self.emit("Instruction_Method", (invocation, enclosing))
          self.emit("MethodInvocation_Method", (invocation, method))
          self.emit(source + "_Base", (invocation, base))
          if kind == "Virtual":
            self.emit("isVirtualMethodInvocation_Insn", (invocation,))
            if method in methods:
              simple, params, _, returns = methods[method]
              self.emit("VirtualMethodInvocation_SimpleName", (invocation, simple))
              self.emit("VirtualMethodInvocation_Descriptor", (invocation, returns + params))
    self.handles.clear()
    self.seen_instructions.clear()
    if self.relations.keys() != SCHEMA.keys():
      raise ValueError("Preparation did not produce the complete canonical input schema")


def prepare(
  source: Path,
  output: Path,
  *,
  program: Path | None = None,
  provenance: dict | None = None,
) -> dict:
  """Prepare a NEW directory; failures leave no published partial dataset.

  Source files are visited in fixed schema order and physical row order, so
  identical raw files produce identical IDs and prepared files. All numeric
  raw columns use the signed int32 oracle domain; only parameter indices are
  retained as numbers in the outputs. Symbol IDs share one global namespace.
  """
  source = Path(source).resolve(strict=True)
  output = Path(output).absolute()
  if not source.is_dir():
    raise NotADirectoryError(source)
  if os.path.lexists(output):
    raise FileExistsError(f"Refusing to overwrite {output}")
  program = (
    Path(program) if program is not None else Path(__file__).resolve().parents[1] / "doop.py"
  )
  program_sha256 = _program_contract(program)
  if provenance is not None and not isinstance(provenance, dict):
    raise TypeError("provenance must be a JSON object")
  sorter = shutil.which("sort")
  if sorter is None:
    raise FileNotFoundError("DOOP preparation requires the external sort command")
  scratch_root = os.environ.get("DOOP_SORT_TMPDIR")
  if scratch_root is not None and not Path(scratch_root).is_dir():
    raise NotADirectoryError(f"DOOP_SORT_TMPDIR is not a directory: {scratch_root}")
  output.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(
    prefix=f".{output.name}.prepare-", dir=output.parent
  ) as temporary:
    work = Path(temporary)
    staging = work / "prepared"
    staging.mkdir()
    adapter = _Preparation(source, staging)
    adapter.project()
    with tempfile.TemporaryDirectory(prefix="doop-sort-", dir=scratch_root or work) as scratch:
      for name, info in adapter.relations.items():
        path = staging / f"{name}.csv"
        subprocess.run(
          [sorter, "-u", "-T", scratch, "-o", str(path), str(path)],
          env=dict(os.environ, LC_ALL="C"),
          check=True,
        )
        info.update(_file_info(path, count_rows=True))
    meta = {key: adapter.intern(symbol) for key, symbol in CONSTANTS.items()}
    _dump(staging / "meta.json", meta)
    _dump(staging / "str2num.json", adapter.symbols)
    source_identity = json.dumps(
      adapter.source_relations, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest = {
      "status": "prepared_not_engine_validated",
      "mode": "symbolic",
      "source": source.name,
      "output": output.name,
      "source_sha256": hashlib.sha256(source_identity).hexdigest(),
      "program": {"path": program.name, "sha256": program_sha256},
      "adapter_sha256": _file_info(Path(__file__))["sha256"],
      "source_relations": adapter.source_relations,
      "relations": adapter.relations,
      "metadata": _file_info(staging / "meta.json"),
      "symbols": _file_info(staging / "str2num.json"),
      "meta": meta,
      "meta_symbols": CONSTANTS,
      "entrypoints": _entrypoints(staging, meta, adapter.symbols),
      "provenance": provenance if provenance is not None else {},
      "semantics": {
        "roots": "Exact MainClass set; no synthetic roots or heaps",
        "normalization": "Known raw DOOP projections; functional shared instruction and method attributes required",
        "descriptors": "returnType(params); already parenthesized parameter lists retained",
        "heap_types": "Explicit HeapAllocation_Type if supplied, otherwise NormalHeap union StringConstant->java.lang.String",
        "set_semantics": "LC_ALL=C sort -u over every complete projected relation; no sampling",
        "numeric_domain": "Signed int32 numeric inputs and positive signed-int32 global symbol IDs",
        "exclusions": "Exact three upstream symbolic methods named in meta_symbols",
      },
    }
    _dump(staging / "manifest.json", manifest)
    # Reserve without overwrite, then replace our empty reservation with the
    # complete directory in one same-filesystem rename. Concurrent preparers
    # cannot both reserve the destination. Cleanup owns only our reservation.
    output.mkdir()
    try:
      os.replace(staging, output)
    except BaseException:
      output.rmdir()
      raise
  return manifest
