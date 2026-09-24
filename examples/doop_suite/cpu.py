"""Compile the instantiated canonical DOOP program and run fresh Souffle fixedpoints.

Requires ``souffle``, its development headers, and a C++17/OpenMP compiler with
zlib and SQLite development libraries. SOUFFLE and CXX select executables;
SOUFFLE_INCLUDE_DIR, CPPFLAGS, CXXFLAGS and LDFLAGS support nonstandard installs.
No binary or generated source is reused from outside the caller's new output.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import traceback
from pathlib import Path

from srdatalog.dsl import ArgKind, Atom, Filter, Negation, Program, Split

_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_INTEGER = r"-?(?:0|[1-9][0-9]*)"
_OPERAND = rf"(?:[A-Za-z_][A-Za-z0-9_]*|{_INTEGER})"
_COMPARISON = re.compile(rf"\s*({_OPERAND})\s*(!=|==)\s*({_OPERAND})\s*\Z")


def _digest(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _json(path: Path, value: dict) -> None:
  path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _integer(value: int) -> str:
  if type(value) is not int or not -(2**31) <= value < 2**31:
    raise ValueError(f"Expected a signed int32 literal, got {value!r}")
  return str(value)


def _variable(name: str) -> str:
  if name == "_":
    return name
  if not isinstance(name, str) or not _VARIABLE.fullmatch(name):
    raise ValueError(f"Unsupported variable: {name!r}")
  return "v_" + name


def _filter(clause: Filter) -> list[str]:
  match = re.fullmatch(r"\s*return\s+(.+);\s*", clause.code)
  if not match:
    raise ValueError(f"Unsupported C++ filter: {clause.code}")
  translated = []
  for part in match.group(1).split("&&"):
    comparison = _COMPARISON.fullmatch(part)
    if not comparison:
      raise ValueError(f"Unsupported C++ filter: {clause.code}")
    left, operator, right = comparison.groups()
    operands = []
    for operand in (left, right):
      if re.fullmatch(_INTEGER, operand):
        operands.append(_integer(int(operand)))
      elif operand != "_" and operand in clause.vars:
        operands.append(_variable(operand))
      else:
        raise ValueError(f"Undeclared filter variable: {operand!r}")
    translated.append(f'{operands[0]} {"=" if operator == "==" else operator} {operands[1]}')
  return translated


def translate_program(program: Program) -> tuple[str, dict]:
  """Export integer set rules, rejecting unsupported semantics rather than guessing.

  SPLIT and GPU plans affect execution only. Multiheads become independent
  rules with the same body; anonymous variables and negation stay intact.
  All IDBs are observable outputs, including intermediates needed for parity.
  """
  relations = {relation.name: relation for relation in program.relations}
  if len(relations) != len(program.relations):
    raise ValueError("Duplicate relation names")
  declarations = []
  schema = []
  for relation in program.relations:
    if not _IDENTIFIER.fullmatch(relation.name):
      raise ValueError(f"Unsupported relation name: {relation.name!r}")
    if (
      relation.arity < 1
      or len(relation.column_types) != relation.arity
      or any(kind is not int for kind in relation.column_types)
      or relation.semiring != "NoProvenance"
    ):
      raise ValueError(f"Unsupported relation type/semiring: {relation.name}")
    columns = ", ".join(f"c{index}:number" for index in range(relation.arity))
    declarations.append(f".decl {relation.name}({columns})")
    if relation.input_file:
      filename = Path(relation.input_file)
      if filename.is_absolute() or ".." in filename.parts:
        raise ValueError(f"Nonrelative input filename: {filename}")
      declarations.append(
        f'.input {relation.name}(IO="file", filename={json.dumps(str(filename))}, delimiter="\\t")'
      )
    else:
      declarations.append(
        f'.output {relation.name}(IO="file", filename="{relation.name}.tsv", delimiter="\\t")'
      )
    schema.append(
      {"name": relation.name, "arity": relation.arity, "input_file": relation.input_file or None}
    )

  def argument(arg) -> str:
    if arg.kind is ArgKind.LVAR:
      return _variable(arg.var_name)
    if arg.kind is ArgKind.CONST:
      literal = _integer(arg.const_value)
      if arg.const_cpp_expr not in (None, literal):
        raise ValueError(f"Nonliteral C++ constant: {arg.const_cpp_expr}")
      return literal
    raise ValueError(f"Unsupported argument: {arg}")

  def atom(value: Atom) -> str:
    if value.rel not in relations or len(value.args) != relations[value.rel].arity:
      raise ValueError(f"Undeclared or malformed atom: {value}")
    return value.rel + "(" + ", ".join(argument(arg) for arg in value.args) + ")"

  rules = []
  rule_map = []
  filters = []
  for index, rule in enumerate(program.rules):
    if rule.count or rule.debug_code or not rule.heads:
      raise ValueError(f"Non-relational rule behavior: {rule.name}")
    body = []
    for clause in rule.body:
      if isinstance(clause, Atom):
        body.append(atom(clause))
      elif isinstance(clause, Negation):
        body.append("!" + atom(clause.atom))
      elif isinstance(clause, Filter):
        translated = _filter(clause)
        body.extend(translated)
        filters.append({"rule": rule.name, "cpp": clause.code, "datalog": translated})
      elif isinstance(clause, Split):
        continue
      else:
        raise ValueError(f"Unsupported clause in {rule.name}: {clause!r}")
    emitted = [atom(head) + (" :- " + ", ".join(body) if body else "") + "." for head in rule.heads]
    rules.extend(emitted)
    rule_map.append(
      {
        "source_index": index,
        "name": rule.name,
        "execution_plans_omitted": len(rule.plans),
        "emitted": emitted,
      }
    )
  text = "\n".join(
    [
      "// Instantiated canonical Python Program; integer set semantics, logical column order.",
      *declarations,
      "",
      *rules,
      "",
    ]
  )
  return text, {
    "relations": schema,
    "source_rule_count": len(program.rules),
    "emitted_rule_count": len(rules),
    "rule_map": rule_map,
    "filters": filters,
  }


def _command(
  argv: list[str], directory: Path, label: str, env: dict[str, str], timeout: int
) -> dict:
  """Keep command/log evidence even if a compiler or run fails or times out."""
  started = time.perf_counter()
  record = {
    "argv": argv,
    "cwd": str(directory),
    "timeout_seconds": timeout,
    "stdout": str(directory / f"{label}.stdout.log"),
    "stderr": str(directory / f"{label}.stderr.log"),
  }
  process = None
  try:
    with Path(record["stdout"]).open("w") as stdout, Path(record["stderr"]).open("w") as stderr:
      process = subprocess.Popen(
        argv,
        cwd=directory,
        env=env,
        stdout=stdout,
        stderr=stderr,
        start_new_session=os.name == "posix",
      )
      try:
        record["returncode"] = process.wait(timeout=timeout)
      except BaseException:
        if os.name == "posix":
          with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        else:
          process.kill()
        process.wait()
        raise
  except subprocess.TimeoutExpired:
    record["returncode"] = "timeout"
    raise TimeoutError(f"CPU {label} exceeded {timeout}s; inspect {directory}") from None
  except BaseException as error:
    record["error"] = f"{type(error).__name__}: {error}"
    if process is not None:
      record["returncode"] = process.returncode
    raise
  finally:
    record["process_wall_seconds"] = time.perf_counter() - started
    _json(directory / f"{label}.process.json", record)
  if record["returncode"] != 0:
    raise RuntimeError(f"CPU {label} exited {record['returncode']}; inspect {directory}")
  return record


def _build(source: Path, directory: Path, threads: int, env: dict[str, str], timeout: int) -> dict:
  souffle = shutil.which(env.get("SOUFFLE", "souffle"))
  compiler = shlex.split(env.get("CXX", "c++"))
  if not souffle:
    raise FileNotFoundError("Souffle not found; install souffle or set SOUFFLE")
  if not compiler or not shutil.which(compiler[0]):
    raise FileNotFoundError("C++ compiler not found; install a C++17/OpenMP compiler or set CXX")
  include = env.get("SOUFFLE_INCLUDE_DIR")
  if include:
    includes = ["-I" + str(Path(include).expanduser().resolve())]
  else:
    # A prefix install keeps bin/ and include/ beside each other. Standard
    # system headers also work through the compiler's normal search paths.
    prefix = Path(souffle).resolve().parent.parent / "include"
    includes = ["-I" + str(prefix)] if (prefix / "souffle/SouffleInterface.h").is_file() else []
  driver = Path(__file__).resolve().with_name("souffle_driver.cpp")
  generated = directory / "doop_reference.cpp"
  executable = directory / "souffle_reference"
  generation = _command(
    [souffle, "-j", str(threads), "-g", str(generated), str(source)],
    directory,
    "generate",
    env,
    timeout,
  )
  command = [
    *compiler,
    "-O3",
    "-DNDEBUG",
    "-std=c++17",
    "-fopenmp",
    "-pthread",
    *shlex.split(env.get("CPPFLAGS", "")),
    *shlex.split(env.get("CXXFLAGS", "")),
    "-D__EMBEDDED_SOUFFLE__",
    *includes,
    str(generated),
    str(driver),
    "-o",
    str(executable),
    *shlex.split(env.get("LDFLAGS", "")),
    "-lz",
    "-lsqlite3",
    "-ldl",
  ]
  compilation = _command(command, directory, "compile", env, timeout)
  result = {
    "status": "built",
    "binary": str(executable),
    "binary_sha256": _digest(executable),
    "factory_name": "doop_reference",
    "driver_sha256": _digest(driver),
    "souffle_program_sha256": _digest(source),
    "generation": generation,
    "compilation": compilation,
  }
  _json(directory / "build.json", result)
  return result


def run_cpu(
  facts: Path,
  output: Path,
  *,
  threads: int = 12,
  timeout: int = 900,
  warmups: int = 1,
  repeats: int = 3,
) -> dict:
  """Run exact CPU fixedpoints; ``output`` must not exist, even if empty.

  Each warmup/repetition starts a fresh process with empty IDBs. Only run()
  is timed as the fixedpoint; load, compilation, counting and final all-IDB
  TSV export are separate. Timeout applies to each build or execution process.
  """
  for name, value, minimum in (
    ("threads", threads, 1),
    ("timeout", timeout, 1),
    ("warmups", warmups, 0),
    ("repeats", repeats, 1),
  ):
    if type(value) is not int or value < minimum:
      raise ValueError(f"{name} must be an integer >= {minimum}")
  if Path(output).is_symlink():
    raise FileExistsError(f"Output already exists: {output}")
  facts, output = Path(facts).resolve(), Path(output).resolve()
  output.mkdir(parents=True, exist_ok=False)
  result = {
    "schema_version": 1,
    "status": "running",
    "backend": "cpu",
    "plan": "baseline",
    "engine": "souffle-compiled",
    "dataset": facts.name,
    "facts": str(facts),
    "threads": threads,
    "warmups": warmups,
    "repeats": repeats,
    "timings_seconds": [],
    "runs": [],
  }
  try:
    manifest_path, metadata_path = facts / "manifest.json", facts / "meta.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(metadata, dict):
      raise ValueError("Prepared manifest and metadata must be JSON objects")
    source = Path(__file__).resolve().parents[1] / "doop.py"
    result.update(
      source_sha256=_digest(source),
      metadata_sha256=_digest(metadata_path),
      input_source_sha256=manifest["source_sha256"],
      input_manifest_sha256=_digest(manifest_path),
    )
    if manifest["program"]["sha256"] != result["source_sha256"]:
      raise ValueError("Canonical program differs from the prepared manifest; prepare again")
    if (
      manifest["metadata"]["path"] != "meta.json"
      or manifest["metadata"]["sha256"] != result["metadata_sha256"]
    ):
      raise ValueError("Prepared metadata differs from manifest")
    spec = importlib.util.spec_from_file_location("_doop_suite_cpu_canonical", source)
    if spec is None or spec.loader is None:
      raise ImportError(f"Cannot load canonical program: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text, exported = translate_program(module.build_doopdb_program(metadata))
    schema = {relation["name"]: relation for relation in exported["relations"]}
    inputs = {name: relation for name, relation in schema.items() if relation["input_file"]}
    idbs = set(schema) - inputs.keys()
    result.update(expected_relations=sorted(schema), input_relations=sorted(inputs))
    from .prepare import SCHEMA

    # The source declares two EDBs unused by the instantiated Program.
    # Keep the complete preparation contract, but execute only active relations.
    if set(manifest["relations"]) != set(SCHEMA) or not inputs.keys() <= SCHEMA.keys():
      raise ValueError("Prepared manifest does not contain the declared input contract")
    input_counts = {}
    for name, relation in inputs.items():
      entry = manifest["relations"][name]
      path = facts / relation["input_file"]
      if entry["path"] != relation["input_file"] or entry["arity"] != relation["arity"]:
        raise ValueError(f"Prepared schema mismatch: {name}")
      if not path.is_file() or path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Prepared input missing or size differs from manifest: {path}")
      if type(entry["rows"]) is not int or entry["rows"] < 0:
        raise ValueError(f"Invalid prepared row count: {name}")
      input_counts[name] = entry["rows"]
    # Do not scan multi-gigabyte inputs here: preparation records hashes and
    # validates int32 set data. Scanning immediately before load warms cache.
    build_dir = output / "build"
    build_dir.mkdir()
    dl = build_dir / "doop_reference.dl"
    dl.write_text(text, encoding="utf-8")
    exported.update(
      source_sha256=result["source_sha256"], metadata_sha256=result["metadata_sha256"]
    )
    _json(build_dir / "export.json", exported)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    temp = build_dir / "tmp"
    temp.mkdir()
    env["TMPDIR"] = str(temp)
    build = _build(dl, build_dir, threads, env, timeout)
    result["build"] = build
    result["generation_seconds"] = build["generation"]["process_wall_seconds"]
    result["compile_seconds"] = build["compilation"]["process_wall_seconds"]
    counts = None
    for index in range(warmups + repeats):
      warmup = index < warmups
      run_dir = output / (f"warmup-{index:03d}" if warmup else f"run-{index - warmups:03d}")
      run_dir.mkdir()
      export = index == warmups + repeats - 1
      timing_path = run_dir / "timing.json"
      command = [
        build["binary"],
        build["factory_name"],
        str(facts),
        str(run_dir / "tuples"),
        str(threads),
        str(timing_path),
        "tsv" if export else "none",
      ]
      process = _command(command, run_dir, "run", env, timeout)
      timing = json.loads(timing_path.read_text(encoding="utf-8"))
      observed = timing["relation_counts"]
      if set(observed) != schema.keys() or any(
        type(n) is not int or n < 0 for n in observed.values()
      ):
        raise ValueError(f"CPU did not retain all canonical relations: {run_dir}")
      if any(observed[name] != count for name, count in input_counts.items()):
        raise ValueError(f"Loaded input counts differ from prepared set counts: {run_dir}")
      if counts is not None and observed != counts:
        raise ValueError(f"Fresh CPU fixedpoints disagree on cardinalities: {run_dir}")
      counts = observed
      if set(timing["exported_relations"]) != (idbs if export else set()):
        raise ValueError(f"CPU export omitted or added a relation: {run_dir}")
      result["runs"].append({"warmup": warmup, "process": process, "timing": timing})
      if not warmup:
        result["timings_seconds"].append(timing["run_seconds"])
      if export:
        result["outputs"] = {name: str(run_dir / "tuples" / f"{name}.tsv") for name in sorted(idbs)}
        for path in result["outputs"].values():
          if not Path(path).is_file():
            raise FileNotFoundError(path)
    result.update(
      status="passed",
      relation_counts=counts,
      load_seconds=[run["timing"]["load_seconds"] for run in result["runs"] if not run["warmup"]],
      export_seconds=result["runs"][-1]["timing"]["export_seconds"],
      page_cache_policy="Uncontrolled; fresh processes, no automatic cache flush; explicit warmups only.",
      input_verification="Prepared manifest sizes/schema and loaded set counts; input hashes recorded at preparation.",
    )
    _json(output / "result.json", result)
    return result
  except BaseException as error:
    result.update(
      status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()
    )
    # Preserve the original failure if the filesystem itself failed.
    with contextlib.suppress(OSError):
      _json(output / "failure.json", result)
    raise


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--facts", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--threads", type=int, default=12)
  parser.add_argument("--timeout", type=int, default=900)
  parser.add_argument("--warmups", type=int, default=1)
  parser.add_argument("--repeats", type=int, default=3)
  args = parser.parse_args()
  print(
    json.dumps(
      run_cpu(
        args.facts,
        args.output,
        threads=args.threads,
        timeout=args.timeout,
        warmups=args.warmups,
        repeats=args.repeats,
      ),
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
