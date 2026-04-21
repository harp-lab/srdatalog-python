'''Main C++ file emission — the top-level compile unit.

Port of `src/srdatalog/codegen.nim:codegenFixpointRuleSets` (the parts
that build `mir_cpp_str`). The main file ties all the per-rule
JIT batch files together into one compilable source containing:

  1. Relation typedefs (AST::RelationSchema per relation)
  2. Database blueprint (AST::Database<rels...>)
  3. Device DB aliases (SemiNaiveDatabase<Blueprint, DeviceRelationType>)
  4. GPU runtime #includes
  5. JitRunner_<rule> forward declaration structs (from complete_runner's
     `decl` output)
  6. Empty namespace <Ruleset>_Plans {}  (placeholder, was TMP-only)
  7. `<Ruleset>_Runner` struct with:
       - using DB = <Ruleset>_DB
       - load_data() (generates CSV input loading for every relation
         with `input_file`)
       - Per-step `step_N(db, max_iterations)` methods — each is just
         the content of `gen_step_body` for that step
       - run(db, max_iterations) dispatcher that calls step_N with
         chrono timing + console log, plus print_size stats for every
         relation tagged `print_size=True`

Top-level entry: `gen_main_file_content(...)` returns the full string.

Scope caveats:
- Skips the Nim-only `cpp_str = ...AST::Fixpoint<...>` AST dump
  (removed from Nim too — the comment at codegen.nim:148 confirms)
- Canonical index tracking for print_size stats is a simplified port
  (uses relation's declared canonical index when available; falls
  back to default column order otherwise).
- User-level `datalog_db(<BlueprintName>, <DBCode>)` emission is
  caller responsibility — not part of this module.
'''
from __future__ import annotations

import srdatalog.mir_types as m
from srdatalog.hir_types import RelationDecl


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _extract_computed_relations(plan: m.MirNode) -> list[str]:
  '''Walk a FixpointPlan or Block and collect the dest relation names
  from ExecutePipeline / ParallelGroup children. Dedup preserving order.
  '''
  instructions: list[m.MirNode] = []
  if isinstance(plan, m.FixpointPlan):
    instructions = list(plan.instructions)
  elif isinstance(plan, m.Block):
    instructions = list(plan.instructions)
  else:
    return []

  out: list[str] = []
  seen: set[str] = set()

  def _add_dests(ep: m.ExecutePipeline) -> None:
    for d in ep.dest_specs:
      if isinstance(d, m.InsertInto) and d.rel_name not in seen:
        seen.add(d.rel_name)
        out.append(d.rel_name)

  for instr in instructions:
    if isinstance(instr, m.ExecutePipeline):
      _add_dests(instr)
    elif isinstance(instr, m.ParallelGroup):
      for op in instr.ops:
        if isinstance(op, m.ExecutePipeline):
          _add_dests(op)
  return out


# -----------------------------------------------------------------------------
# Section emitters
# -----------------------------------------------------------------------------

def gen_relation_typedefs(decls: list[RelationDecl]) -> str:
  '''Emit `using <Name> = AST::RelationSchema<decltype("<Name>"_s),
  <Semiring>, std::tuple<<types>>>;` for each relation decl.'''
  out = ""
  for d in decls:
    types_str = ", ".join(d.types)
    out += (
      f'using {d.rel_name} = AST::RelationSchema<'
      f'decltype("{d.rel_name}"_s), {d.semiring}, '
      f'std::tuple<{types_str}>>;\n'
    )
  return out


def gen_db_alias(ruleset_name: str, decls: list[RelationDecl]) -> str:
  '''`using <Ruleset>_DB = AST::Database<...>;` — matches the Nim
  genRelCpp emission (codegen.nim:224).'''
  names = ", ".join(d.rel_name for d in decls)
  return f"using {ruleset_name}_DB = AST::Database<{names}>;\n"


def gen_kernel_decls_block(
  ruleset_name: str,
  decls: list[RelationDecl],
  runner_decls: dict[str, str],
  cache_dir_hint: str,
) -> str:
  '''Device DB aliases + GPU runtime includes + JitRunner forward
  declaration structs. `runner_decls[rule_name]` is the `decl` string
  returned by `gen_complete_runner` — mirrors Nim's collecting of
  `ruleDecls[ruleName]` in codegen.nim:357.
  '''
  names = ", ".join(d.rel_name for d in decls)
  ext_db = f"{ruleset_name}_DB"
  blueprint = f"{ext_db}_Blueprint"
  device_db = f"{ext_db}_DeviceDB"

  out = "// Device DB type alias (matches batch files)\n"
  out += f"using {blueprint} = SRDatalog::AST::Database<{names}>;\n"
  out += (
    f"using {device_db} = SRDatalog::AST::SemiNaiveDatabase"
    f"<{blueprint}, SRDatalog::GPU::DeviceRelationType>;\n\n"
  )
  # clang-format sorts these alphabetically; emit in that order so
  # the fixture (which gets formatted by the pre-commit hook) stays
  # byte-identical without a reformat step.
  out += '#include "gpu/runtime/gpu_mir_helpers.h"\n'
  out += '#include "gpu/runtime/jit/materialized_join.h"\n'
  out += '#include "gpu/runtime/jit/ws_infrastructure.h"\n'
  out += '#include "gpu/runtime/stream_pool.h"\n'
  out += "using namespace SRDatalog::GPU;\n\n"

  for rule_name, decl_str in runner_decls.items():
    out += "// Forward declaration - defined in JIT batch file\n"
    out += f"// See: {cache_dir_hint}/jit_batch_*.cpp\n"
    out += decl_str
    out += "\n"
  return out


def gen_load_data_method(decls: list[RelationDecl]) -> str:
  '''Emit the `load_data` static template method — reads CSVs for
  every relation with `input_file` set.'''
  out = "  template <typename DB>\n"
  out += "  static void load_data(DB& db, std::string root_dir) {\n"
  for d in decls:
    if d.input_file:
      out += (
        f'    SRDatalog::load_from_file<{d.rel_name}>'
        f'(db, root_dir + "/{d.input_file}");\n'
      )
  out += "  }\n\n"
  return out


def _gen_run_body_per_step(step_idx: int, step: m.MirNode, is_recursive: bool) -> str:
  '''Per-step timing-instrumented call from run(). Matches Nim
  codegen.nim:423-439.'''
  s = str(step_idx)
  rels = _extract_computed_relations(step)
  step_type = "recursive" if is_recursive else "simple"

  out = (
    f"    auto step_{s}_start = std::chrono::high_resolution_clock::now();\n"
  )
  out += f"    step_{s}(db, max_iterations);\n"
  out += f"    auto step_{s}_end = std::chrono::high_resolution_clock::now();\n"
  out += (
    f"    auto step_{s}_duration = "
    f"std::chrono::duration_cast<std::chrono::milliseconds>"
    f"(step_{s}_end - step_{s}_start);\n"
  )
  out += f'    std::cout << "[Step {s} ({step_type})] "'
  if rels:
    out += f' << "Relations: {", ".join(rels)}"'
  out += (
    f' << " completed in " << step_{s}_duration.count() '
    '<< " ms" << std::endl;\n'
  )
  return out


def _gen_final_print_block(
  decls: list[RelationDecl],
  canonical_indices: dict[str, list[int]] | None,
) -> str:
  '''Final `print_size` block — emits size reads for every relation
  tagged `print_size=True`. Uses canonical index when available;
  otherwise falls back to the default column order
  `{0, 1, ..., arity-1}`. Mirrors Nim codegen.nim:449-522.
  '''
  canonical = canonical_indices or {}
  out = ""
  for d in decls:
    if not d.print_size:
      continue
    if d.rel_name in canonical:
      cols = canonical[d.rel_name]
    else:
      cols = list(range(len(d.types)))
    cols_str = ", ".join(str(c) for c in cols)
    out += "    {\n"
    out += f"      SRDatalog::IndexSpec canonical_idx{{{cols_str}}};\n"
    out += (
      f"      auto& rel = get_relation_by_schema<{d.rel_name}, FULL_VER>(db);\n"
    )
    out += "      if (rel.has_index(canonical_idx)) {\n"
    out += "        auto& idx = rel.get_index(canonical_idx);\n"
    out += (
      f'        std::cout << " >>>>>>>>>>>>>>>>> {d.rel_name} : " '
      "<< idx.root().degree() << std::endl;\n"
    )
    out += "      } else {\n"
    out += (
      f'        std::cout << " >>>>>>>>>>>>>>>>> {d.rel_name} : '
      '[Index Missing]" << std::endl;\n'
    )
    out += "      }\n"
    out += "    }\n"
  return out


def gen_runner_struct(
  ruleset_name: str,
  decls: list[RelationDecl],
  mir_program: m.Program,
  step_bodies: list[str],
  canonical_indices: dict[str, list[int]] | None = None,
) -> str:
  '''Emit `<Ruleset>_Runner` — the main orchestrator struct.

  Args:
    ruleset_name: base name (e.g. "TrianglePlan")
    decls: relation declarations (for load_data + print_size)
    mir_program: output of compile_to_mir
    step_bodies: one per step, the full `template <typename DB> static
      void step_N(...)` function body as emitted by gen_step_body.
    canonical_indices: optional relation→column-order map for
      print_size stats. If omitted, falls back to default order.
  '''
  assert len(step_bodies) == len(mir_program.steps)

  out = f"struct {ruleset_name}_Runner {{\n"
  out += f"  using DB = {ruleset_name}_DB;\n\n"
  out += gen_load_data_method(decls)

  # Step bodies — each is a full `template <typename DB> static void step_N(...)`.
  for body in step_bodies:
    out += body

  # run() method.
  out += "  template <typename DB>\n"
  out += (
    "  static void run(DB& db, std::size_t max_iterations = "
    "std::numeric_limits<int>::max()) {\n"
  )
  for i, (step, is_rec) in enumerate(mir_program.steps):
    out += _gen_run_body_per_step(i, step, is_rec)
  out += _gen_final_print_block(decls, canonical_indices)
  out += "  }\n"
  out += "};\n"
  return out


# -----------------------------------------------------------------------------
# Top-level assembly
# -----------------------------------------------------------------------------

def gen_main_file_content(
  ruleset_name: str,
  decls: list[RelationDecl],
  mir_program: m.Program,
  step_bodies: list[str],
  runner_decls: dict[str, str],
  cache_dir_hint: str = "<jit-cache>",
  canonical_indices: dict[str, list[int]] | None = None,
  jit_batch_count: int = 0,
) -> str:
  '''Emit the full main-file string — mirrors Nim's `mir_cpp_str`.

  Structure:
    1. Relation typedefs + DB alias (`genRelCpp`)
    2. `using namespace SRDatalog::mir::dsl;`
    3. Kernel-decls block (device DB + GPU includes + JitRunner fwd decls)
    4. `namespace <Ruleset>_Plans { }` (empty for JIT target)
    5. `<Ruleset>_Runner` struct
    6. `// ======== JIT File-Based Compilation ========` summary
  '''
  out = ""
  out += gen_relation_typedefs(decls)
  out += gen_db_alias(ruleset_name, decls)
  out += "using namespace SRDatalog::mir::dsl;\n"
  out += gen_kernel_decls_block(ruleset_name, decls, runner_decls, cache_dir_hint)
  out += f"namespace {ruleset_name}_Plans {{\n"
  out += "}\n\n"
  out += gen_runner_struct(
    ruleset_name, decls, mir_program, step_bodies, canonical_indices,
  )
  if jit_batch_count > 0:
    out += "\n// ======== JIT File-Based Compilation ========\n"
    out += f"// JIT kernels in {jit_batch_count} batch files\n"
  return out
