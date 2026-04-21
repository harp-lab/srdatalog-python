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

import srdatalog.mir.types as m
from srdatalog.hir.types import RelationDecl


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
  <Semiring>, std::tuple<<types>>>;` for each relation decl. Used in
  the main-file scope where `using namespace SRDatalog::AST` makes
  the unqualified `AST::` resolve.

  Appends a 4th template argument for relations with a non-default
  `index_type` (e.g. `Device2LevelIndex` for fat-fact points-to sets).
  '''
  out = ""
  for d in decls:
    types_str = ", ".join(d.types)
    extra = f", {d.index_type}" if d.index_type else ""
    out += (
      f'using {d.rel_name} = AST::RelationSchema<'
      f'decltype("{d.rel_name}"_s), {d.semiring}, '
      f'std::tuple<{types_str}>{extra}>;\n'
    )
  return out


def gen_db_alias(ruleset_name: str, decls: list[RelationDecl]) -> str:
  '''`using <Ruleset>_DB = AST::Database<...>;` — matches the Nim
  genRelCpp emission (codegen.nim:224).'''
  names = ", ".join(d.rel_name for d in decls)
  return f"using {ruleset_name}_DB = AST::Database<{names}>;\n"


def gen_schema_definitions_for_batch(decls: list[RelationDecl]) -> str:
  '''Schema definitions inlined into each `jit_batch_N.cpp`. Mirrors
  Nim's `collectSchemaDefinitions` output: fully-qualified
  `SRDatalog::AST::RelationSchema` (no `using namespace` umbrella in
  the batch file scope) plus the `Literals` using-directive for the
  `_s` UDL.

  Pass this to `cache.JitBatchManager.set_schema_definitions(...)` /
  `write_jit_project(schema_definitions=...)`.
  '''
  out = "using namespace SRDatalog::AST::Literals;  // For _s string literal\n\n"
  for d in decls:
    types_str = ", ".join(d.types)
    extra = f", {d.index_type}" if d.index_type else ""
    out += (
      f"using {d.rel_name} = SRDatalog::AST::RelationSchema<"
      f'decltype("{d.rel_name}"_s), {d.semiring}, '
      f"std::tuple<{types_str}>{extra}>;\n"
    )
  return out


def gen_db_type_alias_for_batch(
  ruleset_name: str, decls: list[RelationDecl],
) -> str:
  '''DB blueprint + DeviceDB alias inlined into each batch file so
  `JitRunner_<rule>::DB` resolves. Mirrors the `dbTypeAlias` Nim
  passes to `JitBatchManager.setDbTypeAlias`.

  `ruleset_name` here is the **ext_db** name (e.g. "TrianglePlan_DB"),
  not the bare ruleset; the resulting types are `<ruleset>_Blueprint`
  and `<ruleset>_DeviceDB`.
  '''
  names = ", ".join(d.rel_name for d in decls)
  blueprint = f"{ruleset_name}_Blueprint"
  device_db = f"{ruleset_name}_DeviceDB"
  return (
    f"using {blueprint} = SRDatalog::AST::Database<{names}>;\n"
    f"using {device_db} = SRDatalog::AST::SemiNaiveDatabase"
    f"<{blueprint}, SRDatalog::GPU::DeviceRelationType>;\n"
  )


def gen_kernel_decls_block(
  ruleset_name: str,
  decls: list[RelationDecl],
  runner_decls: dict[str, str],
  cache_dir_hint: str,
  standalone_order: bool = False,
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

  # Fragment mode (Nim-compat): alias first, then GPU includes, exactly
  # the same set of includes as the checked-in main-file fixtures.
  # Standalone mode (emit_preamble=True): prepend `gpu/runtime/query.h`
  # and put the whole include block BEFORE the alias so
  # DeviceRelationType is in scope.
  base_gpu_includes = (
    '#include "gpu/runtime/gpu_mir_helpers.h"\n'
    '#include "gpu/runtime/jit/materialized_join.h"\n'
    '#include "gpu/runtime/jit/ws_infrastructure.h"\n'
    '#include "gpu/runtime/stream_pool.h"\n'
    "using namespace SRDatalog::GPU;\n\n"
  )
  alias_block = (
    "// Device DB type alias (matches batch files)\n"
    f"using {blueprint} = SRDatalog::AST::Database<{names}>;\n"
    f"using {device_db} = SRDatalog::AST::SemiNaiveDatabase"
    f"<{blueprint}, SRDatalog::GPU::DeviceRelationType>;\n\n"
  )
  if standalone_order:
    out = (
      '#include "gpu/runtime/query.h"  // DeviceRelationType, init_cuda, copy_host_to_device\n'
      + base_gpu_includes + alias_block
    )
  else:
    out = alias_block + base_gpu_includes

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
# Sharded emit — step bodies & run() moved out of main.cpp into their own TUs
# -----------------------------------------------------------------------------
#
# Problem: putting all 60 step_N methods + the template `run()` that calls
# them inside main.cpp makes main.cpp the parallelism bottleneck — for doop
# it's a single 55 MB .o that takes ~70s serially while the JIT batches
# compile in parallel under it. Nim avoids this by keeping heavy template
# machinery inside each batch file (one runner per file).
#
# Our fix: emit each step_N body AND the run() dispatcher as its own .cpp
# shard, then register them alongside the batch files with the existing
# parallel compile queue. main.cpp keeps only the Runner struct's *non-
# template declarations* and the extern "C" shim that calls into them.

def gen_runner_struct_declonly(
  ruleset_name: str,
  decls: list[RelationDecl],
  mir_program: m.Program,
) -> str:
  '''Non-template Runner struct DECLARATION (for main.cpp).

  All step_N and run() become `static void step_N(DB& db, size_t);` with
  the concrete `DB` typedef resolved to `<Ruleset>_DB_DeviceDB`. Bodies
  live in separate shard files.
  '''
  device_db = f"{ruleset_name}_DB_DeviceDB"
  out = f"struct {ruleset_name}_Runner {{\n"
  out += f"  using DB = {device_db};\n\n"
  out += gen_load_data_method(decls)  # load_data stays a template — cheap
  for i in range(len(mir_program.steps)):
    out += f"  static void step_{i}(DB& db, std::size_t max_iterations);\n"
  out += (
    f"  static void run(DB& db, std::size_t max_iterations = "
    "std::numeric_limits<int>::max());\n"
  )
  out += "};\n"
  return out


def _step_body_template_to_oop(body: str, ruleset_name: str,
                               step_idx: int, device_db: str) -> str:
  '''Rewrite the templated step body produced by `gen_step_body` into
  an out-of-line method definition.

  Input:
      template <typename DB>
      static void step_5(DB& db, std::size_t max_iterations) {
        ... body ...
      }

  Output:
      void DoopPlan_Runner::step_5(DoopPlan_DB_DeviceDB& db, std::size_t max_iterations) {
        ... body ...
      }
  '''
  template_decl = "  template <typename DB>\n"
  sig = (
    f"  static void step_{step_idx}"
    "(DB& db, std::size_t max_iterations) {\n"
  )
  if body.startswith(template_decl + sig):
    remainder = body[len(template_decl + sig):]
  else:
    # Fallback — don't mangle something we didn't recognize.
    raise ValueError(
      f"unexpected step body prefix for step {step_idx}; "
      "adjust _step_body_template_to_oop if gen_step_body changed"
    )
  return (
    f"void {ruleset_name}_Runner::step_{step_idx}"
    f"({device_db}& db, std::size_t max_iterations) {{\n"
    + remainder
  )


def _shared_runner_preamble(
  ruleset_name: str,
  decls: list[RelationDecl],
  runner_decls: dict[str, str],
  extra_index_headers: list[str] | None = None,
) -> str:
  '''Preamble shared by every out-of-line shard: srdatalog.h, plugin
  headers, inline schemas + DB alias, GPU runtime includes, JitRunner
  forward decls, and the Runner struct DECLARATION. Matches the
  content main.cpp needs to compile the same call sites, minus the
  extern "C" shim.
  '''
  out = '#include "srdatalog.h"\n'
  out += 'using namespace SRDatalog;\n'
  out += 'using namespace SRDatalog::AST::Literals;\n\n'
  # GPU runtime FIRST so DeviceRelationType / init_cuda are in scope
  # before the DB alias references them.
  out += '#include "gpu/runtime/query.h"  // DeviceRelationType\n'
  out += '#include "gpu/runtime/gpu_mir_helpers.h"\n'
  out += '#include "gpu/runtime/jit/materialized_join.h"\n'
  out += '#include "gpu/runtime/jit/ws_infrastructure.h"\n'
  out += '#include "gpu/runtime/stream_pool.h"\n'
  out += "using namespace SRDatalog::GPU;\n\n"
  for h in (extra_index_headers or []):
    out += f'#include "{h}"\n'
  if extra_index_headers:
    out += "\n"
  out += gen_schema_definitions_for_batch(decls) + "\n"
  out += gen_db_type_alias_for_batch(f"{ruleset_name}_DB", decls) + "\n"
  # Forward decls for every JitRunner referenced — emit them all so any
  # step body can resolve them without per-shard dependency tracking.
  for _rule_name, decl_str in runner_decls.items():
    out += decl_str
    out += "\n"
  out += "\n"
  # Runner struct declaration — MUST match main.cpp's declaration
  # token-for-token (ODR). That means emitting the SAME
  # `gen_load_data_method` template body, not just a forward decl.
  out += f"struct {ruleset_name}_Runner {{\n"
  out += f"  using DB = {ruleset_name}_DB_DeviceDB;\n\n"
  out += gen_load_data_method(decls)
  # step_N + run decls are appended by the caller (we don't know step
  # count here).
  return out


def gen_step_shard_file(
  ruleset_name: str,
  decls: list[RelationDecl],
  runner_decls: dict[str, str],
  mir_program: m.Program,
  step_bodies: list[str],
  step_idx: int,
  extra_index_headers: list[str] | None = None,
) -> str:
  '''Emit a standalone .cpp for one step_N out-of-line definition.

  Each shard repeats the schema/DB/GPU-runtime preamble so it is a
  valid TU — same cost as the batch files' self-contained layout.
  The gain is parallelism: N shards compile alongside the batches
  instead of serializing through main.cpp.
  '''
  device_db = f"{ruleset_name}_DB_DeviceDB"
  pre = _shared_runner_preamble(ruleset_name, decls, runner_decls,
                                extra_index_headers)
  # Complete the Runner struct declaration with all step_N + run sigs.
  for i in range(len(mir_program.steps)):
    pre += f"  static void step_{i}(DB& db, std::size_t max_iterations);\n"
  pre += (
    "  static void run(DB& db, std::size_t max_iterations = "
    "std::numeric_limits<int>::max());\n"
  )
  pre += "};\n\n"

  body_oop = _step_body_template_to_oop(
    step_bodies[step_idx], ruleset_name, step_idx, device_db,
  )
  return pre + body_oop + "\n"


def gen_run_dispatcher_file(
  ruleset_name: str,
  decls: list[RelationDecl],
  runner_decls: dict[str, str],
  mir_program: m.Program,
  canonical_indices: dict[str, list[int]] | None = None,
  extra_index_headers: list[str] | None = None,
) -> str:
  '''Emit the Runner::run() out-of-line definition in its own TU.
  Contains the step_0..step_N dispatch + print_size block.
  '''
  device_db = f"{ruleset_name}_DB_DeviceDB"
  pre = _shared_runner_preamble(ruleset_name, decls, runner_decls,
                                extra_index_headers)
  for i in range(len(mir_program.steps)):
    pre += f"  static void step_{i}(DB& db, std::size_t max_iterations);\n"
  pre += (
    "  static void run(DB& db, std::size_t max_iterations = "
    "std::numeric_limits<int>::max());\n"
  )
  pre += "};\n\n"
  body = (
    f"void {ruleset_name}_Runner::run({device_db}& db, "
    "std::size_t max_iterations) {\n"
  )
  for i, (step, is_rec) in enumerate(mir_program.steps):
    body += _gen_run_body_per_step(i, step, is_rec)
  body += _gen_final_print_block(decls, canonical_indices)
  body += "}\n"
  return pre + body


# -----------------------------------------------------------------------------
# Top-level assembly
# -----------------------------------------------------------------------------

def gen_main_file_preamble() -> str:
  '''Top-of-file preamble needed for main.cpp to compile as its own TU.
  In Nim's pipeline these using-directives come from the .nim driver;
  for the standalone Python path we emit them explicitly.'''
  return (
    '#include "srdatalog.h"\n'
    'using namespace SRDatalog;\n'
    'using namespace SRDatalog::AST::Literals;  // _s UDL\n'
    '\n'
  )


def gen_main_file_content(
  ruleset_name: str,
  decls: list[RelationDecl],
  mir_program: m.Program,
  step_bodies: list[str],
  runner_decls: dict[str, str],
  cache_dir_hint: str = "<jit-cache>",
  canonical_indices: dict[str, list[int]] | None = None,
  jit_batch_count: int = 0,
  emit_preamble: bool = False,
  extra_index_headers: list[str] | None = None,
  decl_only_runner: bool = False,
) -> str:
  '''Emit the full main-file string — mirrors Nim's `mir_cpp_str`.

  Structure:
    0. (opt) `#include "srdatalog.h"` + `using namespace` directives
       (`emit_preamble=True`, default — needed for the standalone Python
       path where main.cpp is a compilable TU. Pass False to match the
       pure Nim fragment byte-for-byte.)
    1. Relation typedefs + DB alias (`genRelCpp`)
    2. `using namespace SRDatalog::mir::dsl;`
    3. Kernel-decls block (device DB + GPU includes + JitRunner fwd decls)
    4. `namespace <Ruleset>_Plans { }` (empty for JIT target)
    5. `<Ruleset>_Runner` struct
    6. `// ======== JIT File-Based Compilation ========` summary
  '''
  out = ""
  if emit_preamble:
    out += gen_main_file_preamble()
    # Plugin index headers must appear BEFORE schema typedefs so their
    # class names are in scope when used as the 4th template argument.
    for h in (extra_index_headers or []):
      out += f'#include "{h}"\n'
    if extra_index_headers:
      out += "\n"
  out += gen_relation_typedefs(decls)
  out += gen_db_alias(ruleset_name, decls)
  out += "using namespace SRDatalog::mir::dsl;\n"
  out += gen_kernel_decls_block(
    ruleset_name, decls, runner_decls, cache_dir_hint,
    standalone_order=emit_preamble,
  )
  out += f"namespace {ruleset_name}_Plans {{\n"
  out += "}\n\n"
  if decl_only_runner:
    # Step/run bodies live in their own shards — keep main.cpp tiny.
    out += gen_runner_struct_declonly(ruleset_name, decls, mir_program)
  else:
    out += gen_runner_struct(
      ruleset_name, decls, mir_program, step_bodies, canonical_indices,
    )
  if jit_batch_count > 0:
    out += "\n// ======== JIT File-Based Compilation ========\n"
    out += f"// JIT kernels in {jit_batch_count} batch files\n"
  return out


def gen_unity_main_file_content(
  ruleset_name: str,
  decls: list[RelationDecl],
  mir_program: m.Program,
  step_bodies: list[str],
  runner_decls: dict[str, str],
  per_rule_runner_bodies: list[tuple[str, str]],
  *,
  canonical_indices: dict[str, list[int]] | None = None,
  extra_index_headers: list[str] | None = None,
) -> str:
  '''Emit ONE large .cpp that contains everything a project needs —
  preamble, schemas, DB alias, every JitRunner_X struct body, the
  host-side `_Runner` with step bodies, and the extern "C" shim.

  Compiling this single TU is much faster than compiling main.cpp +
  N batch files when PCH is unavailable, because `srdatalog.h` (the
  ~500 KB boost/hana/RMM header stack) parses ONCE instead of N
  times. For doop that's ~100s → ~20s on cold compile.
  '''
  out = '#include "srdatalog.h"\n'
  out += 'using namespace SRDatalog;\n'
  out += 'using namespace SRDatalog::AST::Literals;  // _s UDL\n\n'
  out += '#include <cstdint>\n'
  out += '#include <cooperative_groups.h>\n'
  out += 'namespace cg = cooperative_groups;\n\n'
  # JIT runtime headers — order matches what JIT_COMMON_INCLUDES uses
  # in the batch file path (we KEEP this exact order for ABI parity).
  out += '#include "gpu/device_sorted_array_index.h"\n'
  out += '#include "gpu/runtime/output_context.h"\n'
  out += '#include "gpu/runtime/jit/intersect_handles.h"\n'
  out += '#include "gpu/runtime/jit/jit_executor.h"\n'
  out += '#include "gpu/runtime/jit/materialized_join.h"\n'
  out += '#include "gpu/runtime/jit/ws_infrastructure.h"\n'
  out += '#include "gpu/runtime/query.h"\n'
  out += '#include "gpu/runtime/gpu_mir_helpers.h"\n'
  out += '#include "gpu/runtime/stream_pool.h"\n'
  out += '#include "gpu/init.h"  // SRDatalog::GPU::init_cuda\n\n'
  out += 'using SRDatalog::GPU::JIT::intersect_handles;\n'
  out += 'using namespace SRDatalog::GPU;\n\n'
  # Plugin index headers (Device2LevelIndex, DeviceTvjoinIndex, ...).
  for h in (extra_index_headers or []):
    out += f'#include "{h}"\n'
  if extra_index_headers:
    out += '\n'
  # Schemas + DB alias + device DB alias.
  out += gen_relation_typedefs(decls)
  out += gen_db_alias(ruleset_name, decls)
  out += "using namespace SRDatalog::mir::dsl;\n\n"
  names = ", ".join(d.rel_name for d in decls)
  blueprint = f"{ruleset_name}_DB_Blueprint"
  device_db = f"{ruleset_name}_DB_DeviceDB"
  out += f"using {blueprint} = SRDatalog::AST::Database<{names}>;\n"
  out += (
    f"using {device_db} = SRDatalog::AST::SemiNaiveDatabase"
    f"<{blueprint}, SRDatalog::GPU::DeviceRelationType>;\n\n"
  )

  # All JitRunner_X struct bodies, inline.
  out += "// ========== JitRunner structs (unity-inlined) ==========\n"
  for rule_name, body in per_rule_runner_bodies:
    out += f"// --- {rule_name} ---\n"
    out += body
    out += "\n"

  # Empty plans namespace (placeholder from Nim codegen).
  out += f"namespace {ruleset_name}_Plans {{}}\n\n"

  # Host-side Runner struct with step bodies inline (template form).
  out += gen_runner_struct(
    ruleset_name, decls, mir_program, step_bodies, canonical_indices,
  )
  return out


def gen_extern_c_shim(
  ruleset_name: str,
  decls: list[RelationDecl],
) -> str:
  '''Emit an `extern "C"` shim the Python ctypes loader can call.

  Exposes (all C-ABI, return 0 on success / nonzero on error):
    - `srdatalog_init()`                     — init CUDA
    - `srdatalog_load_csv(rel, path)`        — load_from_file for one relation
    - `srdatalog_run(max_iters)`             — copy-to-device + _Runner::run
    - `srdatalog_shutdown()`                 — free host DB
    - `srdatalog_size(rel_name)`             — FULL_VER canonical index size

  The shim uses a file-scope `HostDB*` holding the live SemiNaiveDatabase
  so Python can stage data via multiple `load_csv` calls before `run`.
  '''
  ext_db = f"{ruleset_name}_DB"
  blueprint = f"{ext_db}_Blueprint"
  host_db = f"{blueprint}_HostDB"

  out = [
    "// ======== Python ctypes shim (extern \"C\") ========",
    '#include "gpu/init.h"  // SRDatalog::GPU::init_cuda',
    "",
    f"using {host_db} = SRDatalog::AST::SemiNaiveDatabase<{blueprint}>;",
    f"static {host_db}* g_host_db = nullptr;",
    "",
    'extern "C" {',
    "",
    "int srdatalog_init() {",
    "  try { SRDatalog::GPU::init_cuda(); return 0; }",
    "  catch (const std::exception& e) {",
    '    std::cerr << "srdatalog_init: " << e.what() << std::endl;',
    "    return 1;",
    "  } catch (...) { return 2; }",
    "}",
    "",
    "int srdatalog_load_csv(const char* rel_name, const char* path) {",
    "  if (!rel_name || !path) return 1;",
    f"  if (!g_host_db) g_host_db = new {host_db}();",
    "  try {",
    "    std::string rn(rel_name);",
  ]
  # Only emit load_from_file dispatch for relations marked with
  # `input_file=...`. Non-input relations (pure IDB, or relations
  # with non-default index types like Device2LevelIndex) fail to
  # instantiate `load_from_file<>` because build_all_indexes requires
  # a ValueRange type the custom index doesn't expose.
  loadable = [d for d in decls if d.input_file]
  first = True
  for d in loadable:
    kw = "if" if first else "else if"
    first = False
    out.append(f'    {kw} (rn == "{d.rel_name}") '
               f'SRDatalog::load_from_file<{d.rel_name}>(*g_host_db, path);')
  if loadable:
    out.append('    else { std::cerr << "srdatalog_load_csv: unknown or non-input relation " << rn << std::endl; return 2; }')
  else:
    out.append('    std::cerr << "srdatalog_load_csv: no loadable relations declared (set input_file= on Relation)" << std::endl;')
    out.append('    return 2;')
  out += [
    "    return 0;",
    "  } catch (const std::exception& e) {",
    '    std::cerr << "srdatalog_load_csv: " << e.what() << std::endl;',
    "    return 3;",
    "  }",
    "}",
    "",
    "// Batch-load every relation declared with input_file from a root directory.",
    "// Matches the Nim driver pattern: single call, relies on _Runner::load_data.",
    "int srdatalog_load_all(const char* data_dir) {",
    "  if (!data_dir) return 1;",
    f"  if (!g_host_db) g_host_db = new {host_db}();",
    "  try {",
    f"    {ruleset_name}_Runner::load_data(*g_host_db, std::string(data_dir));",
    "    return 0;",
    "  } catch (const std::exception& e) {",
    '    std::cerr << "srdatalog_load_all: " << e.what() << std::endl;',
    "    return 2;",
    "  }",
    "}",
    "",
    "int srdatalog_run(unsigned long long max_iters) {",
    "  if (!g_host_db) return 1;",
    "  try {",
    "    auto device_db = SRDatalog::GPU::copy_host_to_device(*g_host_db);",
    f"    {ruleset_name}_Runner::run(device_db, max_iters ? (std::size_t)max_iters : std::numeric_limits<int>::max());",
    "    return 0;",
    "  } catch (const std::exception& e) {",
    '    std::cerr << "srdatalog_run: " << e.what() << std::endl;',
    "    return 2;",
    "  }",
    "}",
    "",
    "unsigned long long srdatalog_size(const char* rel_name) {",
    "  if (!g_host_db || !rel_name) return 0;",
    "  std::string rn(rel_name);",
  ]
  for d in decls:
    cols = ", ".join(str(i) for i in range(len(d.types)))
    out.append(f'  if (rn == "{d.rel_name}") {{')
    out.append(f'    auto& rel = get_relation_by_schema<{d.rel_name}, FULL_VER>(*g_host_db);')
    out.append(f'    SRDatalog::IndexSpec idx{{{cols}}};')
    out.append('    return rel.has_index(idx) ? (unsigned long long)rel.get_index(idx).root().degree() : 0ULL;')
    out.append('  }')
  out += [
    "  return 0;",
    "}",
    "",
    "int srdatalog_shutdown() {",
    "  if (g_host_db) { delete g_host_db; g_host_db = nullptr; }",
    "  return 0;",
    "}",
    "",
    "}  // extern \"C\"",
    "",
  ]
  return "\n".join(out)
