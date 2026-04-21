'''JIT kernel functor wrappers.

Port of src/srdatalog/codegen/target_jit/jit_kernel_functor.nim.

Two kinds of output:

  jit_functor_start / jit_functor_end
    The `struct Kernel_<rule> { ... __device__ void operator()(...) const { ... } };`
    envelope. jit_full_kernel uses these to wrap jit_pipeline output into
    the complete functor struct emitted into a JIT batch file.

  jit_execute_pipeline(node)
    The JitRunner_<rule> wrapper that ExecuteJit dispatches through.
    References the Kernel_<rule> functor + the SourceSpecs / DestSpecs /
    AllSchemas type tuples computed from the ExecutePipeline MIR node.
    TMP-backend helpers (genIndexSpecsTupleType, extractAllIndexSpecs)
    haven't been ported yet — for now the Runner form is stubbed with
    placeholder type tuples and will be finalized once the TMP helpers
    are available or we inline the spec-tuple formatting here.

jit_full_kernel is the core entry: given a rule name and its MIR
pipeline, emit the full kernel functor + inner pipeline body.
'''

from __future__ import annotations

import srdatalog.mir.types as m
from srdatalog.codegen.jit.context import CodeGenContext, new_code_gen_context
from srdatalog.codegen.jit.emit_helpers import (
  assign_handle_positions,
  count_handles_in_pipeline,
)
from srdatalog.codegen.jit.pipeline import jit_pipeline

# -----------------------------------------------------------------------------
# Functor envelope
# -----------------------------------------------------------------------------


def gen_dedup_table_struct(arity: int) -> str:
  '''Emit the DedupTable struct nested inside a rule's kernel scope.

  GPU hash table that:
    - `try_insert(thread_id, v0, v1, ...)` — atomicCAS claims the slot,
      returns true iff this thread won OR the slot was unoccupied.
      Used during the count phase.
    - `check_winner(thread_id, v0, v1, ...)` — read-only lookup of the
      previously-claimed slot. Returns true iff this thread is the
      winner for this tuple. Used during the materialize phase.

  Both paths use linear probing over 128 slots from an FNV-1a hash of
  the arity-sized column tuple. `capacity` is power-of-2 — host-side
  runner sets it up + zero-initializes `hash_slots` between phases.

  Mirrors Nim's DedupTable emission in jit_complete_runner.nim.
  '''
  code = (
    "  // GPU dedup hash table: full 64-bit hash + separate thread_id array\n"
    "  struct DedupTable {\n"
    "    unsigned long long* hash_slots; // full 64-bit hash per slot\n"
    "    uint32_t* tid_slots;            // winner thread_id per slot\n"
    "    uint32_t capacity;              // must be power of 2\n\n"
  )
  # compute_hash
  code += "    __device__ __forceinline__ unsigned long long compute_hash(\n"
  for col in range(arity):
    sep = "," if col < arity - 1 else ")"
    code += f"        uint32_t v{col}{sep}\n"
  code += "    {\n"
  code += "      uint64_t h = 14695981039346656037ULL;\n"
  for col in range(arity):
    code += f"      h ^= (uint64_t)v{col}; h *= 1099511628211ULL;\n"
  code += "      return h | 1ULL; // ensure non-zero\n"
  code += "    }\n\n"

  # try_insert — count phase, atomicCAS claims
  code += "    __device__ __forceinline__ bool try_insert(\n"
  code += "        uint32_t thread_id,\n"
  for col in range(arity):
    sep = "," if col < arity - 1 else ")"
    code += f"        uint32_t v{col}{sep}\n"
  code += "    {\n"
  code += "      unsigned long long h = compute_hash("
  for col in range(arity):
    code += f"v{col}" + (", " if col < arity - 1 else ");\n")
  code += "      uint32_t base = (uint32_t)(h ^ (h >> 32)) & (capacity - 1);\n"
  code += "      for (uint32_t p = 0; p < 128; p++) {\n"
  code += "        uint32_t s = (base + p) & (capacity - 1);\n"
  code += "        unsigned long long old = atomicCAS(&hash_slots[s], 0ULL, h);\n"
  code += "        if (old == 0ULL) { tid_slots[s] = thread_id; return true; } // claimed\n"
  code += "        if (old == h) return false; // same hash = duplicate\n"
  code += "        // old != h: collision with different tuple -> probe next\n"
  code += "      }\n"
  code += "      return true; // probe overflow -> emit (safe)\n"
  code += "    }\n\n"

  # check_winner — materialize phase, read-only
  code += "    __device__ __forceinline__ bool check_winner(\n"
  code += "        uint32_t thread_id,\n"
  for col in range(arity):
    sep = "," if col < arity - 1 else ")"
    code += f"        uint32_t v{col}{sep}\n"
  code += "    {\n"
  code += "      unsigned long long h = compute_hash("
  for col in range(arity):
    code += f"v{col}" + (", " if col < arity - 1 else ");\n")
  code += "      uint32_t base = (uint32_t)(h ^ (h >> 32)) & (capacity - 1);\n"
  code += "      for (uint32_t p = 0; p < 128; p++) {\n"
  code += "        uint32_t s = (base + p) & (capacity - 1);\n"
  code += "        unsigned long long stored = hash_slots[s];\n"
  code += "        if (stored == h) return tid_slots[s] == thread_id; // found: am I winner?\n"
  code += "        if (stored == 0ULL) return true; // not found -> probe overflow, emit\n"
  code += "        // different hash -> probe next (collision resolution)\n"
  code += "      }\n"
  code += "      return true; // probe overflow -> emit\n"
  code += "    }\n"
  code += "  };\n\n"
  return code


def jit_functor_start(
  rule_name: str,
  scalar_mode: bool = False,
  dedup_hash: bool = False,
) -> str:
  '''Open `struct Kernel_<rule> { ... operator()(...) const {`.

  When `dedup_hash=True`, operator() takes an additional
  `DedupTable dedup_table` parameter so the emit paths in
  jit_insert_into can call `dedup_table.try_insert(...)` /
  `.check_winner(...)` to deduplicate output tuples.
  '''
  group_size = 1 if scalar_mode else 32
  mode_comment = (
    "// SCALAR MODE: 1 thread per row, sequential search"
    if scalar_mode
    else "// WARP MODE: 32 threads share 1 row, cooperative search"
  )
  dedup_param = "      DedupTable dedup_table,\n" if dedup_hash else ""
  return (
    mode_comment + "\n"
    f"struct Kernel_{rule_name} {{\n"
    "  static constexpr int kBlockSize = 256;\n"
    f"  static constexpr int kGroupSize = {group_size};\n\n"
    "  template <typename Tile, typename Views, typename ValueType, typename Output>\n"
    "  __device__ void operator()(\n"
    "      Tile& tile,\n"
    "      const Views* views,\n"
    "      const ValueType* __restrict__ root_unique_values,\n"
    "      uint32_t num_unique_root_keys,\n"
    "      uint32_t num_root_keys,\n"
    "      uint32_t warp_id,\n"
    "      uint32_t num_warps,\n"
    f"{dedup_param}"
    "      Output& output\n"
    "  ) const {\n"
  )


def jit_functor_end() -> str:
  return "  }\n};\n"


def jit_kernel_declaration(rule_name: str) -> str:
  '''Declaration-only variant (no body). Not used by the current batch
  file generator but ported for completeness.'''
  return (
    "// Kernel declaration for extern template (implementation in batch file)\n"
    f"struct Kernel_{rule_name} {{\n"
    "  template <typename Tile, typename Views, typename ValueType, typename Output>\n"
    "  __device__ void operator()(\n"
    "      Tile& tile,\n"
    "      const Views* views,\n"
    "      const ValueType* __restrict__ root_unique_values,\n"
    "      uint32_t num_unique_root_keys,\n"
    "      uint32_t num_root_keys,\n"
    "      uint32_t warp_id,\n"
    "      uint32_t num_warps,\n"
    "      Output& output\n"
    "  ) const;\n"
    "};\n\n"
  )


# -----------------------------------------------------------------------------
# Full kernel emit (functor start + pipeline body + functor end)
# -----------------------------------------------------------------------------


def _first_dest_arity(pipeline: list[m.MirNode]) -> int:
  '''Arity of the first InsertInto's column set. Used to size the
  DedupTable's hash function (one v0...vN-1 parameter per column).'''
  for op in pipeline:
    if isinstance(op, m.InsertInto):
      return len(op.vars)
  return 0


def jit_full_kernel(
  rule_name: str,
  pipeline: list[m.MirNode],
  ctx: CodeGenContext,
) -> str:
  '''Complete kernel functor emit — preamble banner + struct envelope +
  jit_pipeline body + envelope close. Sets `ctx.indent = 4` to match
  `operator()` body indentation.

  When `ctx.dedup_hash_enabled=True`, emits the DedupTable struct
  before operator() and adds the `DedupTable dedup_table` parameter
  to the signature so jit_insert_into's dedup path resolves against
  a real in-scope object.
  '''
  num_handles = count_handles_in_pipeline(pipeline)
  banner = (
    "// =============================================================\n"
    f"// JIT-Generated Kernel Functor: {rule_name}\n"
    f"// Handles: {num_handles}\n"
    "// =============================================================\n\n"
  )
  # Open struct + pre-operator() DedupTable when dedup_hash.
  header = banner + jit_functor_start(
    rule_name,
    ctx.scalar_mode,
    dedup_hash=ctx.dedup_hash_enabled,
  )
  # DedupTable struct goes INSIDE the kernel struct but BEFORE
  # operator() — Nim emits it between the constexpr decls and the
  # operator() line. Our jit_functor_start emits constexpr + operator()
  # in one block, so we inject the DedupTable just after the "{"
  # that opens the kernel struct. Simplest: include it right after the
  # banner, outside the functor (standalone struct); operator() then
  # references it by unqualified name. Note: Nim puts it inside the
  # Kernel_X struct (but at module-ish scope for nested access); either
  # way a subsequent reference resolves.
  if ctx.dedup_hash_enabled:
    arity = _first_dest_arity(pipeline)
    # Inject the struct definition right after `struct Kernel_<rule> {`
    # so DedupTable is nested inside. Locate the first `{\n`.
    open_brace = header.find("{\n")
    assert open_brace != -1, "jit_functor_start must contain a struct open brace"
    # Insert after the first two lines (struct + kBlockSize + kGroupSize),
    # matching Nim's placement.
    dedup_struct = gen_dedup_table_struct(arity)
    # Insert after `static constexpr int kGroupSize = ...;\n\n`
    marker = "static constexpr int kGroupSize = "
    marker_pos = header.find(marker)
    if marker_pos != -1:
      newline_after = header.find("\n\n", marker_pos)
      if newline_after != -1:
        insert_at = newline_after + 2
        header = header[:insert_at] + dedup_struct + header[insert_at:]

  result = header
  ctx.indent = 4
  result += jit_pipeline(pipeline, [], ctx)
  result += jit_functor_end()
  return result


def jit_kernel_full(node: m.ExecutePipeline) -> str:
  '''Entry point from an ExecutePipeline MIR node — mirrors Nim's
  jitKernelFull. Uses a fresh context (counting phase default) and
  runs `assign_handle_positions` so pipeline ops get sequential
  handle_start values before emit.

  Flag flow: pull `dedup_hash`, `work_stealing`, `block_group`,
  `count`, `fanout` off the ExecutePipeline node onto `ctx` so
  downstream emitters (jit_insert_into, jit_nested_cartesian_join,
  jit_pipeline dispatch) see the right state.

  Note: this mutates the pipeline in place (matches Nim's `var` param).
  Callers passing a pipeline they plan to reuse should deep-copy first.
  '''
  assert isinstance(node, m.ExecutePipeline)
  pipeline = list(node.pipeline)
  assign_handle_positions(pipeline)

  ctx = new_code_gen_context()
  # Propagate ExecutePipeline feature flags onto the context (Nim does
  # this in jit_complete_runner lines 280-290 but our functor path
  # needs it here).
  ctx.dedup_hash_enabled = node.dedup_hash
  # NOTE: ws_enabled / bg_enabled / is_fan_out_explore are wired when
  # those feature branches land. For now leave them at False — the
  # pipeline emitter will still raise NotImplementedError if a node
  # has one of those flags set but the context doesn't, making the
  # gap visible.
  if ctx.dedup_hash_enabled:
    # Capture the first InsertInto's vars as the hash key set.
    for op in pipeline:
      if isinstance(op, m.InsertInto):
        ctx.dedup_hash_vars = list(op.vars)
        break

  return jit_full_kernel(node.rule_name, pipeline, ctx)


def jit_kernel_definition(
  rule_name: str,
  pipeline: list[m.MirNode],
  ctx: CodeGenContext,
) -> str:
  '''Out-of-line operator() definition for extern-template batch files.
  Ported structurally; we don't exercise this path yet since our
  current batch-file strategy emits the full struct inline.'''
  num_handles = count_handles_in_pipeline(pipeline)
  result = (
    "// =============================================================\n"
    f"// JIT-Generated Kernel Definition: {rule_name}\n"
    "// Out-of-line operator() definition for extern template\n"
    f"// Handles: {num_handles}\n"
    "// =============================================================\n\n"
    "template <typename Tile, typename Views, typename ValueType, typename Output>\n"
    f"__device__ void Kernel_{rule_name}::operator()(\n"
    "    Tile& tile,\n"
    "    const Views* views,\n"
    "    const ValueType* __restrict__ root_unique_values,\n"
    "    uint32_t num_unique_root_keys,\n"
    "    uint32_t num_root_keys,\n"
    "    uint32_t warp_id,\n"
    "    uint32_t num_warps,\n"
    "    Output& output\n"
    ") const {\n"
  )
  ctx.indent = 4
  result += jit_pipeline(pipeline, [], ctx)
  result += "}\n\n"
  return result
