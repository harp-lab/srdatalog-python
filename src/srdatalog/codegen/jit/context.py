'''Code-generation context + C++ expression helpers.

Port of src/srdatalog/codegen/target_jit/jit_base.nim.

The `CodeGenContext` is the big state object threaded through every
emitter in the JIT backend. It tracks bound variables, handle/view
name tables, indentation, thread group size, and the menagerie of
feature-flag state (work-stealing, block-group, tiled Cartesian,
dedup-hash, etc.). Every field mirrors the Nim source so field names
line up 1:1 with emitter ports.

`CodeGenHooks` lets feature-specific modules override emit/materialize
and runner-level hooks without checking flags inline (same role as
Halide's schedule/algorithm split). Defaults are no-ops / identity;
BG / WS / dedup modules will override individual hooks in later commits.

The `gen_*` helpers at the bottom dispatch through the index plugin
registry, so custom index types (like Device2LevelIndex) can override
C++ expression shapes without touching emitter code.
'''
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

from srdatalog.codegen.jit.plugin import (
  plugin_gen_root_handle, plugin_gen_degree, plugin_gen_valid,
  plugin_gen_get_value_at, plugin_gen_get_value, plugin_gen_child,
  plugin_gen_child_range, plugin_gen_iterators,
  plugin_chained_prefix_calls, plugin_chained_prefix_with_last_lower_bound,
)


# -----------------------------------------------------------------------------
# C++ keywords — sanitize variable names to avoid collisions
# -----------------------------------------------------------------------------

CPP_KEYWORDS: frozenset[str] = frozenset({
  "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit",
  "atomic_noexcept", "auto", "bitand", "bitor", "bool", "break", "case", "catch",
  "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const",
  "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await",
  "co_return", "co_yield", "decltype", "default", "delete", "do", "double",
  "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false", "float",
  "for", "friend", "goto", "if", "inline", "int", "long", "mutable", "namespace", "new",
  "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq", "private",
  "protected", "public", "register", "reinterpret_cast", "requires", "return", "short",
  "signed", "sizeof", "static", "static_assert", "static_cast", "struct", "switch",
  "synchronized", "template", "this", "thread_local", "throw", "true", "try", "typedef",
  "typeid", "typename", "union", "unsigned", "using", "virtual", "void", "volatile",
  "wchar_t", "while", "xor", "xor_eq",
})


def sanitize_var_name(name: str) -> str:
  '''Append `_val` to any C++ keyword so it's safe as a C++ identifier.'''
  return name + "_val" if name in CPP_KEYWORDS else name


# -----------------------------------------------------------------------------
# NegPreNarrowInfo
# -----------------------------------------------------------------------------

@dataclass
class NegPreNarrowInfo:
  '''Pre-narrowing info for a negation handle applied before the Cartesian
  loop. Pre-cartesian prefix vars are applied cooperatively once; in-cartesian
  vars are applied per-thread via prefix_seq inside the loop.
  '''
  var_name: str = ""
  pre_vars: list[str] = field(default_factory=list)
  in_cartesian_vars: list[str] = field(default_factory=list)
  pre_consts: list[tuple[int, int]] = field(default_factory=list)  # (col, constValue)
  view_var: str = ""
  rel_name: str = ""
  index_type: str = ""


# -----------------------------------------------------------------------------
# RunnerGenState — captured once per kernel, passed to runner hooks
# -----------------------------------------------------------------------------

@dataclass
class RunnerGenState:
  node: Any = None                                 # MirNode (ExecutePipeline)
  db_type_name: str = ""
  rule_name: str = ""
  runner_prefix: str = ""
  rel_index_types: dict[str, str] = field(default_factory=dict)
  mutable_pipe: list[Any] = field(default_factory=list)
  first_schema: str = ""
  first_version: str = ""
  first_index: list[int] = field(default_factory=list)
  dest_arities: list[int] = field(default_factory=list)
  total_view_count: int = 0
  is_balanced: bool = False
  is_work_stealing: bool = False
  is_block_group: bool = False
  is_dedup_hash: bool = False
  is_count: bool = False


# -----------------------------------------------------------------------------
# CodeGenHooks — feature-specific emit/runner overrides
# -----------------------------------------------------------------------------

@dataclass
class CodeGenHooks:
  '''Feature-specific codegen hooks, resolved once per kernel. Default
  implementations are identity / no-op; BG / WS / dedup modules will
  supply their own.
  '''
  # Emit hooks (decompose jit_insert_into)
  wrap_emit: Optional[Callable[[str, "CodeGenContext"], str]] = None
  emit_count: Optional[Callable[[str, str, bool, "CodeGenContext"], str]] = None
  emit_materialize: Optional[Callable[
    [str, str, list[str], bool, "CodeGenContext"], str
  ]] = None

  # Pipeline hooks (decompose jit_nested_pipeline)
  pre_column_join: Optional[Callable[[Any, "CodeGenContext"], None]] = None
  post_column_join: Optional[Callable[[Any, "CodeGenContext"], None]] = None
  pre_cartesian_join: Optional[Callable[
    [Any, list[Any], "CodeGenContext"], None
  ]] = None

  # Root dispatch hook
  root_column_join: Optional[Callable[[Any, "CodeGenContext", str], str]] = None

  # Runner hooks (decompose jit_complete_runner)
  emit_extra_types: Optional[Callable[[RunnerGenState], str]] = None
  emit_extra_kernels: Optional[Callable[[RunnerGenState], str]] = None
  emit_phase_methods: Optional[Callable[[RunnerGenState], str]] = None
  emit_execute_body: Optional[Callable[[RunnerGenState], str]] = None


def default_hooks() -> CodeGenHooks:
  '''Baseline hook implementations — no-ops / identity. Feature modules
  (BG, WS, dedup) override individual hooks via their own factories.
  '''
  return CodeGenHooks(
    wrap_emit=lambda code, _ctx: code,
    emit_count=None,        # set by feature modules
    emit_materialize=None,
    pre_column_join=lambda _op, _ctx: None,
    post_column_join=lambda _op, _ctx: None,
    pre_cartesian_join=lambda _op, _rest, _ctx: None,
    root_column_join=None,  # set by feature modules
    emit_extra_types=lambda _state: "",
    emit_extra_kernels=lambda _state: "",
    emit_phase_methods=lambda _state: "",
    emit_execute_body=None,  # set by feature modules
  )


# -----------------------------------------------------------------------------
# CodeGenContext — the big state object
# -----------------------------------------------------------------------------

@dataclass
class CodeGenContext:
  '''Threaded through every emitter. Field order + names mirror Nim's
  CodeGenContext 1:1 so port diffs stay local.
  '''
  # -- Core scope tracking --
  bound_vars: list[str] = field(default_factory=list)
  handle_vars: dict[str, str] = field(default_factory=dict)
  view_vars: dict[str, str] = field(default_factory=dict)
  indent: int = 2                                # start at function-body indent
  name_counter: int = 0
  debug: bool = True
  output_var_name: str = "output"                # legacy single-output default
  output_vars: dict[str, str] = field(default_factory=dict)

  # -- Tile / group state --
  group_size: int = 32                           # full warp initially
  tile_var: str = "tile"
  parent_tile_var: str = "tile"
  is_leaf_level: bool = False

  # -- Counting vs materialize phase --
  is_counting: bool = False

  # -- Balanced scan state --
  balanced_idx1: str = ""
  balanced_idx2: str = ""

  # -- JIT mode flag --
  is_jit_mode: bool = False

  # -- Cartesian-loop state --
  inside_cartesian: bool = False
  cartesian_bound_vars: list[str] = field(default_factory=list)

  # -- Tiled Cartesian (2-source) optimization --
  tiled_cartesian_enabled: bool = False
  tiled_cartesian_valid_var: str = ""
  tiled_cartesian_ballot_done: bool = False

  # -- Relation → index type (for plugin dispatch) --
  rel_index_types: dict[str, str] = field(default_factory=dict)

  # -- View slot mapping (multi-view sources) --
  view_slot_offsets: dict[int, int] = field(default_factory=dict)

  # -- Scalar (thread-per-row) mode --
  scalar_mode: bool = False

  # -- Block-group histogram kernel flag --
  bg_histogram_mode: bool = False
  cartesian_as_product: bool = False

  # -- Dedup-hash state --
  dedup_hash_enabled: bool = False
  dedup_hash_vars: list[str] = field(default_factory=list)

  # -- Block-group state --
  bg_enabled: bool = False
  bg_warp_begin_var: str = ""
  bg_warp_end_var: str = ""
  bg_cumulative_var: str = ""
  bg_done_var: str = ""

  # -- Fan-out explore mode --
  is_fan_out_explore: bool = False

  # -- Work-stealing state --
  ws_enabled: bool = False
  ws_level: int = 0
  ws_queue_var: str = ""
  ws_range_board_var: str = ""
  ws_live_handles: list[tuple] = field(default_factory=list)
  # Each entry: (var_name, handle_idx, rel_name, index_spec, prefix_vars,
  # view_slot_expr). Kept as tuple rather than dataclass to stay close to Nim.
  ws_has_cartesian: bool = False
  ws_cartesian_valid_var: str = ""
  ws_cartesian_bound_vars: list[str] = field(default_factory=list)

  # -- Negation pre-narrowing --
  neg_pre_narrow: dict[int, NegPreNarrowInfo] = field(default_factory=dict)

  # -- Feature hooks --
  hooks: CodeGenHooks = field(default_factory=default_hooks)


def new_code_gen_context() -> CodeGenContext:
  '''Fresh context with Nim-matching defaults.'''
  return CodeGenContext()


# -----------------------------------------------------------------------------
# Indentation + scope utilities
# -----------------------------------------------------------------------------

def ind(ctx: CodeGenContext) -> str:
  '''Current indentation string (2-space levels).'''
  return "  " * ctx.indent


def inc_indent(ctx: CodeGenContext) -> None:
  ctx.indent += 1


def dec_indent(ctx: CodeGenContext) -> None:
  ctx.indent = max(0, ctx.indent - 1)


def gen_unique_name(ctx: CodeGenContext, prefix: str) -> str:
  '''Bump the per-context counter and return `<prefix>_<n>`.'''
  ctx.name_counter += 1
  return f"{prefix}_{ctx.name_counter}"


def with_bound_var(ctx: CodeGenContext, var_name: str) -> CodeGenContext:
  '''Return a shallow copy of `ctx` with `var_name` added to `bound_vars`.'''
  import copy
  out = copy.copy(ctx)
  out.bound_vars = list(ctx.bound_vars)
  out.bound_vars.append(var_name)
  return out


def is_var_bound(ctx: CodeGenContext, var_name: str) -> bool:
  return var_name in ctx.bound_vars


def get_rel_index_type(ctx: CodeGenContext, rel_name: str) -> str:
  '''Look up the index type for a relation. Empty string = DSAI default.'''
  return ctx.rel_index_types.get(rel_name, "")


def get_view_slot_base(ctx: CodeGenContext, handle_idx: int) -> int:
  '''Base view slot for a source. Falls back to `handle_idx` when no
  override is set (single-view / legacy case).'''
  return ctx.view_slot_offsets.get(handle_idx, handle_idx)


# -----------------------------------------------------------------------------
# Name / key generators
# -----------------------------------------------------------------------------

def gen_view_access(handle_idx: int) -> str:
  '''`views[i]` — positional view access.'''
  return f"views[{handle_idx}]"


def gen_view_var_name(rel_name: str, handle_idx: int) -> str:
  '''`view_<rel>_<handle>` — readable view variable name.'''
  return f"view_{rel_name}_{handle_idx}"


def gen_handle_var_name(rel_name: str, handle_idx: int, ctx: CodeGenContext) -> str:
  '''Unique handle variable name `h_<rel>_<handle>_<n>`.'''
  return gen_unique_name(ctx, f"h_{rel_name}_{handle_idx}")


def gen_index_spec_key(rel_name: str, index: list[int], version: str = "") -> str:
  '''Key for handle/view lookup: `Rel_<cols joined by _>` optionally
  suffixed with `_<VER>`. Differentiates DELTA from FULL sources that
  share a relation + index.'''
  base = rel_name + "_" + "_".join(str(c) for c in index)
  return base + "_" + version if version else base


def gen_handle_state_key(
  rel_name: str, index: list[int], bound_prefixes: list[str], version: str = "",
) -> str:
  '''Semantic key tying together (rel, idx, bound prefixes, version). Lets
  handle reuse work across different MIR handleIdx values that point at
  the same narrowed trie path.'''
  base = gen_index_spec_key(rel_name, index, version)
  if not bound_prefixes:
    return base
  return base + "_" + "_".join(bound_prefixes)


# -----------------------------------------------------------------------------
# Plugin-dispatched C++ expression wrappers
# -----------------------------------------------------------------------------

def gen_root_handle(view_var: str, index_type: str = "") -> str:
  '''Root handle: `HandleType(0, view.num_rows_, 0)` (DSAI default).'''
  return plugin_gen_root_handle(view_var, index_type)


def gen_root_handle_from_view_idx(view_idx: int, index_type: str = "") -> str:
  '''Shorthand — inline the `views[i]` form.'''
  return plugin_gen_root_handle(gen_view_access(view_idx), index_type)


def gen_degree(handle: str, index_type: str = "") -> str:
  return plugin_gen_degree(handle, index_type)


def gen_valid(handle: str, index_type: str = "") -> str:
  return plugin_gen_valid(handle, index_type)


def gen_get_value_at(handle: str, view_var: str, idx: str, index_type: str = "") -> str:
  return plugin_gen_get_value_at(handle, view_var, idx, index_type)


def gen_get_value(view_var: str, col: int, pos: str, index_type: str = "") -> str:
  return plugin_gen_get_value(view_var, col, pos, index_type)


def gen_child(handle: str, idx: str, index_type: str = "") -> str:
  return plugin_gen_child(handle, idx, index_type)


def gen_child_range(
  handle: str, pos: str, key: str, tile: str, view_var: str, index_type: str = "",
) -> str:
  return plugin_gen_child_range(handle, pos, key, tile, view_var, index_type)


def gen_iterators(handle: str, view_var: str, index_type: str = "") -> str:
  return plugin_gen_iterators(handle, view_var, index_type)


def gen_chained_prefix_calls(
  parent_handle: str,
  prefix_vars: list[str],
  view_var: str,
  cartesian_bound_vars: list[str] | None = None,
  scalar_mode: bool = False,
  index_type: str = "",
) -> str:
  '''Chained .prefix(...) calls. Prefix vars go through sanitize_var_name
  first (keyword escape).'''
  sanitized = [sanitize_var_name(v) for v in prefix_vars]
  return plugin_chained_prefix_calls(
    parent_handle, sanitized, view_var, cartesian_bound_vars, scalar_mode, index_type,
  )


def gen_chained_prefix_with_last_lower_bound(
  parent_handle: str,
  prefix_vars: list[str],
  view_var: str,
  cartesian_bound_vars: list[str] | None = None,
  scalar_mode: bool = False,
  index_type: str = "",
) -> str:
  '''Chained .prefix(...) with last key using .prefix_lower_bound().'''
  sanitized = [sanitize_var_name(v) for v in prefix_vars]
  return plugin_chained_prefix_with_last_lower_bound(
    parent_handle, sanitized, view_var, cartesian_bound_vars, scalar_mode, index_type,
  )


def gen_chained_prefix_calls_seq(
  parent_handle: str, prefix_vars: list[str], view_var: str, index_type: str = "",
) -> str:
  '''All-sequential variant — every key applied via prefix_seq.'''
  sanitized = [sanitize_var_name(v) for v in prefix_vars]
  return plugin_chained_prefix_calls(
    parent_handle, sanitized, view_var, [], True, index_type,
  )
