"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/polonius/polonius_test.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/polonius/polonius_test.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dataset_const import load_meta, resolve_program_consts
from srdatalog.dsl import Filter, Program, Relation, Var

# ----- Relations ----------------------------------------------

subset_base = Relation(
  "subset_base",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="subset_base.facts",
)
cfg_edge = Relation(
  "cfg_edge",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="cfg_edge.facts",
)
loan_issued_at = Relation(
  "loan_issued_at",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="loan_issued_at.facts",
)
universal_region = Relation(
  "universal_region", 1, column_types=(int,), input_file="universal_region.facts"
)
var_used_at = Relation(
  "var_used_at",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="var_used_at.facts",
)
loan_killed_at = Relation(
  "loan_killed_at",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="loan_killed_at.facts",
)
known_placeholder_subset_input = Relation(
  "known_placeholder_subset_input",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="known_placeholder_subset.facts",
)
var_dropped_at = Relation(
  "var_dropped_at",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="var_dropped_at.facts",
)
drop_of_var_derefs_origin = Relation(
  "drop_of_var_derefs_origin",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="drop_of_var_derefs_origin.facts",
)
var_defined_at = Relation(
  "var_defined_at",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="var_defined_at.facts",
)
child_path = Relation(
  "child_path",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="child_path.facts",
)
path_moved_at_base = Relation(
  "path_moved_at_base",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="path_moved_at_base.facts",
)
path_assigned_at_base = Relation(
  "path_assigned_at_base",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="path_assigned_at_base.facts",
)
path_accessed_at_base = Relation(
  "path_accessed_at_base",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="path_accessed_at_base.facts",
)
path_is_var = Relation(
  "path_is_var",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="path_is_var.facts",
)
loan_invalidated_at = Relation(
  "loan_invalidated_at",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="loan_invalidated_at.facts",
)
use_of_var_derefs_origin = Relation(
  "use_of_var_derefs_origin",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="use_of_var_derefs_origin.facts",
)
subset = Relation(
  "subset",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  print_size=True,
  index_type="SRDatalog::GPU::Device2LevelIndex",
)
origin_live_on_entry = Relation(
  "origin_live_on_entry",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
  index_type="SRDatalog::GPU::Device2LevelIndex",
)
origin_contains_loan_on_entry = Relation(
  "origin_contains_loan_on_entry",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  print_size=True,
  index_type="SRDatalog::GPU::Device2LevelIndex",
)
loan_live_at = Relation(
  "loan_live_at",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
errors = Relation(
  "errors",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
placeholder_origin = Relation("placeholder_origin", 1, column_types=(int,), print_size=True)
subset_error = Relation(
  "subset_error",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  print_size=True,
)
known_placeholder_subset = Relation(
  "known_placeholder_subset",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
cfg_node = Relation("cfg_node", 1, column_types=(int,), print_size=True)
var_live_on_entry = Relation(
  "var_live_on_entry",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
  index_type="SRDatalog::GPU::Device2LevelIndex",
)
var_drop_live_on_entry = Relation(
  "var_drop_live_on_entry",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
  index_type="SRDatalog::GPU::Device2LevelIndex",
)
var_maybe_partly_initialized_on_exit = Relation(
  "var_maybe_partly_initialized_on_exit",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
  index_type="SRDatalog::GPU::Device2LevelIndex",
)
var_maybe_partly_initialized_on_entry = Relation(
  "var_maybe_partly_initialized_on_entry",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
  index_type="SRDatalog::GPU::Device2LevelIndex",
)
ancestor_path = Relation(
  "ancestor_path",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
path_moved_at = Relation(
  "path_moved_at",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
path_assigned_at = Relation(
  "path_assigned_at",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
path_accessed_at = Relation(
  "path_accessed_at",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
path_begins_with_var = Relation(
  "path_begins_with_var",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)
path_maybe_initialized_on_exit = Relation(
  "path_maybe_initialized_on_exit",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
  index_type="SRDatalog::GPU::Device2LevelIndex",
)
path_maybe_uninitialized_on_exit = Relation(
  "path_maybe_uninitialized_on_exit",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
  index_type="SRDatalog::GPU::Device2LevelIndex",
)
move_error = Relation(
  "move_error",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
)

# ----- dataset_const declarations -----------------------------

DATASET_CONST_DECLS = {}

# ----- Rules: PoloniusDB -----


def build_poloniusdb_program() -> Program:
  child = Var("child")
  grandparent = Var("grandparent")
  loan = Var("loan")
  origin = Var("origin")
  origin1 = Var("origin1")
  origin2 = Var("origin2")
  origin3 = Var("origin3")
  parent = Var("parent")
  path = Var("path")
  point = Var("point")
  point1 = Var("point1")
  point2 = Var("point2")
  src = Var("src")
  tgt = Var("tgt")
  v = Var("v")
  vr = Var("vr")
  x = Var("x")
  y = Var("y")
  z = Var("z")

  return Program(
    relations=[
      subset_base,
      cfg_edge,
      loan_issued_at,
      universal_region,
      var_used_at,
      loan_killed_at,
      known_placeholder_subset_input,
      var_dropped_at,
      drop_of_var_derefs_origin,
      var_defined_at,
      child_path,
      path_moved_at_base,
      path_assigned_at_base,
      path_accessed_at_base,
      path_is_var,
      loan_invalidated_at,
      use_of_var_derefs_origin,
      subset,
      origin_live_on_entry,
      origin_contains_loan_on_entry,
      loan_live_at,
      errors,
      placeholder_origin,
      subset_error,
      known_placeholder_subset,
      cfg_node,
      var_live_on_entry,
      var_drop_live_on_entry,
      var_maybe_partly_initialized_on_exit,
      var_maybe_partly_initialized_on_entry,
      ancestor_path,
      path_moved_at,
      path_assigned_at,
      path_accessed_at,
      path_begins_with_var,
      path_maybe_initialized_on_exit,
      path_maybe_uninitialized_on_exit,
      move_error,
    ],
    rules=[
      (subset(origin1, origin2, point) <= subset_base(origin1, origin2, point)).named(
        'subset_base_rule'
      ),
      (
        origin_contains_loan_on_entry(origin, loan, point) <= loan_issued_at(loan, origin, point)
      ).named('ocle_issued'),
      (placeholder_origin(origin) <= universal_region(origin)).named('placeholder_origin_rule'),
      (known_placeholder_subset(x, z) <= known_placeholder_subset_input(x, z)).named('kps_seed'),
      (
        known_placeholder_subset(x, z)
        <= known_placeholder_subset(x, y) & known_placeholder_subset(y, z)
      ).named('kps_transitive'),
      (
        subset(origin1, origin3, point)
        <= subset(origin1, origin2, point)
        & subset_base(origin2, origin3, point)
        & Filter(
          (
            'origin1',
            'origin3',
          ),
          "return origin1 != origin3;",
        )
      )
      .named('subset_trans')
      .with_plan(delta=0, work_stealing=True),
      (
        subset(origin1, origin2, point2)
        <= subset(origin1, origin2, point1)
        & cfg_edge(point1, point2)
        & origin_live_on_entry(origin1, point2)
        & origin_live_on_entry(origin2, point2)
      )
      .named('subset_cfg')
      .with_plan(delta=0, var_order=['point1', 'point2', 'origin1', 'origin2']),
      (
        origin_contains_loan_on_entry(origin2, loan, point)
        <= origin_contains_loan_on_entry(origin1, loan, point) & subset(origin1, origin2, point)
      ).named('ocle_subset'),
      (
        origin_contains_loan_on_entry(origin, loan, point2)
        <= origin_contains_loan_on_entry(origin, loan, point1)
        & cfg_edge(point1, point2)
        & ~loan_killed_at(loan, point1)
        & origin_live_on_entry(origin, point2)
      ).named('ocle_cfg'),
      (
        loan_live_at(loan, point)
        <= origin_contains_loan_on_entry(origin, loan, point) & origin_live_on_entry(origin, point)
      ).named('loan_live_at_rule'),
      (errors(loan, point) <= loan_invalidated_at(loan, point) & loan_live_at(loan, point)).named(
        'errors_rule'
      ),
      (
        subset_error(origin1, origin2, point)
        <= subset(origin1, origin2, point)
        & placeholder_origin(origin1)
        & placeholder_origin(origin2)
        & ~known_placeholder_subset(origin1, origin2)
        & Filter(
          (
            'origin1',
            'origin2',
          ),
          "return origin1 != origin2;",
        )
      ).named('subset_error_rule'),
      (cfg_node(point1) <= cfg_edge(point1, Var("_"))).named('cfg_node_from_edge_src'),
      (cfg_node(point2) <= cfg_edge(Var("_"), point2)).named('cfg_node_from_edge_dst'),
      (origin_live_on_entry(origin, point) <= cfg_node(point) & universal_region(origin)).named(
        'ole_universal'
      ),
      (var_live_on_entry(vr, point) <= var_used_at(vr, point)).named('vle_used'),
      (
        var_maybe_partly_initialized_on_entry(vr, point2)
        <= var_maybe_partly_initialized_on_exit(vr, point1) & cfg_edge(point1, point2)
      ).named('vmpie_from_exit'),
      (
        var_drop_live_on_entry(vr, point)
        <= var_dropped_at(vr, point) & var_maybe_partly_initialized_on_entry(vr, point)
      ).named('vdle_dropped'),
      (
        origin_live_on_entry(origin, point)
        <= var_drop_live_on_entry(vr, point) & drop_of_var_derefs_origin(vr, origin)
      ).named('ole_drop'),
      (
        origin_live_on_entry(origin, point)
        <= var_live_on_entry(vr, point) & use_of_var_derefs_origin(vr, origin)
      ).named('ole_use'),
      (
        var_live_on_entry(vr, point1)
        <= var_live_on_entry(vr, point2) & cfg_edge(point1, point2) & ~var_defined_at(vr, point1)
      ).named('vle_cfg'),
      (
        var_drop_live_on_entry(v, src)
        <= var_drop_live_on_entry(v, tgt)
        & cfg_edge(src, tgt)
        & ~var_defined_at(v, src)
        & var_maybe_partly_initialized_on_exit(v, src)
      ).named('vdle_cfg'),
      (ancestor_path(x, y) <= child_path(x, y)).named('ancestor_path_base'),
      (path_moved_at(x, y) <= path_moved_at_base(x, y)).named('path_moved_at_base_rule'),
      (path_assigned_at(x, y) <= path_assigned_at_base(x, y)).named('path_assigned_at_base_rule'),
      (path_accessed_at(x, y) <= path_accessed_at_base(x, y)).named('path_accessed_at_base_rule'),
      (path_begins_with_var(x, vr) <= path_is_var(x, vr)).named('pbwv_base'),
      (
        ancestor_path(grandparent, child)
        <= ancestor_path(parent, child) & child_path(parent, grandparent)
      ).named('ancestor_path_trans'),
      (
        path_moved_at(child, point) <= path_moved_at(parent, point) & ancestor_path(parent, child)
      ).named('path_moved_at_ancestor'),
      (
        path_assigned_at(child, point)
        <= path_assigned_at(parent, point) & ancestor_path(parent, child)
      ).named('path_assigned_at_ancestor'),
      (
        path_accessed_at(child, point)
        <= path_accessed_at(parent, point) & ancestor_path(parent, child)
      ).named('path_accessed_at_ancestor'),
      (
        path_begins_with_var(child, v)
        <= path_begins_with_var(parent, v) & ancestor_path(parent, child)
      ).named('pbwv_ancestor'),
      (path_maybe_initialized_on_exit(path, point) <= path_assigned_at(path, point)).named(
        'pmioe_assigned'
      ),
      (path_maybe_uninitialized_on_exit(path, point) <= path_moved_at(path, point)).named(
        'pmuoe_moved'
      ),
      (
        path_maybe_initialized_on_exit(path, point2)
        <= path_maybe_initialized_on_exit(path, point1)
        & cfg_edge(point1, point2)
        & ~path_moved_at(path, point2)
      )
      .named('pmioe_cfg')
      .with_plan(delta=0, var_order=['point1', 'point2', 'path']),
      (
        path_maybe_uninitialized_on_exit(path, point2)
        <= path_maybe_uninitialized_on_exit(path, point1)
        & cfg_edge(point1, point2)
        & ~path_assigned_at(path, point2)
      )
      .named('pmuoe_cfg')
      .with_plan(delta=0, var_order=['point1', 'point2', 'path']),
      (
        var_maybe_partly_initialized_on_exit(vr, point)
        <= path_maybe_initialized_on_exit(path, point) & path_begins_with_var(path, vr)
      ).named('vmpioe_from_path'),
      (
        move_error(path, tgt) <= path_maybe_uninitialized_on_exit(path, src) & cfg_edge(src, tgt)
      ).named('move_error_rule'),
    ],
  )


def build_poloniusdb(meta_json_path: str) -> tuple[Program, dict[str, int]]:
  """Convenience: build the program, load dataset_consts, substitute."""
  consts = load_meta(meta_json_path, DATASET_CONST_DECLS)
  return resolve_program_consts(build_poloniusdb_program(), consts), consts
