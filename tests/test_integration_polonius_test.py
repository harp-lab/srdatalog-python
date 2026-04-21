'''polonius_test.nim -- borrow-checker analysis (~40 rules, 2 wildcards)'''

from integration_helpers import diff_hir, diff_mir

from srdatalog.dsl import Filter, Program, Relation, Var


def build_polonius_test() -> Program:
  O1, O2, O3 = Var("origin1"), Var("origin2"), Var("origin3")
  POINT, PT1, PT2 = Var("point"), Var("point1"), Var("point2")
  LOAN, ORIGIN = Var("loan"), Var("origin")
  X, Y, Z = Var("x"), Var("y"), Var("z")
  VR, V = Var("vr"), Var("v")
  TGT, SRC = Var("tgt"), Var("src")
  PARENT, CHILD, GRAND = Var("parent"), Var("child"), Var("grandparent")
  PATH = Var("path")
  G1, G2 = Var("_gen1"), Var("_gen2")

  subset_base = Relation("subset_base", 3)
  cfg_edge = Relation("cfg_edge", 2)
  loan_issued_at = Relation("loan_issued_at", 3)
  universal_region = Relation("universal_region", 1)
  var_used_at = Relation("var_used_at", 2)
  loan_killed_at = Relation("loan_killed_at", 2)
  kps_input = Relation("known_placeholder_subset_input", 2)
  var_dropped_at = Relation("var_dropped_at", 2)
  drop_of_var_derefs_origin = Relation("drop_of_var_derefs_origin", 2)
  var_defined_at = Relation("var_defined_at", 2)
  child_path = Relation("child_path", 2)
  path_moved_at_base = Relation("path_moved_at_base", 2)
  path_assigned_at_base = Relation("path_assigned_at_base", 2)
  path_accessed_at_base = Relation("path_accessed_at_base", 2)
  path_is_var = Relation("path_is_var", 2)
  loan_invalidated_at = Relation("loan_invalidated_at", 2)
  use_of_var_derefs_origin = Relation("use_of_var_derefs_origin", 2)

  subset = Relation("subset", 3)
  origin_live_on_entry = Relation("origin_live_on_entry", 2)
  origin_contains_loan_on_entry = Relation("origin_contains_loan_on_entry", 3)
  loan_live_at = Relation("loan_live_at", 2)
  errors = Relation("errors", 2)
  placeholder_origin = Relation("placeholder_origin", 1)
  subset_error = Relation("subset_error", 3)
  known_placeholder_subset = Relation("known_placeholder_subset", 2)
  cfg_node = Relation("cfg_node", 1)
  var_live_on_entry = Relation("var_live_on_entry", 2)
  var_drop_live_on_entry = Relation("var_drop_live_on_entry", 2)
  vmp_init_on_exit = Relation("var_maybe_partly_initialized_on_exit", 2)
  vmp_init_on_entry = Relation("var_maybe_partly_initialized_on_entry", 2)
  ancestor_path = Relation("ancestor_path", 2)
  path_moved_at = Relation("path_moved_at", 2)
  path_assigned_at = Relation("path_assigned_at", 2)
  path_accessed_at = Relation("path_accessed_at", 2)
  path_begins_with_var = Relation("path_begins_with_var", 2)
  pmi_on_exit = Relation("path_maybe_initialized_on_exit", 2)
  pmu_on_exit = Relation("path_maybe_uninitialized_on_exit", 2)
  move_error = Relation("move_error", 2)

  return Program(
    relations=[
      subset_base,
      cfg_edge,
      loan_issued_at,
      universal_region,
      var_used_at,
      loan_killed_at,
      kps_input,
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
      vmp_init_on_exit,
      vmp_init_on_entry,
      ancestor_path,
      path_moved_at,
      path_assigned_at,
      path_accessed_at,
      path_begins_with_var,
      pmi_on_exit,
      pmu_on_exit,
      move_error,
    ],
    rules=[
      (subset(O1, O2, POINT) <= subset_base(O1, O2, POINT)).named("subset_base_rule"),
      (
        origin_contains_loan_on_entry(ORIGIN, LOAN, POINT) <= loan_issued_at(LOAN, ORIGIN, POINT)
      ).named("ocle_issued"),
      (placeholder_origin(ORIGIN) <= universal_region(ORIGIN)).named("placeholder_origin_rule"),
      (known_placeholder_subset(X, Z) <= kps_input(X, Z)).named("kps_seed"),
      (
        known_placeholder_subset(X, Z)
        <= known_placeholder_subset(X, Y) & known_placeholder_subset(Y, Z)
      ).named("kps_transitive"),
      (
        subset(O1, O3, POINT)
        <= subset(O1, O2, POINT)
        & subset_base(O2, O3, POINT)
        & Filter(vars=("origin1", "origin3"), code="return origin1 != origin3;")
      )
      .named("subset_trans")
      .with_plan(delta=0, work_stealing=True),
      (
        subset(O1, O2, PT2)
        <= subset(O1, O2, PT1)
        & cfg_edge(PT1, PT2)
        & origin_live_on_entry(O1, PT2)
        & origin_live_on_entry(O2, PT2)
      )
      .named("subset_cfg")
      .with_plan(delta=0, var_order=("point1", "point2", "origin1", "origin2")),
      (
        origin_contains_loan_on_entry(O2, LOAN, POINT)
        <= origin_contains_loan_on_entry(O1, LOAN, POINT) & subset(O1, O2, POINT)
      ).named("ocle_subset"),
      (
        origin_contains_loan_on_entry(ORIGIN, LOAN, PT2)
        <= origin_contains_loan_on_entry(ORIGIN, LOAN, PT1)
        & cfg_edge(PT1, PT2)
        & ~loan_killed_at(LOAN, PT1)
        & origin_live_on_entry(ORIGIN, PT2)
      ).named("ocle_cfg"),
      (
        loan_live_at(LOAN, POINT)
        <= origin_contains_loan_on_entry(ORIGIN, LOAN, POINT) & origin_live_on_entry(ORIGIN, POINT)
      ).named("loan_live_at_rule"),
      (errors(LOAN, POINT) <= loan_invalidated_at(LOAN, POINT) & loan_live_at(LOAN, POINT)).named(
        "errors_rule"
      ),
      (
        subset_error(O1, O2, POINT)
        <= subset(O1, O2, POINT)
        & placeholder_origin(O1)
        & placeholder_origin(O2)
        & ~known_placeholder_subset(O1, O2)
        & Filter(vars=("origin1", "origin2"), code="return origin1 != origin2;")
      ).named("subset_error_rule"),
      (cfg_node(PT1) <= cfg_edge(PT1, G1)).named("cfg_node_from_edge_src"),
      (cfg_node(PT2) <= cfg_edge(G2, PT2)).named("cfg_node_from_edge_dst"),
      (origin_live_on_entry(ORIGIN, POINT) <= cfg_node(POINT) & universal_region(ORIGIN)).named(
        "ole_universal"
      ),
      (var_live_on_entry(VR, POINT) <= var_used_at(VR, POINT)).named("vle_used"),
      (vmp_init_on_entry(VR, PT2) <= vmp_init_on_exit(VR, PT1) & cfg_edge(PT1, PT2)).named(
        "vmpie_from_exit"
      ),
      (
        var_drop_live_on_entry(VR, POINT)
        <= var_dropped_at(VR, POINT) & vmp_init_on_entry(VR, POINT)
      ).named("vdle_dropped"),
      (
        origin_live_on_entry(ORIGIN, POINT)
        <= var_drop_live_on_entry(VR, POINT) & drop_of_var_derefs_origin(VR, ORIGIN)
      ).named("ole_drop"),
      (
        origin_live_on_entry(ORIGIN, POINT)
        <= var_live_on_entry(VR, POINT) & use_of_var_derefs_origin(VR, ORIGIN)
      ).named("ole_use"),
      (
        var_live_on_entry(VR, PT1)
        <= var_live_on_entry(VR, PT2) & cfg_edge(PT1, PT2) & ~var_defined_at(VR, PT1)
      ).named("vle_cfg"),
      (
        var_drop_live_on_entry(V, SRC)
        <= var_drop_live_on_entry(V, TGT)
        & cfg_edge(SRC, TGT)
        & ~var_defined_at(V, SRC)
        & vmp_init_on_exit(V, SRC)
      ).named("vdle_cfg"),
      (ancestor_path(X, Y) <= child_path(X, Y)).named("ancestor_path_base"),
      (path_moved_at(X, Y) <= path_moved_at_base(X, Y)).named("path_moved_at_base_rule"),
      (path_assigned_at(X, Y) <= path_assigned_at_base(X, Y)).named("path_assigned_at_base_rule"),
      (path_accessed_at(X, Y) <= path_accessed_at_base(X, Y)).named("path_accessed_at_base_rule"),
      (path_begins_with_var(X, VR) <= path_is_var(X, VR)).named("pbwv_base"),
      (
        ancestor_path(GRAND, CHILD) <= ancestor_path(PARENT, CHILD) & child_path(PARENT, GRAND)
      ).named("ancestor_path_trans"),
      (
        path_moved_at(CHILD, POINT) <= path_moved_at(PARENT, POINT) & ancestor_path(PARENT, CHILD)
      ).named("path_moved_at_ancestor"),
      (
        path_assigned_at(CHILD, POINT)
        <= path_assigned_at(PARENT, POINT) & ancestor_path(PARENT, CHILD)
      ).named("path_assigned_at_ancestor"),
      (
        path_accessed_at(CHILD, POINT)
        <= path_accessed_at(PARENT, POINT) & ancestor_path(PARENT, CHILD)
      ).named("path_accessed_at_ancestor"),
      (
        path_begins_with_var(CHILD, V)
        <= path_begins_with_var(PARENT, V) & ancestor_path(PARENT, CHILD)
      ).named("pbwv_ancestor"),
      (pmi_on_exit(PATH, POINT) <= path_assigned_at(PATH, POINT)).named("pmioe_assigned"),
      (pmu_on_exit(PATH, POINT) <= path_moved_at(PATH, POINT)).named("pmuoe_moved"),
      (
        pmi_on_exit(PATH, PT2)
        <= pmi_on_exit(PATH, PT1) & cfg_edge(PT1, PT2) & ~path_moved_at(PATH, PT2)
      )
      .named("pmioe_cfg")
      .with_plan(delta=0, var_order=("point1", "point2", "path")),
      (
        pmu_on_exit(PATH, PT2)
        <= pmu_on_exit(PATH, PT1) & cfg_edge(PT1, PT2) & ~path_assigned_at(PATH, PT2)
      )
      .named("pmuoe_cfg")
      .with_plan(delta=0, var_order=("point1", "point2", "path")),
      (
        vmp_init_on_exit(VR, POINT) <= pmi_on_exit(PATH, POINT) & path_begins_with_var(PATH, VR)
      ).named("vmpioe_from_path"),
      (move_error(PATH, TGT) <= pmu_on_exit(PATH, SRC) & cfg_edge(SRC, TGT)).named(
        "move_error_rule"
      ),
    ],
  )


def test_polonius_test_hir():
  diff_hir(build_polonius_test(), "polonius_test")


def test_polonius_test_mir():
  diff_mir(build_polonius_test(), "polonius_test")


if __name__ == "__main__":
  test_polonius_test_hir()
  test_polonius_test_mir()
  print("polonius_test: OK")
