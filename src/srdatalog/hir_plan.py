'''HIR Pass 4: Join Planning.

For each variant, compute:
  - clause_order: execution order of body clauses (heuristic: deps → delta
    priority → max-overlap with bound vars)
  - var_order: variable binding order (join vars first, starting from the
    delta clause for recursive rules; then remaining join vars in clause
    order; then non-join vars in clause order)
  - access_patterns: per-positive-clause AccessPattern (rel, version,
    access_order, prefix_len, index_cols, clause_idx)
  - negation_patterns: per-negation-clause AccessPattern (version forced
    to FULL)

Mirrors src/srdatalog/hir/join_planner.nim. Not yet ported: user-provided
rule.plans (DSL doesn't support them), split clauses (SplitClause),
balanced partitioning pragmas, IfClause/LetClause/AggClause handling.
'''
from __future__ import annotations
from dataclasses import dataclass, field

from srdatalog.dsl import Rule, Atom, Negation, ArgKind, PlanEntry, Filter, Let, Agg, Split
from srdatalog.hir_types import HirProgram, HirRuleVariant, AccessPattern, Version
from srdatalog.hir_pass import PassInfo, PassLevel, Dialect


# -----------------------------------------------------------------------------
# Rule Analysis
# -----------------------------------------------------------------------------

@dataclass
class RuleAnalysis:
  vars: set[str] = field(default_factory=set)
  clause_vars: list[set[str]] = field(default_factory=list)
  join_vars: set[str] = field(default_factory=set)       # appear in >1 POSITIVE clause
  head_vars: set[str] = field(default_factory=set)


def analyze_rule(rule: Rule) -> RuleAnalysis:
  r = RuleAnalysis()
  positive_count: dict[str, int] = {}
  for body in rule.body:
    cvars: set[str] = set()
    if isinstance(body, Negation):
      # Negation vars are tracked but NOT counted for join_vars (it's a filter, not a join).
      for arg in body.atom.args:
        if arg.kind is ArgKind.LVAR:
          cvars.add(arg.var_name)
          r.vars.add(arg.var_name)
    elif isinstance(body, Atom):
      for arg in body.args:
        if arg.kind is ArgKind.LVAR:
          cvars.add(arg.var_name)
          r.vars.add(arg.var_name)
          positive_count[arg.var_name] = positive_count.get(arg.var_name, 0) + 1
    elif isinstance(body, Agg):
      # Aggregate args count as positive (like RelClause), plus the
      # result_var is a produced positive var. Mirrors Nim analyzeRule.
      for arg in body.args:
        if arg.kind is ArgKind.LVAR:
          cvars.add(arg.var_name)
          r.vars.add(arg.var_name)
          positive_count[arg.var_name] = positive_count.get(arg.var_name, 0) + 1
      cvars.add(body.result_var)
      r.vars.add(body.result_var)
      positive_count[body.result_var] = positive_count.get(body.result_var, 0) + 1
    # Filter / Let: no relation args to analyze; leave cvars empty.
    r.clause_vars.append(cvars)

  for arg in rule.head.args:
    if arg.kind is ArgKind.LVAR:
      r.head_vars.add(arg.var_name)

  for v, c in positive_count.items():
    if c > 1:
      r.join_vars.add(v)
  return r


# -----------------------------------------------------------------------------
# Clause ordering heuristic
# -----------------------------------------------------------------------------

def _clause_lvar_names(body) -> list[str]:
  '''Ordered LVar names for a body clause. Column order for Atom/Negation;
  for Filter the `vars` field (what the filter reads); for Let the single
  `var_name` (what it binds); for Agg: all arg LVars then the result_var
  last (matches Nim's extractClauseVars). Split contributes nothing.
  '''
  if isinstance(body, Atom):
    return [a.var_name for a in body.args if a.kind is ArgKind.LVAR]
  if isinstance(body, Negation):
    return [a.var_name for a in body.atom.args if a.kind is ArgKind.LVAR]
  if isinstance(body, Filter):
    return list(body.vars)
  if isinstance(body, Let):
    return [body.var_name]
  if isinstance(body, Agg):
    names = [a.var_name for a in body.args if a.kind is ArgKind.LVAR]
    names.append(body.result_var)
    return names
  return []  # Split, unknown


def _get_dependencies(body) -> set[str]:
  '''Vars that must be bound before this clause is runnable.
    - Atom, Agg: no deps (both are generators).
    - Negation: safe-negation -> all args must be bound.
    - Filter: every var it references.
    - Let: every var in its `deps` list (NOT the var it binds).
  '''
  if isinstance(body, Negation):
    return {a.var_name for a in body.atom.args if a.kind is ArgKind.LVAR}
  if isinstance(body, Filter):
    return set(body.vars)
  if isinstance(body, Let):
    return set(body.deps)
  return set()


def _get_produced_vars(body) -> set[str]:
  '''Vars newly bound by this clause.
    - Atom: all its LVar args.
    - Let / Agg: the single bound var (`var_name` / `result_var`).
    - Negation / Filter: nothing.
  '''
  if isinstance(body, Atom):
    return {a.var_name for a in body.args if a.kind is ArgKind.LVAR}
  if isinstance(body, Let):
    return {body.var_name}
  if isinstance(body, Agg):
    return {body.result_var}
  return set()


def compute_clause_order(rule: Rule, delta_idx: int = -1) -> list[int]:
  '''Pick body clause execution order using the Nim heuristic:
    1. Dependency-gated runnable set (sorted by source idx for tie-breaking).
    2. Delta clause first when runnable (for recursive variants).
    3. Max overlap of clause vars with currently-bound vars.
  '''
  bound: set[str] = set()
  scheduled: list[int] = []
  remaining: set[int] = set(range(len(rule.body)))

  while remaining:
    runnable = sorted(
      i for i in remaining if _get_dependencies(rule.body[i]) <= bound
    )
    if not runnable:
      # Deadlock fallback (shouldn't happen for stratified DSL input, but
      # matches Nim behavior): prefer an Atom, else lowest index.
      atoms = [i for i in sorted(remaining) if isinstance(rule.body[i], Atom)]
      runnable = [atoms[0] if atoms else sorted(remaining)[0]]

    # Priority 2.1: Filters get first crack so the planner pushes selection
    # DOWN (drops ineligible bindings ASAP).
    best = -1
    for r in runnable:
      if isinstance(rule.body[r], Filter):
        best = r
        break

    # Priority 2.2: delta clause (seed for recursive variants).
    if best == -1 and delta_idx in runnable:
      best = delta_idx
    elif best == -1:
      # Priority 2.3: max overlap of clause args with already-bound vars.
      max_overlap = -1
      for r in runnable:
        body = rule.body[r]
        if isinstance(body, Atom):
          overlap = sum(1 for v in _clause_lvar_names(body) if v in bound)
        else:
          overlap = 0
        if overlap > max_overlap:
          max_overlap = overlap
          best = r
      if best == -1:
        best = runnable[0]

    scheduled.append(best)
    remaining.discard(best)
    bound.update(_get_produced_vars(rule.body[best]))

  return scheduled


# -----------------------------------------------------------------------------
# Variable ordering heuristic
# -----------------------------------------------------------------------------

def compute_var_order_from_clauses(
  rule: Rule, clause_order: list[int], join_vars: set[str], delta_idx: int = -1
) -> list[str]:
  '''Derive variable binding order from clause execution order.

  Pass 1 (recursive variants only): join vars appearing in the delta clause.
  Pass 2: remaining join vars, in clauseOrder.
  Pass 3: non-join vars, in clauseOrder.
  '''
  result: list[str] = []
  seen: set[str] = set()

  if 0 <= delta_idx < len(rule.body):
    for v in _clause_lvar_names(rule.body[delta_idx]):
      if v in join_vars and v not in seen:
        seen.add(v)
        result.append(v)

  for idx in clause_order:
    for v in _clause_lvar_names(rule.body[idx]):
      if v in join_vars and v not in seen:
        seen.add(v)
        result.append(v)

  for idx in clause_order:
    for v in _clause_lvar_names(rule.body[idx]):
      if v not in join_vars and v not in seen:
        seen.add(v)
        result.append(v)

  return result


# -----------------------------------------------------------------------------
# Access pattern computation
# -----------------------------------------------------------------------------

def compute_access_pattern(
  body, version: Version, join_vars: set[str], var_order: list[str], clause_idx: int
) -> AccessPattern:
  '''Build the AccessPattern for one body clause (Atom or Negation).

  - access_order = var_order filtered to clause vars, plus any remaining
    clause vars appended in a deterministic (sorted) order.
  - prefix_len: Atoms → count of join vars; Negations → count of non-wildcard vars.
  - index_cols: maps access_order to column positions; then completes to full arity.
  - For Negations: prepend constant-column indices and force version=FULL (caller).
  '''
  if isinstance(body, (Atom, Negation)):
    atom_args = body.atom.args if isinstance(body, Negation) else body.args
    rel = body.atom.rel if isinstance(body, Negation) else body.rel
  else:
    # IfClause/LetClause etc. — not supported yet.
    return AccessPattern(rel_name="", version=version, clause_idx=clause_idx)

  clause_vars: set[str] = set()
  const_args: list[tuple[int, int]] = []
  for col, arg in enumerate(atom_args):
    if arg.kind is ArgKind.LVAR:
      clause_vars.add(arg.var_name)
    elif arg.kind is ArgKind.CONST:
      const_args.append((col, arg.const_value))

  access_order: list[str] = [v for v in var_order if v in clause_vars]
  seen = set(access_order)
  # Remaining clause vars (shouldn't happen for auto-generated var_order, but
  # can with user-provided plans that have wildcard holes). Sort for determinism.
  for v in sorted(clause_vars - seen):
    access_order.append(v)

  if isinstance(body, Atom):
    prefix_len = sum(1 for v in access_order if v in join_vars)
  else:  # Negation: wildcard vars (starting with "_") don't count toward prefix.
    prefix_len = sum(1 for v in access_order if not v.startswith("_"))

  # index_cols: position-in-atom for each var in access_order, then any
  # missing columns appended to complete full arity.
  index_cols: list[int] = []
  for v in access_order:
    for col, arg in enumerate(atom_args):
      if arg.kind is ArgKind.LVAR and arg.var_name == v:
        index_cols.append(col)
        break
  for col in range(len(atom_args)):
    if col not in index_cols:
      index_cols.append(col)

  if isinstance(body, Negation):
    const_cols = [c for c, _ in const_args]
    index_cols = const_cols + [c for c in index_cols if c not in const_cols]

  return AccessPattern(
    rel_name=rel,
    version=version,
    access_order=access_order,
    index_cols=index_cols,
    prefix_len=prefix_len,
    clause_idx=clause_idx,
    const_args=const_args,
  )


# -----------------------------------------------------------------------------
# Derive clause order from user-provided var_order (when the plan omits it)
# -----------------------------------------------------------------------------

def derive_clause_order_from_var_order(
  rule: Rule, var_order: list[str], delta_idx: int = -1
) -> list[int]:
  '''Mirror deriveClauseOrderFromVarOrder in join_planner.nim.

  Walks var_order left to right; for each not-yet-bound variable, picks a
  runnable clause that introduces it (preferring the delta clause when
  applicable; source-order ties otherwise). Any unscheduled clauses (filters,
  disconnected negations) are appended at the end in a runnable sweep.
  '''
  scheduled: list[int] = []
  remaining: set[int] = set(range(len(rule.body)))
  bound: set[str] = set()

  def clause_vars(body) -> set[str]:
    return set(_clause_lvar_names(body))

  def can_schedule(idx: int) -> bool:
    return _get_dependencies(rule.body[idx]) <= bound

  for target in var_order:
    if target in bound:
      continue
    candidates = [
      idx for idx in remaining
      if can_schedule(idx) and target in clause_vars(rule.body[idx])
    ]
    if not candidates:
      continue
    if delta_idx in candidates:
      picked = delta_idx
    else:
      candidates.sort()
      picked = candidates[0]
    scheduled.append(picked)
    remaining.discard(picked)
    bound.update(clause_vars(rule.body[picked]))

  # Sweep remaining (filters/negations not referenced by var_order).
  # Prefer Filter clauses first (push-down policy consistent with the main
  # heuristic). Nim's sweep iterates a HashSet whose hash-bucket order
  # happens to surface Filter before Negation in common cases; the
  # explicit priority here matches that behavior deterministically.
  while remaining:
    scheduled_this_round = False
    for idx in sorted(remaining):
      if can_schedule(idx) and isinstance(rule.body[idx], Filter):
        scheduled.append(idx)
        remaining.discard(idx)
        bound.update(clause_vars(rule.body[idx]))
        scheduled_this_round = True
        break
    if scheduled_this_round:
      continue
    for idx in sorted(remaining):
      if can_schedule(idx):
        scheduled.append(idx)
        remaining.discard(idx)
        bound.update(clause_vars(rule.body[idx]))
        scheduled_this_round = True
        break
    if not scheduled_this_round:
      # Deadlock fallback: force-pick lowest remaining.
      idx = min(remaining)
      scheduled.append(idx)
      remaining.discard(idx)
      bound.update(clause_vars(rule.body[idx]))

  return scheduled


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def _find_plan(rule: Rule, delta_idx: int) -> PlanEntry | None:
  for p in rule.plans:
    if p.delta == delta_idx:
      return p
  return None


def detect_split(rule: Rule) -> int:
  '''Body index of the Split clause, or -1 if absent.'''
  for i, clause in enumerate(rule.body):
    if isinstance(clause, Split):
      return i
  return -1


def compute_temp_vars(rule: Rule, split_at: int) -> list[str]:
  '''Variables that cross a rule's split boundary: bound above the split
  AND used below (body clauses or head). Mirrors Nim's computeTempVars.

  Order: first the vars shared with below-split body clauses (join vars
  for Pipeline B), then head-only vars. Sorted within each group for
  determinism (Nim uses HashSet iteration; for singleton / small-set
  cases the fixture passes either way).
  '''
  vars_above: set[str] = set()
  for i in range(split_at):
    clause = rule.body[i]
    if isinstance(clause, (Atom, Negation)):
      vars_above.update(_clause_lvar_names(clause))

  below_body_vars: set[str] = set()
  for i in range(split_at + 1, len(rule.body)):
    clause = rule.body[i]
    if isinstance(clause, Atom):   # Nim only counts RelClause for belowBodyVars
      below_body_vars.update(_clause_lvar_names(clause))

  vars_below: set[str] = set(below_body_vars)
  # Also include head args
  for a in rule.head.args:
    if a.kind is ArgKind.LVAR:
      vars_below.add(a.var_name)
  # Include negation clauses' vars from below (Nim does this)
  for i in range(split_at + 1, len(rule.body)):
    clause = rule.body[i]
    if isinstance(clause, Negation):
      vars_below.update(_clause_lvar_names(clause))

  result: list[str] = []
  # Pass 1: vars_above ∩ below_body_vars ∩ vars_below (join vars)
  for v in sorted(vars_above):
    if v in below_body_vars and v in vars_below:
      result.append(v)
  # Pass 2: vars_above ∩ vars_below minus below_body_vars (head-only)
  for v in sorted(vars_above):
    if v in vars_below and v not in below_body_vars:
      result.append(v)
  return result


def _plan_variant(v: HirRuleVariant) -> None:
  rule = v.original_rule
  analysis = analyze_rule(rule)
  d = v.delta_idx  # -1 for base variants

  plan = _find_plan(rule, d)
  if plan is not None and plan.var_order:
    var_order = list(plan.var_order)
    if plan.clause_order:
      clause_order = list(plan.clause_order)
    else:
      clause_order = derive_clause_order_from_var_order(rule, var_order, delta_idx=d)
  else:
    clause_order = compute_clause_order(rule, delta_idx=d)
    var_order = compute_var_order_from_clauses(
      rule, clause_order, analysis.join_vars, delta_idx=d
    )

  # Propagate pragma flags whenever a plan is attached — the pragma
  # branch above is gated on `plan.var_order`, but pragmas like
  # `work_stealing: true` frequently appear on plan entries that have
  # no custom var_order (e.g. Polonius subset_trans). Using the flags
  # regardless of var_order matches Nim's planJoins which copies the
  # pragma set unconditionally.
  if plan is not None:
    v.fanout = plan.fanout
    v.work_stealing = plan.work_stealing
    v.block_group = plan.block_group
    v.dedup_hash = plan.dedup_hash
    v.balanced_root = list(plan.balanced_root)
    v.balanced_sources = list(plan.balanced_sources)

  v.count = rule.count
  v.clause_order = clause_order
  v.var_order = var_order
  v.join_vars = analysis.join_vars

  # Split-rule metadata. Auto-detect the `split` marker in the body and
  # attach temp-rel fields so downstream passes (temp decl synthesis,
  # temp index registration, split-aware lowering) can key off them.
  split_at = detect_split(rule)
  v.split_at = split_at
  if split_at >= 0 and rule.name:
    v.temp_vars = compute_temp_vars(rule, split_at)
    v.temp_rel_name = f"_temp_{rule.name}"

  for k, body in enumerate(rule.body):
    version = v.clause_versions[k]
    pattern = compute_access_pattern(body, version, analysis.join_vars, var_order, k)
    if not pattern.rel_name:
      continue
    if isinstance(body, Negation):
      pattern.version = Version.FULL
      v.negation_patterns.append(pattern)
    else:
      v.access_patterns.append(pattern)


def plan_joins(hir: HirProgram) -> HirProgram:
  '''HIR Pass 4 entry. Mutates and returns the HirProgram.'''
  for stratum in hir.strata:
    for v in stratum.base_variants:
      _plan_variant(v)
    for v in stratum.recursive_variants:
      _plan_variant(v)
  return hir


class JoinPlannerPass:
  info = PassInfo(
    name="JoinPlanning",
    level=PassLevel.HIR_TRANSFORM,
    order=200,
    source_dialect=Dialect.HIR,
    target_dialect=Dialect.HIR,
  )

  def run(self, hir: HirProgram) -> HirProgram:
    return plan_joins(hir)
