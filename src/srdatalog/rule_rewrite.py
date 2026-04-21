'''Rule-rewrite passes (HIR Pass 0 / Pass 1). Mirror
src/srdatalog/hir/constant_rewriting.nim and head_constant_rewriting.nim.

Semi-join optimization (Pass 1.5) is 385 lines and is deferred to a
later turn; it doesn't fire on simple programs anyway.

Counters are per-pass-invocation (reset each call). Nim uses a
`compileTime` counter that persists across a single macro compilation —
for fixtures the tool compiles once, so starting at 0 matches.
'''

from __future__ import annotations

import dataclasses

from srdatalog.dsl import (
  ArgKind,
  Atom,
  ClauseArg,
  Filter,
  Let,
  Negation,
  PlanEntry,
  Rule,
)
from srdatalog.hir.pass_ import Dialect, PassInfo, PassLevel
from srdatalog.hir.types import RelationDecl
from srdatalog.provenance import compiler_gen


def _atom_has_const(atom: Atom) -> bool:
  return any(a.kind is ArgKind.CONST for a in atom.args)


def rewrite_constants(
  rules: list[Rule],
  decls: list[RelationDecl],
) -> tuple[list[Rule], list[RelationDecl]]:
  '''Pass 0: body-constant rewriting.

  For each positive body clause with constant args, replace every Const
  with a fresh LVar (_cN) and append a Filter clause asserting equality
  immediately after the rewritten atom. Matches Nim's output byte-for-byte
  when fresh-name counters start at 0 (default).
  '''
  counter = 0
  new_rules: list[Rule] = []
  for rule in rules:
    new_body: list = []
    for clause in rule.body:
      if isinstance(clause, Atom) and _atom_has_const(clause):
        new_args: list[ClauseArg] = []
        filter_vars: list[str] = []
        filter_parts: list[str] = []
        for arg in clause.args:
          if arg.kind is ArgKind.CONST:
            fresh = f"_c{counter}"
            counter += 1
            new_args.append(ClauseArg(kind=ArgKind.LVAR, var_name=fresh))
            filter_vars.append(fresh)
            filter_parts.append(f"{fresh} == {arg.const_cpp_expr}")
          else:
            new_args.append(arg)
        new_body.append(Atom(rel=clause.rel, args=tuple(new_args)))
        filter_code = "return " + " && ".join(filter_parts) + ";"
        new_body.append(Filter(vars=tuple(filter_vars), code=filter_code))
      else:
        new_body.append(clause)
    new_rules.append(dataclasses.replace(rule, body=tuple(new_body)))
  return new_rules, decls


def rewrite_head_constants(
  rules: list[Rule],
  decls: list[RelationDecl],
) -> tuple[list[Rule], list[RelationDecl]]:
  '''Pass 1: head-constant rewriting.

  For each head with constant args, replace each Const with a fresh LVar
  (_hcN) and append a Let clause to the body binding that variable to the
  constant's C++ expression. Runs AFTER body-constant rewriting.
  '''
  counter = 0
  new_rules: list[Rule] = []
  for rule in rules:
    new_head_args: list[ClauseArg] = []
    extra_body: list[Let] = []
    needs_rewrite = False
    for arg in rule.head.args:
      if arg.kind is ArgKind.CONST:
        fresh = f"_hc{counter}"
        counter += 1
        new_head_args.append(ClauseArg(kind=ArgKind.LVAR, var_name=fresh))
        extra_body.append(Let(var_name=fresh, code=arg.const_cpp_expr, deps=()))
        needs_rewrite = True
      else:
        new_head_args.append(arg)
    if needs_rewrite:
      new_head = Atom(rel=rule.head.rel, args=tuple(new_head_args))
      new_body = tuple(rule.body) + tuple(extra_body)
      new_rules.append(dataclasses.replace(rule, head=new_head, body=new_body))
    else:
      new_rules.append(rule)
  return new_rules, decls


# -----------------------------------------------------------------------------
# Pipeline pass wrappers
# -----------------------------------------------------------------------------


class ConstantRewritePass:
  info = PassInfo(
    name="ConstantRewrite",
    level=PassLevel.RULE_REWRITE,
    order=0,
    source_dialect=Dialect.HIR,
    target_dialect=Dialect.HIR,
  )

  def run(self, rules, decls):
    return rewrite_constants(rules, decls)


class HeadConstantRewritePass:
  info = PassInfo(
    name="HeadConstantRewrite",
    level=PassLevel.RULE_REWRITE,
    order=1,
    source_dialect=Dialect.HIR,
    target_dialect=Dialect.HIR,
  )

  def run(self, rules, decls):
    return rewrite_head_constants(rules, decls)


# -----------------------------------------------------------------------------
# Semi-join optimization (Pass 1.5 in Nim's compileToHir)
# -----------------------------------------------------------------------------


def _clause_vars(clause) -> set[str]:
  '''Var names in an Atom or Negation; empty for Filter/Let.'''
  if isinstance(clause, Atom):
    return {a.var_name for a in clause.args if a.kind is ArgKind.LVAR}
  if isinstance(clause, Negation):
    return {a.var_name for a in clause.atom.args if a.kind is ArgKind.LVAR}
  return set()


def _lvar_name(arg: ClauseArg) -> str:
  return arg.var_name if arg.kind is ArgKind.LVAR else ""


def _is_semi_join_candidate(filt, target) -> bool:
  '''filt is a semi-join filter for target iff its var set is a proper
  subset of target's and both are positive relation clauses.
  '''
  if not isinstance(filt, Atom) or not isinstance(target, Atom):
    return False
  fvars = _clause_vars(filt)
  tvars = _clause_vars(target)
  if len(fvars) == 0 or len(fvars) >= len(tvars):
    return False
  return fvars <= tvars


def optimize_semi_joins(
  rules: list[Rule],
  decls: list[RelationDecl],
) -> tuple[list[Rule], list[RelationDecl]]:
  '''Single pass. Rule must have semi_join=True and body.len > 2 to be
  considered. For each candidate target/filter pair, synthesise an
  `_SJ_Target_Filter_<keptIndices>` relation whose generator rule is
  `Target ⋈ Filter`, and replace the target clause (dropping the filter
  clause) in the original rule.

  Mirrors src/srdatalog/hir/semi_join_optimization.nim. Fixed-point
  iteration lives in SemiJoinPass.run (Nim does it in compileToHir).
  '''
  new_rules: list[Rule] = []
  generated_rules: list[Rule] = []
  generated_decls: list[RelationDecl] = []
  generated_cache: set[str] = set()

  decls_map = {d.rel_name: d for d in decls}

  for rule in rules:
    if not rule.semi_join or len(rule.body) <= 2:
      new_rules.append(rule)
      continue

    body = list(rule.body)
    replaced: set[int] = set()  # body indices removed
    rewrites: dict[int, tuple[Atom, int]] = {}  # target_idx -> (new_atom, filter_idx)

    # Pass 1: find each target -> filter opportunity.
    for i, target in enumerate(body):
      if not isinstance(target, Atom):
        continue
      if target.rel.startswith("_SJ_"):
        continue  # never double-optimise

      for j, filt in enumerate(body):
        if i == j or not _is_semi_join_candidate(filt, target):
          continue

        filter_vars = _clause_vars(filt)

        # Vars shared with head or other body clauses.
        shared: set[str] = set()
        for a in rule.head.args:
          v = _lvar_name(a)
          if v in filter_vars:
            shared.add(v)
        for k, other in enumerate(body):
          if k in (i, j):
            continue
          for v in filter_vars:
            if v in _clause_vars(other):
              shared.add(v)

        # Keep target columns whose var is NOT a filter-only var.
        kept_idx: list[int] = []
        for arg_idx, arg in enumerate(target.args):
          v = _lvar_name(arg)
          if v not in filter_vars or v in shared:
            kept_idx.append(arg_idx)

        suffix = "".join(f"_{k}" for k in kept_idx)
        new_rel_name = f"_SJ_{target.rel}_{filt.rel}{suffix}"

        # Generate decl + rule only once for this relation name.
        if new_rel_name not in generated_cache:
          generated_cache.add(new_rel_name)

          # Semiring / types inherited from target (if available).
          semiring = "NoProvenance"
          types: list[str] = []
          if target.rel in decls_map:
            orig = decls_map[target.rel]
            semiring = orig.semiring
            for k in kept_idx:
              if k < len(orig.types):
                types.append(orig.types[k])

          generated_decls.append(
            RelationDecl(
              rel_name=new_rel_name,
              types=types,
              semiring=semiring,
              is_generated=True,
            )
          )

          fresh = [f"v{k}" for k in range(len(target.args))]

          # _SJ_X(v_kept...) :- Target(v0..vN), Filter(filter_args@target_pos).
          gen_head = Atom(
            rel=new_rel_name,
            args=tuple(ClauseArg(kind=ArgKind.LVAR, var_name=fresh[k]) for k in kept_idx),
          )
          gen_target = Atom(
            rel=target.rel,
            args=tuple(
              ClauseArg(kind=ArgKind.LVAR, var_name=fresh[k]) for k in range(len(target.args))
            ),
          )
          # Map each filter arg to its corresponding target column's fresh var.
          filter_args: list[ClauseArg] = []
          for f_arg in filt.args:
            fv = _lvar_name(f_arg)
            for ti, t_arg in enumerate(target.args):
              if _lvar_name(t_arg) == fv:
                filter_args.append(
                  ClauseArg(
                    kind=ArgKind.LVAR,
                    var_name=fresh[ti],
                  )
                )
                break
          gen_filter = Atom(rel=filt.rel, args=tuple(filter_args))

          prov = compiler_gen(
            parent_rule=rule.name or "",
            derived_from=target.rel,
            transform_pass="semi_join",
          )
          generated_rules.append(
            Rule(
              head=gen_head,
              body=(gen_target, gen_filter),
              name=f"{new_rel_name}_Gen",
              is_generated=True,
              prov=prov,
            )
          )

        # Build the replacement clause for the original rule.
        prov = compiler_gen(
          parent_rule=rule.name or "",
          derived_from=target.rel,
          transform_pass="semi_join",
        )
        new_atom = Atom(
          rel=new_rel_name,
          args=tuple(target.args[k] for k in kept_idx),
          prov=prov,
        )
        rewrites[i] = (new_atom, j)
        replaced.add(j)
        break  # one opt per target

    # Pass 2: build the rewritten body + clause-index mapping.
    new_body: list = []
    idx_map: dict[int, int] = {}
    filter_to_target = {fi: ti for ti, (_, fi) in rewrites.items()}
    next_idx = 0
    for i, clause in enumerate(body):
      if i in replaced:
        # Filter removed — but it'll map to the _SJ_ index the target now owns.
        idx_map[i] = -2 if i in filter_to_target else -1
        continue
      if i in rewrites:
        idx_map[i] = next_idx
        fi = rewrites[i][1]
        if fi in idx_map and idx_map[fi] == -2:
          idx_map[fi] = next_idx
        next_idx += 1
        new_body.append(rewrites[i][0])
      else:
        idx_map[i] = next_idx
        next_idx += 1
        new_body.append(clause)

    # Pass 3: translate plans (delta / clause_order / var_order).
    body_vars: set[str] = set()
    for c in new_body:
      body_vars |= _clause_vars(c)

    translated_plans: list[PlanEntry] = []
    for plan in rule.plans:
      new_delta = plan.delta
      if plan.delta >= 0:
        if plan.delta not in idx_map:
          continue
        mapped = idx_map[plan.delta]
        if mapped < 0:
          continue
        new_delta = mapped
      new_clause_order = tuple(
        idx_map[k] for k in plan.clause_order if k in idx_map and idx_map[k] >= 0
      )
      new_var_order = tuple(v for v in plan.var_order if v in body_vars)
      translated_plans.append(
        dataclasses.replace(
          plan,
          delta=new_delta,
          clause_order=new_clause_order,
          var_order=new_var_order,
        )
      )

    new_rules.append(
      dataclasses.replace(
        rule,
        body=tuple(new_body),
        plans=tuple(translated_plans),
      )
    )

  # Generated rules go FIRST so stratification schedules them upstream of
  # their consumers. Generated decls append to the existing list.
  return generated_rules + new_rules, list(decls) + generated_decls


class SemiJoinPass:
  '''Runs `optimize_semi_joins` to a fixed point. Nim handles the outer
  loop in compileToHir; we keep it inside the pass for symmetry with the
  other rule-rewrites.
  '''

  info = PassInfo(
    name="SemiJoinOpt",
    level=PassLevel.RULE_REWRITE,
    order=2,
    source_dialect=Dialect.HIR,
    target_dialect=Dialect.HIR,
  )

  def run(self, rules, decls):
    while True:
      prev = len(rules)
      rules, decls = optimize_semi_joins(rules, decls)
      if len(rules) == prev:
        return rules, decls
