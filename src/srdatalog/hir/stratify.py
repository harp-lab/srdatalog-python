'''HIR Pass 1: Stratification.

Input:  Program (DSL-level: relations + rules)
Output: HirProgram with strata populated (scc_members, is_recursive, stratum_rules)

Mirrors src/srdatalog/hir/stratification.nim. Key behavior: for a recursive SCC
that has some rules with no SCC dependency in their body (true base case) and
some with SCC dependency (recursive step), we emit TWO separate strata -- base
first, then recursive -- so the base runs once before the fixpoint loop.

The subsequent fusion pass merges consecutive non-recursive strata that have no
inter-dependency, enabling parallel execution at the MIR/codegen level.

NOTE: Python's set iteration order depends on the hash-random seed across runs.
To keep SCC ordering reproducible, we sort every set iteration that feeds into
downstream ordering (dependency edges, SCC membership iteration).
'''

from __future__ import annotations

from srdatalog.dsl import Agg, Atom, Negation, Rule
from srdatalog.hir.pass_ import Dialect, PassInfo, PassLevel
from srdatalog.hir.types import HirProgram, HirStratum, RelationDecl


def _body_relations(rule: Rule) -> set[str]:
  '''Relation names referenced in a rule's body (Atom, Negation, Agg).
  Filter / Let clauses don't reference a relation — they're inline
  predicates / bindings — and are skipped.
  '''
  out: set[str] = set()
  for b in rule.body:
    if isinstance(b, Negation):
      out.add(b.atom.rel)
    elif isinstance(b, Atom) or isinstance(b, Agg):
      out.add(b.rel)
    # Filter / Let: no relation reference, skip.
  return out


def _build_dep_graph(rules: list[Rule]) -> dict[str, set[str]]:
  '''head_rel -> set of body rels (only IDBs: rels that appear as some rule's head).
  Multi-head rules contribute every head_rel -> body-rels edge.'''
  idbs: set[str] = {h.rel for r in rules for h in r.heads}
  graph: dict[str, set[str]] = {rel: set() for rel in idbs}
  for r in rules:
    body_idbs = {b for b in _body_relations(r) if b in idbs}
    for h in r.heads:
      graph[h.rel] |= body_idbs
  return graph


def _compute_sccs(rules: list[Rule], graph: dict[str, set[str]]) -> list[set[str]]:
  '''Tarjan's SCC. Returns SCCs in reverse topological order (bottom-up).

  Starting nodes are iterated in rule-definition order, and neighbors in
  sorted order, so output is reproducible across runs.
  '''
  index: dict[str, int] = {}
  lowlink: dict[str, int] = {}
  on_stack: set[str] = set()
  stack: list[str] = []
  counter = [0]
  sccs: list[set[str]] = []

  def strongconnect(v: str) -> None:
    index[v] = counter[0]
    lowlink[v] = counter[0]
    counter[0] += 1
    stack.append(v)
    on_stack.add(v)
    for w in sorted(graph.get(v, set())):
      if w not in index:
        strongconnect(w)
        lowlink[v] = min(lowlink[v], lowlink[w])
      elif w in on_stack:
        lowlink[v] = min(lowlink[v], index[w])
    if lowlink[v] == index[v]:
      new_scc: set[str] = set()
      while True:
        w = stack.pop()
        on_stack.discard(w)
        new_scc.add(w)
        if w == v:
          break
      sccs.append(new_scc)

  seen: set[str] = set()
  for r in rules:
    for h in r.heads:
      name = h.rel
      if name in graph and name not in seen:
        seen.add(name)
        if name not in index:
          strongconnect(name)
  return sccs


def _merge_sccs_for_multi_head(sccs: list[set[str]], rules: list[Rule]) -> list[set[str]]:
  '''Union SCCs that share a multi-head rule. Mirrors
  mergeSCCsForMultiHeadRules in src/srdatalog/hir/stratification.nim.
  A rule with N>1 heads must emit all heads from a single pipeline, so
  every head-rel's SCC is fused into one.
  '''
  rel_to_scc: dict[str, int] = {}
  for i, scc in enumerate(sccs):
    for rel in scc:
      rel_to_scc[rel] = i

  parent = list(range(len(sccs)))

  def find(x: int) -> int:
    while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    return x

  def union(a: int, b: int) -> None:
    ra, rb = find(a), find(b)
    if ra != rb:
      parent[ra] = rb

  for rule in rules:
    if len(rule.heads) > 1:
      head_sccs = [rel_to_scc[h.rel] for h in rule.heads if h.rel in rel_to_scc]
      for i in range(1, len(head_sccs)):
        union(head_sccs[0], head_sccs[i])

  # Rebuild, preserving original order by first-occurrence of each root.
  merged: dict[int, set[str]] = {}
  root_order: list[int] = []
  for i, scc in enumerate(sccs):
    root = find(i)
    if root not in merged:
      merged[root] = set()
      root_order.append(root)
    merged[root] |= scc
  return [merged[r] for r in root_order]


def _is_recursive_scc(scc: set[str], graph: dict[str, set[str]]) -> bool:
  if len(scc) > 1:
    return True
  for rel in scc:  # singleton
    if rel in graph.get(rel, set()):
      return True
  return False


def _rules_for_scc(rules: list[Rule], scc: set[str]) -> list[Rule]:
  return [r for r in rules if any(h.rel in scc for h in r.heads)]


def _split_base_rec(scc_rules: list[Rule], scc: set[str]) -> tuple[list[Rule], list[Rule]]:
  '''Partition SCC rules into (base, recursive) by whether the body references
  any SCC member. Base rules have no SCC dependency; they can run once before
  the fixpoint loop.
  '''
  base, rec = [], []
  for r in scc_rules:
    if _body_relations(r) & scc:
      rec.append(r)
    else:
      base.append(r)
  return base, rec


def _stratum_depends_on(s: HirStratum, produced: set[str]) -> bool:
  for r in s.stratum_rules:
    if _body_relations(r) & produced:
      return True
  return False


def _fuse_independent_strata(strata: list[HirStratum]) -> list[HirStratum]:
  '''Merge consecutive non-recursive strata with no inter-dependency.

  A non-recursive stratum that is the BASE of a recursive SCC is never fused
  (it must stay pinned right before its recursive sibling so the evaluator
  runs them in sequence).
  '''
  if len(strata) <= 1:
    return strata

  recursive_members: set[str] = set()
  for s in strata:
    if s.is_recursive:
      recursive_members.update(s.scc_members)

  def is_fusable(s: HirStratum) -> bool:
    if s.is_recursive:
      return False
    return not (s.scc_members & recursive_members)

  fused: list[HirStratum] = []
  i = 0
  while i < len(strata):
    current = strata[i]
    if not is_fusable(current):
      fused.append(current)
      i += 1
      continue
    produced = set(current.scc_members)
    j = i + 1
    while j < len(strata):
      nxt = strata[j]
      if not is_fusable(nxt):
        break
      if _stratum_depends_on(nxt, produced):
        break
      current.scc_members.update(nxt.scc_members)
      current.stratum_rules.extend(nxt.stratum_rules)
      if nxt.before_hook:
        current.before_hook += nxt.before_hook
      if nxt.after_hook:
        current.after_hook += nxt.after_hook
      current.is_generated = current.is_generated and nxt.is_generated
      produced = set(current.scc_members)
      j += 1
    fused.append(current)
    i = j
  return fused


def stratify(rules: list[Rule], decls: list[RelationDecl]) -> HirProgram:
  '''HIR Pass 1 entry point. Takes (rules, decls) after any rule-rewrite
  passes; produces a HirProgram with strata populated. Mirrors stratify()
  in src/srdatalog/hir/stratification.nim.

  This is the fixed entry of the HIR pipeline — it's the point where
  (rules, decls) becomes HirProgram. See `hir_pass.Pipeline.compile_to_hir`.
  '''
  graph = _build_dep_graph(rules)
  sccs = _compute_sccs(rules, graph)
  sccs = _merge_sccs_for_multi_head(sccs, rules)

  strata: list[HirStratum] = []
  for scc in sccs:
    scc_rules = _rules_for_scc(rules, scc)
    if not scc_rules:
      continue
    if _is_recursive_scc(scc, graph):
      base_rules, rec_rules = _split_base_rec(scc_rules, scc)
      if base_rules:
        strata.append(
          HirStratum(scc_members=set(scc), is_recursive=False, stratum_rules=base_rules)
        )
      if rec_rules:
        strata.append(HirStratum(scc_members=set(scc), is_recursive=True, stratum_rules=rec_rules))
    else:
      strata.append(HirStratum(scc_members=set(scc), is_recursive=False, stratum_rules=scc_rules))

  strata = _fuse_independent_strata(strata)
  return HirProgram(strata=strata, relation_decls=decls)


# -----------------------------------------------------------------------------
# Pass wrapper — lets the Pipeline treat stratification uniformly even though
# it's signature-wise distinct (it's the HIR entry, not a transform).
# -----------------------------------------------------------------------------


class StratificationPass:
  info = PassInfo(
    name="Stratification",
    level=PassLevel.HIR_TRANSFORM,
    order=0,
    source_dialect=Dialect.HIR,
    target_dialect=Dialect.HIR,
  )

  def run(self, rules: list[Rule], decls: list[RelationDecl]) -> HirProgram:
    return stratify(rules, decls)
