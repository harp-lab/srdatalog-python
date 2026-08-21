# Trace-parity contract (v1)

**Scope:** the minimal VulReasoner subset. **Parties:** the PyReason oracle
(as consumed by VulReasoner's `trace_conversion.py` / `filter_reasoning_trace.py`)
and any SRDatalog implementation claiming trace parity (e.g. the
`srdatalog.pyreason` frontend of PR #2).

This document defines *what "the SRDatalog trace matches PyReason" means* —
precisely enough to implement against and to test mechanically. It deliberately
does **not** require reproducing PyReason's `get_rule_trace` operation
chronology; §5 proves that within scope no chronology is needed.

Field names below refer to VulReasoner's `ReasoningStep`
(`vulreasoner-mindset/src/trace_conversion.py`), the struct every downstream
consumer reads. Empirical claims about PyReason's CSV output were verified
against real traces in `vulreasoner-mindset/outputs/traces_api/CWE_121_MVP2/`.

---

## 1. Scoped program fragment

The contract covers exactly the programs expressible in
`minimal_vulreasoner/`:

| Element | Allowed forms |
|---|---|
| Rules | the six analyst rules of `rules/analyst_rules.csv`: head `analystAt(CB2):paired_minimum_bounds_ann_fn`, delay `<-1`, body = `analystAt(CB1):[0.25,1]`, `hasLabel(CB1,·):[0.1,1]`, `hasLabel(CB2,·):[0.1,1]`, `connector(·,·):[0.1,1]`, `stepFrom(CB1,CB2)` |
| Annotation fn | `paired_minimum_bounds_ann_fn` only |
| Closed world | `analystAt` (fixed finite domain) |
| Facts | static `hasLabel(entity,label)`; timed `analystAt(seed)` on `[0,1]`; timed `stepFrom(bᵢ,bᵢ₊₁)` on `[i+1,i+2]`; KG attributes via `load_graphml` + `save_graph_attributes_to_trace` |
| Settings | `atom_trace=True`, `allow_ground_rules=True`, `save_graph_attributes_to_trace=True`, `persistent=False` (default) |
| Run | `pr.reason(timesteps=END_TIME)` |

**Excluded** (out of contract, not merely untested): `<-0` rules (label
rules), the `closed_rule.csv` pair (`future`, negated-head
`inconsistent_rule`), `set_static` heads, binary heads with `infer_edges`
domain growth, the `obs` layer and ground-atom registration, `persistent=True`,
multi-KG orchestration, and `generalized_minimum_bounds_ann_fn`. Every one of
these invalidates at least one lemma in §5; extending the contract to them
requires a new version, not a relaxation.

## 2. The canonical trace record

Both engines must be renderable into a common form. One **record** corresponds
to one row of PyReason's rule-trace CSVs (one `ReasoningStep`):

```
Record = (
  time            : int,                     # logical timestep
  label           : str,                     # predicate name
  term            : Node(id) | Edge(a, b),   # 1-ary → Node, 2-ary → Edge
  due_to          : str,                     # rule name or fact name
  triggered_by    : "Fact" | "Rule",
  consistent      : bool,
  clause_sets     : tuple[frozenset[Term], ...],  # per body-clause position; () for facts
  inconsistency_message : None,              # always None in scope
)
```

plus, per record, `old_bound` and `new_bound` (handled separately in §4 —
they are *not* part of the canonical identity because intermediate bounds are
order-sensitive when several updates hit one key).

Canonicalization rules:

- **term**: PyReason writes 1-ary atoms to the node CSV (`term = {"node": id}`)
  and 2-ary atoms to the edge CSV (`term = {"edge": (a, b)}`). The SRDatalog
  side must classify identically by predicate arity.
- **clause_sets**: parse each `Clause-i` cell with `ast.literal_eval`; map each
  grounding through `grounding_to_term` semantics (bare string or 1-tuple →
  Node, n-tuple → Edge); collect **as a set**, position-aligned with the
  rule's body clauses in source order. Ordering *inside* a clause cell is
  PyReason predicate-map admission order and is explicitly **not** part of the
  contract. Facts have `clause_sets = ()`.
- **key** of a record group: `(label, term, time)` — this matches
  `filter_reasoning_trace.index_steps` (note: that index ignores `time`; the
  contract still groups by time because §3's multiset equality subsumes the
  coarser index).

## 3. Tier 1 — firing-multiset equality (REQUIRED)

> The **multiset** of canonical records produced by the two engines over the
> full run must be equal.

This is the load-bearing requirement. It implies, without further clauses:

- `num_inferences` parity (it is `len(combined_trace)`),
- `input_valid` parity (§6; all records must have `consistent = True`),
- `base_filter` output parity — the filter's seeds (`due_to` prefix
  `analyst-rule`), recursion gate (`triggered_by == "Rule"`), body-predicate
  walk over `clause_sets`, and keep-index over `(label, term)` are all
  functions of the canonical multiset only,
- secondary-filter (`label_filtering` / `workflow_filtering` /
  `analyst_filtering`) parity, given Tier 2 below.

The record multiset decomposes into two families, each independently checkable:

**Fact records** — deterministic re-assertion schedule (verified against real
traces; a consequence of `persistent=False` worlds resetting to `[0,1]` each
step):

| Source | Records emitted |
|---|---|
| static fact (incl. every `hasLabel` workflow fact) | one per `t ∈ [0, END_TIME]`, `due_to` = fact name |
| timed fact on `[t0, t1]` (`initial-control`, `workflow-edge-i`) | one per `t ∈ [t0, t1]`, `due_to` = fact name |
| GraphML attribute (KG nodes/edges, e.g. `can_cause`) | one per `t ∈ [0, END_TIME]`, `due_to = "graph-attribute-fact"` |

All fact records: `triggered_by = "Fact"`, `clause_sets = ()`,
`consistent = True`.

**Rule records** — by §5, within scope each `(rule, head-grounding, t)` fires
at most once, and its clause sets are the *complete* qualified-grounding sets
at `t`, uniquely determined by the inputs. So the expected rule-record multiset
is: for every analyst rule `r`, head grounding `h`, and timestep `t+1 ≤
END_TIME` such that `r`'s body is satisfied at `t` (all five clauses have a
consistent grounding meeting their interval thresholds, `analystAt` read
closed-world), exactly one record

```
(t+1, "analystAt", Node(h), name(r), "Rule", True,
 (Q₁, Q₂, Q₃, Q₄, Q₅))
```

where `Qᵢ` is the full set of groundings of clause `i` qualified at `t` **in
the rule-consistent join** (the sets PyReason's `atom_trace` records: every
grounding that participates in some satisfying assignment, not only the
ARG-MAX winner). Recording only the winning witness is a contract violation —
`base_filter`'s transitive walk visits *every* grounding in every clause set.

## 4. Tier 2 — bounds (REQUIRED, order-tolerant)

Per key `(label, term, time)`:

1. **Final bound — exact.** The last `new_bound` after all updates at that key
   must be equal across engines. For `analystAt` keys this is precisely the
   temporal interval map already validated by `validate_minimal_parity.py`.
   For fact keys it is the fact's bound (`[1,1]`).
2. **Chain consistency — per engine (self-check).** Within one engine's
   ordered records for a key: the first `old_bound` is the reset value
   `[0,1]`; each subsequent `old_bound` equals the previous `new_bound`; the
   last `new_bound` is the final bound.
3. **Interleaving — NOT compared.** When ≥ 2 records share a key (two analyst
   rules deriving the same head at the same t), the sequence of intermediate
   `old_bound → new_bound` transitions is engine-order-dependent and is
   excluded from comparison. (Consequence of interval intersection being
   commutative/associative: the final bound is order-free; the intermediates
   are not.) No downstream consumer reads intermediates: `base_filter` never
   touches bounds; `workflow_filtering` reads only `new_bound[1]` of
   `stepFrom` records, which are single-record fact keys.
4. **Float equality** = equality of IEEE-754 float32 bit patterns
   (`float32_to_u32`) after parsing; the PyReason side parses the CSV decimal
   strings to float64 first. This matches the digest convention already used
   by `benchmark_srdatalog.py`.

The annotation semantics behind rule-record bounds (already implemented and
value-validated, restated here as the normative definition): per firing, each
satisfying body grounding is a witness; a witness's candidate interval is
`[min, min]-pairing` per `paired_minimum_bounds_ann_fn`; the firing's
contributed interval is that of the witness with maximal lower bound, ties
broken by **least admission rank** (PyReason first-match); the world update is
`old_bound ∩ contributed`.

## 5. Why no chronology is needed (the snapshot-stability argument)

The PR's conversion report declines to reproduce "callback trace chronology"
— same-snapshot witness groups and global `Fixed-Point-Operation` numbers.
Within the scoped fragment this problem dissolves:

- **Lemma 1 (no intra-timestep feedback).** Every in-scope rule carries delay
  `<-1`: bodies are evaluated against the state at `t`, heads land at `t+1`.
  The state at `t` is fully determined before any rule evaluation at `t`
  begins — it consists of fact assertions scheduled at `t` (§3 table) and
  delayed heads computed from `t-1`. No in-scope derivation can change a set
  another in-scope body reads at the same `t`.
- **Lemma 2 (stable qualified sets).** Therefore the qualified set `Qᵢ` of
  every clause at `(rule, head, t)` is the unique fixpoint-independent set
  determined by the inputs — there is no "growing snapshot" across
  iterations. (The multi-row-per-key rows observed in full-VulReasoner traces
  are caused by `<-0` label rules growing `hasLabel` within a timestep —
  excluded from scope.)
- **Corollary.** Each `(rule, head, t)` yields at most one record, computable
  without replaying PyReason's operational loop. The trace is a pure function
  of the inputs.

This is why `Fixed-Point-Operation` is excluded (§7): it is the only trace
column whose value is chronology, and nothing consumes it.

## 6. Derived-output requirements

For the per-KG result shape (`run_reasoning_for_kg`):

| Field | Requirement |
|---|---|
| `input_valid` | `True` on both engines (no inconsistency sources in scope); any `consistent = False` record is a contract violation, not a tolerated difference |
| `num_inferences` | equal (follows from Tier 1) |
| `reasoning_trace` (base-filtered) | equal as canonical multisets (follows from Tiers 1–2) |
| `loaded_rules` | the six analyst rules, parsed per `parse_rules_from_file`, in file order — required because `base_filter` maps `due_to` → body predicates through it |
| `metrics`, `fixed_point_operation` | excluded (§7) |

## 7. Explicitly out of contract

- `Fixed-Point-Operation` values (SRDatalog may synthesize any int, e.g. `0`
  or a per-time counter; must still be parseable as int).
- Row **order** in CSVs / JSON arrays, and ordering inside clause cells.
  Comparison is multiset-based after canonicalization; byte-identical output
  is a non-goal. (If byte-stable artifacts are wanted later, apply one shared
  normalizer — sorted records, sorted clause sets — to *both* engines'
  outputs; do not chase PyReason's native ordering.)
- Trace CSV filenames/timestamps, edge-vs-node file column counts, float
  *formatting* (only float32 values are compared).
- Intermediate `old_bound → new_bound` interleavings on multi-record keys
  (§4.3).
- Everything in §1 "Excluded".

## 8. Conformance procedure

A conforming implementation is verified by a differential harness
(`validate_trace_parity.py`, companion to `validate_minimal_parity.py`):

1. **Oracle:** run the minimal example under PyReason, `pr.save_rule_trace`,
   load CSVs with `trace_conversion` semantics, canonicalize per §2.
2. **Candidate:** run the SRDatalog implementation; emit records per §3 (fact
   schedule from the input manifest; rule records from the materialized
   candidate/selected relations plus clause-set queries), canonicalize per §2.
3. **Compare:** (a) Tier-1 multiset equality — report missing / extra records
   grouped by key; (b) Tier-2 final-bound map equality under float32 bit
   equality; (c) chain self-checks per engine; (d) run `base_filter` +
   secondary filters over both canonical traces and assert equal kept
   multisets — this guards the theorem-level claims in §3 empirically.
4. **Pass** = all of (a)–(d) on: the checked-in `CWE_121_MVP2` workflow, and
   the deterministic stress workloads of `stress_workload.py` at ≥ 2 sizes.

Divergence triage order: a Tier-1 mismatch on *fact* records is an ingest/
schedule bug; on *rule* records with equal clause sets it is a threshold/CWA
read bug; a clause-set mismatch is a qualified-set (not winner-only)
materialization bug; a Tier-2-only mismatch is an annotation-fn or
intersection bug.

## 9. Versioning

This is **v1**, scoped to §1. Any widening — `<-0` rules, negation/
`inconsistent_rule`, `set_static`, persistence, `obs` — must be a new
contract version with its own stability argument replacing §5 (Lemma 1 fails
for `<-0` rules and for the undelayed `inconsistent_rule`; conflict repair
additionally breaks §4.3's commutativity argument).
