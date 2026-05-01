# Datalog IR Architecture: Multi-Dialect Compilation with Denotational Semantics

This document specifies the IR design for the datalog compiler. The
core decision: **the IR is a stack of phases, but each phase hosts
multiple dialects**. Different data structures, parallelism strategies,
and hardware backends are *separate dialects* with independent lifetimes
in the codebase. New dialects can be added without touching existing
ones.

The architecture is validated against four flexibility axes:

1. **Data structure** — sorted-array, hash-trie, $K$-level LSM,
   union-find, bitmap, …
2. **Join processing** — generic-join (leapfrog), binary, materialized,
   balanced-scan.
3. **Hardware backend** — NVIDIA CUDA, AMD HIP, CPU+TBB, future
   targets (Apple Metal, SYCL).
4. **Parallel mode** — warp-strided, block-group, atomic-WS, TBB
   parallel-for, task-parallel.

Each axis becomes an *orthogonal* family of dialects. A program's
compilation pipeline is a chain of independent dialect-choice
decisions, not a Cartesian explosion of templates.

The codegen target is **C++ source code** consumed by the existing JIT
runtime. There is no LLVM/MLIR dependency; the IR machinery is
implemented in Python (~600 LOC) with each dialect at ~500–1500 LOC.

## 0. Reading map

| Part | Section | What it answers |
|---|---|---|
| **I** Architecture | §1–4 | What the compiler looks like; what a dialect is. |
| **II** Semantics | §5–8 | Denotation; the correctness theorem; refinement. |
| **III** IIR-sorted-array | §9–12 | The first complete dialect — types, lowerings, proofs, target backends. |
| **IV** Other dialects | §13–15 | Sketches of LSM⟨K⟩, union-find, bitmap. Validates pluggability. |
| **V** Cross-cutting | §16–18 | Parallelism, target, memory dialects. |
| **VI** Machinery | §19–22 | Registry, pattern matching, pass driver, verifier. |
| **VII** Roadmap | §23+ | Implementation order with risk gates. |

# Part I: Compilation Architecture

## 1. Pipeline overview

```
        HIR (rules, atoms, fixpoints — purely semantic)
         │
         ▼  [HIR→MIR pass: rule rewrite, magic sets, semi-naive plumbing]
         │
        MIR (physical join plans — generic, dialect-free)
         │
         ▼  [MIR→IR position: per-relation dialect choice]
         │
   ┌─────┴─────┬─────────┬─────────────────────────┐
   │           │         │                         │
   ▼           ▼         ▼                         ▼
relation.       relation.    relation.           relation.
sorted_array     lsm⟨K⟩     uf                  bitmap
   │           │         │                         │
   └─────┬─────┴─────────┴─────────────────────────┘
         │
         ▼  [target lowering, parameterized by parallelism dialect]
         │
   ┌─────┴─────┬─────────┬─────────────────────────┐
   ▼           ▼         ▼                         ▼
target.cuda    target.hip    target.cpp_tbb     target.cpp_omp
   │           │         │                         │
   └─────┬─────┴─────────┴─────────────────────────┘
         │
         ▼
       C++ source (consumed by JIT runtime)
```

Phases (HIR, MIR, IR position, target) are *positions* in the pipeline.
Dialects are *families that occupy a position*. The IR position can
host multiple data-structure dialects in the same program — different
relations may live in different dialects.

## 2. The four flexibility axes

| Axis | Mechanism | Where it lives |
|---|---|---|
| Data structure | Choice of low-level dialect per relation | MIR→IR pass |
| Join processing | Rewrite rules within MIR | MIR-internal passes |
| Hardware backend | Choice of target dialect | IR→target pass |
| Parallel mode | `ParallelFor` strategy parameter (a sub-dialect) | within each dialect |

Independence claim: the choice along one axis does not constrain the
others. An LSM⟨K⟩ relation can be compiled with TBB parallel-for; a
sorted-array relation with CUDA warp-strided. The compiler driver
makes each decision independently, guided by per-relation metadata
and target configuration.

## 3. The dialect ABI

Every dialect $\mathcal{D}$ provides:

$$
\mathcal{D} = \langle \mathcal{T}_\mathcal{D},\ \mathcal{O}_\mathcal{D},\ \mathcal{L}_\mathcal{D}^{\text{out}},\ \mathcal{R}_\mathcal{D},\ \mathcal{V}_\mathcal{D} \rangle
$$

- $\mathcal{T}_\mathcal{D}$ — types: state, handle, view, etc.
- $\mathcal{O}_\mathcal{D}$ — operations on those types.
- $\mathcal{L}_\mathcal{D}^{\text{out}}$ — lowering rules out of the dialect, into other
  dialects (in particular, into target dialects).
- $\mathcal{R}_\mathcal{D}$ — internal rewrite rules.
- $\mathcal{V}_\mathcal{D}$ — verifier predicates (well-formedness).

Each component is a *value* in the dialect-registry sense: registered
at compiler init time, looked up by pattern matching during lowering.

## 4. Pluggability invariants

Three properties make the IR genuinely extensible:

### Property P1 — Open dialect registration

Adding a new dialect = providing $\langle \mathcal{T}, \mathcal{O}, \mathcal{L}^{\text{out}}, \mathcal{R}, \mathcal{V} \rangle$
and calling `compiler.register_dialect(...)`. **No edits to existing
dialects.** No central enum, no central type-switch.

### Property P2 — Polymorphic relation references

MIR ops never name a dialect. A relation is referenced by *symbolic
name*; the symbol table maps names to dialect-bound state types. The
MIR→IR pass looks up each source's actual dialect at lowering time.

This is what makes the data-structure axis open. If MIR encoded
"this source is sorted-array," every new dialect would require MIR
edits. Because MIR encodes only "this source is named `path`," the
lowering pass dispatches per-source.

### Property P3 — Dialect-local pass libraries

Lowering rules and rewrites are typed: $\mathcal{L} : \mathcal{D}_1.\text{Op} \to \mathcal{D}_2.\text{Op}^*$.
A pass belongs to *one* dialect — the one whose ops it consumes
or produces. There is no global pass that knows about all dialects;
the pass driver composes them by type.

Adding a dialect ships ~5 lowering passes, ~3 rewrite passes, ~1
verifier. Bounded scope. No risk of breaking existing dialects.

# Part II: Semantic Foundations

## 5. Domains

$$
\begin{aligned}
V &\quad\text{interned values (uint32 / uint64)} \\
\text{Tuple} &= V^{*} \\
\text{Schema} &= \langle \text{Name},\ \text{Arity},\ \text{AttrTs},\ \mathcal{S} \rangle, \quad \mathcal{S} = (V, \oplus, \otimes, 0_\mathcal{S}, 1_\mathcal{S})\ \text{a semiring} \\
\text{Relation} &= \langle \text{Schema},\ \text{MultiSet}_\mathcal{S}\langle \text{Tuple} \rangle \rangle \\
\text{Version} &::= \mathbf{FULL} \mid \mathbf{DELTA} \mid \mathbf{NEW} \\
\text{DB} &: (\text{Schema} \times \text{Version}) \to \text{Relation} \\
\beta &: \text{Var} \rightharpoonup V, \quad \beta\cdot s \quad\text{(binding tagged with } s \in \mathcal{S}\text{)} \\
B &: \text{Bindings}_\mathcal{S} = \text{MultiSet}_\mathcal{S}\langle \text{Binding} \rangle \\
\text{Output} &= \text{MultiSet}_\mathcal{S} \langle \text{Schema} \times \text{Tuple} \rangle
\end{aligned}
$$

Notation:
- $\uplus$ — multiset union, combining $\oplus$ on duplicate keys.
- $\bigotimes_i$ — semiring product over an indexed family.
- $|M| = \sum_{x \in M} \mathrm{sr}(x)$ — semiring-weighted cardinality.
- $\beta[v \mapsto x]$ — extending a binding.
- $\pi_O$ — projection of a dialect-annotated result onto the `Output` component.

## 6. MIR semantics (the spec)

$\llbracket \cdot \rrbracket_M : \text{MirOp}^{*} \times \text{DB} \times \text{Iteration} \times \text{Bindings}_\mathcal{S} \to \text{Output}$

Defined by structural induction. Empty pipeline:

$$\llbracket [\,] \rrbracket_M(\text{db}, i, B) = \emptyset$$

Cons cases:

$$
\begin{aligned}
\llbracket \mathbf{Scan}(\bar{v}, s) :: r \rrbracket_M(\text{db}, i, B) &= \llbracket r \rrbracket_M\bigl(\text{db}, i, \biguplus_{\beta\cdot \sigma \in B} \{\text{ext}(\beta, \bar{v}, t) \cdot (\sigma \otimes \sigma_t) \mid (t, \sigma_t) \in \text{rows}(s, \beta)\}\bigr) \\
\llbracket \mathbf{CJ}(v, \bar{s}) :: r \rrbracket_M(\text{db}, i, B) &= \llbracket r \rrbracket_M\bigl(\text{db}, i, \biguplus_{\beta\cdot \sigma \in B}\ \biguplus_{x \in \bigcap_{s \in \bar{s}} \text{vals}(s, \beta)} \{\beta[v \mapsto x] \cdot \sigma'\}\bigr) \\
\llbracket \mathbf{Cart}(\bar{v}, \bar{s}) :: r \rrbracket_M(\text{db}, i, B) &= \llbracket r \rrbracket_M\bigl(\text{db}, i, \biguplus_{\beta\cdot \sigma \in B}\ \biguplus_{(x_k) \in \prod_s \text{vals}(s, \beta)} \{\beta[\bar{v} \mapsto (x_k)] \cdot \sigma'\}\bigr) \\
\llbracket \mathbf{Filter}(p) :: r \rrbracket_M(\text{db}, i, B) &= \llbracket r \rrbracket_M(\text{db}, i, \{\beta\cdot \sigma \in B \mid p(\beta)\}) \\
\llbracket \mathbf{Bind}(v, e) :: r \rrbracket_M(\text{db}, i, B) &= \llbracket r \rrbracket_M(\text{db}, i, \{\beta[v \mapsto e(\beta)] \cdot \sigma \mid \beta\cdot\sigma \in B\}) \\
\llbracket \mathbf{Neg}(\rho, \bar{u}) :: r \rrbracket_M(\text{db}, i, B) &= \llbracket r \rrbracket_M\bigl(\text{db}, i, \{\beta\cdot\sigma \in B \mid \neg \exists t \in \rho@i.\ t.\text{prefix} = \bar{u}(\beta)\}\bigr) \\
\llbracket \mathbf{Agg}(v, s, \bar{u}, f) :: r \rrbracket_M(\text{db}, i, B) &= \llbracket r \rrbracket_M\bigl(\text{db}, i, \{\beta[v \mapsto \text{fold}(f, \text{vals}(s, \beta))] \cdot \sigma\}\bigr) \\
\llbracket \mathbf{II}(\rho, \bar{v}, \alpha) :: r \rrbracket_M(\text{db}, i, B) &= \{(\rho, \pi(\beta, \bar{v})) \cdot \sigma \mid \beta\cdot\sigma \in B\} \uplus \llbracket r \rrbracket_M(\text{db}, i, B)
\end{aligned}
$$

This is the spec. **No lowering may produce an Output that differs
from this multiset** (modulo the refinement defined in §8).

## 7. Multi-dialect lowering correctness

The classical lowering theorem states: a lowering $L$ is correct iff
$\llbracket p \rrbracket_M = \pi_O\, \llbracket L(p) \rrbracket_{IR}$. With multiple dialects this generalizes.

**Definition.** A *dialect-bound program* is a triple $(p, \tau, R)$
where $p$ is an MIR program, $\tau$ is a symbol table mapping each
relation name in $p$ to a dialect, and $R$ is a per-dialect realization
(the actual state values at runtime).

**Theorem (multi-dialect lowering correctness).** For every dialect
$\mathcal{D}$ and every dialect-bound program $(p, \tau, R)$ where $\tau$ binds all
$p$'s relations to $\mathcal{D}$:

$$
\forall\, \text{db},\, i,\, B,\, V,\, W \text{ a complete partition.}\quad
\llbracket p \rrbracket_M(\text{db}, i, B)\ =_{\uplus}\ \pi_O\, \llbracket L_\mathcal{D}(p) \rrbracket_\mathcal{D}(\text{db}, i, B, R, W, V)
$$

where $L_\mathcal{D}$ is $\mathcal{D}$'s lowering function. Each dialect proves this
theorem **independently**, and the proofs do not refer to other
dialects.

**Compositional case.** When $\tau$ binds $p$'s relations to several
dialects $\mathcal{D}_1, \ldots, \mathcal{D}_n$, the compiler driver applies a *per-source
lowering*: each MIR op's lowering is parameterized over the source's
dialect. Correctness follows by induction on the MIR op tree, using
each dialect's individual theorem at the leaves.

## 8. Refinement and parallelism

**Definition (refinement).** $M \sqsubseteq_\uplus M'$ iff $M$ and $M'$ are
equal as multisets (any sequential ordering admissible).

**Theorem (parallel correctness).** Let $\pi$ be a complete partition
of $B$ over $W$ (i.e., $\biguplus_{w \in W} \pi(B, w) = B$, no duplication, no omission).
Then for any $\uplus$-distributive pipeline $p$:

$$
\biguplus_{w \in W} \llbracket p \rrbracket_M(\text{db}, i, \pi(B, w))\ =_\uplus\ \llbracket p \rrbracket_M(\text{db}, i, B)
$$

All MIR ops are $\uplus$-distributive by inspection.

**Implication.** Strategy-level differences in `ParallelFor` (warp-strided
vs block-group vs TBB-for vs atomic-WS) are *cost decisions*, not
*correctness decisions*. The semantic function collapses them.

# Part III: The IIR-sorted-array dialect

This is the first complete dialect, ported from the existing GPU JIT
emitter. It serves as the proof-of-concept and the byte-equivalence
gate for the rewrite.

## 9. Types and operations

### Types ($\mathcal{T}_{sa}$)

$$
\begin{aligned}
\text{sa}\langle T \rangle &\quad\text{sorted-array state for schema } T \\
\text{sa\_view}\langle T \rangle &\quad\text{a single sorted-array view (one column ordering)} \\
\text{sa\_handle}\langle T \rangle &\quad\text{node handle: } \langle \text{view},\ \text{lo},\ \text{hi},\ \text{depth} \rangle
\end{aligned}
$$

### Operations ($\mathcal{O}_{sa}$)

```
sa.root(v: sa_view⟨T⟩)                   -> sa_handle⟨T⟩
sa.pref(h: sa_handle⟨T⟩, k: V)           -> sa_handle⟨T⟩
sa.child(h: sa_handle⟨T⟩, j: int)        -> sa_handle⟨T⟩
sa.hint(v: sa_view⟨T⟩, lo, hi, d: int)   -> sa_handle⟨T⟩
sa.values(h: sa_handle⟨T⟩)               -> stream⟨V⟩
sa.degree(h: sa_handle⟨T⟩)               -> int
sa.valid(h: sa_handle⟨T⟩)                -> bool
sa.exists(h: sa_handle⟨T⟩)               -> bool        // generalized is_leaf
sa.get_val(v: sa_view⟨T⟩, j: int)        -> V
sa.get_val_at(h: sa_handle⟨T⟩, j: int)   -> V
```

Plus the cross-cutting infrastructure consumed by the dialect:

```
ParallelFor(W, π, body)        // parallelism dialect
ScanGS(j, n, κ, body)          // grid-stride scan
IterURV(driver, body)          // iterate unique root values
IterSeg(view_count, body)      // multi-view segment loop (e.g., HEAD+FULL)
Phase(κ ∈ {C, M}, body)        // counting / materialize scope
WithDedup(table, body)         // dedup-table scope
Cartesian(vars, sources, body) // cart-join semantic scope
If(p, body), Let(v, e, body)
WriteOutput(rel, vals, sr)
AddCount(expr)                 // count-mode shortcut
IndexDecompose(φ, [d_k], (j_k))
```

## 10. Per-MIR-op lowering rules ($\mathcal{L}_{sa}^{\text{out}}$)

Each rule has the form $\text{LHS}_M \rightsquigarrow \text{RHS}$ with proof and design check.

### 10.1 Scan

**Rule.**

$$
\mathbf{Scan}(\bar{v}, s) :: r \quad\rightsquigarrow\quad
\mathbf{ParallelFor}(W,\ \pi_W,\ \mathbf{ScanGS}(j,\ \text{sa.degree}(\text{sa.root}(\nu_s)),\ \kappa,\
   \mathbf{Bind}(\bar{v} = \text{sa.get\_val}(\nu_s, j))\ ::\ L(r)))
$$

**Proof.** $\mathbf{Scan}$ binds $\bar{v}$ to each row of $s$. $\mathbf{ScanGS}$
enumerates $j \in [0, \deg)$; $\text{sa.get\_val}$ materializes the row.
$\mathbf{ParallelFor}$ over a complete partition is the identity on
$\text{Output}$ multisets (Theorem §8). $\blacksquare$

**Design check.** Splitting $\mathbf{ParallelFor}$ from $\mathbf{ScanGS}$
keeps the parallelism strategy orthogonal — same scan body lowers to
GPU warp-strided or CPU TBB by swapping $\pi_W$ alone.

### 10.2 ColumnJoin (single source, root)

**Rule.** For $\mathbf{CJ}(v, [s])$ at the top of a pipeline:

$$
\mathbf{CJ}(v, [s]) :: r \quad\rightsquigarrow\quad
\mathbf{ParallelFor}(\dots,\ \mathbf{ScanGS}(j,\ \text{sa.degree}(H_s),\ \kappa,\
   \mathbf{Bind}(v = \text{sa.get\_val}(\nu_s, j))\ ::\
   \mathbf{Bind}(c = \text{sa.child}(H_s, j))\ ::\ L(r)))
$$

with $H_s = \text{sa.pref}(\text{sa.root}(\nu_s), \beta(\bar{u}_s))$ if $s$ has prefix vars,
else $H_s = \text{sa.root}(\nu_s)$.

**Proof.** $\mathbf{CJ}$ with one source is a degenerate intersection.
RHS enumerates $\text{vals}(s)$ via $j$ and binds. Nested ops in $r$
navigate from this position via the captured child handle $c$. $\blacksquare$

**Design check.** $\text{sa.child}$ and $\text{sa.pref}$ are both
navigation but **must remain separate**: $\text{sa.child}(H, j)$ is
$O(1)$ slot arithmetic, while reformulating it as
$\text{sa.pref}(H, \text{sa.get\_val\_at}(H, j))$ would force a binary
search per child — a real perf regression.

### 10.3 ColumnJoin (multi-source, root)

**Rule.** For $\mathbf{CJ}(v, [s_1, \ldots, s_n])$ with $n \geq 2$:

$$
\mathbf{CJ}(v, \bar{s}) :: r \quad\rightsquigarrow\quad
\mathbf{ParallelFor}(\dots,\ \mathbf{IterURV}(s_1,\
   \mathbf{Bind}(v = \text{root\_val})\ ::\
   \mathbf{IntersectHandles}(\{H_1, \ldots, H_n\},\ L(r))))
$$

where $H_1$ uses $\text{sa.hint}$ for tight binary search:

$$
H_1 = \text{sa.pref}(\text{sa.hint}(\nu_{s_1},\ \text{hint}_\ell(j_v),\ \text{hint}_h(j_v, |\text{uniq}(s_1)|),\ 0),\ \text{root\_val})
$$

and $H_k = \text{sa.pref}(\text{sa.root}(\nu_{s_k}),\ \text{root\_val})$ for $k > 1$.

**Precondition.** $j_v$ is the index of `root_val` in
`unique_root_values`; the hint range $[\text{hint}_\ell, \text{hint}_h)$ contains
every row $r$ where $\text{row}_r.\text{col}_0 = \text{root\_val}$.

**Proof.** From §6,

$$
\llbracket \mathbf{CJ}(v, \bar{s}) \rrbracket_M = \biguplus_\beta \biguplus_{x \in \bigcap_s \text{vals}(s, \beta)} \{\beta[v \mapsto x]\}
$$

At the root, $\beta = \bot$, so $\text{vals}(s_1, \bot) = \text{uniq}(s_1)$ and

$$
\bigcap_s \text{vals}(s, \bot) = \{x \in \text{uniq}(s_1) \mid \forall k > 1.\ \text{sa.valid}(\text{sa.pref}(\text{sa.root}(\nu_{s_k}), x))\}
$$

This is exactly what $\mathbf{IterURV}(s_1, \cdot)$ enumerates with
$\mathbf{IntersectHandles}$ providing the validity guard. $\text{sa.hint}$
denotes the same handle as $\text{sa.root}$ when the precondition
holds. $\blacksquare$

**Design check.**
- The driver $s_1$ is **not free** — chosen at lowering time, typically
  by smallest cardinality. `IterURV` carries the choice as a node field.
- $\text{sa.hint}$'s correctness is *cross-node*: it depends on the
  surrounding `IterURV` providing a correct unique-values array.
  Either restrict $\text{sa.hint}$ syntactically to that scope, or the
  rewrite that introduces it must verify the invariant.

### 10.4 ColumnJoin (single source, nested)

**Rule.** When the surrounding context has parent handle $H_p$:

$$
\mathbf{CJ}(v, [s]) :: r \quad\rightsquigarrow\quad
\mathbf{ScanGS}(j,\ \text{sa.degree}(H_p),\ \kappa,\
   \mathbf{Bind}(v = \text{sa.get\_val\_at}(H_p, j))\ ::\
   \mathbf{Bind}(c = \text{sa.child}(H_p, j))\ ::\ L(r))
$$

**Proof.** Same as 10.2 with $H_p$ in place of $\mathbf{Root}$. $\blacksquare$

**Design check.** The "parent handle" is a *binding from the lexical
scope*, not a runtime lookup. The current emitter uses string-keyed
handle dicts; the IR binds handles via $\mathbf{Let}$ scopes. This
makes negation pre-narrow (§10.8) a free hoist of `Let`.

### 10.5 CartesianJoin

**Rule.**

$$
\mathbf{Cart}(\bar{v}, \bar{s}) :: r \quad\rightsquigarrow\quad
\mathbf{ParallelFor}(\dots,\ \mathbf{ScanGS}(\phi,\ \prod_k \text{sa.degree}(H_{s_k}),\ \kappa,\
   \mathbf{IndexDecompose}(\phi, [\text{sa.degree}(H_{s_k})], (j_k))\ ::\
   \mathbf{Bind}(\bar{v} = (\text{sa.get\_val\_at}(H_{s_k}, j_k))_k)\ ::\ L(r)))
$$

**Proof.** $\prod_k [0, \deg_k)$ is in bijection with $[0, \prod_k \deg_k)$ via
row-major encoding; $\mathbf{IndexDecompose}$ inverts it. $\blacksquare$

### 10.6 Filter

$\mathbf{Filter}(p) :: r \rightsquigarrow \mathbf{If}(p, L(r))$. Direct.

### 10.7 Bind

$\mathbf{Bind}(v, e) :: r \rightsquigarrow \mathbf{Let}(v, e, L(r))$. Direct.

### 10.8 Negation

**Rule.**

$$
\mathbf{Neg}(\rho, \bar{u}) :: r \quad\rightsquigarrow\quad
\mathbf{Let}(H = \text{sa.pref}(\text{sa.root}(\nu_\rho),\ \bar{u}(\beta)),\
   \mathbf{If}(\neg\, \text{sa.exists}(H),\ L(r)))
$$

**Proof.** $\mathbf{Neg}$ keeps $\beta$ iff no tuple matches the prefix.
$\text{sa.exists}(\text{sa.pref}(\dots))$ is true iff a tuple matches.
$\blacksquare$

**Design check.** $\text{sa.exists}$ is the generalized
existence test; for sorted-array it lowers to `is_leaf`. Other
dialects implement `exists` differently (UF: `same`; bitmap: `test`).
Keeping `exists` as the dialect op rather than `is_leaf` keeps
sorted-array vocabulary out of negation.

### 10.9 Aggregate

$$
\mathbf{Agg}(v, s, \bar{u}, f) :: r \quad\rightsquigarrow\quad
\mathbf{Let}(H = \text{sa.pref}(\text{sa.root}(\nu_s), \bar{u}(\beta)),\
   \mathbf{Let}(v = \mathbf{Fold}(f, \text{sa.values}(H)),\ L(r)))
$$

**Design check.** $\mathbf{Fold}(f, \dots)$ is singular; specialization
for $f \in \{\text{count}, \text{sum}, \text{min}, \text{max}\}$ happens in target
lowering, not as separate IR nodes.

### 10.10 InsertInto

$$
\mathbf{II}(\rho, \bar{v}, \alpha) :: r \quad\rightsquigarrow\quad
\mathbf{WriteOutput}(\rho,\ (\beta(v_k))_k,\ \sigma_\beta)\ ::\ L(r)
$$

**Design check.** Output dedup is a *separate scope*, not a write
attribute. $\mathbf{WithDedup}(\text{table}, b)$ wraps a region; nested
$\mathbf{WriteOutput}$s within use the table.

### 10.11 Pipeline mode

**GPU two-phase.**

$$
p \rightsquigarrow \mathbf{Phase}(\mathbf{C}, L(p))\ ;\ \text{prefix\_scan}\ ;\ \mathbf{Phase}(\mathbf{M}, L(p)\,[\text{output}\mapsto\text{offsets}])
$$

**CPU one-phase (default).**

$$
p \rightsquigarrow \mathbf{Phase}(\mathbf{M}_{\text{tl}}, L(p))\ ;\ \mathbf{Consolidate}
$$

where $\mathbf{M}_{\text{tl}}$ writes to thread-local chunked buffers and
$\mathbf{Consolidate}$ memcpys + parallel-sorts into the destination.

**Design check.** $\mathbf{Phase}$ must be a *scope*, not a per-node
flag, because the same $L(p)$ subtree appears under both
$\mathbf{Phase}(\mathbf{C}, \cdot)$ and $\mathbf{Phase}(\mathbf{M}, \cdot)$ in two-phase emission.

## 11. Internal rewrite rules ($\mathcal{R}_{sa}$)

Each rewrite is $(\text{LHS}, \text{RHS}, \text{Pre})$ with $\llbracket \text{LHS} \rrbracket = \llbracket \text{RHS} \rrbracket$
under $\text{Pre}$.

### R1 — count-as-product

$$
\mathbf{Phase}(\mathbf{C},\ \mathbf{Cart}(\bar{v}, \bar{s},\ [\mathbf{II}])) \rightsquigarrow \mathbf{AddCount}\bigl(\textstyle\prod_s \text{sa.degree}(s)\bigr)
$$

**Pre.** Body is purely $\mathbf{II}$ (no Filter, Negation, Aggregate).
Cardinality is multiplicative over Cartesian product.

### R2 — unused-var elision under counting

$$
\mathbf{Phase}(\mathbf{C},\ \mathbf{Bind}(v = \text{sa.get\_val\_at}(\dots)) :: r) \rightsquigarrow \mathbf{Phase}(\mathbf{C},\ r[v \mapsto \bot])
$$

**Pre.** $v \notin \text{free}(r)$. In counting mode, only cardinality
matters; deterministic extensions of $\beta$ don't change it.

### R3 — hint introduction

$$
\text{sa.pref}(\text{sa.root}(\nu),\ \text{root\_val}) \rightsquigarrow \text{sa.pref}(\text{sa.hint}(\nu, \ell, h, 0),\ \text{root\_val})
$$

**Pre.** Surrounded by $\mathbf{IterURV}(\text{driver}=s_1)$ where $\nu$ is $s_1$'s
view; $\ell, h$ computed from `root_val` index in `unique_root_values`.

### R4 — negation pre-narrow

$$
\mathbf{Cart}(\dots,\ \mathbf{Neg}(\rho, \bar{u}) :: r) \rightsquigarrow \mathbf{Let}(H = \text{sa.pref}(\dots, \bar{u}_{\text{free}}),\ \mathbf{Cart}(\dots,\ \mathbf{Neg}'(\rho, H, \bar{u}_{\text{cart}}) :: r))
$$

**Pre.** $\bar{u}_{\text{free}} \subseteq \text{vars bound before Cart}$;
$\bar{u}_{\text{cart}} \subseteq \text{vars bound by Cart}$;
$\bar{u} = \bar{u}_{\text{free}} \cdot \bar{u}_{\text{cart}}$.
Hoists the partial prefix out of the Cartesian loop.

### R5 — phase-specialization (two-phase)

$$
\mathbf{Phase}(\mathbf{M}, b) \quad =_\uplus \quad \begin{aligned}&\mathbf{Phase}(\mathbf{C}, b);\\ &\text{offsets} = \text{prefix\_scan}(\text{counts}); \\ &\mathbf{Phase}(\mathbf{M}, b\,[\mathbf{WriteOutput} \mapsto \text{write at offset}])\end{aligned}
$$

**Pre.** $b$ is deterministic across re-evaluations (true for pure datalog).

## 12. Lowering to target dialects ($\mathcal{L}_{sa}^{\text{out}}$, target side)

Each $sa$ op has lowerings into each target dialect.

### 12.1 Into target.cuda

| sa op | Emits |
|---|---|
| `sa.root(v)` | `auto h = NodeHandle<...>(v.begin(), v.end(), 0);` |
| `sa.pref(h, k)` | `h = h.prefix(k, tile, view);` (cooperative-group binary search) |
| `sa.child(h, j)` | `h.child(j)` (pointer arithmetic on internal range) |
| `sa.hint(v, lo, hi, d)` | `auto h = NodeHandle<...>(lo, hi, d);` (range-based ctor) |
| `sa.values(h)` | tile-coalesced loop over `[h.begin(), h.end())` |
| `sa.degree(h)` | `h.degree()` |
| `sa.exists(h)` | `h.is_leaf()` |
| `ParallelFor(.., warp_strided, b)` | `for (i = warp_id; i < n; i += num_warps) { b }` |
| `IterSeg(K, b)` | `for (int seg = 0; seg < K; ++seg) { ... b }` |
| `WriteOutput` (under Phase(M)) | atomic-bumped write at offset |
| `WriteOutput` (under Phase(C)) | `output_ctx.add_count(1)` |

### 12.2 Into target.cpp_tbb

| sa op | Emits |
|---|---|
| `sa.root(v)` | `auto h = SortedArrayIndex::NodeHandle{v.impl(), 0, v.size, 0};` |
| `sa.pref(h, k)` | `h = h.prefix(k);` (single-thread binary search) |
| `sa.values(h)` | `for (auto v : h.values()) { ... }` (sequential loop) |
| `ParallelFor(.., tbb_for, b)` | `tbb::parallel_for(blocked_range<...>{0, n}, [&](auto r) { for (i in r) b });` |
| `WriteOutput` (under Phase(M_tl)) | `local_out.push_row(...)` |
| `Consolidate` | parallel memcpy chunks → arena destination + `tbb::parallel_sort` |

### 12.3 Byte-equivalence gate

Before any new feature ships, the new IIR-sorted-array → target.cuda
lowering must produce **byte-equivalent** C++ to the existing emitter
on every test fixture (modulo deterministic name renaming). This is
the no-regression test for the rewrite.

# Part IV: Other data-structure dialects

These dialects validate Property P1 (open registration). Each is
independent of `relation.sorted_array` and of the others.

## 13. relation.lsm⟨K⟩ dialect — proof of pluggability

A $K$-level Log-Structured Merge tree. Levels $L_0, L_1, \ldots, L_{K-1}$ with
geometric size growth. Inserts go to $L_0$; periodic compactions merge
adjacent levels.

### 13.1 Types

$$
\begin{aligned}
\text{lsm}\langle T, K \rangle &\quad \text{state with } K \text{ levels and a compaction policy} \\
\text{lsm\_tier\_view}\langle T \rangle &\quad \text{a single level's storage view} \\
\text{lsm\_handle}\langle T, K \rangle &\quad \text{cursors across all } K \text{ tiers}
\end{aligned}
$$

### 13.2 Operations

```
lsm.empty⟨T, K⟩(policy)                  -> lsm⟨T, K⟩
lsm.tier_view(s: lsm⟨T, K⟩, k: int)      -> lsm_tier_view⟨T⟩
lsm.probe(s: lsm⟨T, K⟩, key: V)          -> lsm_handle⟨T, K⟩
lsm.merged_values(h: lsm_handle⟨T, K⟩)   -> stream⟨V⟩
lsm.exists(h: lsm_handle⟨T, K⟩)          -> bool
lsm.insert_l0(s: lsm⟨T, K⟩, batch: …)    -> lsm⟨T, K⟩
lsm.compact(s: lsm⟨T, K⟩, k_lo, k_hi)    -> lsm⟨T, K⟩
lsm.bloom_test(s: lsm⟨T, K⟩, k, key)     -> bool        // optional
```

### 13.3 Lowering MIR → LSM

```
@lowering(MirColumnJoin)
when sources are LSM-bound:
  ColumnJoin(v, [s_lsm]) :: r ⟼
    Let(h = lsm.probe(s_lsm, prefix_eval),
        ScanForEach(v in lsm.merged_values(h), L(r)))
```

The dispatch is per-source. If only some of $\bar{s}$ are LSM, the
lowering uses LSM ops for those and IIR-sorted-array ops for the
others — mixed-dialect within a single ColumnJoin.

### 13.4 Lowering LSM → target

`lsm.merged_values(h)` lowers differently per target:

- `target.cuda`: emit $K$ separate cursor scans + warp-coalesced merge.
- `target.cpp_tbb`: emit a loser-tree of size $K$ for sequential merge.

Both lowerings denote the same multiset (multiset union over tiers).

### 13.5 Internal rewrites

```
R-LSM-1: lsm.compact(lsm.compact(s, [a, b]), [b, c]) → lsm.compact(s, [a, c])
R-LSM-2: lsm.probe(s, k1) :: lsm.probe(_, k2) is type-illegal — verifier catches.
R-LSM-3: lsm.insert_l0(s, batch) followed by lsm.probe(s, k):
         compaction may have moved data — this is correct because lsm.probe
         is over all tiers. No fence needed.
```

### 13.6 What did NOT change

- `relation.sorted_array` dialect: untouched.
- `target.cuda`, `target.cpp_tbb`: each gained one set of LSM-op lowerings,
  but didn't change for sorted-array.
- MIR: untouched (relations bind to LSM via the symbol table).
- HIR: untouched.
- Compiler driver: gained one `register_dialect(LsmKDialect)`.

This is the demonstration of Property P1.

## 14. relation.uf dialect

Union-find for equivalence-class relations. Recognized via an HIR-level
pre-pass that detects reflexive/symmetric/transitive rule shapes.

### 14.1 Types

$$
\begin{aligned}
\text{uf}\langle T \rangle &\quad \text{equivalence-class state over } T \\
\text{ufrep}\langle T \rangle &\quad \text{a representative element (validated against a specific uf}\langle T \rangle\text{)}
\end{aligned}
$$

### 14.2 Operations

```
uf.empty⟨T⟩()                   -> uf⟨T⟩
uf.add(s: uf⟨T⟩, x: T)          -> uf⟨T⟩       // idempotent on existing x
uf.find(s: uf⟨T⟩, x: T)         -> ufrep⟨T⟩
uf.union(s: uf⟨T⟩, x: T, y: T)  -> uf⟨T⟩
uf.same(s: uf⟨T⟩, x: T, y: T)   -> bool
uf.members(s: uf⟨T⟩, x: T)      -> stream⟨T⟩
```

### 14.3 HIR pre-pass for equivalence detection

Patterns recognized:

```
eq(x, y) :- p(x, y).                    // base
eq(x, y) :- eq(y, x).                   // symmetric
eq(x, z) :- eq(x, y), eq(y, z).         // transitive
```

Combined: the rule set defines an equivalence relation. The pre-pass
rewrites the HIR to:

```
HirEquivalenceRelation(rel=eq, source=p)
```

which the MIR→IR lowering picks up and emits as UF dialect.

### 14.4 Mixed-dialect lowering

For a rule using $\text{eq}$ in a join with another (sorted-array) relation:

```
result(x, z) :- foo(x, y), eq(y, z).

ColumnJoin(z, [foo, eq]) ⟼  // foo is sorted-array, eq is uf
   ScanGS(j, sa.degree(sa.root(ν_foo)), κ,
      Bind(y = sa.get_val(ν_foo, j),
        Bind(rep = uf.find(s_eq, y),
          ScanForEach(z in uf.members(s_eq, rep), body))))
```

The IIR mixes sorted-array ops and UF ops in one flow. Type system
enforces: $\text{uf.find}$ returns $\text{ufrep}$, which cannot be
passed to $\text{sa.pref}$ (wrong type). This catches the
"used a representative as a sortable key" bug at lowering, not runtime.

### 14.5 Negation lowers via uf.same

For a rule with $\neg \text{eq}(x, y)$:

```
Neg(eq, [x, y]) ⟼ If(¬ uf.same(s_eq, β(x), β(y)), body)
```

Note: the $sa$ dialect's $\text{sa.exists}$ is replaced by
$\text{uf.same}$ in this lowering. Both are realizations of "the
existence test for negation"; the dialect chooses how.

## 15. relation.bitmap dialect (sketch)

For dense small-domain relations. Faster than sorted-array when
$|\text{domain}| \leq \text{some threshold}$ (e.g., 64K).

```
bitmap⟨T⟩                              // bitvec state
bitmap.set(b, x), bitmap.test(b, x)
bitmap.intersect(a, b), bitmap.union(a, b)
bitmap.iter_set_bits(b)                // enumerate set positions
```

Lowering MIR → bitmap dialect for relations annotated `dense<small_domain>`:
join becomes `bitmap.intersect`, scan becomes `bitmap.iter_set_bits`.
Targets emit AVX-512 popcount + tzcnt, SVE for ARM, CUDA `__popc`.

# Part V: Cross-cutting dialects

## 16. Parallelism dialects

Each strategy is its own dialect, contributing one $\pi_W$ to
`ParallelFor`.

| Dialect | Partition shape | Use case |
|---|---|---|
| `par.data.warp_strided` | `for (i = warp_id; i < n; i += num_warps)` | GPU baseline |
| `par.data.block_group` | binary-search cumulative work + row-proportional | GPU skew |
| `par.data.atomic_ws` | per-warp atomic claim | GPU dynamic load balancing |
| `par.data.tbb_for` | `tbb::parallel_for(blocked_range, ...)` | CPU baseline |
| `par.task` | TBB task graphs | CPU pipeline-level parallelism |
| `par.simd` | AVX/NEON vectorization in inner loop | inner-loop SIMD |
| `par.scalar` | no parallelism | reference / debugging |

Strategy-specific runtime metadata (e.g., block-group's
`bg_cumulative_work[]`) lives in the strategy's lowering, not as
top-level IR nodes.

## 17. Target dialects

Each target lowers an entire IR program (after data-structure and
parallelism dialects have applied) to a specific C++ flavor.

| Dialect | Emits | Status |
|---|---|---|
| `target.cuda` | `__device__` kernels, cooperative_groups, atomicCAS | existing |
| `target.hip` | HIP-flavored CUDA via shim | partially via vendored shim |
| `target.cpp_tbb` | TBB parallel-for, std::pmr arenas | new (CPU JIT) |
| `target.cpp_omp` | `#pragma omp parallel` | alternative |
| `target.metal` | Apple Metal shading language | future |
| `target.sycl` | SYCL kernels | future |

A target dialect's job: take an IR program, walk it, emit C++ source.
The dialect knows about its target's runtime calling conventions,
memory model, and intrinsics. It does NOT know about source dialects
beyond the lowering rules registered.

## 18. Memory dialect

Used by all data-structure dialects for storage management.

```
mem.arena_create(initial_bytes, hugepage_hint) -> arena
mem.arena_alloc(arena, bytes, align)            -> ptr
mem.arena_reset(arena)                          -> arena
mem.thread_local_arena⟨T⟩()                     -> per-thread arena
mem.hugepage_resource()                         -> memory_resource
```

Lowerings:
- `target.cpp_tbb`: $\text{mem.arena\_create}$ → `pmr::monotonic_buffer_resource` + `madvise(MADV_HUGEPAGE)`
- `target.cuda`: $\text{mem.arena\_create}$ → `cudaMallocAsync` + custom pool

Cross-cutting because every data-structure dialect uses the same memory
abstraction; targets specialize it.

# Part VI: IR machinery

The minimum infrastructure needed to make the dialect framework work.
Total ~600 LOC of Python, dialect-agnostic.

## 19. Dialect registry

```python
class Dialect:
    name: str
    types: list[Type]
    ops: list[OpDef]
    lowerings: list[Lowering]   # to other dialects
    rewrites: list[Rewrite]     # within this dialect
    verifier: Callable[[Op], list[Error]]

class Compiler:
    def register_dialect(self, d: Dialect) -> None: ...
    def lookup_lowering(self, src_op_kind, dst_dialect=None) -> list[Lowering]: ...
    def lookup_rewrites(self, op_kind) -> list[Rewrite]: ...
```

The registry is the single source of truth for what dialects exist. No
central enum or type-switch — just a dict-keyed lookup at lowering time.

## 20. Pattern matching

Python 3.10+ `match` works for the bulk; a small helper provides
multi-node patterns.

```python
@lowering(matches=MirColumnJoin)
def lower_cj(op, ctx):
    match op:
        case MirColumnJoin(var=v, sources=[single]):
            return _lower_single_source_cj(v, single, ctx)
        case MirColumnJoin(var=v, sources=many) if len(many) >= 2:
            return _lower_multi_source_cj(v, many, ctx)
```

Multi-node patterns (e.g., R4 negation pre-narrow) match a *subtree*:

```python
@rewrite
def negation_pre_narrow(prog):
    for cart_node in prog.find_all(MirCart):
        for neg in cart_node.body.iter_of(MirNeg):
            free, cart_bound = split_prefix_vars(neg.prefix_vars, cart_node.bound_vars)
            if free:
                hoist_let(cart_node, neg, free, cart_bound)
```

## 21. Pass driver

```python
class PassDriver:
    def run(self, prog: Program) -> Program:
        # 1. Run rewrites on each dialect to fixpoint
        for dialect in self.compiler.dialects:
            prog = self.run_rewrites_to_fixpoint(prog, dialect)
        # 2. Lower stage by stage (HIR→MIR→data-structure dialects→target)
        for stage in self.lowering_stages:
            prog = self.lower(prog, stage)
        # 3. Verify after each lowering
        return prog
```

The driver doesn't know about specific dialects — it operates on the
registry. New dialects participate in the right lowering stages by
declaring their `lowerings` and `stage` metadata.

## 22. Verification

Per-dialect verifier predicates run after each lowering pass. Catch
type errors, broken invariants, malformed IR.

```python
def verify_sa_dialect(op):
    match op:
        case SaPref(handle=h, key=k):
            assert isinstance(h.type, SaHandle), f"sa.pref expects sa_handle, got {h.type}"
            assert isinstance(k.type, ValueType), f"sa.pref expects value, got {k.type}"
        case SaHint(view=v, lo=l, hi=h, d=d):
            assert is_inside(SaHint, IterURV), "sa.hint must be inside IterURV scope"
        ...
```

The `is_inside` predicate enables Property P3's syntactic restriction
(see §10.3 design check on `sa.hint`'s cross-node correctness).

# Part VII: Implementation roadmap

Execution order is not arbitrary — earlier stages set up invariants
later stages depend on. The riskiest stage is **3** (porting GPU codegen
under byte-equivalence); the rest is mostly mechanical.

## 23. Stage 1 — IR machinery

Build the dialect-agnostic infrastructure. No new compiler features yet.

- §19 registry, §20 pattern matching, §21 pass driver, §22 verifier.
- Total: ~600 LOC Python.
- Test: register a no-op dialect, verify the driver runs cleanly.

**Note on reference interpreters.** An earlier draft proposed a
Python interpreter for $\llbracket \cdot \rrbracket_M$ as a separate stage. It
was dropped: an interpreter validates the *spec*, not the *compiled
code*. Correctness gates that matter all run against compiled
output. The transitive trust chain is enough:

```
existing emitter (battle-tested by 887 tests)
        ↓ byte-equivalence (Stage 2 gate)
new IIR-sorted-array → target.cuda
        ↓ runtime equivalence
new dialects (LSM⟨K⟩, etc.)
```

If a property-test loop later turns out to need faster ground truth
than compile-and-run, build a tightly scoped interpreter then.

## 24. Stage 2 — Port GPU codegen as IIR-sorted-array → target.cuda

The high-stakes step. Re-derive every emission template in the
existing emitter from the IR lowering rules in §10. Validate by
**byte-equivalence** with the current emitter on every fixture
(modulo name renaming).

- IIR-sorted-array dialect: ~800 LOC.
- target.cuda lowering: ~600 LOC.
- Property-based tests for each rewrite rule (R1–R5): ~300 LOC.
- Test gate: every fixture in [tests/](../tests/) must pass with
  byte-equivalent emitted C++.

If any fixture fails byte-equivalence, the IR design is wrong —
revisit the corresponding rule. **Do not move past this gate** with
non-equivalent emission; that's the rewrite-with-regression failure
mode the architecture is supposed to prevent.

## 25. Stage 3 — CPU TBB target

Add `target.cpp_tbb` lowering for the IIR-sorted-array dialect. New
parallelism choices: `par.data.tbb_for` instead of `par.data.warp_strided`.
New mode: $\mathbf{Phase}(\mathbf{M}_{\text{tl}})$ with chunked thread-local output +
$\mathbf{Consolidate}$.

- target.cpp_tbb: ~400 LOC.
- par.data.tbb_for dialect: ~200 LOC.
- Memory dialect (mem.arena, hugepage): ~200 LOC.
- Test gate: same MIR programs used in Stage 2 should now also
  compile to TBB and produce identical outputs to the existing CPU
  TMP executor on all fixtures.

This is the parallel CPU JIT discussed in earlier design
conversations. By this point it's a small new lowering on top of
established infrastructure, not a parallel implementation effort.

## 26. Stage 4 — Add LSM⟨K⟩ dialect

The proof-of-pluggability stage. Add a new data-structure dialect
without touching existing ones.

- relation.lsm dialect (§13): ~1000 LOC.
- Lowerings: lsm → target.cuda (~300 LOC), lsm → target.cpp_tbb (~300 LOC).
- Test gate: a relation in a test fixture is annotated as LSM;
  verify the emitted C++ uses LSM ops; verify program output matches
  the same fixture run with sorted-array (cross-dialect equivalence).

If Stage 4 requires changes to sorted-array, target dialects, or MIR,
the architecture failed Property P1 — revisit before proceeding.

## 27. Stage 5+ — Additional dialects on demand

- relation.uf (§14) when an equivalence-relation use case appears.
- relation.bitmap (§15) for dense-domain relations.
- target.metal, target.sycl, target.hip when targeting new hardware.
- par.simd for vectorized inner loops.

Each is independent. None blocks others.

## 28. Risk gates summary

| Gate | What it proves | How |
|---|---|---|
| Stage 1 done | Dialect machinery is the right size | Registry test passes |
| Stage 2 done | Rewrite preserves all GPU optimizations | Byte-equivalence with existing emitter on all fixtures |
| Stage 3 done | CPU JIT works | Runtime equivalence with existing TMP executor on all fixtures |
| Stage 4 done | Architecture admits new dialects | LSM⟨K⟩ added with no edits to existing dialects + cross-dialect output equivalence |

The Stage 2 gate is non-negotiable. If it fails, the design is wrong,
not the implementation.

# Appendix A: Open questions

A1. **Per-relation dialect choice.** Currently the choice is made
   before lowering, encoded in the symbol table. Should the choice
   be revisable by an IR pass (e.g., based on profiling)? Probably
   yes; the symbol table can be mutated by an IR pass.

A2. **Provenance semirings and rewrite ordering.** Some semirings
   (e.g., formal-power-series provenance) may have weaker
   commutativity properties. Some rewrites assume $\uplus$
   commutativity; document the precondition per rule.

A3. **Cross-dialect rewrites.** Are there optimizations that span
   dialect boundaries? E.g., "the result of $\text{uf.find}$ feeds into
   $\text{sa.pref}$" might be optimized into a fused op. Probably yes,
   but defer to a later stage; cross-dialect rewrites complicate the
   pass dependency story.

A4. **Optimization passes already in production.** The current emitter
   has feature flags (`fan_out_explore`, `tiled_cartesian_enabled`,
   `bg_histogram_mode`). Each must map to either an IR node, a
   rewrite, or a strategy parameter. Audit before Stage 2.

# Appendix B: Glossary

- **Dialect** — a coherent set of types, ops, lowerings, rewrites,
  and a verifier; registered with the compiler at init time.
- **Phase / Position** — a stage in the pipeline (HIR, MIR, IR
  position, target). Multiple dialects can occupy the same position.
- **Lowering** — a transformation from one dialect's ops to another's.
- **Rewrite** — a transformation within a single dialect.
- **Verifier** — a per-dialect predicate that catches malformed IR.
- **Property P1/P2/P3** — pluggability invariants from §4.
- **Byte-equivalence gate** — Stage 2's no-regression test; emitted C++
  must match existing emitter output on all fixtures.
