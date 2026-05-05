---
orphan: true
---

# Stage 4 — IIR semantic vocabulary

Operational companion to [`milestones.md` § Stage 4](./milestones.md).
The milestones doc has the high-level task index; this doc is the
binding inventory + design.

## 0. Why this stage exists

Stage 3A's framework realization (the renderer registry, decorators,
dispatcher, verifier wiring) is **structural** — it gives every dialect
a registered surface. But the IIR vocabulary that flows through it is
**incomplete**: `sorted_array/lowerings.py` produces 46
`RawString(text="<C++>")` sites in addition to the structured ops. The
codegen's renderer for `RawString` is a passthrough. So for the
majority of code paths, the elaborate per-op dispatch is just doing
string-concatenation with extra steps.

The renderer registry only delivers value for ops with structured
representations. Until RawString is replaced with proper IIR ops, the
codegen has no semantic information to translate per-target — adding
a CPU/WASM target would render every RawString as identical C++.

Stage 4 closes this gap.

## 1. Inventory: 46 RawString sites in `sorted_array/lowerings.py`

Categorized by what the embedded text represents semantically.
Line numbers refer to `src/srdatalog/ir/dialects/relation/sorted_array/lowerings.py`
as of HEAD at Stage 4 start.

### Category A — Bare identifier (1 site)

Just a variable name.

| Line | Text | Replacement |
|---|---|---|
| 824 | `'num_unique_root_keys'` | `VarRef(name='num_unique_root_keys')` |

**Replacement op:** `VarRef` — already exists in `iir.cf`.

### Category B — Member-method call (4 sites; 2 distinct)

Object method invocation, no args.

| Line | Text | Pattern |
|---|---|---|
| 1171 | `'{ctx.tile_var}.thread_rank()'` | `<obj>.<method>()` |
| 1178 | `'{ctx.tile_var}.size()'` | `<obj>.<method>()` |
| 1674 | `'{ctx.tile_var}.thread_rank()'` (dup) | |
| 1681 | `'{ctx.tile_var}.size()'` (dup) | |

**Replacement op:** `MemberCall(obj: Op, method: str, args: tuple[Op, ...])` (in new `iir.expr` sub-dialect or extend `iir.cf`). Renders to `obj.method(args)`.

### Category C — Arithmetic expression (5 sites; 3 distinct)

Variadic multiplication, division, modulo.

| Line | Text | Pattern |
|---|---|---|
| 504 | `' * '.join(degree_var_names)` | n-ary mul |
| 536 | `'{flat_idx_var} / {degree_var_names[1]}'` | binary div |
| 543 | `'{flat_idx_var} % {degree_var_names[1]}'` | binary mod |
| 1216 | `' * '.join(degree_var_names)` (dup) | n-ary mul |
| 1737 | `' * '.join(degree_var_names)` (dup) | n-ary mul |

**Replacement op:** `BinOp(op: str, lhs: Op, rhs: Op)` for binary; `NaryOp(op: str, args: tuple[Op, ...])` for n-ary, OR n-ary as fold of binary.

**Open decision (S4.2):** generic `BinOp(op_str, lhs, rhs)` vs per-operator (`Mul`, `Div`, `Mod`). Generic = fewer ops, less type info. Per-operator = sharper render dispatch, more ceremony.

Recommendation: **generic** — `BinOp("*", lhs, rhs)`. Simpler. Per-operator can be added later if a target needs different rendering (`__umul64hi` for CUDA mul, etc.).

### Category D — Array index (1 site)

Subscript expression.

| Line | Text | Pattern |
|---|---|---|
| 685 | `'root_unique_values[{y_idx_var}]'` | `<arr>[<idx>]` |

**Replacement op:** `IndexExpr(arr: Op, idx: Op)`. Renders to `arr[idx]`.

### Category E — Compound arithmetic (1 site)

Multiple operations combined.

| Line | Text |
|---|---|
| 724 | `'{src_view}.num_rows_ - (num_unique_root_keys - {y_idx_var} - 1)'` |

**Replacement:** decompose into `BinOp("-", MemberAccess(src_view, "num_rows_"), BinOp("-", BinOp("-", VarRef("num_unique_root_keys"), VarRef(y_idx_var)), IntLit(1)))`. Needs `MemberAccess(obj: Op, member: str)` and `IntLit(n: int)` in addition to `BinOp`.

### Category F — Ternary expression (2 sites)

Conditional expression in assignment context.

| Line | Text |
|---|---|
| 729 | `'{hint_hi} = ({hint_hi} <= {src_view}.num_rows_) ? {hint_hi} : {src_view}.num_rows_;'` |
| 735 | `'{hint_hi} = ({hint_hi} > {hint_lo}) ? {hint_hi} : {src_view}.num_rows_;'` |

**Replacement op:** `Ternary(cond: Op, then_: Op, else_: Op)` — the conditional expression. Combined with `Assign(target: str, value: Op)` for the statement form. Renders to `<target> = <cond> ? <then> : <else>;`.

### Category G — If-return / if-continue with NON-inverted condition (6 sites; 2 distinct patterns)

| Line | Text | Pattern |
|---|---|---|
| 488 | `'if ({validity_parts}) return;'` | `if (cond) return;` |
| 508 | `'if ({total_var} == 0) return;'` | `if (cond) return;` |
| 1201 | `'if ({validity_parts}) continue;'` | `if (cond) continue;` |
| 1220 | `'if ({total_var} == 0) continue;'` | `if (cond) continue;` |
| 1722 | (dup of 1201) | |
| 1741 | (dup of 1220) | |

**Existing `IfReturnIfNot(cond)` and `IfContinueIfNot(cond)` are inverted** (`if (!cond) return;`). These six sites are non-inverted.

**Replacement options:**
1. Add `IfReturn(cond)` / `IfContinue(cond)` ops (mirror of existing).
2. Keep `IfReturnIfNot` / `IfContinueIfNot` and invert the condition: `IfReturnIfNot(BoolNot(cond))`. Requires new `BoolNot(expr)` op.

Recommendation: **Option 1** — add `IfReturn` / `IfContinue`. Cleaner; no double-negation in the IIR.

### Category H — Boolean fold / accumulating AND (3 sites)

Reduction pattern: `acc = acc && cond`.

| Line | Text |
|---|---|
| 1009 | `'{fold_var} = {fold_var} && ({cond_expr});'` |
| 1372 | `'{fold_var} = {fold_var} && (!{check_var}.valid());'` |
| 1434 | `'{fold_var} = {fold_var} && (!{neg_handle_var}.valid());'` |

**Replacement op:** could decompose to `Assign(target, BinOp("&&", VarRef(target), cond))`. Or introduce `AndAssign(target: str, expr: Op)` for the compound form. Decomposition is more general; AndAssign reads cleaner at the IR level.

Recommendation: **decompose to Assign + BinOp**. Avoids multiplying ops.

### Category I — Validity-check expression as condition (3 sites)

Used inside `If(cond=..., body=...)`.

| Line | Text |
|---|---|
| 1013 | `cond=RawString(text=cond_expr)` (cond_expr is parameterized) |
| 1377 | `cond=RawString(text=f'!{check_var}.valid()')` |
| 1439 | `cond=RawString(text=f'!{neg_handle_var}.valid()')` |

**Replacement:** structured `BoolNot(MemberCall(check_var, "valid"))`. Needs `BoolNot(expr)` op.

### Category J — Bind expression with user-supplied code (1 site, partial)

| Line | Text |
|---|---|
| 1017 | `Bind(name=var, expr=RawString(text=head.code))` — `head.code` is user-supplied filter code |

**Status:** stays as RawString (user-injected code; the IIR has no semantic info about it). This one is **legitimate** — Filter's `.code` field is opaque user code. May want to wrap as `UserCode(text)` to distinguish from internal RawStrings.

### Category K — Tile-dispatch expression (2 sites)

| Line | Text |
|---|---|
| 1187 | (multi-line, complex) |
| 1693 | similar |

Need deeper read to fully classify; likely combinations of arithmetic + ternary.

### Category L — Multi-line emission blocks (~20 sites)

The hardest tier. These are where the lowering "gives up" on structured emission and just appends raw C++ chunks.

Major clusters:

**L1 — Cap/short-circuit cluster** (lines 1799-1827):
- `'{'` (bare brace)
- `'(uint64_t){total_expr}'` (cast)
- `'static_cast<uint32_t>(cap_total)'` (static cast)
- `'}'` (close brace)
- A multi-arg complex expression for cap calculation

**L2 — Dedup-table cluster** (lines 2136-2186):
- `'{ bool _p = dedup_table.try_insert(thread_id, {args_str});'` — opens block with declaration
- `'  if (_p) {'` — partial if
- `'{out_var}++;'`, `'{out_var}.emit_direct();'` — increment / call
- `'if ({ctx.tile_var}.thread_rank() == 0) {{'` — partial guard
- `'{'` — bare brace
- `'  uint32_t pos = atomicAdd(atomic_write_pos, 1u);'` — declaration
- `'  out_data_0[(pos + out_base_0) + {col} * out_stride_0] = {name};'` — assignment
- `'}'`, `'} }'` — close braces
- `'{out_var}.emit_direct({sanitized});'` — call
- `'{out_var}.emit_warp_coalesced(...)'` — multi-arg call
- `'{out_var}++;'` (count phase)

These need their own structured ops:
- `Assign(target, value)` — bare assignment
- `Decl(type, name, init)` — variable declaration with initializer
- `MemberCall` (already proposed)
- `Cast(type, expr)` — `static_cast<T>(expr)`
- `Block` (already exists in iir.cf)

The "bare brace" RawStrings (`'{'`, `'}'`, `'} }'`) are scope-opening/closing — these should disappear when the surrounding structured ops produce proper Block scopes.

### Category M — Handle-alias comments (multiple)

| Line | Text |
|---|---|
| (1187, 1693, etc.) | `f'auto {handle_var_names[i]} = {alias_targets[i]};  // reusing narrowed handle'` |

These are inline declarations with comments. Replacement: `Decl(type='auto', name=..., init=...)` with optional comment field. Or `Bind` (already exists) — but Bind takes `expr: Op`, and the alias targets are textual (`alias_targets[i]` is a string). Need to first lift the alias-target generation into structured ops (transitively pushes the work).

## 2. Implementation order

Mirrors `milestones.md` § Stage 4 task index, with the inventory's
categorization fleshing out each task.

### S4.1 — bare identifiers (~1 site)

Trivial. Convert Category A's single site to `VarRef`.

**Risk:** zero. **Test gate:** byte-equivalence.

### S4.2 — arithmetic + comparisons (Categories C, E, H)

Define `BinOp(op: str, lhs: Op, rhs: Op)` in new `dialects/iir/expr.py` (or extend `iir/cf`). Add CUDA renderer that emits `<lhs> <op> <rhs>` (with parens as needed).

Replace ~5-8 sites. Decompose Category H's compound assignments to `Assign(target, BinOp("&&", VarRef(target), expr))`.

**Open question:** add `IntLit(n: int)` op too? Some BinOp call sites use literal `0`, `1`. Could use `RawString("0")` (a regression) or inline as `int` field on a sub-op (less general). Cleanest: `IntLit`.

### S4.3 — IndexExpr + MemberAccess (Categories D, K)

Define `IndexExpr(arr: Op, idx: Op)` and `MemberAccess(obj: Op, member: str)`. Renders `arr[idx]` and `obj.member`.

Replace ~6 sites including Category E's compound `src_view.num_rows_ - ...`.

### S4.4 — IfReturn / IfContinue (Category G)

Add `IfReturn(cond: Op)` and `IfContinue(cond: Op)` ops in `iir.cf`.
Replace 6 sites.

### S4.5 — Ternary + BoolNot (Categories F, I)

Add `Ternary(cond, then_, else_)` and `BoolNot(expr)` ops.
Replace ~5 sites.

### S4.6 — Multi-line emission blocks (Category L)

Hardest tier. Per-cluster analysis:

- **L1 cap/short-circuit**: factor into `Block + Decl + Cast + Assign`.
  ~5 sites cleared.
- **L2 dedup-table**: factor into `Block + Decl + MemberCall + IfBlock`.
  Likely needs a `DedupTryInsert(table_var, thread_var, args)` higher-level op
  that bundles the full pattern. ~10 sites cleared.

### S4.7 — Discipline test pinning RawString count

Add `tests/test_iir_no_raw_string.py` asserting that `lowerings.py`
contains at most N `RawString` instances (where N = the count after
S4.6 lands). New uses caught in CI.

Acceptable residual: legitimate user-code wrapping (Category J,
Filter's `head.code`). May want a separate `UserCode(text)` op to
distinguish.

### S4.8 — R1–R5 rewrites as Rewrite instances

Now-tractable. Each rewrite from `ir_lowering_semantics.md` §11
operates on structured ops. Wrap as `Rewrite` instances on
`sorted_array.DIALECT.rewrites` via the `@rewrite` decorator.

Was deferred from Stage 3A's S3A.5; the deferral was justified
because R1–R5 transform IR that included RawString — couldn't
cleanly rewrite text. After S4.6, the IR is structured.

### S4.9 — Op-level dispatch in PassDriver

Now there's a real consumer (the structured ops with semantically
meaningful renderers). Implement tree-walking dispatch in
`PassDriver.run` using `core/strategy.py` combinators. Production
code can opt into framework-driven compilation.

## 3. Open design decisions

These benefit from explicit resolution before code starts:

1. **Generic `BinOp` vs per-operator ops.** Recommendation: generic.
   See Category C.
2. **New `iir.expr` sub-dialect vs extend `iir.cf`.** New dialect is
   cleaner separation (control flow vs expressions); extending iir.cf
   keeps everything in one place. Recommendation: **new `iir.expr`** —
   matches the natural distinction in `emit()` vs `emit_expr()` (which
   already exists in the renderer).
3. **`IntLit`, `StrLit`, `BoolLit` as sub-ops vs literal fields.**
   Some call sites would benefit from typed literals. Recommendation:
   add `IntLit(value: int)` and `BoolLit(value: bool)` for clarity;
   defer `StrLit` until a use case appears.
4. **Compound assignment ops (`AndAssign`, `OrAssign`, etc.) vs decompose.**
   Recommendation: decompose. Avoids op proliferation.
5. **`MemberCall(obj, method, args)` vs `MemberAccess + Call` separately.**
   Recommendation: **bundle as `MemberCall`** — `obj.method(args)` is
   one syntactic form in C++; splitting it adds a Call op for marginal
   benefit.

## 4. What this plan deliberately defers

- **Per-Codegen plugin registry (S3A.8 A6 half).** Stays as a separate
  follow-up beyond Stage 3A.
- **Stage 3B** (HIR/MIR onto Op/Type subclasses). Re-evaluate after
  Stage 4 lands; the structural ops Stage 4 introduces may be reusable
  for HIR/MIR.
- **iir.cf `GridStrideLoop` / `ParallelFor` impurity** (CUDA-flavored
  ops in cross-target IIR). Separate decision; tied to whether a
  second target lands.
- **Filter's `head.code` (Category J)** — user-supplied code; stays
  as RawString or wraps as `UserCode(text)`.
