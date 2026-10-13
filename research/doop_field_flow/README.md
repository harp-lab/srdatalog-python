# Exact field-flow formulation for DOOP field loads

`examples/doop.py` `build_doopdb_program(meta, field_flow=True)` replaces the
recursive object-field container (`IFPT_Store` → `VPT_LoadField`) with upstream
Doop's `HeapInterproceduralAssign` formulation: `StoreObjFrom`, `FieldFlow`,
`VPT_FieldFlow`, and `InstanceFieldPointsTo` as a non-recursive output after the
VarPointsTo SCC. Default (`field_flow=False`) is unchanged.

The repository builder produces a CPU logical query byte-identical to the
measured `harness/candidate.py` (checked with `doop_suite.cpu.translate_program`
on Soot metadata).

## Why: skew is mostly duplicated work

`skew_analysis/` (DuckDB over corrected final FULL relations in
`/data/doop-corrected-xlarge`; final-fixpoint structure, not per-launch DELTA work;
fair share = total / 9,088 warps of the profiled 1136×256 launch):

| VPT_LoadField | Soot | Jython |
|---|---:|---:|
| Largest root (`baseheap`) unit / fair share | 14.6× | 4.5× |
| Largest `(baseheap, sig, base)` unit / fair share | 0.36× | 0.56× |
| Witnesses repeating an identical (field-content set, load-destination set) | 99.2% | 93.7% |
| Witnesses: original → field-flow | 3,048M → 32.9M | 1,085M → 24.5M |

Virtual dispatch on Soot: root (`base`) unit 60.8× fair share; `(base, invocation)`
2.4×; `(base, invocation, heap type)` 1.3×. Untouched by this rewrite.

## Measurement

Frozen source `07f0f19` (`/data/doop-next-exp/hotspots/source`), same prepared
inputs and baseline libraries as the post-split hotspot study. RTX 6000 Ada,
`cuda_async`; one excluded warmup plus three measured fresh processes per
variant, alternating order; fixedpoint medians, seconds. Jython uses the bitmap
plan on `VPT_Assign` and on `VPT_FieldFlow`.

| Dataset | Baseline | Field flow | Change |
|---|---:|---:|---:|
| Soot | 11.968 | 9.551 | −20.2% |
| Jython | 15.091 | 13.498 | −10.6% |
| Scala | 9.236 | 9.399 | +1.8% |
| Kotlin | 10.194 | 10.433 | +2.3% |

Correctness (`correctness/*.json`): all 37 original IDBs, including
`InstanceFieldPointsTo`, equal the corrected CPU oracle as complete sorted tuple
sets; `ReachableStoreInstanceField`, `StoreObjFrom` and `FieldFlow` equal their
definitions over the oracle's VarPointsTo (SQL EXCEPT both directions).

Profiles (`profiles/`, Nsight Systems; kernel sums overlap and are not elapsed
time): Soot GPU kernel union 10.36 → 7.91 s. On Scala/Kotlin the removed field
rules (≈3.4/3.0 s kernel sum) ran off the critical path; union rose slightly
(7.60 → 7.67 s, 8.91 → 9.12 s), array-store rules slowed, `FieldFlow_rule`
(1.36 s Scala) is the new field hotspot, and the post-fixpoint `IFPT_Output`
launch costs 0.22–0.43 s.

## Reproduction

`harness/` is the run directory copy (`/data/doop-field-flow`); paths are absolute
to `/data`. `driver.py build <ds> field-flow`, then `paired.py <ds>...`;
`profile-baselines.py field-flow <ds>...`. Compare requires the corrected CPU
oracle under `/data/doop-corrected-xlarge/cpu`.

## Open

- Plan/partition `FieldFlow_rule` (skewed at `(object, field)`; 384× fair share on Jython).
- Report with and without `InstanceFieldPointsTo` output materialization.
- 21-application campaign; a Soufflé baseline using the same formulation for a
  same-query CPU comparison.
