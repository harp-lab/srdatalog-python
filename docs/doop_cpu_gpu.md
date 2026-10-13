# DOOP CPU/GPU Speedup Summary

## Measurement scope

These results compare native artifact-query CPU configurations with the exact tuned SRDatalog GPU query (`07f0f19`). CPU configurations use 12 threads/workers. Hardware: AMD Threadripper PRO 5945WX (12 physical cores, 24 logical CPUs) and NVIDIA RTX 6000 Ada (48 GiB class).

Each timing is the median of three fresh processes after one excluded warmup, with CPU-idle gating and interference checks. Twelve configured workers does not imply twelve fully utilized physical cores; workers were not explicitly pinned one per physical core.

**Ratio = CPU median process time / GPU median process time**, calculated from unrounded measurements. Process times include engine initialization, loading, execution, and teardown. Compilation, input preparation, and output-file serialization are excluded.

Native CPU processes read the artifact's full 34-relation numeric schema; GPU processes read 37 projected relations. **These are engine-only ratios, not common-starting-point pipeline end-to-end speedups.**

## Per-workload ratios

| Workload | GPU vs. Souffle 12T | GPU vs. FlowLog 12 workers |
|---|---:|---:|
| Xalan | 3.24x | 1.54x |
| ZXing | 4.24x | 2.12x |
| BioJava | 3.66x | 1.77x |
| PMD | 4.43x | 3.11x [1] |
| Bloat | 4.48x | 3.62x |
| Sunflow | 6.34x | 4.29x |
| Clojure | 4.80x | 4.37x |
| Chart | 6.65x | 5.61x |
| Groovy | 6.40x | 5.15x |
| Javac | 6.71x | 4.81x [1] |
| Spring | 7.00x | 5.18x [1] |
| Batik | 8.99x | 7.55x |
| Eclipse | 8.37x | 9.09x |
| H2 | 9.62x | 8.22x |
| FOP | 9.30x | 8.06x |
| JRuby | 10.11x | 7.92x |
| PDFBox | 14.83x | 14.20x |
| Soot [2] | 22.05x | 15.71x |
| Jython [2] | 29.87x | 50.10x |
| Scala [2] | 24.02x | 24.16x |
| Kotlin [2] | 31.22x | Memory limit; no ratio |

[1] Native FlowLog has extra `MethodLookup` tuples on PMD/Javac/Spring (+6/+2/+2, respectively). The other 11 shared output relations match exactly. These entries are descriptive timing ratios, not full-equivalence speedup claims.

[2] Available native counts match, but no new complete native tuple comparison was performed for these four xlarge workloads. Kotlin FlowLog did not finish: it reached the configured 96 GiB host-memory cap during warmup. It has no median and no numeric speedup. The previously established GPU exact-result and query/input identity proofs remain retained. Xlarge ratios are descriptive, not new full-tuple-equivalence claims.

On the strictly validated subsets, geometric-mean engine-process ratios are **6.47x against Souffle (17 workloads)** and **4.98x against FlowLog (14 workloads)**. These exclude the native counts-only xlarge cases; the FlowLog subset also excludes the three mismatches.

## Does larger mean better for the GPU?

The observed advantage increases across the historical analysis-result tiers:

| Analysis-result tier | Workloads | Geometric-mean Souffle/GPU process-time ratio |
|---|---:|---:|
| Small | 6 | 4.30x |
| Medium | 5 | 6.26x |
| Large | 6 | 10.02x |
| Xlarge [2] | 4 | 26.51x |

**These tiers describe derived `VarPointsTo` cardinality, not input file size.** The xlarge aggregate is descriptive and carries the validation qualification above.

Raw input size alone does not explain the results:

- BioJava: **8.10 million input facts**, **3.66x** ratio against Souffle.
- Soot: fewer inputs—**7.58 million facts**—but a **22.05x** measured ratio.

Derived output volume, recursive work, fan-out, skew, and execution plans matter substantially. These are different applications, not a controlled experiment scaling one application's input.

**[INFERENCE]** Heavier fixedpoint workloads can better amortize GPU initialization and transfer costs and expose more parallel work. The observations are consistent with that explanation, but do not isolate its causal contribution or prove that increasing input size always increases speedup.

## Memory and end-to-end limits

All 21 GPU cases completed on one 48 GiB-class GPU. Maximum observed sampled process-resident VRAM was **41.78 GiB**; maximum CUDA-pool allocated high-water was **40.66 GiB**. GPU engine-process host peak RSS was at most **0.64 GiB**. Resident VRAM was sampled every 20 ms and is a lower bound; pool and resident measurements overlap and must not be added. This does not establish that GPU memory use is always lower than CPU memory use.

A separately validated original-artifact Xalan case—not the current-corpus Xalan row—illustrates the end-to-end limitation:

| Configuration | Engine-process time |
|---|---:|
| Native FlowLog | 2.449 s |
| Native Souffle | 5.240 s |
| GPU on projected inputs | 1.460 s |
| Fresh projection plus GPU execution, measured as one pipeline | 15.767 s |

All twelve shared output sets match in that anchor case. Nevertheless, the current Python projection makes the one-shot pipeline slower than native CPU execution. Compilation is also excluded from these measurements. Do not present the engine-only ratios as universal one-shot end-to-end acceleration.

## Suggested grant wording

> Preliminary measurements show that the GPU advantage increases substantially on workloads with larger derived analysis state, suggesting that GPU acceleration is particularly promising for computation-intensive program analyses. On full-tuple-verified subsets, CPU/GPU engine-process ratios have geometric means of 6.47 against 12-thread Souffle and 4.98 against 12-worker FlowLog. All GPU workloads complete within a single 48 GiB-class device. These results motivate research in representation-aware query rewriting and GPU execution; preprocessing, compilation, and device-memory capacity remain important end-to-end constraints. Native-output discrepancies and validation coverage are disclosed explicitly.

## Evidence and reusable tables

- [Complete timing and memory measurements (CSV)](benchmark_data/doop_cpu_gpu.csv)
- [LaTeX timing and speedup table](benchmark_data/doop_timings.tex)
- [LaTeX memory table](benchmark_data/doop_memory.tex)

The LaTeX fragments require `booktabs` and `graphicx`. Include both to resolve
cross-references. The CSV preserves unrounded numbers, measured failures, and
validation status; it contains no workstation-specific file paths.

Measured GPU revision: `07f0f19`. Native FlowLog engine:
`5acbc01749e8225cbde766ff047a9ded3b66cc0d`; Rust 1.89, batch Present.
Souffle: 2.4, 64-bit, OpenMP. Artifact revision:
`abf3e47f8e6da8fa641da3a67deeca1c73701cda`; reproduction revision:
`66c88aff9d72fdee6201af3e418c6e484d03b856`.

Reference artifact: <https://github.com/flowlog-rs/vldb26-artifact>.

Detailed commands, input/binary hashes, comparison proofs, and raw logs remain
in the external experiment archive `/data/doop-next-exp/final-report/` and its
referenced directories. They are not included in Git. The tables above are a
snapshot of those measurements, not a claim that they were rerun on later commits.
