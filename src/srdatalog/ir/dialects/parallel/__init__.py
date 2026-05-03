'''Parallelism-strategy dialects.

Each strategy lowers a `ParallelFor` body to a target-specific work-
distribution shape:

  par.data.warp_strided   GPU baseline grid-stride-by-warp loop
  par.data.block_group    GPU work-balanced — per-block cumulative-work
                          binary search, intra-block warp redistribution
  par.data.atomic_ws      GPU work-stealing — WCOJTask queue + range board
  par.data.tbb_for        CPU TBB parallel_for with blocked_range
  par.scalar              No parallelism (debug / reference)

See docs/ir_lowering_semantics.md §16 for the full taxonomy. The
strategies are independent dialects; each contributes one $\\pi_W$
to `iir.cf.ParallelFor` via lowering registration.
'''
