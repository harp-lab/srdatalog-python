'''srdatalog.ir — the IR pyramid.

  ir.core       — generic dialect machinery (Op, Dialect, strategies,
                  pattern matchers, verifier).
  ir.hir        — high-level IR (Program, Rule, Atom). The user-facing
                  datalog representation that the front-end produces.
  ir.mir        — mid-level IR (Pipeline, ColumnJoin, Scan, CartesianJoin,
                  InsertInto). The join-shape representation that
                  HIR lowering produces.
  ir.dialects   — emission-level IR. Each sub-package is a dialect:
                    iir.cf                  control-flow scaffold
                    parallel.data.block_group  block-group strategy
                    relation.d2l            2-level index
                    relation.sorted_array   single-array index
                    target.cuda             CUDA emission target

The compile pipeline flows top-to-bottom:

  HIR (Program) -> MIR (Pipeline) -> IIR (Block, Bind, ...) -> C++ source

via `srdatalog.compile.compile_pipeline` / `compile_runner`.
'''
