"""Exact field-content rewrite of the frozen 07f0f19 DOOP query (upstream
Doop's HeapInterproceduralAssign formulation).

Replaced rules:
  IFPT_Store:    IFPT(heap,fld,baseheap) <- Reachable(m), Store(frm,base,fld,m), VPT(heap,frm), VPT(baseheap,base)
  VPT_LoadField: VPT(heap,to) <- RLIF(base,fld,to), VPT(baseheap,base), IFPT(heap,fld,baseheap)
With:
  ReachableStoreInstanceField(base,fld,frm) <- Reachable(m), Store(frm,base,fld,m)          (non-recursive)
  StoreObjFrom(baseheap,fld,frm)            <- RSIF(base,fld,frm), VPT(baseheap,base)
  FieldFlow(frm,to)                          <- RLIF(base,fld,to), VPT(baseheap,base), StoreObjFrom(baseheap,fld,frm)
  VPT_FieldFlow: VPT(heap,to)                <- FieldFlow(frm,to), VPT(heap,frm)
  IFPT_Output:   IFPT(heap,fld,baseheap)     <- StoreObjFrom(baseheap,fld,frm), VPT(heap,frm)   (after the VPT SCC)

Equivalence: VPT(heap,to) from a field load holds iff some (base,fld,to) load and
(frm,base',fld) store share a heap baseheap in pts(base) and pts(base'), and heap in
pts(frm) -- exactly the original IFPT-mediated join. IFPT itself is unchanged as a set.
FieldFlow uses Assign's (frm,to) column order so VPT_FieldFlow has VPT_Assign's shape;
the Jython bitmap plan is applied to it as to VPT_Assign.
"""
from doop_suite.gpu import _program as baseline_program
from srdatalog import Program, Relation, Var
import doop

REMOVED = ('VPT_LoadField', 'IFPT_Store')


def program(meta, plan):
  baseline = baseline_program(meta, plan)
  names = [r.name for r in baseline.rules]
  assert all(names.count(n) == 1 for n in REMOVED), names
  rsif = Relation('ReachableStoreInstanceField', 3, column_types=(int, int, int), print_size=True)
  sof = Relation('StoreObjFrom', 3, column_types=(int, int, int), print_size=True,
                 index_type='SRDatalog::GPU::Device2LevelIndex')
  flow = Relation('FieldFlow', 2, column_types=(int, int), print_size=True)
  base, fld, frm, to = Var('base'), Var('fld'), Var('frm'), Var('to')
  heap, baseheap, inmeth = Var('heap'), Var('baseheap'), Var('inmeth')
  VPT, IFPT = doop.VarPointsTo, doop.InstanceFieldPointsTo
  propagate = (VPT(heap, to) <= flow(frm, to) & VPT(heap, frm)).named('VPT_FieldFlow')
  if plan == 'bitmap':
    propagate = propagate.with_plan(delta=0, dedup_bitmap=True).with_plan(delta=1, dedup_bitmap=True)
  rules = [r for r in baseline.rules if r.name not in REMOVED] + [
    (rsif(base, fld, frm) <= doop.Reachable(inmeth) & doop.StoreInstanceField(frm, base, fld, inmeth))
    .named('ReachableStoreInstanceField_rule'),
    (sof(baseheap, fld, frm) <= rsif(base, fld, frm) & VPT(baseheap, base)).named('StoreObjFrom_rule'),
    (flow(frm, to) <= doop.ReachableLoadInstanceField(base, fld, to) & VPT(baseheap, base)
     & sof(baseheap, fld, frm))
    .named('FieldFlow_rule')
    .with_plan(delta=2, var_order=['baseheap', 'fld', 'base', 'frm', 'to']),
    propagate,
    (IFPT(heap, fld, baseheap) <= sof(baseheap, fld, frm) & VPT(heap, frm)).named('IFPT_Output'),
  ]
  return Program(rules=rules)
