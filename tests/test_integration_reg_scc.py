'''reg_scc.nim -- 10 rules, 13 wildcards, plan arrays'''
from integration_helpers import diff_hir, diff_mir
from srdatalog.dsl import Var, Relation, Program, PlanEntry


def build_reg_scc() -> Program:
  EA_DEF = Var("eaDef"); EA_USED = Var("eaUsed"); MVAR = Var("mvar")
  INDEX = Var("index"); BLK = Var("blk"); BLOCK_USED = Var("blockUsed")
  VAR_ID = Var("varIdentity")
  NXT_BLK = Var("nxtBlk"); NEXT_EA = Var("nextEaUsed"); NEXT_IDX = Var("nextIndex")
  CALLEE = Var("callee"); REG = Var("reg"); BLOCK_END = Var("blockEnd")
  EA_CALL = Var("eaCall"); INDEX_USED = Var("indexUsed")
  PREV_BLK = Var("prevBlock"); PREV_END = Var("prevBlockEnd")
  RET_BLK = Var("retBlock"); RET_END = Var("retBlockEnd")
  G1 = Var("_gen1")
  G2 = Var("_gen2")
  G3 = Var("_gen3")
  G4 = Var("_gen4")
  G5 = Var("_gen5")
  G6 = Var("_gen6")
  G7 = Var("_gen7")
  G8 = Var("_gen8")
  G9 = Var("_gen9")
  G10 = Var("_gen10")
  G11 = Var("_gen11")
  G12 = Var("_gen12")
  G13 = Var("_gen13")

  block_next = Relation("BlockNext", 3)
  direct_call = Relation("DirectCall", 2)
  arch_ret = Relation("ArchReturnReg", 1)
  blkld = Relation("RegDefUseBlockLastDef", 3)
  defined_in = Relation("RegDefUseDefinedInBlock", 2)
  flow_def = Relation("RegDefUseFlowDef", 4)
  live_var_def = Relation("RegDefUseLiveVarDef", 4)
  ref_in = Relation("RegDefUseRefInBlock", 2)
  ret_block_end = Relation("RegDefUseReturnBlockEnd", 4)
  used = Relation("RegDefUseUsed", 3)
  used_in_block = Relation("RegDefUseUsedInBlock", 4)
  def_used = Relation("RegDefUseDefUsed", 4)
  ret_val_used = Relation("RegDefUseReturnValUsed", 5)
  live_var_used = Relation("RegDefUseLiveVarUsed", 4)
  live_at_prior = Relation("RegDefUseLiveVarAtPriorUsed", 3)
  live_at_end = Relation("RegDefUseLiveVarAtBlockEnd", 3)

  return Program(
    relations=[
      block_next, direct_call, arch_ret, blkld, defined_in, flow_def,
      live_var_def, ref_in, ret_block_end, used, used_in_block,
      def_used, ret_val_used, live_var_used, live_at_prior, live_at_end,
    ],
    rules=[
      (def_used(EA_DEF, MVAR, EA_USED, INDEX) <=
       used(EA_USED, MVAR, INDEX) & blkld(EA_USED, EA_DEF, MVAR)
      ).named("RegDefUsed1"),
      (def_used(EA_DEF, VAR_ID, EA_USED, INDEX) <=
       live_at_end(BLK, BLOCK_USED, MVAR)
       & live_var_def(BLK, VAR_ID, MVAR, EA_DEF)
       & live_var_used(BLOCK_USED, MVAR, EA_USED, INDEX)
      ).named("RegDefUsed2"),
      (def_used(EA_DEF, MVAR, NEXT_EA, NEXT_IDX) <=
       live_at_prior(EA_USED, NXT_BLK, MVAR)
       & def_used(EA_DEF, MVAR, EA_USED, G1)
       & live_var_used(NXT_BLK, MVAR, NEXT_EA, NEXT_IDX)
      ).named("RegDefUsed3").with_plans([
        PlanEntry(delta=0, var_order=("eaUsed", "mvar", "nxtBlk", "eaDef",
                                       "nextEaUsed", "nextIndex")),
        PlanEntry(delta=1, var_order=("eaUsed", "mvar", "nxtBlk", "eaDef",
                                       "nextEaUsed", "nextIndex")),
        PlanEntry(delta=2, var_order=("nxtBlk", "mvar", "eaUsed", "eaDef",
                                       "nextEaUsed", "nextIndex")),
      ]),
      (def_used(EA_DEF, REG, EA_USED, INDEX) <=
       ret_val_used(G2, CALLEE, REG, EA_USED, INDEX)
       & ret_block_end(CALLEE, G3, G4, BLOCK_END)
       & blkld(BLOCK_END, EA_DEF, REG)
      ).named("RegDefUsed4"),
      (ret_val_used(EA_CALL, CALLEE, REG, EA_USED, INDEX_USED) <=
       arch_ret(REG)
       & def_used(EA_CALL, REG, EA_USED, INDEX_USED)
       & direct_call(EA_CALL, CALLEE)
      ).named("RetValUsed"),
      (live_var_used(BLK, MVAR, EA_USED, INDEX) <=
       used_in_block(BLK, EA_USED, MVAR, INDEX)
       & ~blkld(EA_USED, G5, MVAR)
      ).named("LiveVarUsed1"),
      (live_var_used(RET_BLK, REG, EA_USED, INDEX) <=
       ret_block_end(CALLEE, G6, RET_BLK, RET_END)
       & ~blkld(RET_END, G7, REG)
       & ret_val_used(G8, CALLEE, REG, EA_USED, INDEX)
      ).named("LiveVarUsed2"),
      (live_at_prior(EA_USED, BLOCK_USED, MVAR) <=
       live_at_end(BLK, BLOCK_USED, MVAR)
       & used_in_block(BLK, EA_USED, MVAR, G9)
       & ~defined_in(BLK, MVAR)
      ).named("LiveVarAtPriorUsed"),
      (live_at_end(PREV_BLK, BLK, MVAR) <=
       block_next(PREV_BLK, PREV_END, BLK)
       & live_var_used(BLK, MVAR, G10, G11)
       & ~flow_def(PREV_END, MVAR, BLK, G12)
      ).named("LiveVarAtBlockEnd1"),
      (live_at_end(PREV_BLK, BLOCK_USED, MVAR) <=
       live_at_end(BLK, BLOCK_USED, MVAR)
       & ~ref_in(BLK, MVAR)
       & block_next(PREV_BLK, G13, BLK)
      ).named("LiveVarAtBlockEnd2"),
    ],
  )


def test_reg_scc_hir():
  diff_hir(build_reg_scc(), "reg_scc")


def test_reg_scc_mir():
  diff_mir(build_reg_scc(), "reg_scc")


if __name__ == "__main__":
  test_reg_scc_hir()
  test_reg_scc_mir()
  print("reg_scc: OK")
