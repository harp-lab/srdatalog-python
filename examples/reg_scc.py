"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/ddisasm/reg_scc.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/ddisasm/reg_scc.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dataset_const import load_meta, resolve_program_consts
from srdatalog.dsl import Program, Relation, Var

# ----- Relations ----------------------------------------------

BlockNext = Relation(
  "BlockNext",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="Block_next.csv",
)
DirectCall = Relation(
  "DirectCall",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Direct_call.csv",
)
ArchReturnReg = Relation("ArchReturnReg", 1, column_types=(int,), input_file="Arch_return_reg.csv")
RegDefUseBlockLastDef = Relation(
  "RegDefUseBlockLastDef",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="Reg_def_use_block_last_def.csv",
)
RegDefUseDefinedInBlock = Relation(
  "RegDefUseDefinedInBlock",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Reg_def_use_defined_in_block.csv",
)
RegDefUseFlowDef = Relation(
  "RegDefUseFlowDef",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  input_file="Reg_def_use_flow_def.csv",
)
RegDefUseLiveVarDef = Relation(
  "RegDefUseLiveVarDef",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  input_file="Reg_def_use_live_var_def.csv",
)
RegDefUseRefInBlock = Relation(
  "RegDefUseRefInBlock",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Reg_def_use_ref_in_block.csv",
)
RegDefUseReturnBlockEnd = Relation(
  "RegDefUseReturnBlockEnd",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  input_file="Reg_def_use_return_block_end.csv",
)
RegDefUseUsed = Relation(
  "RegDefUseUsed",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="Reg_def_use_used.csv",
)
RegDefUseUsedInBlock = Relation(
  "RegDefUseUsedInBlock",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  input_file="Reg_def_use_used_in_block.csv",
)
RegDefUseDefUsed = Relation(
  "RegDefUseDefUsed",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
  index_type="SRDatalog::GPU::Device2LevelIndex",
)
RegDefUseReturnValUsed = Relation(
  "RegDefUseReturnValUsed",
  5,
  column_types=(
    int,
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
RegDefUseLiveVarUsed = Relation(
  "RegDefUseLiveVarUsed",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
  index_type="SRDatalog::GPU::Device2LevelIndex",
)
RegDefUseLiveVarAtPriorUsed = Relation(
  "RegDefUseLiveVarAtPriorUsed",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  print_size=True,
  index_type="SRDatalog::GPU::Device2LevelIndex",
)
RegDefUseLiveVarAtBlockEnd = Relation(
  "RegDefUseLiveVarAtBlockEnd",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  print_size=True,
  index_type="SRDatalog::GPU::Device2LevelIndex",
)

# ----- dataset_const declarations -----------------------------

DATASET_CONST_DECLS = {}

# ----- Rules: RegSccDB -----


def build_regsccdb_program() -> Program:
  blk = Var("blk")
  blockEnd = Var("blockEnd")
  blockUsed = Var("blockUsed")
  callee = Var("callee")
  eaCall = Var("eaCall")
  eaDef = Var("eaDef")
  eaUsed = Var("eaUsed")
  index = Var("index")
  indexUsed = Var("indexUsed")
  mvar = Var("mvar")
  nextEaUsed = Var("nextEaUsed")
  nextIndex = Var("nextIndex")
  nxtBlk = Var("nxtBlk")
  prevBlock = Var("prevBlock")
  prevBlockEnd = Var("prevBlockEnd")
  reg = Var("reg")
  retBlock = Var("retBlock")
  retBlockEnd = Var("retBlockEnd")
  varIdentity = Var("varIdentity")

  return Program(
    relations=[
      BlockNext,
      DirectCall,
      ArchReturnReg,
      RegDefUseBlockLastDef,
      RegDefUseDefinedInBlock,
      RegDefUseFlowDef,
      RegDefUseLiveVarDef,
      RegDefUseRefInBlock,
      RegDefUseReturnBlockEnd,
      RegDefUseUsed,
      RegDefUseUsedInBlock,
      RegDefUseDefUsed,
      RegDefUseReturnValUsed,
      RegDefUseLiveVarUsed,
      RegDefUseLiveVarAtPriorUsed,
      RegDefUseLiveVarAtBlockEnd,
    ],
    rules=[
      (
        RegDefUseDefUsed(eaDef, mvar, eaUsed, index)
        <= RegDefUseUsed(eaUsed, mvar, index) & RegDefUseBlockLastDef(eaUsed, eaDef, mvar)
      ).named('RegDefUsed1'),
      (
        RegDefUseDefUsed(eaDef, varIdentity, eaUsed, index)
        <= RegDefUseLiveVarAtBlockEnd(blk, blockUsed, mvar)
        & RegDefUseLiveVarDef(blk, varIdentity, mvar, eaDef)
        & RegDefUseLiveVarUsed(blockUsed, mvar, eaUsed, index)
      ).named('RegDefUsed2'),
      (
        RegDefUseDefUsed(eaDef, mvar, nextEaUsed, nextIndex)
        <= RegDefUseLiveVarAtPriorUsed(eaUsed, nxtBlk, mvar)
        & RegDefUseDefUsed(eaDef, mvar, eaUsed, Var("_"))
        & RegDefUseLiveVarUsed(nxtBlk, mvar, nextEaUsed, nextIndex)
      )
      .named('RegDefUsed3')
      .with_plan(
        delta=0, var_order=['eaUsed', 'mvar', 'nxtBlk', 'eaDef', 'nextEaUsed', 'nextIndex']
      )
      .with_plan(
        delta=1, var_order=['eaUsed', 'mvar', 'nxtBlk', 'eaDef', 'nextEaUsed', 'nextIndex']
      )
      .with_plan(
        delta=2, var_order=['nxtBlk', 'mvar', 'eaUsed', 'eaDef', 'nextEaUsed', 'nextIndex']
      ),
      (
        RegDefUseDefUsed(eaDef, reg, eaUsed, index)
        <= RegDefUseReturnValUsed(Var("_"), callee, reg, eaUsed, index)
        & RegDefUseReturnBlockEnd(callee, Var("_"), Var("_"), blockEnd)
        & RegDefUseBlockLastDef(blockEnd, eaDef, reg)
      ).named('RegDefUsed4'),
      (
        RegDefUseReturnValUsed(eaCall, callee, reg, eaUsed, indexUsed)
        <= ArchReturnReg(reg)
        & RegDefUseDefUsed(eaCall, reg, eaUsed, indexUsed)
        & DirectCall(eaCall, callee)
      ).named('RetValUsed'),
      (
        RegDefUseLiveVarUsed(blk, mvar, eaUsed, index)
        <= RegDefUseUsedInBlock(blk, eaUsed, mvar, index)
        & ~RegDefUseBlockLastDef(eaUsed, Var("_"), mvar)
      ).named('LiveVarUsed1'),
      (
        RegDefUseLiveVarUsed(retBlock, reg, eaUsed, index)
        <= RegDefUseReturnBlockEnd(callee, Var("_"), retBlock, retBlockEnd)
        & ~RegDefUseBlockLastDef(retBlockEnd, Var("_"), reg)
        & RegDefUseReturnValUsed(Var("_"), callee, reg, eaUsed, index)
      ).named('LiveVarUsed2'),
      (
        RegDefUseLiveVarAtPriorUsed(eaUsed, blockUsed, mvar)
        <= RegDefUseLiveVarAtBlockEnd(blk, blockUsed, mvar)
        & RegDefUseUsedInBlock(blk, eaUsed, mvar, Var("_"))
        & ~RegDefUseDefinedInBlock(blk, mvar)
      ).named('LiveVarAtPriorUsed'),
      (
        RegDefUseLiveVarAtBlockEnd(prevBlock, blk, mvar)
        <= BlockNext(prevBlock, prevBlockEnd, blk)
        & RegDefUseLiveVarUsed(blk, mvar, Var("_"), Var("_"))
        & ~RegDefUseFlowDef(prevBlockEnd, mvar, blk, Var("_"))
      ).named('LiveVarAtBlockEnd1'),
      (
        RegDefUseLiveVarAtBlockEnd(prevBlock, blockUsed, mvar)
        <= RegDefUseLiveVarAtBlockEnd(blk, blockUsed, mvar)
        & ~RegDefUseRefInBlock(blk, mvar)
        & BlockNext(prevBlock, Var("_"), blk)
      ).named('LiveVarAtBlockEnd2'),
    ],
  )


def build_regsccdb(meta_json_path: str) -> tuple[Program, dict[str, int]]:
  """Convenience: build the program, load dataset_consts, substitute."""
  consts = load_meta(meta_json_path, DATASET_CONST_DECLS)
  return resolve_program_consts(build_regsccdb_program(), consts), consts
