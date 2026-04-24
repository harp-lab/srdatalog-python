"""Auto-generated from /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/ddisasm/ddisasm.nim by tools/nim_to_dsl.py.
Do not edit manually — regenerate via:

    python tools/nim_to_dsl.py /home/stargazermiao/workspace/SRDatalog/integration_tests/examples/ddisasm/ddisasm.nim --out <this file>
"""

from __future__ import annotations

from srdatalog.dsl import Const, Filter, Program, Relation, SPLIT, Var

# ----- Relations ----------------------------------------------

ArchMemoryAccess = Relation(
  "ArchMemoryAccess",
  6,
  column_types=(
    int,
    int,
    int,
    int,
    int,
    int,
  ),
  input_file="Arch_memory_access_truncate.csv",
)
ArchRegRegArithOp = Relation(
  "ArchRegRegArithOp",
  6,
  column_types=(
    int,
    int,
    int,
    int,
    int,
    int,
  ),
  input_file="Arch_reg_reg_arithmetic_operation.csv",
)
ArchReturnReg = Relation("ArchReturnReg", 1, column_types=(int,), input_file="Arch_return_reg.csv")
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
BlockLastInstr = Relation(
  "BlockLastInstr",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Block_last_instruction.csv",
)
CodeInBlock = Relation(
  "CodeInBlock",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="Code_in_block.csv",
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
MayFallthrough = Relation(
  "MayFallthrough",
  2,
  column_types=(
    int,
    int,
  ),
  input_file="May_fallthrough.csv",
)
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
RegUsedFor = Relation(
  "RegUsedFor",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="Reg_used_for.csv",
)
RelJumpTableEntryCandidate = Relation(
  "RelJumpTableEntryCandidate",
  7,
  column_types=(
    int,
    int,
    int,
    int,
    int,
    int,
    int,
  ),
  input_file="Relative_jump_table_entry_candidate.csv",
)
StackDefUseDef = Relation(
  "StackDefUseDef",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="Stack_def_use_def.csv",
)
StackDefUseDefinedInBlock = Relation(
  "StackDefUseDefinedInBlock",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="Stack_def_use_defined_in_block.csv",
)
StackDefUseLiveVarDef = Relation(
  "StackDefUseLiveVarDef",
  6,
  column_types=(
    int,
    int,
    int,
    int,
    int,
    int,
  ),
  input_file="Stack_def_use_live_var_def.csv",
)
StackDefUseRefInBlock = Relation(
  "StackDefUseRefInBlock",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="Stack_def_use_ref_in_block.csv",
)
StackDefUseUsedInBlock = Relation(
  "StackDefUseUsedInBlock",
  5,
  column_types=(
    int,
    int,
    int,
    int,
    int,
  ),
  input_file="Stack_def_use_used_in_block.csv",
)
StackDefUseUsed = Relation(
  "StackDefUseUsed",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  input_file="Stack_def_use_used.csv",
)
StackDefUseLiveVarUsedEDB = Relation(
  "StackDefUseLiveVarUsedEDB",
  8,
  column_types=(
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
  ),
  input_file="Stack_def_use_live_var_used.csv",
)
JumpTableStart = Relation(
  "JumpTableStart",
  5,
  column_types=(
    int,
    int,
    int,
    int,
    int,
  ),
  input_file="Jump_table_start.csv",
)
DefUsedForAddressEDB = Relation(
  "DefUsedForAddressEDB",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  input_file="Def_used_for_address.csv",
)
StackDefUseBlockLastDef = Relation(
  "StackDefUseBlockLastDef",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  input_file="Stack_def_use_block_last_def.csv",
)
JumpTableTarget = Relation(
  "JumpTableTarget",
  2,
  column_types=(
    int,
    int,
  ),
  print_size=True,
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
RegRegArithOpDefs = Relation(
  "RegRegArithOpDefs",
  8,
  column_types=(
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
DefUsedForAddress = Relation(
  "DefUsedForAddress",
  3,
  column_types=(
    int,
    int,
    int,
  ),
  print_size=True,
)
StackDefUseDefUsed = Relation(
  "StackDefUseDefUsed",
  6,
  column_types=(
    int,
    int,
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)
StackDefUseLiveVarAtBlockEnd = Relation(
  "StackDefUseLiveVarAtBlockEnd",
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
StackDefUseLiveVarAtPriorUsed = Relation(
  "StackDefUseLiveVarAtPriorUsed",
  4,
  column_types=(
    int,
    int,
    int,
    int,
  ),
  print_size=True,
)

# ----- Rules: DdisasmDB -----


def build_ddisasmdb_program(meta: dict[str, int]) -> Program:
  """Build the program, consuming `meta` for dataset_const values.

  `meta` is a `{json_key: int_value}` dict — typically
  `json.load(open("batik_meta.json"))` or similar. Each declared
  dataset_const binds to a Python-local `Const(meta[key])` at the top
  of this function; any missing key raises KeyError loudly here
  instead of surfacing as silent wrong integers downstream.
  """
  blk = Var("blk")
  blockEnd = Var("blockEnd")
  blockUsed = Var("blockUsed")
  callee = Var("callee")
  defVarp = Var("defVarp")
  defVarr = Var("defVarr")
  dest = Var("dest")
  ea = Var("ea")
  eaCall = Var("eaCall")
  eaDef = Var("eaDef")
  eaDef1 = Var("eaDef1")
  eaDef2 = Var("eaDef2")
  eaLoad = Var("eaLoad")
  eaSt = Var("eaSt")
  eaUsed = Var("eaUsed")
  index = Var("index")
  indexUsed = Var("indexUsed")
  mtype = Var("mtype")
  mult = Var("mult")
  mvar = Var("mvar")
  nextEaUsed = Var("nextEaUsed")
  nextIndex = Var("nextIndex")
  nxtBlk = Var("nxtBlk")
  offset = Var("offset")
  prevBlock = Var("prevBlock")
  prevBlockEnd = Var("prevBlockEnd")
  reg = Var("reg")
  reg1 = Var("reg1")
  reg2 = Var("reg2")
  regBaseLoad = Var("regBaseLoad")
  regBaseSt = Var("regBaseSt")
  regDef = Var("regDef")
  retBlock = Var("retBlock")
  retBlockEnd = Var("retBlockEnd")
  size = Var("size")
  stackPosLoad = Var("stackPosLoad")
  stackPosSt = Var("stackPosSt")
  tableStart = Var("tableStart")
  usedVarp = Var("usedVarp")
  usedVarr = Var("usedVarr")
  varDefp = Var("varDefp")
  varDefr = Var("varDefr")
  varIdentity = Var("varIdentity")
  varUsedp = Var("varUsedp")
  varUsedr = Var("varUsedr")
  varp = Var("varp")
  varr = Var("varr")

  # dataset_consts — Python bindings, resolved from meta.json keys.
  LOAD = Const(meta["LOAD"])
  NONE_ACCESS = Const(meta["NONE"])
  PC_RELATIVE = Const(meta["PCRelative"])
  STORE = Const(meta["STORE"])

  return Program(
    relations=[
      ArchMemoryAccess,
      ArchRegRegArithOp,
      ArchReturnReg,
      BlockNext,
      BlockLastInstr,
      CodeInBlock,
      DirectCall,
      MayFallthrough,
      RegDefUseBlockLastDef,
      RegDefUseDefinedInBlock,
      RegDefUseFlowDef,
      RegDefUseLiveVarDef,
      RegDefUseRefInBlock,
      RegDefUseReturnBlockEnd,
      RegDefUseUsed,
      RegDefUseUsedInBlock,
      RegUsedFor,
      RelJumpTableEntryCandidate,
      StackDefUseDef,
      StackDefUseDefinedInBlock,
      StackDefUseLiveVarDef,
      StackDefUseRefInBlock,
      StackDefUseUsedInBlock,
      StackDefUseUsed,
      StackDefUseLiveVarUsedEDB,
      JumpTableStart,
      DefUsedForAddressEDB,
      StackDefUseBlockLastDef,
      JumpTableTarget,
      RegDefUseDefUsed,
      RegDefUseReturnValUsed,
      RegDefUseLiveVarUsed,
      RegDefUseLiveVarAtPriorUsed,
      RegDefUseLiveVarAtBlockEnd,
      RegRegArithOpDefs,
      DefUsedForAddress,
      StackDefUseDefUsed,
      StackDefUseLiveVarAtBlockEnd,
      StackDefUseLiveVarAtPriorUsed,
    ],
    rules=[
      (
        JumpTableTarget(ea, dest)
        <= JumpTableStart(ea, size, tableStart, Var("_"), Var("_"))
        & RelJumpTableEntryCandidate(Var("_"), tableStart, size, Var("_"), dest, Var("_"), Var("_"))
      ).named('JumpTableTargetRule'),
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
      (
        RegRegArithOpDefs(ea, regDef, eaDef1, reg1, eaDef2, reg2, mult, offset)
        <= DefUsedForAddress(ea, regDef, Var("_"))
        & ArchRegRegArithOp(ea, regDef, reg1, reg2, mult, offset)
        & Filter(
          (
            'reg1',
            'reg2',
          ),
          "return reg1 != reg2;",
        )
        & RegDefUseDefUsed(eaDef1, reg1, ea, Var("_"))
        & Filter(
          (
            'ea',
            'eaDef1',
          ),
          "return ea != eaDef1;",
        )
        & RegDefUseDefUsed(eaDef2, reg2, ea, Var("_"))
        & Filter(
          (
            'ea',
            'eaDef2',
          ),
          "return ea != eaDef2;",
        )
      ).named('RegRegArithDefs'),
      (DefUsedForAddress(ea, reg, PC_RELATIVE) <= DefUsedForAddressEDB(ea, reg, PC_RELATIVE)).named(
        'DefUsedAddr1'
      ),
      (
        DefUsedForAddress(eaDef, reg, mtype)
        <= RegDefUseDefUsed(eaDef, reg, ea, Var("_")) & RegUsedFor(ea, reg, mtype)
      )
      .named('DefUsedAddr2')
      .with_plan(var_order=['ea', 'reg', 'eaDef', 'mtype']),
      (
        DefUsedForAddress(eaDef, reg, mtype)
        <= DefUsedForAddress(eaUsed, Var("_"), mtype)
        & RegDefUseDefUsed(eaDef, reg, eaUsed, Var("_"))
      ).named('DefUsedAddr3'),
      (
        DefUsedForAddress(eaDef, reg1, mtype)
        <= DefUsedForAddress(eaLoad, reg2, mtype)
        & ArchMemoryAccess(LOAD, eaLoad, reg2, regBaseLoad, NONE_ACCESS, stackPosLoad)
        & StackDefUseDefUsed(eaSt, regBaseSt, stackPosSt, eaLoad, regBaseLoad, stackPosLoad)
        & ArchMemoryAccess(STORE, eaSt, reg1, regBaseSt, NONE_ACCESS, stackPosSt)
        & RegDefUseDefUsed(eaDef, reg1, eaSt, Var("_"))
      ).named('DefUsedAddr4'),
      (
        StackDefUseDefUsed(eaDef, varr, varp, eaUsed, varr, varp)
        <= StackDefUseUsed(eaUsed, varr, varp, Var("_"))
        & StackDefUseBlockLastDef(eaUsed, eaDef, varr, varp)
      ).named('StackDefUsed1'),
      (
        StackDefUseDefUsed(eaDef, defVarr, defVarp, eaUsed, varUsedr, varUsedp)
        <= StackDefUseLiveVarAtBlockEnd(blk, blockUsed, varr, varp)
        & StackDefUseLiveVarDef(blk, defVarr, defVarp, varr, varp, eaDef)
        & StackDefUseLiveVarUsedEDB(
          blockUsed, varr, varp, varUsedr, varUsedp, eaUsed, Var("_"), Var("_")
        )
      ).named('StackDefUsed2'),
      (
        StackDefUseDefUsed(eaDef, defVarr, defVarp, eaUsed, usedVarr, usedVarp)
        <= StackDefUseLiveVarUsedEDB(
          ea, defVarr, defVarp, usedVarr, usedVarp, eaUsed, Var("_"), Var("_")
        )
        & MayFallthrough(eaDef, ea)
        & CodeInBlock(eaDef, blk)
        & CodeInBlock(ea, blk)
        & StackDefUseDef(eaDef, defVarr, defVarp)
      ).named('StackDefUsed3'),
      (
        StackDefUseDefUsed(eaDef, varDefr, varDefp, nextEaUsed, varUsedr, varUsedp)
        <= StackDefUseLiveVarAtPriorUsed(eaUsed, nxtBlk, varr, varp)
        & StackDefUseDefUsed(eaDef, varDefr, varDefp, eaUsed, varr, varp)
        & StackDefUseLiveVarUsedEDB(
          nxtBlk, varr, varp, varUsedr, varUsedp, nextEaUsed, Var("_"), Var("_")
        )
      )
      .named('StackDefUsed4')
      .with_plan(delta=1, dedup_hash=True),
      (
        StackDefUseLiveVarAtBlockEnd(prevBlock, blockUsed, varr, varp)
        <= StackDefUseLiveVarAtBlockEnd(blk, blockUsed, varr, varp)
        & ~StackDefUseRefInBlock(blk, varr, varp)
        & ~RegDefUseDefinedInBlock(blk, varr)
        & SPLIT
        & BlockNext(prevBlock, Var("_"), blk)
      )
      .named('StackLiveVarBlockEnd1')
      .with_plan(delta=0, var_order=['blk', 'varr', 'varp', 'blockUsed', 'prevBlock']),
      (
        StackDefUseLiveVarAtBlockEnd(prevBlock, blk, varr, varp)
        <= BlockNext(prevBlock, Var("_"), blk)
        & StackDefUseLiveVarUsedEDB(
          blk, varr, varp, Var("_"), Var("_"), Var("_"), Var("_"), Var("_")
        )
      ).named('StackLiveVarBlockEnd2'),
      (
        StackDefUseLiveVarAtPriorUsed(eaUsed, blockUsed, varr, varp)
        <= StackDefUseLiveVarAtBlockEnd(blk, blockUsed, varr, varp)
        & StackDefUseUsedInBlock(blk, eaUsed, varr, varp, Var("_"))
        & ~RegDefUseDefinedInBlock(blk, varr)
        & ~StackDefUseDefinedInBlock(blk, varr, varp)
      ).named('StackLiveVarPriorUsed'),
    ],
  )
