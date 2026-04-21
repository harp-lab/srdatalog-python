# run from root

from srdatalog.mir_commands import *
from srdatalog.mir_schema import * 
from srdatalog.srdatalog_program import SRDatalogProgram

RREL = "RRel"
SREL = "SRel" 
TREL = "TRel"
ZREL = "ZRel" 

dest_1_index_spec = IndexSpec(FactIndex(ZREL, [0, 1, 2]), Version.FULL) 
instructions = MirInstructions(
    [
      Block(
          [dest_1_index_spec],
          [
            ExecutePipeline(
                sources=[
                  IndexSpec(FactIndex(RREL, [0, 1]), Version.FULL),
                  IndexSpec(FactIndex(TREL, [1, 0]), Version.FULL),
                  IndexSpec(FactIndex(RREL, [0, 1]), Version.FULL),
                  IndexSpec(FactIndex(SREL, [0, 1]), Version.FULL),
                  IndexSpec(FactIndex(SREL, [0, 1]), Version.FULL),
                  IndexSpec(FactIndex(TREL, [1, 0]), Version.FULL)
                ],
                dests=[dest_1_index_spec],
                body=[
                  ColumnJoin(vars=("x"),
                              sources=[
                                  ColumnSource(FactIndex(RREL, [0,1]), Version.FULL),
                                  ColumnSource(FactIndex(TREL, [1,0]), Version.FULL)
                            ]),
                  ColumnJoin(vars=("y"),
                              sources=[
                                  ColumnSource(FactIndex(RREL, [0,1]), Version.FULL, prefix=('x')),
                                  ColumnSource(FactIndex(SREL, [0,1]), Version.FULL)
                            ]),
                  ColumnJoin(vars=("z"),
                              sources=[
                                  ColumnSource(FactIndex(SREL, [0,1]), Version.FULL, prefix=('y')),
                                  ColumnSource(FactIndex(TREL, [1,0]), Version.FULL, prefix=('x'))
                            ]),

                  InsertInto(fact=FactIndex(ZREL, [0,1,2]), version=Version.NEW, terms=("x", "y", "z"))
                ]
            ),

            RebuildIndex(FactIndex(ZREL, [0,1,2]), Version.NEW),
            CheckSize(ZREL, Version.NEW),
            ComputeDelta(FactIndex(ZREL, [0,1,2]), Version.NEW),
            ClearRelation(ZREL, Version.NEW),
            MergeIndex(FactIndex(ZREL, [0,1,2]), Version.FULL)
          ],
          recursive=False
      )
    ]
)
database = SchemaDefinition(
    facts=[
    FactDefinition(RREL, [int, int], pragmas={Pragma.INPUT: "RRel.csv"}),
    FactDefinition(SREL, [int, int, int], pragmas={Pragma.INPUT: "SRel.csv"}),
    FactDefinition(TREL, [int, int, int], pragmas={Pragma.INPUT: "TRel.csv"}),
    FactDefinition(ZREL, [int, int, int])
    ]
)
program = SRDatalogProgram(
    name="Triangle",
    database=database,
    instructions=instructions
)

# Make sure to run compile_to_file at least once, but if you run this multiple times w/out changing the program
# then you can use set_file_location to avoid the (slow) compilation step and just load the existing binary and header
program.compile_to_file()
#program.set_file_location(binary_loc="/home/miakerchen/SRDatalog/python/output/Triangle.so", header_loc="/home/miakerchen/SRDatalog/python/output/Triangle.h", compile_type="shared")

triangle = program.open_ffi()

triangle.load_data("/home/miakerchen/SRDatalog/python/test_data/triangle")
triangle.run()