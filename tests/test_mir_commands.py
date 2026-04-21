import unittest
import sys
from pathlib import Path

# Add the parent directory to the path so we can import mir_commands

from srdatalog.mir.commands import (
    FactIndex, Version,
    RebuildIndex,
    MergeIndex,
    CheckSize,
    ComputeDelta,
    ClearRelation,
    MergeRelation,
    Scan,
    InsertInto,
    IndexSpec,
    ExecutePipeline,
    Block,
    ColumnJoin,
    ColumnSource,
    CartesianJoin,
    MirInstructions,
    Aggregate,
    Negate,
    Filter,
    CppHook,
    MISSING_HANDLE,
    generate_single_vhm,
    generate_multi_vhm
)

Relation_PointsTo = "PointsTo"  # just a convenience definition
Relation_AddressOf = "AddressOf" 

def print_diff(expected, actual):
    
    # for debugging diff
    expected = expected.replace("<", "\n<")
    actual = actual.replace("<", "\n<")
    print("expected: \n\n\n"+expected)
    print("actual:\n\n\n"+actual)
    

class TestFactIndex(unittest.TestCase):
    """Test cases for FactIndex class"""

    def setUp(self):
        """Set up test fixtures"""
        self.fact_index = FactIndex(Relation_PointsTo, [0, 1])

    def test_fact_index_creation(self):
        """Test FactIndex can be created with name and args"""
        self.assertEqual(self.fact_index.name, Relation_PointsTo)
        self.assertEqual(self.fact_index.args, [0, 1])

    def test_fact_index_str(self):
        """Test FactIndex string representation"""
        expected = "PointsTo, 0, 1"
        self.assertEqual(str(self.fact_index), expected)

    def test_fact_index_ftype(self):
        """Test FactIndex ftype method"""
        expected = "SRDatalog::mir::dsl::index<PointsTo, 0, 1>"
        self.assertEqual(self.fact_index.ftype(), expected)


class TestVersion(unittest.TestCase):
    """Test cases for Version enum"""

    def test_version_new_ver(self):
        """Test NEW_VER version enum"""
        self.assertEqual(Version.NEW.name, "NEW")
        self.assertEqual(Version.NEW.number, "2")
        self.assertEqual(Version.NEW.method, "newt")
        self.assertEqual(Version.NEW.code, "NEW_VER")

    def test_version_delta_ver(self):
        """Test DELTA_VER version enum"""
        self.assertEqual(Version.DELTA.name, "DELTA")
        self.assertEqual(Version.DELTA.number, "1")
        self.assertEqual(Version.DELTA.method, "delta")
        self.assertEqual(Version.DELTA.code, "DELTA_VER")

    def test_version_full_ver(self):
        """Test FULL_VER version enum"""
        self.assertEqual(Version.FULL.name, "FULL")
        self.assertEqual(Version.FULL.number, "0")
        self.assertEqual(Version.FULL.method, "full")
        self.assertEqual(Version.FULL.code, "FULL_VER")


class TestRebuildIndex(unittest.TestCase):
    """Test cases for RebuildIndex class"""

    def setUp(self):
        """Set up test fixtures"""
        self.fact = FactIndex(Relation_PointsTo, [0, 1])
        self.rebuild = RebuildIndex(fact=self.fact, version=Version.NEW)

    def test_rebuild_index_creation(self):
        """Test RebuildIndex can be created"""
        self.assertEqual(self.rebuild.fact.name, Relation_PointsTo)
        self.assertEqual(self.rebuild.version, Version.NEW)

    def test_rebuild_index_str(self):
        """Test RebuildIndex string representation"""
        expected = "rebuild_index(SRDatalog::mir::dsl::index<PointsTo, 0, 1>().newt())"
        self.assertEqual(str(self.rebuild), expected)


class TestMergeIndex(unittest.TestCase):
    """Test cases for MergeIndex class"""

    def setUp(self):
        """Set up test fixtures"""
        self.fact = FactIndex(Relation_PointsTo, [0, 1])
        self.merge = MergeIndex(fact=self.fact, version=Version.DELTA)

    def test_merge_index_creation(self):
        """Test MergeIndex can be created"""
        self.assertEqual(self.merge.fact.name, Relation_PointsTo)
        self.assertEqual(self.merge.version, Version.DELTA)

    def test_merge_index_str(self):
        """Test MergeIndex string representation"""
        expected = "merge_index(SRDatalog::mir::dsl::index<PointsTo, 0, 1>().delta())"
        self.assertEqual(str(self.merge), expected)


class TestCheckSize(unittest.TestCase):
    """Test cases for CheckSize class"""

    def setUp(self):
        """Set up test fixtures"""
        self.fact = FactIndex(Relation_PointsTo, [0, 1])
        self.check = CheckSize(fact_name=self.fact.name, version=Version.FULL)

    def test_check_size_str(self):
        """Test CheckSize string representation"""
        expected = "check_size<PointsTo, FULL_VER>()"
        self.assertEqual(str(self.check), expected)


class TestComputeDelta(unittest.TestCase):
    """Test cases for ComputeDelta class"""

    def setUp(self):
        """Set up test fixtures"""
        self.fact = FactIndex(Relation_PointsTo, [0, 1])
        self.compute = ComputeDelta(fact=self.fact, version=Version.NEW)

    def test_compute_delta_str(self):
        """Test ComputeDelta string representation"""
        expected = "compute_delta(SRDatalog::mir::dsl::index<PointsTo, 0, 1>().newt())"
        self.assertEqual(str(self.compute), expected)


class TestClearRelation(unittest.TestCase):
    """Test cases for ClearRelation class"""

    def setUp(self):
        """Set up test fixtures"""
        self.fact = FactIndex(Relation_PointsTo, [0, 1])
        self.clear = ClearRelation(fact_name=self.fact.name, version=Version.DELTA)

    def test_clear_relation_str(self):
        """Test ClearRelation string representation"""
        expected = "clear_relation<PointsTo, DELTA_VER>()"
        self.assertEqual(str(self.clear), expected)


class TestMergeRelation(unittest.TestCase):
    """Test cases for MergeRelation class"""

    def setUp(self):
        """Set up test fixtures"""
        self.fact = FactIndex(Relation_PointsTo, [0, 1])
        self.merge = MergeRelation(fact_name=self.fact.name)

    def test_merge_relation_str(self):
        """Test MergeRelation string representation"""
        expected = "merge_relation<PointsTo>()"
        self.assertEqual(str(self.merge), expected)


class TestScan(unittest.TestCase):
    """Test cases for Scan class"""

    def test_scan_with_index_and_full_version(self):
        """Test Scan creation with index arguments and FULL version"""
        fact = FactIndex(Relation_AddressOf, [0, 1])
        scan = Scan(fact=fact, version=Version.FULL, vars=("y", "x"), prefix=())
        
        output = str(scan)
        expected='SRDatalog::mir::dsl::scan_h<'+str(MISSING_HANDLE)+', decltype(boost::hana::make_map())>(SRDatalog::mir::dsl::vars("y"_v, "x"_v),SRDatalog::mir::dsl::index<AddressOf, 0, 1>().full())'

        self.assertEqual(expected, output)


    # (scan :vars (y x) :index (AddressOf 0 1) :ver FULL :prefix ())
    def test_scan_with_index_and_full_version_regex(self):
        """Test Scan creation with index arguments and FULL version"""
        fact = FactIndex(Relation_AddressOf, [0, 1])
        scan = Scan(fact=fact, version=Version.FULL, vars=("y", "x"), prefix=())
        
        output = str(scan)
        expected = 'SRDatalog::mir::dsl::scan_h<'+str(MISSING_HANDLE)+', decltype(boost::hana::make_map())>(SRDatalog::mir::dsl::vars("y"_v, "x"_v),SRDatalog::mir::dsl::index<AddressOf, 0, 1>().full())'
        
        self.assertEqual(expected, output)

class TestInsertInto(unittest.TestCase):
    """Test cases for InsertInto class"""

    def test_insert_into_with_dedup_index(self):
        """Test InsertInto with dedup index (index arguments)"""
        fact = FactIndex(Relation_PointsTo, [1, 0])
        insert = InsertInto(fact=fact, version=Version.NEW, terms=("y", "x"))
        
        expected = 'insert_into<PointsTo, NEW_VER, decltype(SRDatalog::mir::dsl::index<PointsTo, 1, 0>().newt())>("y"_v, "x"_v)'
        self.assertEqual(str(insert), expected)

    def test_insert_into_with_full_version(self):
        """Test InsertInto with FULL_VER version"""
        fact = FactIndex(Relation_AddressOf, [0, 1])
        insert = InsertInto(fact=fact, version=Version.FULL, terms=("a", "b"))
        
        output = str(insert)
        
        self.assertIn("insert_into<AddressOf, FULL_VER", output)
        self.assertIn("SRDatalog::mir::dsl::index<AddressOf, 0, 1>().full()", output)
        self.assertIn('"a"_v', output)
        self.assertIn('"b"_v', output)

    def test_insert_into_multiple_terms(self):
        """Test InsertInto with multiple terms"""
        fact = FactIndex("MultiRel", [0, 1, 2])
        insert = InsertInto(fact=fact, version=Version.NEW, terms=("t1", "t2", "t3", "t4"))
        
        output = str(insert)
        
        self.assertIn('"t1"_v', output)
        self.assertIn('"t2"_v', output)
        self.assertIn('"t3"_v', output)
        self.assertIn('"t4"_v', output)

    def test_insert_into_empty_args_uses_simple_index(self):
        """Test InsertInto with empty args uses simple index format"""
        fact = FactIndex("SimpleRel", [])
        insert = InsertInto(fact=fact, version=Version.FULL, terms=("x",))
        
        output = str(insert)
        
        # When args are empty, should use simple index format
        self.assertIn("index<SimpleRel>()", output)
        self.assertIn("FULL_VER", output)



        
# (index-spec :schema AddressOf :index (0 1) :ver FULL))
# std::tuple<SRDatalog::mir::IndexSpecT<AddressOf, std::integer_sequence<int, 0, 1>, 0>>

class TestIndexSpec(unittest.TestCase):
    """Test cases for IndexSpec class"""

    def test_index_spec(self):
        """"""
        fact = FactIndex(Relation_AddressOf, [0, 1])
        insert = IndexSpec(fact=fact, version=Version.FULL)
        
        expected = 'SRDatalog::mir::IndexSpecT<AddressOf, std::integer_sequence<int, 0, 1>, 0>'
        self.assertEqual(str(insert), expected)



class TestExecutePipeline(unittest.TestCase):

  def test_nonrecursive_pipeline(self):
    structure = ExecutePipeline(
        sources=[
            IndexSpec(fact=FactIndex(Relation_AddressOf, [0, 1]), version=Version.FULL)
        ],
        dests=[
            IndexSpec(fact=FactIndex(Relation_PointsTo, [1, 0]), version=Version.FULL)
        ],
        body=[
            Scan(
                fact=FactIndex(Relation_AddressOf, [0, 1]),
                version=Version.FULL,
                vars=("y", "x"),
                prefix=()
            ),
            InsertInto(
                fact=FactIndex(Relation_PointsTo, [1, 0]),
                version=Version.NEW,
                terms=("y", "x")
            )
        ]
    )
    
    expected='execute(pipeline<std::tuple<SRDatalog::mir::IndexSpecT<AddressOf,std::integer_sequence<int,0,1>,0>>,std::tuple<SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,1,0>,0>>,std::tuple<AddressOf>>(SRDatalog::mir::dsl::scan_h<'+str(MISSING_HANDLE)+',decltype(boost::hana::make_map())>(SRDatalog::mir::dsl::vars("y"_v,"x"_v),SRDatalog::mir::dsl::index<AddressOf,0,1>().full()),insert_into<PointsTo,NEW_VER,decltype(SRDatalog::mir::dsl::index<PointsTo,1,0>().newt())>("y"_v,"x"_v)))'
    actual = str(structure).replace(" ", "").replace("\n", "")  # remove whitespace formatting

    # makes the diff a little easier to parse
    actual = actual.replace("<", "\n<")
    expected = expected.replace("<", "\n<")

    self.assertEqual(actual, expected)



class TestBlock(unittest.TestCase):

  def test_andersen_simplest(self):
    structure = Block (
        
      dests=[IndexSpec(fact=FactIndex(Relation_PointsTo, [1, 0]), version=Version.FULL)],
        
      instructions=[
          ExecutePipeline(
            sources=[
                IndexSpec(fact=FactIndex(Relation_AddressOf, [0, 1]), version=Version.FULL)
            ],
            dests=[
                IndexSpec(fact=FactIndex(Relation_PointsTo, [1, 0]), version=Version.FULL)
            ],
            body=[
                InsertInto(
                    fact=FactIndex(Relation_PointsTo, [1, 0]),
                    version=Version.NEW,
                    terms=("y", "x")
                )
            ]
          ),

          RebuildIndex(fact=FactIndex(Relation_PointsTo, [1, 0]), version=Version.NEW),
          RebuildIndex(fact=FactIndex(Relation_PointsTo, [0, 1]), version=Version.NEW)
      ]
      
    )

    # same nim vs python ordering
    expected='fixpoint_plan<std::tuple<PointsTo>,std::tuple<SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,1,0>,0>>,std::tuple<SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,1,0>,2>,SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,0,1>,2>>>(execute(pipeline<std::tuple<SRDatalog::mir::IndexSpecT<AddressOf,std::integer_sequence<int,0,1>,0>>,std::tuple<SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,1,0>,0>>,std::tuple<AddressOf>>(insert_into<PointsTo,NEW_VER,decltype(SRDatalog::mir::dsl::index<PointsTo,1,0>().newt())>("y"_v,"x"_v))),rebuild_index(SRDatalog::mir::dsl::index<PointsTo,1,0>().newt()),rebuild_index(SRDatalog::mir::dsl::index<PointsTo,0,1>().newt()));'
    actual = str(structure).replace(" ", "").replace("\n", "")  # remove whitespace formatting
    
    self.assertEqual(actual, expected)

  def test_andersen_basecase(self):
      

    structure = Block (
        
      dests=[IndexSpec(fact=FactIndex(Relation_PointsTo, [1, 0]), version=Version.FULL)],
        
      instructions=[
          ExecutePipeline(
            sources=[ IndexSpec(fact=FactIndex(Relation_AddressOf, [0, 1]), version=Version.FULL) ],
            dests=[ IndexSpec(fact=FactIndex(Relation_PointsTo, [1, 0]), version=Version.FULL) ],
            body=[
                Scan(
                    fact=FactIndex(Relation_AddressOf, [0, 1]),
                    version=Version.FULL,
                    vars=("y", "x"),
                    prefix=()             
                    ),
                InsertInto(
                    fact=FactIndex(Relation_PointsTo, [1, 0]),
                    version=Version.NEW,
                    terms=("y", "x")
                )
            ]
          ),

          RebuildIndex(fact=FactIndex(Relation_PointsTo, [1, 0]), version=Version.NEW),
          RebuildIndex(fact=FactIndex(Relation_PointsTo, [0, 1]), version=Version.NEW),
          CheckSize(fact_name=Relation_PointsTo, version=Version.NEW),
          ComputeDelta(fact=FactIndex(Relation_PointsTo, [1, 0]), version=Version.NEW),
          ClearRelation(fact_name=Relation_PointsTo, version=Version.NEW),
          MergeIndex(fact=FactIndex(Relation_PointsTo, [1, 0]), version=Version.FULL),
          MergeIndex(fact=FactIndex(Relation_PointsTo, [0, 1]), version=Version.FULL)
      ]
      
    )




    expected='fixpoint_plan<std::tuple<PointsTo>,std::tuple<SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,1,0>,0>>,std::tuple<SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,1,0>,2>,SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,0,1>,2>>>(execute(pipeline<std::tuple<SRDatalog::mir::IndexSpecT<AddressOf,std::integer_sequence<int,0,1>,0>>,std::tuple<SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,1,0>,0>>,std::tuple<AddressOf>>(SRDatalog::mir::dsl::scan_h<'+str(MISSING_HANDLE)+',decltype(boost::hana::make_map())>(SRDatalog::mir::dsl::vars("y"_v,"x"_v),SRDatalog::mir::dsl::index<AddressOf,0,1>().full()),insert_into<PointsTo,NEW_VER,decltype(SRDatalog::mir::dsl::index<PointsTo,1,0>().newt())>("y"_v,"x"_v))),rebuild_index(SRDatalog::mir::dsl::index<PointsTo,1,0>().newt()),rebuild_index(SRDatalog::mir::dsl::index<PointsTo,0,1>().newt()),check_size<PointsTo,NEW_VER>(),compute_delta(SRDatalog::mir::dsl::index<PointsTo,1,0>().newt()),clear_relation<PointsTo,NEW_VER>(),merge_index(SRDatalog::mir::dsl::index<PointsTo,1,0>().full()),merge_index(SRDatalog::mir::dsl::index<PointsTo,0,1>().full()));'
    actual = str(structure).replace(" ", "").replace("\n", "")  # remove whitespace formatting


    self.assertEqual(actual, expected)

  # def test_andersen_recursivestep(self):
  #     pass

class TestColumnSource(unittest.TestCase):
    
    
    def test_noprefix(self):

      structure = ColumnSource(fact=FactIndex(Relation_PointsTo, [0, 1]), version=Version.DELTA)

      expected = 'SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index<PointsTo, 0, 1>().delta())'

      self.assertEqual(str(structure), expected)
    
    
    def test_prefix(self):

      structure = ColumnSource(fact=FactIndex(Relation_PointsTo, [0, 1]), version=Version.DELTA, prefix=("x"))

      expected = 'SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index<PointsTo,0,1>().delta(),"x"_v)'
      actual = str(structure).replace(" ", "").replace("\n", "")  # remove whitespace formatting
      self.assertEqual(actual, expected)

class TestColumnJoin(unittest.TestCase):
    

    def test_prefix(self):
        
        structure = ColumnJoin(
            vars=("z"),
            sources=[
                ColumnSource(FactIndex(Relation_PointsTo, [0,1]), Version.DELTA),
                ColumnSource(FactIndex("Assign", [1,0]), Version.FULL)
            ]
        )

        expected='SRDatalog::mir::dsl::column_join_h<'+str(MISSING_HANDLE)+', decltype(boost::hana::make_map())>("z"_v, SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index<PointsTo, 0, 1>().delta()), SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index<Assign, 1, 0>().full()))'
        actual = str(structure).replace(" ", "").replace("\n", "")  # remove whitespace formatting
        expected = expected.replace(" ", "").replace("\n", "")  # remove whitespace formatting

        self.assertEqual(actual, expected)
  # SRDatalog::mir::dsl::column_join_h<0, decltype(boost::hana::make_map())>("z"_v, SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index<PointsTo, 0, 1>().delta()), SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index<Assign, 1, 0>().full()))



class TestCartesianJoin(unittest.TestCase):
    
    def test_two_prefix(self):

      structure = CartesianJoin(
          vars=("y", "w"),
          sources=[
              ColumnSource(FactIndex("Load", [1,0]), Version.FULL, prefix=("x")),
              ColumnSource(FactIndex("PointsTo", [0,1]), Version.FULL, prefix=("z"))
          ]
      )
      expected='SRDatalog::mir::dsl::cartesian_join_h<'+str(MISSING_HANDLE)+', decltype(boost::hana::make_map(boost::hana::make_pair(boost::hana::type_c<SRDatalog::AST::Var<decltype("x"_s)>>, std::integer_sequence<std::size_t, 0>{}), boost::hana::make_pair(boost::hana::type_c<SRDatalog::AST::Var<decltype("z"_s)>>, std::integer_sequence<std::size_t, 1>{})))>(SRDatalog::mir::dsl::vars("y"_v, "w"_v), SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index<Load, 1, 0>().full(), "x"_v), SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index<PointsTo, 0, 1>().full(), "z"_v))'
      actual = str(structure).replace(" ", "").replace("\n", "")  # remove whitespace formatting
      expected = expected.replace(" ", "").replace("\n", "")  # remove whitespace formatting

      self.assertEqual(actual, expected)

class TestMirInstructions(unittest.TestCase):

  def setUp(self):


    self.structure_nohandles = [Block (
        
    dests=[IndexSpec(fact=FactIndex(Relation_PointsTo, [1, 0]), version=Version.FULL)],
        
      instructions=[
          ExecutePipeline(
            sources=[
                IndexSpec(fact=FactIndex(Relation_AddressOf, [0, 1]), version=Version.FULL)
            ],
            dests=[
                IndexSpec(fact=FactIndex(Relation_PointsTo, [1, 0]), version=Version.FULL)
            ],
            body=[
                InsertInto(
                    fact=FactIndex(Relation_PointsTo, [1, 0]),
                    version=Version.NEW,
                    terms=("y", "x")
                )
            ]
          ),

          RebuildIndex(fact=FactIndex(Relation_PointsTo, [1, 0]), version=Version.NEW),
          RebuildIndex(fact=FactIndex(Relation_PointsTo, [0, 1]), version=Version.NEW)
      ]
      
    )]




    
    self.structure_handles = [Block (
        
      dests=[IndexSpec(fact=FactIndex(Relation_PointsTo, [1, 0]), version=Version.FULL)],
        
      instructions=[
          ExecutePipeline(
            sources=[ IndexSpec(fact=FactIndex(Relation_AddressOf, [0, 1]), version=Version.FULL) ],
            dests=[ IndexSpec(fact=FactIndex(Relation_PointsTo, [1, 0]), version=Version.FULL) ],
            body=[
                Scan(
                    fact=FactIndex(Relation_AddressOf, [0, 1]),
                    version=Version.FULL,
                    vars=("y", "x"),
                    prefix=()             
                    ),
                InsertInto(
                    fact=FactIndex(Relation_PointsTo, [1, 0]),
                    version=Version.NEW,
                    terms=("y", "x")
                )
            ]
          ),

          RebuildIndex(fact=FactIndex(Relation_PointsTo, [1, 0]), version=Version.NEW),
          RebuildIndex(fact=FactIndex(Relation_PointsTo, [0, 1]), version=Version.NEW),
          CheckSize(fact_name=Relation_PointsTo, version=Version.NEW),
          ComputeDelta(fact=FactIndex(Relation_PointsTo, [1, 0]), version=Version.NEW),
          ClearRelation(fact_name=Relation_PointsTo, version=Version.NEW),
          MergeIndex(fact=FactIndex(Relation_PointsTo, [1, 0]), version=Version.FULL),
          MergeIndex(fact=FactIndex(Relation_PointsTo, [0, 1]), version=Version.FULL)
      ]
      
    )]
    self.program_nh = MirInstructions(structure=self.structure_nohandles)
    self.program_h = MirInstructions(structure=self.structure_handles)

  def test_program_assignment(self):
    

    self.assertEqual(self.structure_nohandles[0].instructions[0].body[0].program, self.program_nh)


  def test_program_nohandles(self):
    
    # the order of certain factors is different in this program vs what the nim generated
    #as_produced_by_nim ='fixpoint_plan<std::tuple<PointsTo>,std::tuple<SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,1,0>,0>>,std::tuple<SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,0,1>,2>,SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,1,0>,2>>>(execute(pipeline<std::tuple<SRDatalog::mir::IndexSpecT<AddressOf,std::integer_sequence<int,0,1>,0>>,std::tuple<SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,1,0>,0>>,std::tuple<AddressOf>>(insert_into<PointsTo,NEW_VER,decltype(SRDatalog::mir::dsl::index<PointsTo,1,0>().newt())>("y"_v,"x"_v))),rebuild_index(SRDatalog::mir::dsl::index<PointsTo,1,0>().newt()),rebuild_index(SRDatalog::mir::dsl::index<PointsTo,0,1>().newt()));'
    expected ='constexprautostep_0=fixpoint_plan <std::tuple <PointsTo>,std::tuple <SRDatalog::mir::IndexSpecT <PointsTo,std::integer_sequence <int,1,0>,0>>,std::tuple <SRDatalog::mir::IndexSpecT <PointsTo,std::integer_sequence <int,1,0>,2>,SRDatalog::mir::IndexSpecT <PointsTo,std::integer_sequence <int,0,1>,2>>>(execute(pipeline <std::tuple <SRDatalog::mir::IndexSpecT <AddressOf,std::integer_sequence <int,0,1>,0>>,std::tuple <SRDatalog::mir::IndexSpecT <PointsTo,std::integer_sequence <int,1,0>,0>>,std::tuple <AddressOf>>(insert_into <PointsTo,NEW_VER,decltype(SRDatalog::mir::dsl::index <PointsTo,1,0>().newt())>("y"_v,"x"_v))),rebuild_index(SRDatalog::mir::dsl::index <PointsTo,1,0>().newt()),rebuild_index(SRDatalog::mir::dsl::index <PointsTo,0,1>().newt()));usingstep_0_t=decltype(step_0);'
    actual = str(self.program_nh.cpp()).replace(" ", "").replace("\n", "")  # remove whitespace formatting

    actual = actual.replace(" ", "").replace("\n", "")  # remove whitespace formatting
    expected = expected.replace(" ", "").replace("\n", "")  # remove whitespace formatting

    self.assertEqual(expected, actual)
    

  def test_program_handles(self):
    
    # the order of certain factors is different in this program vs what the nim generated
    expected='constexprautostep_0=fixpoint_plan<std::tuple<PointsTo>,std::tuple<SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,1,0>,0>>,std::tuple<SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,1,0>,2>,SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,0,1>,2>>>(execute(pipeline<std::tuple<SRDatalog::mir::IndexSpecT<AddressOf,std::integer_sequence<int,0,1>,0>>,std::tuple<SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,1,0>,0>>,std::tuple<AddressOf>>(SRDatalog::mir::dsl::scan_h<0,decltype(boost::hana::make_map())>(SRDatalog::mir::dsl::vars("y"_v,"x"_v),SRDatalog::mir::dsl::index<AddressOf,0,1>().full()),insert_into<PointsTo,NEW_VER,decltype(SRDatalog::mir::dsl::index<PointsTo,1,0>().newt())>("y"_v,"x"_v))),rebuild_index(SRDatalog::mir::dsl::index<PointsTo,1,0>().newt()),rebuild_index(SRDatalog::mir::dsl::index<PointsTo,0,1>().newt()),check_size<PointsTo,NEW_VER>(),compute_delta(SRDatalog::mir::dsl::index<PointsTo,1,0>().newt()),clear_relation<PointsTo,NEW_VER>(),merge_index(SRDatalog::mir::dsl::index<PointsTo,1,0>().full()),merge_index(SRDatalog::mir::dsl::index<PointsTo,0,1>().full()));usingstep_0_t=decltype(step_0);'
    #self.program.cursors_allocated = 3
    actual = str(self.program_h.cpp()).replace(" ", "").replace("\n", "")  # remove whitespace formatting

    actual = actual.replace(" ", "").replace("\n", "")  # remove whitespace formatting
    expected = expected.replace(" ", "").replace("\n", "")  # remove whitespace formatting

    self.assertEqual(self.program_h.structure[0].instructions[0].body[0].program, self.program_h)
    self.assertEqual(expected, actual)
    

  def test_program_adjusted_handles(self):
    
    expected='constexprautostep_0=fixpoint_plan<std::tuple<PointsTo>,std::tuple<SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,1,0>,0>>,std::tuple<SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,1,0>,2>,SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,0,1>,2>>>(execute(pipeline<std::tuple<SRDatalog::mir::IndexSpecT<AddressOf,std::integer_sequence<int,0,1>,0>>,std::tuple<SRDatalog::mir::IndexSpecT<PointsTo,std::integer_sequence<int,1,0>,0>>,std::tuple<AddressOf>>(SRDatalog::mir::dsl::scan_h<3,decltype(boost::hana::make_map())>(SRDatalog::mir::dsl::vars("y"_v,"x"_v),SRDatalog::mir::dsl::index<AddressOf,0,1>().full()),insert_into<PointsTo,NEW_VER,decltype(SRDatalog::mir::dsl::index<PointsTo,1,0>().newt())>("y"_v,"x"_v))),rebuild_index(SRDatalog::mir::dsl::index<PointsTo,1,0>().newt()),rebuild_index(SRDatalog::mir::dsl::index<PointsTo,0,1>().newt()),check_size<PointsTo,NEW_VER>(),compute_delta(SRDatalog::mir::dsl::index<PointsTo,1,0>().newt()),clear_relation<PointsTo,NEW_VER>(),merge_index(SRDatalog::mir::dsl::index<PointsTo,1,0>().full()),merge_index(SRDatalog::mir::dsl::index<PointsTo,0,1>().full()));usingstep_0_t=decltype(step_0);'
    self.program_h.cursors_allocated = 3
    actual = str(self.program_h.cpp()).replace(" ", "").replace("\n", "")  # remove whitespace formatting

    actual = actual.replace(" ", "").replace("\n", "")  # remove whitespace formatting
    expected = expected.replace(" ", "").replace("\n", "")  # remove whitespace formatting

    self.assertEqual(self.program_h.structure[0].instructions[0].body[0].program, self.program_h)
    self.assertEqual(expected, actual)


class TestCppHook(unittest.TestCase):
    
    def test_andersen(self):

      structure = CppHook(
          code="""
            auto& points_to_delta = get_relation_by_schema<PointsTo, DELTA_VER>(db);
            auto& points_to_delta_idx = points_to_delta.get_index({{0,1}});
            auto& points_to_full = get_relation_by_schema<PointsTo, FULL_VER>(db);
            auto& points_to_full_idx = points_to_full.get_index({{0,1}});
            std::cout << "  PointsTo delta: " << points_to_delta_idx.root().degree()
                      << ", full: " << points_to_full_idx.root().degree() << std::endl;  
          """,
          label="Store"
          )
        
      expected='     inject_cpp_hook([] __host__ (auto& db) { /* Store */       auto& points_to_delta = get_relation_by_schema<PointsTo, DELTA_VER>(db);       auto& points_to_delta_idx = points_to_delta.get_index({{0,1}});       auto& points_to_full = get_relation_by_schema<PointsTo, FULL_VER>(db);       auto& points_to_full_idx = points_to_full.get_index({{0,1}});       std::cout << "  PointsTo delta: " << points_to_delta_idx.root().degree()                 << ", full: " << points_to_full_idx.root().degree() << std::endl;        })'
      expected = expected.replace(" ", "").replace("\n", "")  # remove whitespace formatting
      actual = str(structure).replace(" ", "").replace("\n", "")  # remove whitespace formatting


      self.assertEqual(expected, actual)

class TestAggregate(unittest.TestCase):
    
    def test_agg_noprogram(self):
        
        pass
    
class TestFilter(unittest.TestCase):
    def test_filterjoin_basic(self):
        
      structure = Filter(vars=('Ctr1', 'N1', 'Ctr2', 'N2'), code="return (Ctr1 * 10 + N1) > (Ctr2 * 10 + N2);")
      expected = 'filter(vars("Ctr1"_v, "N1"_v, "Ctr2"_v, "N2"_v), [](auto Ctr1, auto N1, auto Ctr2, auto N2) { return (Ctr1 * 10 + N1) > (Ctr2 * 10 + N2); })'
      expected = expected.replace(" ", "").replace("\n", "")  # remove whitespace formatting
      actual = str(structure).replace(" ", "").replace("\n", "")  # remove whitespace formatting

      self.assertEqual(expected, actual)
      

class TestNegate(unittest.TestCase):

  def test_negationtest_basic(self):


    structure = Negate(fact=FactIndex("HasNextSibling", [0,1]), version=Version.FULL, prefix=("StartCtr", "StartN"))
    expected='SRDatalog::mir::dsl::negation_h<'+str(MISSING_HANDLE)+', decltype(boost::hana::make_map())>(SRDatalog::mir::dsl::index<HasNextSibling, 0, 1>().full(), SRDatalog::mir::dsl::vars("StartCtr"_v, "StartN"_v))'
     
    expected = expected.replace(" ", "").replace("\n", "")  # remove whitespace formatting
    actual = str(structure).replace(" ", "").replace("\n", "")  # remove whitespace formatting

    self.assertEqual(expected, actual)


class TestGenerateSingleVhm(unittest.TestCase):
    """Test cases for generate_single_vhm function"""

    def test_empty_prefixes(self):
        """Test with no prefixes - should return empty map"""
        result = generate_single_vhm(prefixes=(), cursor=0)
        self.assertEqual(result, "decltype(boost::hana::make_map())")

    def test_single_prefix(self):
        """Test with a single prefix"""
        result = generate_single_vhm(prefixes=("x",), cursor=0)
        self.assertIn("boost::hana::make_map(", result)
        self.assertIn('"x"_s', result)
        self.assertIn("std::integer_sequence<std::size_t, 0>", result)

    def test_multiple_prefixes(self):
        """Test with multiple prefixes"""
        result = generate_single_vhm(prefixes=("x", "y", "z"), cursor=0)
        self.assertIn("boost::hana::make_map(", result)
        self.assertIn('"x"_s', result)
        self.assertIn('"y"_s', result)
        self.assertIn('"z"_s', result)
        self.assertIn("std::integer_sequence<std::size_t, 0>", result)

    def test_different_cursor_position(self):
        """Test with different cursor positions"""
        result = generate_single_vhm(prefixes=("a",), cursor=5)
        self.assertIn("std::integer_sequence<std::size_t, 5>", result)

    def test_empty_prefixes_non_zero_cursor(self):
        """Test empty prefixes with non-zero cursor - should still return empty map"""
        result = generate_single_vhm(prefixes=(), cursor=3)
        self.assertEqual(result, "decltype(boost::hana::make_map())")

    def test_prefix_formatting_contains_var_type(self):
        """Test that the result contains proper Var type wrapping"""
        result = generate_single_vhm(prefixes=("myvar",), cursor=2)
        self.assertIn("SRDatalog::AST::Var<decltype(", result)
        self.assertIn('"myvar"_s)>', result)

    def test_with_special_characters_in_prefix(self):
        """Test prefixes with underscores and numbers"""
        result = generate_single_vhm(prefixes=("var_1", "x2y"), cursor=0)
        self.assertIn('"var_1"_s', result)
        self.assertIn('"x2y"_s', result)

    def test_single_vhm_consistency(self):
        """Test that calling single_vhm multiple times with same input is consistent"""
        prefixes = ("a", "b", "c")
        result1 = generate_single_vhm(prefixes=prefixes, cursor=5)
        result2 = generate_single_vhm(prefixes=prefixes, cursor=5)
        self.assertEqual(result1, result2)


class TestGenerateMultiVhm(unittest.TestCase):
    """Test cases for generate_multi_vhm function"""

    def test_empty_sources(self):
        """Test with no sources - should return empty map"""
        result = generate_multi_vhm(sources=[], cursor=0)
        self.assertEqual(result, "decltype(boost::hana::make_map())")

    def test_single_source_single_prefix(self):
        """Test with a single source with a single prefix"""
        fact = FactIndex(name="TestFact", args=[0, 1])
        source = ColumnSource(fact=fact, version=Version.NEW, prefix=("x",))
        result = generate_multi_vhm(sources=[source], cursor=0)

        self.assertIn("boost::hana::make_map(", result)
        self.assertIn('"x"_s', result)
        self.assertIn("std::integer_sequence<std::size_t, 0>", result)

    def test_single_source_multiple_prefixes(self):
        """Test with a single source with multiple prefixes"""
        fact = FactIndex(name="TestFact", args=[0, 1])
        source = ColumnSource(fact=fact, version=Version.NEW, prefix=("x", "y"))
        result = generate_multi_vhm(sources=[source], cursor=0)

        self.assertIn("boost::hana::make_map(", result)
        self.assertIn('"x"_s', result)
        self.assertIn('"y"_s', result)

    def test_multiple_sources_different_prefixes(self):
        """Test with multiple sources, each with different prefixes"""
        fact1 = FactIndex(name="Fact1", args=[0])
        fact2 = FactIndex(name="Fact2", args=[1])
        source1 = ColumnSource(fact=fact1, version=Version.NEW, prefix=("x",))
        source2 = ColumnSource(fact=fact2, version=Version.NEW, prefix=("y",))

        result = generate_multi_vhm(sources=[source1, source2], cursor=0)

        self.assertIn("boost::hana::make_map(", result)
        self.assertIn('"x"_s', result)
        self.assertIn('"y"_s', result)
        # Different sources should get different cursor indices
        self.assertIn("std::integer_sequence<std::size_t, 0>", result)
        self.assertIn("std::integer_sequence<std::size_t, 1>", result)

    def test_multiple_sources_shared_prefix(self):
        """Test with multiple sources sharing the same prefix variable"""
        fact1 = FactIndex(name="Fact1", args=[0])
        fact2 = FactIndex(name="Fact2", args=[1])
        source1 = ColumnSource(fact=fact1, version=Version.NEW, prefix=("x",))
        source2 = ColumnSource(fact=fact2, version=Version.NEW, prefix=("x",))

        result = generate_multi_vhm(sources=[source1, source2], cursor=0)

        self.assertIn("boost::hana::make_map(", result)
        self.assertIn('"x"_s', result)
        # Same prefix should map to multiple cursors
        self.assertIn("std::integer_sequence<std::size_t, 0, 1>", result)

    def test_different_cursor_start_position(self):
        """Test that cursor start position is correctly applied"""
        fact = FactIndex(name="TestFact", args=[0])
        source = ColumnSource(fact=fact, version=Version.NEW, prefix=("x",))
        result = generate_multi_vhm(sources=[source], cursor=5)

        self.assertIn("std::integer_sequence<std::size_t, 5>", result)

    def test_multiple_sources_cursor_increment(self):
        """Test that each source increments the cursor correctly"""
        fact1 = FactIndex(name="Fact1", args=[0])
        fact2 = FactIndex(name="Fact2", args=[1])
        fact3 = FactIndex(name="Fact3", args=[2])
        source1 = ColumnSource(fact=fact1, version=Version.NEW, prefix=("a",))
        source2 = ColumnSource(fact=fact2, version=Version.NEW, prefix=("b",))
        source3 = ColumnSource(fact=fact3, version=Version.NEW, prefix=("c",))

        result = generate_multi_vhm(sources=[source1, source2, source3], cursor=10)

        self.assertIn("std::integer_sequence<std::size_t, 10>", result)
        self.assertIn("std::integer_sequence<std::size_t, 11>", result)
        self.assertIn("std::integer_sequence<std::size_t, 12>", result)

    def test_source_with_empty_prefix(self):
        """Test source with empty prefix is gracefully handled"""
        fact = FactIndex(name="TestFact", args=[0])
        source = ColumnSource(fact=fact, version=Version.NEW, prefix=())
        result = generate_multi_vhm(sources=[source], cursor=0)

        # Should still return empty map if no prefixes
        self.assertEqual(result, "decltype(boost::hana::make_map())")

    def test_multi_vhm_with_many_sources(self):
        """Test multi_vhm with many sources"""
        sources = []
        for i in range(10):
            fact = FactIndex(name=f"Fact{i}", args=[i])
            source = ColumnSource(fact=fact, version=Version.NEW, prefix=(f"var{i}",))
            sources.append(source)

        result = generate_multi_vhm(sources=sources, cursor=0)

        # Check that all variables appear
        for i in range(10):
            self.assertIn(f'"var{i}"_s', result)

    def test_multi_vhm_complex_prefix_names(self):
        """Test multi_vhm with complex variable names"""
        fact = FactIndex(name="TestFact", args=[0])
        source = ColumnSource(fact=fact, version=Version.NEW, prefix=("myVar_123", "anotherVar_XYZ"))
        result = generate_multi_vhm(sources=[source], cursor=0)

        self.assertIn('"myVar_123"_s', result)
        self.assertIn('"anotherVar_XYZ"_s', result)

    def test_multi_vhm_consistency(self):
        """Test that calling multi_vhm multiple times with same input is consistent"""
        fact = FactIndex(name="TestFact", args=[0])
        source = ColumnSource(fact=fact, version=Version.NEW, prefix=("x", "y"))
        sources = [source]

        result1 = generate_multi_vhm(sources=sources, cursor=2)
        result2 = generate_multi_vhm(sources=sources, cursor=2)
        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()