import unittest

# Add the parent directory to the path so we can import mir_commands
from srdatalog.mir.commands import *
from srdatalog.mir.schema import *
from srdatalog.srdatalog_program import SRDatalogProgram

RREL = "RRel"
SREL = "SRel"
TREL = "TRel"
ZREL = "ZRel"


def print_diff(expected, actual):
  # for debugging diff
  expected = expected.replace("<", "\n<")
  actual = actual.replace("<", "\n<")
  print("expected: \n\n\n" + expected)
  print("actual:\n\n\n" + actual + "\n\n\n")


class TestFullProgram(unittest.TestCase):
  def setUp(self):
    dest_1_index_spec = IndexSpec(FactIndex(ZREL, [0, 1, 2]), Version.FULL)
    self.structure = MirInstructions(
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
                IndexSpec(FactIndex(TREL, [1, 0]), Version.FULL),
              ],
              dests=[dest_1_index_spec],
              body=[
                ColumnJoin(
                  vars=("x"),
                  sources=[
                    ColumnSource(FactIndex(RREL, [0, 1]), Version.FULL),
                    ColumnSource(FactIndex(TREL, [1, 0]), Version.FULL),
                  ],
                ),
                ColumnJoin(
                  vars=("y"),
                  sources=[
                    ColumnSource(FactIndex(RREL, [0, 1]), Version.FULL, prefix=('x')),
                    ColumnSource(FactIndex(SREL, [0, 1]), Version.FULL),
                  ],
                ),
                ColumnJoin(
                  vars=("z"),
                  sources=[
                    ColumnSource(FactIndex(SREL, [0, 1]), Version.FULL, prefix=('y')),
                    ColumnSource(FactIndex(TREL, [1, 0]), Version.FULL, prefix=('x')),
                  ],
                ),
                InsertInto(
                  fact=FactIndex(ZREL, [0, 1, 2]), version=Version.NEW, terms=("x", "y", "z")
                ),
              ],
            ),
            RebuildIndex(FactIndex(ZREL, [0, 1, 2]), Version.NEW),
            CheckSize(ZREL, Version.NEW),
            ComputeDelta(FactIndex(ZREL, [0, 1, 2]), Version.NEW),  # **
            ClearRelation(ZREL, Version.NEW),
            MergeIndex(FactIndex(ZREL, [0, 1, 2]), Version.FULL),  # **
          ],
          recursive=False,
        )
      ]
    )
    self.database = SchemaDefinition(
      facts=[
        FactDefinition(RREL, [int, int]),
        FactDefinition(SREL, [int, int, int]),
        FactDefinition(TREL, [int, int, int]),
        FactDefinition(ZREL, [int, int, int]),
      ]
    )
    self.program = SRDatalogProgram(
      name="Triangle", database=self.database, instructions=self.structure
    )

  def test_generate_triangle_fixpointplan_from_mir_program(self):
    # chosen because it's the shortest and seems the least labor intensive lol

    actual = self.structure.cpp()
    # print(actual)
    actual = actual.replace(" ", "").replace("\n", "")
    # expected_nim = 'namespace TrianglePlan_Plans {   constexpr auto step_0 = fixpoint_plan<std::tuple<ZRel>, std::tuple<SRDatalog::mir::IndexSpecT<ZRel, std::integer_sequence<int, 0, 1, 2>, 0>>, std::tuple<SRDatalog::mir::IndexSpecT<TRel, std::integer_sequence<int, 1, 0>, 0>, SRDatalog::mir::IndexSpecT<SRel, std::integer_sequence<int, 0, 1>, 0>, SRDatalog::mir::IndexSpecT<ZRel, std::integer_sequence<int, 0, 1, 2>, 2>, SRDatalog::mir::IndexSpecT<RRel, std::integer_sequence<int, 0, 1>, 0>>>(     execute(pipeline<std::tuple<SRDatalog::mir::IndexSpecT<RRel, std::integer_sequence<int, 0, 1>, 0>, SRDatalog::mir::IndexSpecT<TRel, std::integer_sequence<int, 1, 0>, 0>, SRDatalog::mir::IndexSpecT<RRel, std::integer_sequence<int, 0, 1>, 0>, SRDatalog::mir::IndexSpecT<SRel, std::integer_sequence<int, 0, 1>, 0>, SRDatalog::mir::IndexSpecT<SRel, std::integer_sequence<int, 0, 1>, 0>, SRDatalog::mir::IndexSpecT<TRel, std::integer_sequence<int, 1, 0>, 0>>, std::tuple<SRDatalog::mir::IndexSpecT<ZRel, std::integer_sequence<int, 0, 1, 2>, 0>>, std::tuple<SRel, TRel, RRel>>(     SRDatalog::mir::dsl::column_join_h<0, decltype(boost::hana::make_map())>("x"_v, SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index<RRel, 0, 1>().full()), SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index<TRel, 1, 0>().full())),     SRDatalog::mir::dsl::column_join_h<2, decltype(boost::hana::make_map(boost::hana::make_pair(boost::hana::type_c<SRDatalog::AST::Var<decltype("x"_s)>>, std::integer_sequence<std::size_t, 2>{})))>("y"_v, SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index<RRel, 0, 1>().full(), "x"_v), SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index<SRel, 0, 1>().full())),     SRDatalog::mir::dsl::column_join_h<4, decltype(boost::hana::make_map(boost::hana::make_pair(boost::hana::type_c<SRDatalog::AST::Var<decltype("x"_s)>>, std::integer_sequence<std::size_t, 5>{}), boost::hana::make_pair(boost::hana::type_c<SRDatalog::AST::Var<decltype("y"_s)>>, std::integer_sequence<std::size_t, 4>{})))>("z"_v, SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index<SRel, 0, 1>().full(), "y"_v), SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index<TRel, 1, 0>().full(), "x"_v)),     insert_into<ZRel, NEW_VER, decltype(SRDatalog::mir::dsl::index<ZRel, 0, 1, 2>().newt())>("x"_v, "y"_v, "z"_v)   )),     rebuild_index(SRDatalog::mir::dsl::index<ZRel, 0, 1, 2>().newt()),     check_size<ZRel, NEW_VER>(),     compute_delta(SRDatalog::mir::dsl::index<ZRel, 0, 1, 2>().newt()),     clear_relation<ZRel, NEW_VER>(),     merge_index(SRDatalog::mir::dsl::index<ZRel, 0, 1, 2>().full())   );   using step_0_t = decltype(step_0); }'
    expected = 'constexprautostep_0=fixpoint_plan <std::tuple <ZRel>,std::tuple <SRDatalog::mir::IndexSpecT <ZRel,std::integer_sequence <int,0,1,2>,0>>,std::tuple <SRDatalog::mir::IndexSpecT <RRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <TRel,std::integer_sequence <int,1,0>,0>,SRDatalog::mir::IndexSpecT <SRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <ZRel,std::integer_sequence <int,0,1,2>,2>>>(execute(pipeline <std::tuple <SRDatalog::mir::IndexSpecT <RRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <TRel,std::integer_sequence <int,1,0>,0>,SRDatalog::mir::IndexSpecT <RRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <SRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <SRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <TRel,std::integer_sequence <int,1,0>,0>>,std::tuple <SRDatalog::mir::IndexSpecT <ZRel,std::integer_sequence <int,0,1,2>,0>>,std::tuple <RRel,TRel,SRel>>(SRDatalog::mir::dsl::column_join_h <0,decltype(boost::hana::make_map())>("x"_v,SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <RRel,0,1>().full()),SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <TRel,1,0>().full())),SRDatalog::mir::dsl::column_join_h <2,decltype(boost::hana::make_map(boost::hana::make_pair(boost::hana::type_c <SRDatalog::AST::Var <decltype("x"_s)>>,std::integer_sequence <std::size_t,2>{})))>("y"_v,SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <RRel,0,1>().full(),"x"_v),SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <SRel,0,1>().full())),SRDatalog::mir::dsl::column_join_h <4,decltype(boost::hana::make_map(boost::hana::make_pair(boost::hana::type_c <SRDatalog::AST::Var <decltype("y"_s)>>,std::integer_sequence <std::size_t,4>{}),boost::hana::make_pair(boost::hana::type_c <SRDatalog::AST::Var <decltype("x"_s)>>,std::integer_sequence <std::size_t,5>{})))>("z"_v,SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <SRel,0,1>().full(),"y"_v),SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <TRel,1,0>().full(),"x"_v)),insert_into <ZRel,NEW_VER,decltype(SRDatalog::mir::dsl::index <ZRel,0,1,2>().newt())>("x"_v,"y"_v,"z"_v))),rebuild_index(SRDatalog::mir::dsl::index <ZRel,0,1,2>().newt()),check_size <ZRel,NEW_VER>(),compute_delta(SRDatalog::mir::dsl::index <ZRel,0,1,2>().newt()),clear_relation <ZRel,NEW_VER>(),merge_index(SRDatalog::mir::dsl::index <ZRel,0,1,2>().full()));usingstep_0_t=decltype(step_0);'
    expected = expected.replace(" ", "").replace("\n", "")

    # print_diff(expected=expected, actual=actual)

    self.assertEqual(expected, actual)

  def test_generate_triangle_fixpointplan(self):
    actual = self.program._generate_fixpoint_plans()
    # print(actual)
    actual = actual.replace(" ", "").replace("\n", "")
    expected = 'namespace Triangle_Plans {constexprautostep_0=fixpoint_plan <std::tuple <ZRel>,std::tuple <SRDatalog::mir::IndexSpecT <ZRel,std::integer_sequence <int,0,1,2>,0>>,std::tuple <SRDatalog::mir::IndexSpecT <RRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <TRel,std::integer_sequence <int,1,0>,0>,SRDatalog::mir::IndexSpecT <SRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <ZRel,std::integer_sequence <int,0,1,2>,2>>>(execute(pipeline <std::tuple <SRDatalog::mir::IndexSpecT <RRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <TRel,std::integer_sequence <int,1,0>,0>,SRDatalog::mir::IndexSpecT <RRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <SRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <SRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <TRel,std::integer_sequence <int,1,0>,0>>,std::tuple <SRDatalog::mir::IndexSpecT <ZRel,std::integer_sequence <int,0,1,2>,0>>,std::tuple <RRel,TRel,SRel>>(SRDatalog::mir::dsl::column_join_h <0,decltype(boost::hana::make_map())>("x"_v,SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <RRel,0,1>().full()),SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <TRel,1,0>().full())),SRDatalog::mir::dsl::column_join_h <2,decltype(boost::hana::make_map(boost::hana::make_pair(boost::hana::type_c <SRDatalog::AST::Var <decltype("x"_s)>>,std::integer_sequence <std::size_t,2>{})))>("y"_v,SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <RRel,0,1>().full(),"x"_v),SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <SRel,0,1>().full())),SRDatalog::mir::dsl::column_join_h <4,decltype(boost::hana::make_map(boost::hana::make_pair(boost::hana::type_c <SRDatalog::AST::Var <decltype("y"_s)>>,std::integer_sequence <std::size_t,4>{}),boost::hana::make_pair(boost::hana::type_c <SRDatalog::AST::Var <decltype("x"_s)>>,std::integer_sequence <std::size_t,5>{})))>("z"_v,SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <SRel,0,1>().full(),"y"_v),SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <TRel,1,0>().full(),"x"_v)),insert_into <ZRel,NEW_VER,decltype(SRDatalog::mir::dsl::index <ZRel,0,1,2>().newt())>("x"_v,"y"_v,"z"_v))),rebuild_index(SRDatalog::mir::dsl::index <ZRel,0,1,2>().newt()),check_size <ZRel,NEW_VER>(),compute_delta(SRDatalog::mir::dsl::index <ZRel,0,1,2>().newt()),clear_relation <ZRel,NEW_VER>(),merge_index(SRDatalog::mir::dsl::index <ZRel,0,1,2>().full()));usingstep_0_t=decltype(step_0);}'
    expected = expected.replace(" ", "").replace("\n", "")

    self.assertEqual(expected, actual)

  def test_generate_triangle_schema(self):
    actual = self.program._generate_schema()
    # print(actual)
    actual = actual.replace(" ", "").replace("\n", "")
    expected = '  using RRel = AST::RelationSchema<decltype("RRel"_s), BooleanSR, std::tuple<int, int>>; using SRel = AST::RelationSchema<decltype("SRel"_s), BooleanSR, std::tuple<int, int, int>>; using TRel = AST::RelationSchema<decltype("TRel"_s), BooleanSR, std::tuple<int, int, int>>; using ZRel = AST::RelationSchema<decltype("ZRel"_s), BooleanSR, std::tuple<int, int, int>>; using TriangleDB = AST::Database<RRel, SRel, TRel, ZRel>; using TrianglePlan_DB = AST::Database<RRel, SRel, TRel, ZRel>; using namespace SRDatalog::mir::dsl;'
    expected = expected.replace(" ", "").replace("\n", "")

    self.assertEqual(expected, actual)

  def test_generate_triangle_runner(self):
    actual = self.program._generate_fixpoint_runner()
    # print(actual)
    actual = actual.replace(" ", "").replace("\n", "")
    expected = 'struct Triangle_Runner {   template <typename DB>   static void load_data(DB& db, std::string root_dir) {   }    template <typename DB>   static void step_0(DB& db, std::size_t max_iterations) {     SRDatalog::GPU::execute_gpu_mir_query<typename Triangle_Plans::step_0_t::type>(db, 1);   }   template <typename DB>   static void run(DB& db, std::size_t max_iterations = std::numeric_limits<int>::max()) {     auto step_0_start = std::chrono::high_resolution_clock::now();     step_0(db, max_iterations);     auto step_0_end = std::chrono::high_resolution_clock::now();     auto step_0_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_0_end - step_0_start);     std::cout << "[Step 0 (simple)] " << "Relations: ZRel" << " completed in " << step_0_duration.count() << " ms" << std::endl;   } };'
    expected = expected.replace(" ", "").replace("\n", "")

    # print_diff(expected=expected, actual=actual)
    self.assertEqual(expected, actual)

  def test_generate_program(self):
    actual = self.program.generate()
    expected = '#include "srdatalog.h" #include "runtime/io.h" using namespace SRDatalog; using namespace SRDatalog::AST::Literals; using string = std::string; using Arena = boost::container::pmr::monotonic_buffer_resource;  #include "gpu/runtime/query.h" #include "gpu/gpu_api.h" #include "gpu/init.h" #include "gpu/runtime/gpu_fixpoint_executor.h" #include <chrono>usingRRel=AST::RelationSchema <decltype("RRel"_s),BooleanSR,std::tuple <int,int>>;usingSRel=AST::RelationSchema <decltype("SRel"_s),BooleanSR,std::tuple <int,int,int>>;usingTRel=AST::RelationSchema <decltype("TRel"_s),BooleanSR,std::tuple <int,int,int>>;usingZRel=AST::RelationSchema <decltype("ZRel"_s),BooleanSR,std::tuple <int,int,int>>;usingTriangleDB=AST::Database <RRel,SRel,TRel,ZRel>;usingTrianglePlan_DB=AST::Database <RRel,SRel,TRel,ZRel>;usingnamespaceSRDatalog::mir::dsl;namespaceTriangle_Plans{constexprautostep_0=fixpoint_plan <std::tuple <ZRel>,std::tuple <SRDatalog::mir::IndexSpecT <ZRel,std::integer_sequence <int,0,1,2>,0>>,std::tuple <SRDatalog::mir::IndexSpecT <RRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <TRel,std::integer_sequence <int,1,0>,0>,SRDatalog::mir::IndexSpecT <SRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <ZRel,std::integer_sequence <int,0,1,2>,2>>>(execute(pipeline <std::tuple <SRDatalog::mir::IndexSpecT <RRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <TRel,std::integer_sequence <int,1,0>,0>,SRDatalog::mir::IndexSpecT <RRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <SRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <SRel,std::integer_sequence <int,0,1>,0>,SRDatalog::mir::IndexSpecT <TRel,std::integer_sequence <int,1,0>,0>>,std::tuple <SRDatalog::mir::IndexSpecT <ZRel,std::integer_sequence <int,0,1,2>,0>>,std::tuple <RRel,TRel,SRel>>(SRDatalog::mir::dsl::column_join_h <0,decltype(boost::hana::make_map())>("x"_v,SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <RRel,0,1>().full()),SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <TRel,1,0>().full())),SRDatalog::mir::dsl::column_join_h <2,decltype(boost::hana::make_map(boost::hana::make_pair(boost::hana::type_c <SRDatalog::AST::Var <decltype("x"_s)>>,std::integer_sequence <std::size_t,2>{})))>("y"_v,SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <RRel,0,1>().full(),"x"_v),SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <SRel,0,1>().full())),SRDatalog::mir::dsl::column_join_h <4,decltype(boost::hana::make_map(boost::hana::make_pair(boost::hana::type_c <SRDatalog::AST::Var <decltype("y"_s)>>,std::integer_sequence <std::size_t,4>{}),boost::hana::make_pair(boost::hana::type_c <SRDatalog::AST::Var <decltype("x"_s)>>,std::integer_sequence <std::size_t,5>{})))>("z"_v,SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <SRel,0,1>().full(),"y"_v),SRDatalog::mir::dsl::column_source(SRDatalog::mir::dsl::index <TRel,1,0>().full(),"x"_v)),insert_into <ZRel,NEW_VER,decltype(SRDatalog::mir::dsl::index <ZRel,0,1,2>().newt())>("x"_v,"y"_v,"z"_v))),rebuild_index(SRDatalog::mir::dsl::index <ZRel,0,1,2>().newt()),check_size <ZRel,NEW_VER>(),compute_delta(SRDatalog::mir::dsl::index <ZRel,0,1,2>().newt()),clear_relation <ZRel,NEW_VER>(),merge_index(SRDatalog::mir::dsl::index <ZRel,0,1,2>().full()));usingstep_0_t=decltype(step_0);}structTriangle_Runner{template <typename DB>staticvoidload_data(DB&db,std::stringroot_dir){}template <typename DB>staticvoidstep_0(DB&db,std::size_tmax_iterations){SRDatalog::GPU::execute_gpu_mir_query <typenameTriangle_Plans::step_0_t::type>(db,1);}template <typename DB>staticvoidrun(DB&db,std::size_tmax_iterations=std::numeric_limits <int>::max()){autostep_0_start=std::chrono::high_resolution_clock::now();step_0(db,max_iterations);autostep_0_end=std::chrono::high_resolution_clock::now();autostep_0_duration=std::chrono::duration_cast <std::chrono::milliseconds>(step_0_end-step_0_start);std::cout < <"[Step0(simple)]" < <"Relations:ZRel" < <"completedin" < <step_0_duration.count() < <"ms" < <std::endl;}};intmain(intargc,char**args,char**env){}  extern "C" {    using DBHandle = void *;    DBHandle db_new() {     return new SemiNaiveDatabase<TriangleDB>();   }    void db_free(DBHandle h) {     // must be called by program whenever the database is no longer needed to avoid memory leaks     auto ptr = static_cast<SemiNaiveDatabase<TriangleDB> *>(h);     delete ptr;   }    void load(DBHandle h, const char *root_dir) {     auto &db = *static_cast<SemiNaiveDatabase<TriangleDB> *>(h);     Triangle_Runner::load_data(db, std::string(root_dir));   }    void run(DBHandle h, size_t max_iters) {     auto &db = *static_cast<SemiNaiveDatabase<TriangleDB> *>(h);     Triangle_Runner::run(db, max_iters);   }    } // extern "C"'
    actual = actual.replace(" ", "").replace("\n", "")
    expected = expected.replace(" ", "").replace("\n", "")

    print_diff(expected=expected, actual=actual)
    self.assertEqual(expected, actual)
