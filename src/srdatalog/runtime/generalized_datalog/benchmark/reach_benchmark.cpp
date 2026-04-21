#include "srdatalog.h"
#include "test_util.h"
#include <chrono>
#include <iostream>

// .in
// .decl Source(id: number)
// .input Source.csv

// .decl Arc(x: number, y: number)
// .input Arc.csv

// .printsize
// .decl Reach(id: number)

// .rule
// Reach(y) :- Source(y).
// Reach(y) :- Reach(x), Arc(x, y).

int main(int argc, char** argv) {
  namespace sd = SRDatalog;
  namespace ast = sd::AST;

  using ast::database;
  using ast::fixpoint;
  using ast::non_iterative;
  using ast::rel;
  using ast::relation;
  using ast::SemiNaiveDatabase;
  using ast::operator""_v;
  using ast::Literals::operator""_s;

  using ::BooleanSR;
  using sd::execute_query;
  using sd::get_relation_by_schema;
  using sd::load_from_file;
  using sd::TestUtil::find_project_root;

  using SR = BooleanSR;
  using Arena = boost::container::pmr::monotonic_buffer_resource;
  Arena arena_full;
  Arena arena_delta;
  Arena arena_newt;

  // using SourceSchema = RelationSchema<decltype("Source"_s), SR, std::tuple<int>>;
  // using ArcSchema = RelationSchema<decltype("Arc"_s), SR, std::tuple<int, int>>;
  // using ReachSchema = RelationSchema<decltype("Reach"_s), SR, std::tuple<int>>;
  // using ReachBlueprint = Database<SourceSchema, ArcSchema, ReachSchema>;

  // using x_ = Var<decltype("x"_s)>;
  // using y_ = Var<decltype("y"_s)>;

  // using ReachRuleInit =
  //     Rule<std::tuple<Clause<ReachSchema, NEW_VER, y_>>,
  //          std::tuple<Clause<SourceSchema, FULL_VER, y_>, Clause<ReachSchema, FULL_VER, x_, y_>>,
  //          JoinPlan<std::tuple<x_, y_>>>;
  // using ReachBaseSet = NonIterativeRuleSets<ReachRuleInit>;
  // using ReachTransitiveRule =
  //     Rule<std::tuple<Clause<ReachSchema, NEW_VER, y_>>,
  //          std::tuple<Clause<ReachSchema, DELTA_VER, x_, y_>, Clause<ArcSchema, FULL_VER, x_,
  //          y_>>, JoinPlan<std::tuple<x_, y_>>>;
  // using ReachFixpoint = Fixpoint<ReachTransitiveRule>;

  // SemiNaiveDatabase<ReachBlueprint> db(&arena_full, &arena_delta, &arena_newt);

  // // read first argument as folder path
  // std::string folder_path = argv[1];
  // std::string source_file_path = folder_path + "/Source.csv";
  // std::string arc_file_path = folder_path + "/Arc.csv";

  // load_from_file<SourceSchema>(db, source_file_path);
  // load_from_file<ArcSchema>(db, arc_file_path);
  // auto start_time = std::chrono::high_resolution_clock::now();
  // LOG_INFO << "Starting base set execution";
  // execute_query<ReachBaseSet>(db);
  // LOG_INFO << "Base set execution complete";
  // execute_query<ReachFixpoint>(db);
  // auto end_time = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
  // std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
}
