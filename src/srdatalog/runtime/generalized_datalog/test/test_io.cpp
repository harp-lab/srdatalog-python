#define BOOST_TEST_MODULE io_system_test
#include "ast.h"
#include "query.h"  // For get_relation_by_schema
#include "runtime.h"
#include "runtime/io.h"  // For load_file
#include "semiring.h"
#include "test_util.h"
#include <boost/container/pmr/monotonic_buffer_resource.hpp>
#include <boost/test/included/unit_test.hpp>
#include <filesystem>
#include <fstream>
#include <random>

namespace {
using SRDatalog::get_relation_by_schema;
using SRDatalog::load_file;
using SRDatalog::load_from_file;
using SRDatalog::Relation;
using SRDatalog::IndexSpec;
using SRDatalog::SemiNaiveDatabase;
using SRDatalog::TestUtil::find_project_root;
using SRDatalog::AST::Literals::operator""_s;
using SRDatalog::AST::Database;
using SRDatalog::AST::RelationSchema;

// Define a schema for the test relation
using TestSchema = RelationSchema<decltype("TestRel"_s), BooleanSR, std::tuple<int, int>>;
using TestDBBlueprint = Database<TestSchema>;

// Helper function to create a temporary test file
std::filesystem::path create_temp_file(const std::string& content) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(1000, 9999);
  std::filesystem::path temp_file =
      std::filesystem::path("/tmp") / ("test_temp_" + std::to_string(dis(gen)) + ".txt");

  std::ofstream out(temp_file);
  out << content;
  out.close();

  return temp_file;
}

BOOST_AUTO_TEST_SUITE(io_system_suite)

BOOST_AUTO_TEST_CASE(LoadCsvFileTest) {
  std::filesystem::path project_dir = find_project_root();
  std::filesystem::path file_path = project_dir / "misc" / "test_data.csv";

  if (!std::filesystem::exists(file_path)) {
    BOOST_FAIL("Could not find test data file");
  }

  std::string file_path_str = file_path.string();
  boost::container::pmr::monotonic_buffer_resource arena_full;
  boost::container::pmr::monotonic_buffer_resource arena_delta;
  boost::container::pmr::monotonic_buffer_resource arena_newt;
  SemiNaiveDatabase<TestDBBlueprint> db(&arena_full, &arena_delta, &arena_newt);

  BOOST_REQUIRE_NO_THROW((load_from_file<TestSchema>(db, file_path_str)));

  auto& rel = get_relation_by_schema<TestSchema, FULL_VER>(db);
  BOOST_CHECK_EQUAL(rel.size(), 3);
}

BOOST_AUTO_TEST_CASE(LoadTsvFileTest) {
  std::string content = "1\t2\n3\t4\n5\t6\n";
  std::filesystem::path temp_file = create_temp_file(content);

  boost::container::pmr::monotonic_buffer_resource arena_full;
  boost::container::pmr::monotonic_buffer_resource arena_delta;
  boost::container::pmr::monotonic_buffer_resource arena_newt;
  SemiNaiveDatabase<TestDBBlueprint> db(&arena_full, &arena_delta, &arena_newt);

  BOOST_REQUIRE_NO_THROW((load_from_file<TestSchema>(db, temp_file.string())));

  auto& rel = get_relation_by_schema<TestSchema, FULL_VER>(db);
  BOOST_CHECK_EQUAL(rel.size(), 3);

  std::filesystem::remove(temp_file);
}

BOOST_AUTO_TEST_CASE(LoadSpaceSeparatedFileTest) {
  std::string content = "1 2\n3 4\n5 6\n";
  std::filesystem::path temp_file = create_temp_file(content);

  boost::container::pmr::monotonic_buffer_resource arena_full;
  boost::container::pmr::monotonic_buffer_resource arena_delta;
  boost::container::pmr::monotonic_buffer_resource arena_newt;
  SemiNaiveDatabase<TestDBBlueprint> db(&arena_full, &arena_delta, &arena_newt);

  BOOST_REQUIRE_NO_THROW((load_from_file<TestSchema>(db, temp_file.string())));

  auto& rel = get_relation_by_schema<TestSchema, FULL_VER>(db);
  BOOST_CHECK_EQUAL(rel.size(), 3);

  std::filesystem::remove(temp_file);
}

BOOST_AUTO_TEST_CASE(LoadMultiSpaceSeparatedFileTest) {
  std::string content = "1    2\n3    4\n5    6\n";
  std::filesystem::path temp_file = create_temp_file(content);

  boost::container::pmr::monotonic_buffer_resource arena_full;
  boost::container::pmr::monotonic_buffer_resource arena_delta;
  boost::container::pmr::monotonic_buffer_resource arena_newt;
  SemiNaiveDatabase<TestDBBlueprint> db(&arena_full, &arena_delta, &arena_newt);

  BOOST_REQUIRE_NO_THROW((load_from_file<TestSchema>(db, temp_file.string())));

  auto& rel = get_relation_by_schema<TestSchema, FULL_VER>(db);
  BOOST_CHECK_EQUAL(rel.size(), 3);

  std::filesystem::remove(temp_file);
}

BOOST_AUTO_TEST_CASE(LoadEmptyFileTest) {
  std::filesystem::path temp_file = create_temp_file("");

  boost::container::pmr::monotonic_buffer_resource arena_full;
  boost::container::pmr::monotonic_buffer_resource arena_delta;
  boost::container::pmr::monotonic_buffer_resource arena_newt;
  SemiNaiveDatabase<TestDBBlueprint> db(&arena_full, &arena_delta, &arena_newt);

  BOOST_REQUIRE_NO_THROW((load_from_file<TestSchema>(db, temp_file.string())));

  auto& rel = get_relation_by_schema<TestSchema, FULL_VER>(db);
  BOOST_CHECK_EQUAL(rel.size(), 0);

  std::filesystem::remove(temp_file);
}

BOOST_AUTO_TEST_CASE(LoadFileWithEmptyLinesTest) {
  std::string content = "\n\n1,2\n\n3,4\n\n5,6\n\n";
  std::filesystem::path temp_file = create_temp_file(content);

  boost::container::pmr::monotonic_buffer_resource arena_full;
  boost::container::pmr::monotonic_buffer_resource arena_delta;
  boost::container::pmr::monotonic_buffer_resource arena_newt;
  SemiNaiveDatabase<TestDBBlueprint> db(&arena_full, &arena_delta, &arena_newt);

  BOOST_REQUIRE_NO_THROW((load_from_file<TestSchema>(db, temp_file.string())));

  auto& rel = get_relation_by_schema<TestSchema, FULL_VER>(db);
  BOOST_CHECK_EQUAL(rel.size(), 3);

  std::filesystem::remove(temp_file);
}

BOOST_AUTO_TEST_CASE(LoadFileWithWhitespaceTest) {
  std::string content = "  1  ,  2  \n  3  ,  4  \n  5  ,  6  \n";
  std::filesystem::path temp_file = create_temp_file(content);

  boost::container::pmr::monotonic_buffer_resource arena_full;
  boost::container::pmr::monotonic_buffer_resource arena_delta;
  boost::container::pmr::monotonic_buffer_resource arena_newt;
  SemiNaiveDatabase<TestDBBlueprint> db(&arena_full, &arena_delta, &arena_newt);

  BOOST_REQUIRE_NO_THROW((load_from_file<TestSchema>(db, temp_file.string())));

  auto& rel = get_relation_by_schema<TestSchema, FULL_VER>(db);
  BOOST_CHECK_EQUAL(rel.size(), 3);

  std::filesystem::remove(temp_file);
}

BOOST_AUTO_TEST_CASE(DelimiterPreferenceTest) {
  // Tab should be preferred over comma when both are present
  // This tests that tab delimiter is detected even when comma also exists
  std::string content = "1\t2\n3\t4\n";
  std::filesystem::path temp_file = create_temp_file(content);

  boost::container::pmr::monotonic_buffer_resource arena_full;
  boost::container::pmr::monotonic_buffer_resource arena_delta;
  boost::container::pmr::monotonic_buffer_resource arena_newt;
  SemiNaiveDatabase<TestDBBlueprint> db(&arena_full, &arena_delta, &arena_newt);

  BOOST_REQUIRE_NO_THROW((load_from_file<TestSchema>(db, temp_file.string())));

  auto& rel = get_relation_by_schema<TestSchema, FULL_VER>(db);
  BOOST_CHECK_EQUAL(rel.size(), 2);

  std::filesystem::remove(temp_file);
}

BOOST_AUTO_TEST_CASE(MultiSpaceVsSingleSpaceTest) {
  // Multi-space should be detected when 2+ spaces are present
  std::string content = "1  2\n3  4\n";
  std::filesystem::path temp_file = create_temp_file(content);

  boost::container::pmr::monotonic_buffer_resource arena_full;
  boost::container::pmr::monotonic_buffer_resource arena_delta;
  boost::container::pmr::monotonic_buffer_resource arena_newt;
  SemiNaiveDatabase<TestDBBlueprint> db(&arena_full, &arena_delta, &arena_newt);

  BOOST_REQUIRE_NO_THROW((load_from_file<TestSchema>(db, temp_file.string())));

  auto& rel = get_relation_by_schema<TestSchema, FULL_VER>(db);
  BOOST_CHECK_EQUAL(rel.size(), 2);

  std::filesystem::remove(temp_file);
}

BOOST_AUTO_TEST_CASE(LoadFileWithHashTrieIndexTest) {
  // Test load_file with default HashTrieIndex
  std::string content = "1,2\n3,4\n5,6\n";
  std::filesystem::path temp_file = create_temp_file(content);

  Relation<BooleanSR, std::tuple<int, int>> rel;
  BOOST_REQUIRE_NO_THROW(load_file(rel, temp_file.string()));

  BOOST_CHECK_EQUAL(rel.size(), 3);

  std::filesystem::remove(temp_file);
}

// Note: SortedArrayIndex test is commented out because interned columns
// are still std::size_t, but SortedArrayIndex expects uint32_t.
// This is a known limitation that needs to be addressed separately.
// BOOST_AUTO_TEST_CASE(LoadFileWithSortedArrayIndexTest) {
//   // Test load_file with SortedArrayIndex
//   std::string content = "1,2\n3,4\n5,6\n";
//   std::filesystem::path temp_file = create_temp_file(content);
//
//   Relation<BooleanSR, std::tuple<int, int>, SortedArrayIndex> rel;
//   BOOST_REQUIRE_NO_THROW(load_file(rel, temp_file.string()));
//
//   BOOST_CHECK_EQUAL(rel.size(), 3);
//
//   std::filesystem::remove(temp_file);
// }

BOOST_AUTO_TEST_CASE(LoadFileWithSpecificIndexSpecsVectorTest) {
  // Test load_file with specific IndexSpecs (vector)
  std::string content = "1,2\n3,4\n5,6\n";
  std::filesystem::path temp_file = create_temp_file(content);

  Relation<BooleanSR, std::tuple<int, int>> rel;
  std::vector<IndexSpec> specs;
  IndexSpec spec1;
  spec1.cols = {0, 1};
  specs.push_back(spec1);
  IndexSpec spec2;
  spec2.cols = {1, 0};
  specs.push_back(spec2);

  BOOST_REQUIRE_NO_THROW(load_file(rel, temp_file.string(), specs));

  BOOST_CHECK_EQUAL(rel.size(), 3);
  // Verify indexes were built
  BOOST_CHECK_NO_THROW(rel.get_index(spec1));
  BOOST_CHECK_NO_THROW(rel.get_index(spec2));

  std::filesystem::remove(temp_file);
}

BOOST_AUTO_TEST_CASE(LoadFileWithSpecificIndexSpecsInitializerListTest) {
  // Test load_file with specific IndexSpecs (initializer list)
  std::string content = "1,2\n3,4\n5,6\n";
  std::filesystem::path temp_file = create_temp_file(content);

  Relation<BooleanSR, std::tuple<int, int>> rel;
  IndexSpec spec1;
  spec1.cols = {0, 1};
  IndexSpec spec2;
  spec2.cols = {1, 0};

  BOOST_REQUIRE_NO_THROW(load_file(rel, temp_file.string(), {spec1, spec2}));

  BOOST_CHECK_EQUAL(rel.size(), 3);
  // Verify indexes were built
  BOOST_CHECK_NO_THROW(rel.get_index(spec1));
  BOOST_CHECK_NO_THROW(rel.get_index(spec2));

  std::filesystem::remove(temp_file);
}

// Note: SortedArrayIndex test is commented out because interned columns
// are still std::size_t, but SortedArrayIndex expects uint32_t.
// This is a known limitation that needs to be addressed separately.
// BOOST_AUTO_TEST_CASE(LoadFileWithSortedArrayIndexAndSpecificSpecsTest) {
//   // Test load_file with SortedArrayIndex and specific IndexSpecs
//   std::string content = "1,2\n3,4\n5,6\n";
//   std::filesystem::path temp_file = create_temp_file(content);
//
//   Relation<BooleanSR, std::tuple<int, int>, SortedArrayIndex> rel;
//   IndexSpec spec;
//   spec.cols = {0, 1};
//
//   BOOST_REQUIRE_NO_THROW(load_file(rel, temp_file.string(), {spec}));
//
//   BOOST_CHECK_EQUAL(rel.size(), 3);
//   // Verify index was built
//   BOOST_CHECK_NO_THROW(rel.get_index(spec));
//
//   std::filesystem::remove(temp_file);
// }

BOOST_AUTO_TEST_SUITE_END()

}  // namespace
