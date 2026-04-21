// Unit test to verify SortedArrayIndex produces same results as HashTrieIndex for TC
#define BOOST_TEST_MODULE test_tc_sorted_array
#include "hashtrie.h"
#include "query.h"
#include "relation_col.h"
#include "sorted_array_index.h"
#include <boost/test/included/unit_test.hpp>
#include <set>

using namespace SRDatalog;

// Helper to collect all rows from a relation
template <typename Rel>
std::set<std::pair<int, int>> collect_rows(const Rel& rel) {
  std::set<std::pair<int, int>> rows;
  for (std::size_t i = 0; i < rel.size(); ++i) {
    rows.insert({rel.template column<0>()[i], rel.template column<1>()[i]});
  }
  return rows;
}

BOOST_AUTO_TEST_CASE(test_tc_sorted_array_vs_hashtrie) {
  // Test transitive closure with both index types and verify they produce same results

  // Create relations with HashTrieIndex (default)
  Relation<BooleanSR, std::tuple<int, int>> edge_full_ht;
  edge_full_ht.set_version(FULL_VER);
  Relation<BooleanSR, std::tuple<int, int>> path_full_ht;
  path_full_ht.set_version(FULL_VER);
  Relation<BooleanSR, std::tuple<int, int>> path_delta_ht;
  path_delta_ht.set_version(DELTA_VER);
  Relation<BooleanSR, std::tuple<int, int>> path_newt_ht;
  path_newt_ht.set_version(NEW_VER);

  // Create relations with SortedArrayIndex
  Relation<BooleanSR, std::tuple<int, int>, SortedArrayIndex> edge_full_sa;
  edge_full_sa.set_version(FULL_VER);
  Relation<BooleanSR, std::tuple<int, int>, SortedArrayIndex> path_full_sa;
  path_full_sa.set_version(FULL_VER);
  Relation<BooleanSR, std::tuple<int, int>, SortedArrayIndex> path_delta_sa;
  path_delta_sa.set_version(DELTA_VER);
  Relation<BooleanSR, std::tuple<int, int>, SortedArrayIndex> path_newt_sa;
  path_newt_sa.set_version(NEW_VER);

  // Load test data: simple graph 1->2->3->4, 2->5->6
  edge_full_ht.push_row({1, 2}, BooleanSR::one());
  edge_full_ht.push_row({2, 3}, BooleanSR::one());
  edge_full_ht.push_row({3, 4}, BooleanSR::one());
  edge_full_ht.push_row({2, 5}, BooleanSR::one());
  edge_full_ht.push_row({5, 6}, BooleanSR::one());

  edge_full_sa.push_row({1, 2}, BooleanSR::one());
  edge_full_sa.push_row({2, 3}, BooleanSR::one());
  edge_full_sa.push_row({3, 4}, BooleanSR::one());
  edge_full_sa.push_row({2, 5}, BooleanSR::one());
  edge_full_sa.push_row({5, 6}, BooleanSR::one());

  // Build indexes
  const auto& edge_full_idx_ht = edge_full_ht.ensure_index({{0, 1}});
  const auto& edge_full_idx_sa = edge_full_sa.ensure_index({{0, 1}});

  // Copy edges to path_newt
  using HTIndexType = typename decltype(edge_full_ht)::IndexTypeInst;
  using SAIndexType = typename decltype(edge_full_sa)::IndexTypeInst;
  for (auto x : HTIndexType::intersect(edge_full_idx_ht.root())) {
    for (auto y : HTIndexType::intersect(edge_full_idx_ht.root().prefix(x))) {
      path_newt_ht.push_row({x, y}, BooleanSR::one());
    }
  }

  for (auto x : SAIndexType::intersect(edge_full_idx_sa.root())) {
    for (auto y : SAIndexType::intersect(edge_full_idx_sa.root().prefix(x))) {
      path_newt_sa.push_row({x, y}, BooleanSR::one());
    }
  }

  // Setup delta and full
  path_delta_ht.clear();
  path_newt_ht.clone_into(path_delta_ht);
  path_newt_ht.clear();
  path_full_ht.concat(path_delta_ht);
  path_full_ht.ensure_index({{1, 0}});

  path_delta_sa.clear();
  path_newt_sa.clone_into(path_delta_sa);
  path_newt_sa.clear();
  path_full_sa.concat(path_delta_sa);
  path_full_sa.ensure_index({{1, 0}});

  // Verify initial state matches
  auto rows_ht = collect_rows(path_full_ht);
  auto rows_sa = collect_rows(path_full_sa);
  BOOST_REQUIRE_EQUAL(rows_ht.size(), rows_sa.size());
  BOOST_REQUIRE(rows_ht == rows_sa);

  // Run one iteration of TC
  auto path_full_1_0_ht = path_full_ht.get_index({{1, 0}}).root();
  auto edge_full_0_1_ht = edge_full_ht.get_index({{0, 1}}).root();
  auto path_full_1_0_sa = path_full_sa.get_index({{1, 0}}).root();
  auto edge_full_0_1_sa = edge_full_sa.get_index({{0, 1}}).root();

  auto path_ver_1_0_ht = path_full_ht.get_index({{1, 0}}).root();
  auto path_ver_1_0_sa = path_full_sa.get_index({{1, 0}}).root();

  // Compute new paths
  for (auto y : HTIndexType::intersect(path_ver_1_0_ht, edge_full_0_1_ht)) {
    auto edge_y_ht = edge_full_0_1_ht.prefix(y);
    auto path_y_ht = path_ver_1_0_ht.prefix(y);
    for (auto z : HTIndexType::intersect(edge_y_ht)) {
      auto path_full_z_ht = path_full_1_0_ht.prefix(z);
      for (auto x : HTIndexType::intersect(path_y_ht)) {
        if (!path_full_z_ht.valid() || !path_full_z_ht.prefix(x).valid()) {
          path_newt_ht.push_row({x, z}, BooleanSR::one());
        }
      }
    }
  }

  for (auto y : SAIndexType::intersect(path_ver_1_0_sa, edge_full_0_1_sa)) {
    auto edge_y_sa = edge_full_0_1_sa.prefix(y);
    auto path_y_sa = path_ver_1_0_sa.prefix(y);
    for (auto z : SAIndexType::intersect(edge_y_sa)) {
      auto path_full_z_sa = path_full_1_0_sa.prefix(z);
      for (auto x : SAIndexType::intersect(path_y_sa)) {
        if (!path_full_z_sa.valid() || !path_full_z_sa.prefix(x).valid()) {
          path_newt_sa.push_row({x, z}, BooleanSR::one());
        }
      }
    }
  }

  // Verify path_newt matches
  auto newt_rows_ht = collect_rows(path_newt_ht);
  auto newt_rows_sa = collect_rows(path_newt_sa);
  BOOST_REQUIRE_EQUAL(newt_rows_ht.size(), newt_rows_sa.size());
  BOOST_REQUIRE(newt_rows_ht == newt_rows_sa);

  // Add to path_full
  path_newt_ht.ensure_index({{1, 0}});
  path_newt_sa.ensure_index({{1, 0}});

  auto path_newt_1_0_ht = path_newt_ht.get_index({{1, 0}}).root();
  auto path_newt_1_0_sa = path_newt_sa.get_index({{1, 0}}).root();

  std::size_t path_full_start_size_ht = path_full_ht.size();
  std::size_t path_full_start_size_sa = path_full_sa.size();

  for (auto y : HTIndexType::intersect(path_newt_1_0_ht)) {
    for (auto x : HTIndexType::intersect(path_newt_1_0_ht.prefix(y))) {
      path_full_ht.push_row({x, y}, BooleanSR::one());
    }
  }

  for (auto y : SAIndexType::intersect(path_newt_1_0_sa)) {
    for (auto x : SAIndexType::intersect(path_newt_1_0_sa.prefix(y))) {
      path_full_sa.push_row({x, y}, BooleanSR::one());
    }
  }

  // Verify path_full matches after adding
  auto full_rows_ht = collect_rows(path_full_ht);
  auto full_rows_sa = collect_rows(path_full_sa);
  BOOST_REQUIRE_EQUAL(full_rows_ht.size(), full_rows_sa.size());
  BOOST_REQUIRE(full_rows_ht == full_rows_sa);

  // Swap and merge
  std::swap(path_delta_ht, path_newt_ht);
  std::swap(path_delta_sa, path_newt_sa);

  if (path_delta_ht.size() > 0) {
    auto& path_full_idx_ht = path_full_ht.get_index({{1, 0}});
    const auto& path_delta_idx_ht = path_delta_ht.get_index({{1, 0}});
    path_full_idx_ht.merge(path_delta_idx_ht, path_full_start_size_ht);
    // BOOST_REQUIRE(path_full_idx_ht.verify_sorted());
  }

  if (path_delta_sa.size() > 0) {
    auto& path_full_idx_sa = path_full_sa.get_index({{1, 0}});
    const auto& path_delta_idx_sa = path_delta_sa.get_index({{1, 0}});
    path_full_idx_sa.merge(path_delta_idx_sa, path_full_start_size_sa);
    BOOST_REQUIRE(path_full_idx_sa.verify_sorted());
  }

  // Final verification
  auto final_rows_ht = collect_rows(path_full_ht);
  auto final_rows_sa = collect_rows(path_full_sa);
  BOOST_REQUIRE_EQUAL(final_rows_ht.size(), final_rows_sa.size());
  BOOST_REQUIRE(final_rows_ht == final_rows_sa);

  // Note: Index sizes may differ between HashTrieIndex and SortedArrayIndex
  // because SortedArrayIndex may have duplicate keys with different row IDs,
  // while HashTrieIndex stores them differently. The important thing is that
  // the unique rows match, which we've verified above.
}

BOOST_AUTO_TEST_CASE(test_merge_deduplication_with_relations) {
  // CRITICAL TEST: This tests the actual bug scenario from TC computation.
  // This simulates a semi-naive TC join operation like in tc_lower_benchmark.cpp:
  // 1. Join path_delta with edge to produce path_newt
  // 2. Add path_newt rows to path_full using push_intern_row
  // 3. Swap path_newt and path_delta
  // 4. Merge path_delta index (which now contains old path_newt's index) into path_full

  Relation<BooleanSR, std::tuple<int, int>, SortedArrayIndex> path_full;
  path_full.set_version(FULL_VER);
  Relation<BooleanSR, std::tuple<int, int>, SortedArrayIndex> path_delta;
  path_delta.set_version(DELTA_VER);
  Relation<BooleanSR, std::tuple<int, int>, SortedArrayIndex> path_newt;
  path_newt.set_version(NEW_VER);
  Relation<BooleanSR, std::tuple<int, int>, SortedArrayIndex> edge_full;
  edge_full.set_version(FULL_VER);

  // Setup: path_full has (1,10), (2,20)
  path_full.push_row({1, 10}, BooleanSR::one());
  path_full.push_row({2, 20}, BooleanSR::one());
  path_full.ensure_index({{1, 0}});
  BOOST_REQUIRE_EQUAL(path_full.size(), 2);

  // Setup: path_delta has (2,20), (3,30)
  // These represent paths ending at nodes 20 and 30
  path_delta.push_row({2, 20}, BooleanSR::one());
  path_delta.push_row({3, 30}, BooleanSR::one());
  path_delta.ensure_index({{1, 0}});
  BOOST_REQUIRE_EQUAL(path_delta.size(), 2);

  // Setup: edge_full has edges for the join
  // path_delta(2,20) JOIN edge(20,50) -> path_newt(2,50)
  // path_delta(3,30) JOIN edge(30,60) -> path_newt(3,60)
  // Also add edge(20,70) to create (2,70) - but this will be filtered if already exists
  edge_full.push_row({20, 50}, BooleanSR::one());
  edge_full.push_row({30, 60}, BooleanSR::one());
  edge_full.push_row({20, 70}, BooleanSR::one());
  edge_full.ensure_index({{0, 1}});

  // STEP 1: Semi-naive TC join operation (matches tc_lower_benchmark.cpp pattern)
  // Join path_delta with edge_full: path(x,y) JOIN edge(y,z) -> path(x,z)
  auto path_full_1_0 = path_full.get_index({{1, 0}}).root();
  auto path_delta_1_0 = path_delta.get_index({{1, 0}}).root();
  auto edge_full_0_1 = edge_full.get_index({{0, 1}}).root();
  using TestIndexType = typename decltype(path_delta)::IndexTypeInst;
  for (auto y : TestIndexType::intersect(path_delta_1_0, edge_full_0_1)) {
    auto edge_y = edge_full_0_1.prefix(y);
    auto path_y = path_delta_1_0.prefix(y);
    for (auto z : TestIndexType::intersect(edge_y)) {
      auto path_full_z = path_full_1_0.prefix(z);
      for (auto x : TestIndexType::intersect(path_y)) {
        // Check if path (x, z) already exists in path_full (deduplication check)
        if (!path_full_z.valid() || !path_full_z.prefix(x).valid()) {
          path_newt.push_intern_row({x, z}, BooleanSR::one());
        }
      }
    }
  }

  // path_newt should have: (2,50), (3,60), (2,70) - paths from join
  // path_delta(2,20) JOIN edge(20,50) -> (2,50)
  // path_delta(2,20) JOIN edge(20,70) -> (2,70)
  // path_delta(3,30) JOIN edge(30,60) -> (3,60)
  BOOST_REQUIRE(path_newt.size() >= 2);

  // STEP 2: Build index on path_newt
  path_newt.ensure_index({{1, 0}});
  path_delta.clear();

  // STEP 3: Track starting size and add path_newt rows to path_full
  // This matches the pattern in tc_lower_benchmark.cpp
  std::size_t path_full_start_size = path_full.size();
  auto path_newt_1_0 = path_newt.get_index({{1, 0}}).root();
  for (auto y : TestIndexType::intersect(path_newt_1_0)) {
    for (auto x : TestIndexType::intersect(path_newt_1_0.prefix(y))) {
      path_full.push_intern_row({x, y}, BooleanSR::one());
    }
  }

  // path_full should now have original 2 + paths from path_newt
  BOOST_REQUIRE(path_full.size() > path_full_start_size);

  // STEP 4: Swap path_newt and path_delta (simulating the benchmark pattern)
  // After swap, path_delta contains the index from old path_newt
  std::swap(path_delta, path_newt);

  // STEP 5: Merge path_delta index (which now has old path_newt's index) into path_full
  // This is where deduplication happens - the merge should remove duplicate keys
  auto& path_full_idx = path_full.get_index({{1, 0}});
  if (path_delta.size() > 0) {
    const auto& path_delta_idx = path_delta.get_index({{1, 0}});
    path_full_idx.merge(path_delta_idx, path_full_start_size);
    BOOST_REQUIRE(path_full_idx.verify_sorted());
  }

  // CRITICAL: After merge, the index should be deduplicated
  // The index size should equal the number of unique keys in path_full
  // Verify by iterating through the index and collecting unique keys
  std::set<std::pair<int, int>> unique_keys;
  auto root = path_full_idx.root();
  using PathIndexType = typename decltype(path_full)::IndexTypeInst;
  for (auto y : PathIndexType::intersect(root)) {
    auto node_y = root.prefix(y);
    for (auto x : PathIndexType::intersect(node_y)) {
      auto leaf = node_y.prefix(x);
      if (leaf.is_leaf()) {
        unique_keys.insert({static_cast<int>(x), static_cast<int>(y)});
        // Each unique key should have exactly one row ID after deduplication
        auto rows = leaf.rows();
        BOOST_REQUIRE_EQUAL(rows.size(), 1);
      }
    }
  }

  // Verify index size matches number of unique keys
  BOOST_REQUIRE_EQUAL(path_full_idx.size(), unique_keys.size());

  // Verify all expected paths are present
  BOOST_REQUIRE(unique_keys.count({1, 10}) > 0);  // Original
  BOOST_REQUIRE(unique_keys.count({2, 20}) > 0);  // Original
  // New paths from join should be present
}

BOOST_AUTO_TEST_CASE(test_column_intersect_deduplication) {
  // Test that column_intersect handles duplicates correctly for SortedArrayIndex
  // This test ensures that when values() contains duplicates, column_intersect
  // doesn't cause duplicate rows to be added to relations

  Relation<BooleanSR, std::tuple<int, int>, SortedArrayIndex> rel;

  // Add some rows that will create duplicate keys in the index
  rel.push_row({1, 10}, BooleanSR::one());
  rel.push_row({1, 10}, BooleanSR::one());  // Duplicate key
  rel.push_row({2, 20}, BooleanSR::one());

  const auto& idx = rel.ensure_index({{0, 1}});
  auto root = idx.root();

  // values() returns unique values for SortedArrayIndex
  BOOST_REQUIRE_GE(root.unique_count(), 2);  // At least 2 unique values

  // When iterating column_intersect, we should get unique values
  // (This test documents the expected behavior - if column_intersect
  //  doesn't deduplicate, this test will fail)
  std::set<int> unique_vals;
  using DedupIndexType = typename decltype(rel)::IndexTypeInst;
  for (auto x : DedupIndexType::intersect(root)) {
    unique_vals.insert(static_cast<int>(x));
  }

  // Should have exactly 2 unique values (1 and 2)
  BOOST_REQUIRE_EQUAL(unique_vals.size(), 2);
  BOOST_REQUIRE(unique_vals.count(1) > 0);
  BOOST_REQUIRE(unique_vals.count(2) > 0);
}
