#define BOOST_TEST_MODULE hashmap_index_test
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <ranges>
#include <span>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "column.h"
#include "hashmap.h"
#include "index.h"
#include "query.h"  // For Prefix
#include "semiring.h"
#include "tmp.h"

// Include implementation for auto return type deduction
#include "hashmap.ipp"

using namespace SRDatalog;
using namespace SRDatalog::index;

namespace {

// ============================================================================
// Test Cases
// ============================================================================

BOOST_AUTO_TEST_CASE(test_basic_construction_and_build) {
  using SR = BooleanSR;
  using HMI = HashmapIndex<SR, std::tuple<int, int, int>>;
  HMI idx(default_memory_resource());

  BOOST_REQUIRE(idx.empty());
  BOOST_REQUIRE_EQUAL(idx.size(), 0);

  // Build test data
  std::array<Vector<uint32_t>, 3> env;
  env[0] = {1, 2, 2, 3, 3};
  env[1] = {10, 20, 40, 30, 50};
  env[2] = {100, 200, 300, 400, 500};

  const auto enc_span = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size()),
      std::span<const uint32_t>(env[2].data(), env[2].size())};

  IndexSpec spec{{0, 1, 2}};  // Must index all columns
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  BOOST_REQUIRE_EQUAL(idx.size(), 5);
  BOOST_REQUIRE(!idx.empty());
}

BOOST_AUTO_TEST_CASE(test_root_node_and_navigation) {
  using SR = BooleanSR;
  using HMI = HashmapIndex<SR, std::tuple<int, int, int>>;
  HMI idx(default_memory_resource());

  std::array<Vector<uint32_t>, 3> env;
  env[0] = {1, 2, 2, 3, 3};
  env[1] = {10, 20, 40, 30, 50};
  env[2] = {100, 200, 300, 400, 500};

  const auto enc_span = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size()),
      std::span<const uint32_t>(env[2].data(), env[2].size())};

  IndexSpec spec{{0, 1, 2}};  // Must index all columns
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  auto root = idx.root();
  BOOST_REQUIRE(root.valid());
  BOOST_REQUIRE(!root.is_leaf());
  BOOST_REQUIRE_EQUAL(root.depth(), 0);

  // Check values at root - should return unique values (1, 2, 3)
  auto root_vals = root.values();
  std::vector<uint32_t> unique_vals;
  for (auto val : root_vals) {
    unique_vals.push_back(val);
  }
  std::sort(unique_vals.begin(), unique_vals.end());  // Set is unordered, so sort for comparison
  BOOST_REQUIRE_EQUAL(unique_vals.size(), 3);
  BOOST_REQUIRE_EQUAL(unique_vals[0], 1u);
  BOOST_REQUIRE_EQUAL(unique_vals[1], 2u);
  BOOST_REQUIRE_EQUAL(unique_vals[2], 3u);
}

BOOST_AUTO_TEST_CASE(test_prefix_navigation) {
  using SR = BooleanSR;
  using HMI = HashmapIndex<SR, std::tuple<int, int, int>>;
  HMI idx(default_memory_resource());

  std::array<Vector<uint32_t>, 3> env;
  env[0] = {1, 2, 2, 3, 3};
  env[1] = {10, 20, 40, 30, 50};
  env[2] = {100, 200, 300, 400, 500};

  const auto enc_span = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size()),
      std::span<const uint32_t>(env[2].data(), env[2].size())};

  IndexSpec spec{{0, 1, 2}};  // Must index all columns
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // Navigate to key=2
  auto root = idx.root();
  auto n2 = root.prefix(2);
  BOOST_REQUIRE(n2.valid());
  BOOST_REQUIRE(!n2.is_leaf());
  BOOST_REQUIRE_EQUAL(n2.depth(), 1);

  // Check values at n2 (should be unique: 20, 40)
  auto n2_vals = n2.values();
  std::vector<uint32_t> collected_n2_vals;
  for (auto val : n2_vals) {
    collected_n2_vals.push_back(val);
  }
  std::sort(collected_n2_vals.begin(),
            collected_n2_vals.end());  // Set is unordered, so sort for comparison
  BOOST_REQUIRE_EQUAL(collected_n2_vals.size(), 2);
  BOOST_REQUIRE_EQUAL(collected_n2_vals[0], 20u);
  BOOST_REQUIRE_EQUAL(collected_n2_vals[1], 40u);

  // Navigate to (2, 20) first, then to (2, 20, 200) - should be leaf (with 3 columns, leaf is at depth 3)
  auto n3 = n2.prefix(20);
  BOOST_REQUIRE(n3.valid());
  BOOST_REQUIRE(!n3.is_leaf());
  BOOST_REQUIRE_EQUAL(n3.depth(), 2);
  
  auto leaf = n3.prefix(200);
  BOOST_REQUIRE(leaf.valid());
  BOOST_REQUIRE(leaf.is_leaf());
  BOOST_REQUIRE_EQUAL(leaf.depth(), 3);

  // Check rows (set semantics - single row ID)
  auto rows = leaf.rows();
  BOOST_REQUIRE_EQUAL(rows.size(), 1);
}

BOOST_AUTO_TEST_CASE(test_prefix_lookup) {
  using SR = BooleanSR;
  using HMI = HashmapIndex<SR, std::tuple<int, int, int>>;
  HMI idx(default_memory_resource());

  std::array<Vector<uint32_t>, 3> env;
  env[0] = {1, 2, 2, 3, 3};
  env[1] = {10, 20, 40, 30, 50};
  env[2] = {100, 200, 300, 400, 500};

  const auto enc_span = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size()),
      std::span<const uint32_t>(env[2].data(), env[2].size())};

  IndexSpec spec{{0, 1, 2}};  // Must index all columns
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // Full key lookup (with 3 columns, need all 3 values)
  const auto key_220 = Prefix<int, int, int>{2, 20, 200}.encoded();
  auto h = idx.prefix_lookup(key_220);
  BOOST_REQUIRE(h.valid());
  BOOST_REQUIRE(h.is_leaf());

  // Partial key lookup
  const auto key_2 = Prefix<int>{2}.encoded();
  auto h2 = idx.prefix_lookup(key_2);
  BOOST_REQUIRE(h2.valid());
  BOOST_REQUIRE(!h2.is_leaf());

  // Non-existent key
  const auto key_999 = Prefix<int>{999}.encoded();
  auto h999 = idx.prefix_lookup(key_999);
  BOOST_REQUIRE(!h999.valid());
}

BOOST_AUTO_TEST_CASE(test_contains_value) {
  using SR = BooleanSR;
  using HMI = HashmapIndex<SR, std::tuple<int, int, int>>;
  HMI idx(default_memory_resource());

  std::array<Vector<uint32_t>, 3> env;
  env[0] = {1, 2, 2, 3, 3};
  env[1] = {10, 20, 40, 30, 50};
  env[2] = {100, 200, 300, 400, 500};

  const auto enc_span = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size()),
      std::span<const uint32_t>(env[2].data(), env[2].size())};

  IndexSpec spec{{0, 1, 2}};  // Must index all columns
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  auto root = idx.root();
  BOOST_REQUIRE(root.contains_value(1));
  BOOST_REQUIRE(root.contains_value(2));
  BOOST_REQUIRE(root.contains_value(3));
  BOOST_REQUIRE(!root.contains_value(999));
}

BOOST_AUTO_TEST_CASE(test_degree) {
  using SR = BooleanSR;
  using HMI = HashmapIndex<SR, std::tuple<int, int, int>>;
  HMI idx(default_memory_resource());

  std::array<Vector<uint32_t>, 3> env;
  env[0] = {1, 2, 2, 3, 3};
  env[1] = {10, 20, 40, 30, 50};
  env[2] = {100, 200, 300, 400, 500};

  const auto enc_span = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size()),
      std::span<const uint32_t>(env[2].data(), env[2].size())};

  IndexSpec spec{{0, 1, 2}};  // Must index all columns
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  auto root = idx.root();
  BOOST_REQUIRE_EQUAL(root.degree(), 3);  // Unique values: 1, 2, 3

  auto n2 = root.prefix(2);
  BOOST_REQUIRE_EQUAL(n2.degree(), 2);  // Unique values: 20, 40
}

BOOST_AUTO_TEST_CASE(test_clear) {
  using SR = BooleanSR;
  using HMI = HashmapIndex<SR, std::tuple<int, int, int>>;
  HMI idx(default_memory_resource());

  std::array<Vector<uint32_t>, 3> env;
  env[0] = {1, 2, 3};
  env[1] = {10, 20, 30};
  env[2] = {100, 200, 300};

  const auto enc_span = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size()),
      std::span<const uint32_t>(env[2].data(), env[2].size())};

  IndexSpec spec{{0, 1, 2}};  // Must index all columns
  idx.build_from_encoded(spec, enc_span, default_memory_resource());
  BOOST_REQUIRE_EQUAL(idx.size(), 3);

  idx.clear();
  BOOST_REQUIRE_EQUAL(idx.size(), 0);
  BOOST_REQUIRE(idx.empty());
}

BOOST_AUTO_TEST_CASE(test_merge_two_non_empty_indexes) {
  using SR = BooleanSR;
  using HMI = HashmapIndex<SR, std::tuple<int, int, int>>;
  HMI idx1(default_memory_resource());
  HMI idx2(default_memory_resource());

  // Build first index: (1,10), (3,30)
  std::array<Vector<uint32_t>, 3> env1;
  env1[0] = {1, 3};
  env1[1] = {10, 30};
  env1[2] = {100, 300};

  const auto enc_span1 = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env1[0].data(), env1[0].size()),
      std::span<const uint32_t>(env1[1].data(), env1[1].size()),
      std::span<const uint32_t>(env1[2].data(), env1[2].size())};

  IndexSpec spec{{0, 1, 2}};  // Must index all columns
  idx1.build_from_encoded(spec, enc_span1, default_memory_resource());
  BOOST_REQUIRE_EQUAL(idx1.size(), 2);

  // Build second index: (2,20), (4,40)
  std::array<Vector<uint32_t>, 3> env2;
  env2[0] = {2, 4};
  env2[1] = {20, 40};
  env2[2] = {200, 400};

  const auto enc_span2 = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env2[0].data(), env2[0].size()),
      std::span<const uint32_t>(env2[1].data(), env2[1].size()),
      std::span<const uint32_t>(env2[2].data(), env2[2].size())};

  idx2.build_from_encoded(spec, enc_span2, default_memory_resource());
  BOOST_REQUIRE_EQUAL(idx2.size(), 2);

  // Merge idx2 into idx1 with offset 100
  idx1.merge(idx2, 100);
  BOOST_REQUIRE_EQUAL(idx1.size(), 4);

  // Verify merged index has all keys
  auto root = idx1.root();
  BOOST_REQUIRE(root.valid());
  auto root_vals = root.values();
  std::vector<uint32_t> collected_root_vals;
  for (auto val : root_vals) {
    collected_root_vals.push_back(val);
  }
  BOOST_REQUIRE_EQUAL(collected_root_vals.size(), 4);
}

BOOST_AUTO_TEST_CASE(test_bytes_used) {
  using SR = BooleanSR;
  using HMI = HashmapIndex<SR, std::tuple<int, int, int>>;
  HMI idx(default_memory_resource());

  BOOST_REQUIRE_EQUAL(idx.bytes_used(), 0);

  std::array<Vector<uint32_t>, 3> env;
  env[0] = {1, 2, 3};
  env[1] = {10, 20, 30};
  env[2] = {100, 200, 300};

  const auto enc_span = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size()),
      std::span<const uint32_t>(env[2].data(), env[2].size())};

  IndexSpec spec{{0, 1, 2}};  // Must index all columns
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  auto bytes = idx.bytes_used();
  BOOST_REQUIRE_GT(bytes, 0);
}

BOOST_AUTO_TEST_CASE(test_intersect) {
  using SR = BooleanSR;
  using HMI = HashmapIndex<SR, std::tuple<int, int, int>>;
  HMI idx1(default_memory_resource());
  HMI idx2(default_memory_resource());

  // Build first index
  std::array<Vector<uint32_t>, 3> env1;
  env1[0] = {1, 2, 2, 3, 3};
  env1[1] = {10, 20, 40, 30, 50};
  env1[2] = {100, 200, 300, 400, 500};

  const auto enc_span1 = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env1[0].data(), env1[0].size()),
      std::span<const uint32_t>(env1[1].data(), env1[1].size()),
      std::span<const uint32_t>(env1[2].data(), env1[2].size())};

  IndexSpec spec{{0, 1, 2}};  // Must index all columns
  idx1.build_from_encoded(spec, enc_span1, default_memory_resource());

  // Build second index with overlapping keys
  std::array<Vector<uint32_t>, 3> env2;
  env2[0] = {2, 2, 3};
  env2[1] = {20, 99, 30};
  env2[2] = {200, 999, 300};

  const auto enc_span2 = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env2[0].data(), env2[0].size()),
      std::span<const uint32_t>(env2[1].data(), env2[1].size()),
      std::span<const uint32_t>(env2[2].data(), env2[2].size())};

  idx2.build_from_encoded(spec, enc_span2, default_memory_resource());

  // Test intersection
  auto root1 = idx1.root();
  auto n2_1 = root1.prefix(2);  // children {20, 40}
  auto root2 = idx2.root();
  auto n2_2 = root2.prefix(2);  // children {20, 99}

  // Intersect should find common value 20
  auto view = HMI::intersect(n2_1, n2_2);
  auto count = static_cast<std::size_t>(std::ranges::distance(view));
  BOOST_REQUIRE_EQUAL(count, 1u);

  // Verify the intersection contains 20
  std::vector<uint32_t> intersection;
  for (auto val : view) {
    intersection.push_back(val);
  }
  BOOST_REQUIRE_EQUAL(intersection[0], 20u);
}

BOOST_AUTO_TEST_CASE(test_set_semantics_leaf_node) {
  using SR = BooleanSR;
  using HMI = HashmapIndex<SR, std::tuple<int, int>>;
  HMI idx(default_memory_resource());

  // Build index with duplicate keys - set semantics means only one row per key
  std::array<Vector<uint32_t>, 2> env;
  env[0] = {1, 1, 1, 2, 2};
  env[1] = {10, 10, 10, 20, 20};

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};  // Must index all columns (2 columns for this relation)
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // Navigate to leaf (1, 10) - with 2 columns, leaf is at depth 2
  auto root = idx.root();
  auto n1 = root.prefix(1);
  auto leaf = n1.prefix(10);

  BOOST_REQUIRE(leaf.valid());
  BOOST_REQUIRE(leaf.is_leaf());

  // Set semantics: should have exactly 2 unique keys: (1,10) and (2,20)
  BOOST_REQUIRE_EQUAL(idx.size(), 2);
  // The leaf node for (1,10) should have 1 row ID (we store just one per unique key for efficiency)
  auto rows = leaf.rows();
  BOOST_REQUIRE_EQUAL(rows.size(), 1);
}

BOOST_AUTO_TEST_CASE(test_join_pattern_with_intersect) {
  // Test the exact pattern used in benchmarks: intersect() with values()
  // This simulates: for (auto x : intersect(root)) { for (auto y : intersect(root.prefix(x))) { ... } }
  using SR = BooleanSR;
  using HMI = HashmapIndex<SR, std::tuple<int, int>>;
  HMI idx(default_memory_resource());

  // Create data: (1,10), (1,20), (1,30), (2,40), (2,50), (3,60)
  std::array<Vector<uint32_t>, 2> env;
  env[0] = {1, 1, 1, 2, 2, 3};
  env[1] = {10, 20, 30, 40, 50, 60};

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};  // Must index all columns
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // Verify index size is correct (should be 6 unique keys)
  BOOST_REQUIRE_EQUAL(idx.size(), 6);

  // Pattern: for (auto x : intersect(root)) { for (auto y : intersect(root.prefix(x))) { ... } }
  auto root = idx.root();
  using IndexType = HMI;

  std::set<std::pair<uint32_t, uint32_t>> collected_pairs;
  std::size_t x_count = 0;
  std::size_t y_count = 0;
  for (auto x : IndexType::intersect(root)) {
    x_count++;
    auto node_x = root.prefix(x);
    BOOST_REQUIRE(node_x.valid());
    for (auto y : IndexType::intersect(node_x)) {
      y_count++;
      collected_pairs.insert({x, y});
    }
  }

  // Should collect all pairs: {(1,10), (1,20), (1,30), (2,40), (2,50), (3,60)}
  BOOST_REQUIRE_EQUAL(collected_pairs.size(), 6);
  BOOST_REQUIRE_EQUAL(x_count, 3);  // 3 unique x values: 1, 2, 3
  BOOST_REQUIRE_EQUAL(y_count, 6);  // 6 total y values across all x
  BOOST_REQUIRE(collected_pairs.count({1, 10}));
  BOOST_REQUIRE(collected_pairs.count({1, 20}));
  BOOST_REQUIRE(collected_pairs.count({1, 30}));
  BOOST_REQUIRE(collected_pairs.count({2, 40}));
  BOOST_REQUIRE(collected_pairs.count({2, 50}));
  BOOST_REQUIRE(collected_pairs.count({3, 60}));
}

BOOST_AUTO_TEST_CASE(test_join_pattern_with_duplicates) {
  // Test join pattern with duplicate keys - should still return unique pairs
  using SR = BooleanSR;
  using HMI = HashmapIndex<SR, std::tuple<int, int>>;
  HMI idx(default_memory_resource());

  // Create data with duplicates: (1,10), (1,10), (1,20), (2,40), (2,40), (3,60)
  std::array<Vector<uint32_t>, 2> env;
  env[0] = {1, 1, 1, 2, 2, 3};
  env[1] = {10, 10, 20, 40, 40, 60};

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // With set semantics, should have 4 unique keys: (1,10), (1,20), (2,40), (3,60)
  BOOST_REQUIRE_EQUAL(idx.size(), 4);

  // Join pattern should return unique pairs only
  auto root = idx.root();
  using IndexType = HMI;

  std::set<std::pair<uint32_t, uint32_t>> collected_pairs;
  for (auto x : IndexType::intersect(root)) {
    auto node_x = root.prefix(x);
    BOOST_REQUIRE(node_x.valid());
    for (auto y : IndexType::intersect(node_x)) {
      collected_pairs.insert({x, y});
    }
  }

  // Should collect only unique pairs: {(1,10), (1,20), (2,40), (3,60)}
  BOOST_REQUIRE_EQUAL(collected_pairs.size(), 4);
  BOOST_REQUIRE(collected_pairs.count({1, 10}));
  BOOST_REQUIRE(collected_pairs.count({1, 20}));
  BOOST_REQUIRE(collected_pairs.count({2, 40}));
  BOOST_REQUIRE(collected_pairs.count({3, 60}));
}

BOOST_AUTO_TEST_CASE(test_tc_join_pattern) {
  // Test the exact TC join pattern from benchmark
  // Pattern: for (auto y : intersect(path_ver_1_0, edge_full_0_1)) {
  //            for (auto z : intersect(edge_y)) {
  //              for (auto x : intersect(path_y)) { ... }
  //            }
  //          }
  using SR = BooleanSR;
  using HMI = HashmapIndex<SR, std::tuple<int, int>>;
  
  // Create path relation: (1,10), (1,20), (2,30)
  HMI path_idx(default_memory_resource());
  std::array<Vector<uint32_t>, 2> path_env;
  path_env[0] = {1, 1, 2};
  path_env[1] = {10, 20, 30};
  const auto path_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(path_env[0].data(), path_env[0].size()),
      std::span<const uint32_t>(path_env[1].data(), path_env[1].size())};
  IndexSpec path_spec{{1, 0}};  // Index on columns 1,0 (reversed)
  path_idx.build_from_encoded(path_spec, path_span, default_memory_resource());
  BOOST_REQUIRE_EQUAL(path_idx.size(), 3);

  // Create edge relation: (10,100), (20,200), (30,300)
  HMI edge_idx(default_memory_resource());
  std::array<Vector<uint32_t>, 2> edge_env;
  edge_env[0] = {10, 20, 30};
  edge_env[1] = {100, 200, 300};
  const auto edge_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(edge_env[0].data(), edge_env[0].size()),
      std::span<const uint32_t>(edge_env[1].data(), edge_env[1].size())};
  IndexSpec edge_spec{{0, 1}};  // Index on columns 0,1
  edge_idx.build_from_encoded(edge_spec, edge_span, default_memory_resource());
  BOOST_REQUIRE_EQUAL(edge_idx.size(), 3);

  // TC join: find y where path has (?,y) and edge has (y,?)
  // path_ver_1_0 is index on columns 1,0: root gives all y values
  // edge_full_0_1 is index on columns 0,1: root gives all y values
  auto path_root = path_idx.root();  // Index on (1,0), root gives y values
  auto edge_root = edge_idx.root();  // Index on (0,1), root gives y values

  using IndexType = HMI;
  std::set<std::pair<uint32_t, uint32_t>> results;
  
  // Pattern: for (auto y : intersect(path_root, edge_root))
  for (auto y : IndexType::intersect(path_root, edge_root)) {
    // path_y = path_root.prefix(y) gives all x where path has (x,y)
    // edge_y = edge_root.prefix(y) gives all z where edge has (y,z)
    auto path_y = path_root.prefix(y);
    auto edge_y = edge_root.prefix(y);
    
    BOOST_REQUIRE(path_y.valid());
    BOOST_REQUIRE(edge_y.valid());
    
    // for (auto z : intersect(edge_y))
    for (auto z : IndexType::intersect(edge_y)) {
      // for (auto x : intersect(path_y))
      for (auto x : IndexType::intersect(path_y)) {
        results.insert({x, z});
      }
    }
  }

  // Expected results based on data:
  // path: (1,10), (1,20), (2,30)
  // edge: (10,100), (20,200), (30,300)
  // y=10: path has (1,10), edge has (10,100) -> (1,100)
  // y=20: path has (1,20), edge has (20,200) -> (1,200)
  // y=30: path has (2,30), edge has (30,300) -> (2,300)
  BOOST_REQUIRE_EQUAL(results.size(), 3);
  BOOST_REQUIRE(results.count({1, 100}));
  BOOST_REQUIRE(results.count({1, 200}));
  BOOST_REQUIRE(results.count({2, 300}));
}

BOOST_AUTO_TEST_CASE(test_index_size_matches_iteration_count) {
  // Test that index size matches the number of unique pairs when iterating
  // This verifies that size() correctly reflects unique keys
  using SR = BooleanSR;
  using HMI = HashmapIndex<SR, std::tuple<int, int>>;
  HMI idx(default_memory_resource());

  // Create data with some duplicates
  std::array<Vector<uint32_t>, 2> env;
  env[0] = {1, 1, 1, 2, 2, 3, 3, 3};
  env[1] = {10, 10, 20, 40, 40, 60, 60, 70};

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // Count unique pairs by iterating
  auto root = idx.root();
  using IndexType = HMI;
  std::set<std::pair<uint32_t, uint32_t>> unique_pairs;
  
  for (auto x : IndexType::intersect(root)) {
    auto node_x = root.prefix(x);
    for (auto y : IndexType::intersect(node_x)) {
      unique_pairs.insert({x, y});
    }
  }

  // Index size should match the number of unique pairs
  BOOST_REQUIRE_EQUAL(idx.size(), unique_pairs.size());
  BOOST_REQUIRE_EQUAL(idx.size(), 5);  // (1,10), (1,20), (2,40), (3,60), (3,70)
}

}  // namespace
