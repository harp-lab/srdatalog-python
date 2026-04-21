#define BOOST_TEST_MODULE implicit_btree_index_test
#include <boost/test/included/unit_test.hpp>

#include <array>
#include <cstddef>
#include <ranges>
#include <span>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "column.h"
#include "eytzinger.h"
#include "index.h"
#include "query.h"  // For Prefix
#include "semiring.h"
#include "tmp.h"

using namespace SRDatalog;

namespace {

// ============================================================================
// Helper Functions (reused from sorted_array_index tests)
// ============================================================================

// Helper to collect all (key, row_id) pairs from an index
template <typename Index>
std::set<std::pair<std::vector<uint32_t>, uint32_t>> collect_index_entries(const Index& idx) {
  std::set<std::pair<std::vector<uint32_t>, uint32_t>> entries;

  auto root = idx.root();
  if (!root.valid())
    return entries;

  // For arity 2 index
  for (auto k0 : root.values()) {
    auto n1 = root.prefix(k0);
    if (n1.valid() && !n1.is_leaf()) {
      for (auto k1 : n1.values()) {
        auto leaf = n1.prefix(k1);
        if (leaf.valid() && leaf.is_leaf()) {
          for (auto row_id : leaf.rows()) {
            entries.insert({{static_cast<uint32_t>(k0), static_cast<uint32_t>(k1)},
                            static_cast<uint32_t>(row_id)});
          }
        }
      }
    }
  }

  return entries;
}

// ============================================================================
// Test Cases
// ============================================================================

BOOST_AUTO_TEST_CASE(test_basic_construction_and_build) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int, int>>;
  IBT idx(default_memory_resource());

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

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  BOOST_REQUIRE_EQUAL(idx.size(), 5);
  BOOST_REQUIRE(!idx.empty());
}

BOOST_AUTO_TEST_CASE(test_build_deduplicates_full_keys) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int, int>>;
  IBT idx(default_memory_resource());

  // Two identical (1,10,*) tuples, plus distinct (1,20,*) and (2,20,*)
  std::array<Vector<uint32_t>, 3> env;
  env[0] = {1, 1, 1, 2};      // key column (with duplicates)
  env[1] = {10, 10, 20, 20};  // second indexed column
  env[2] = {100, 100, 200, 300};

  const auto enc_span = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size()),
      std::span<const uint32_t>(env[2].data(), env[2].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // We started with 4 tuples but (1,10,*) appears twice -> 3 unique (full key) tuples.
  BOOST_REQUIRE_EQUAL(idx.size(), 3);

  auto root = idx.root();
  BOOST_REQUIRE(root.valid());

  // At depth 0, values() should return unique values: (1, 2)
  auto root_vals = root.values();
  std::vector<uint32_t> collected_vals(root_vals.begin(), root_vals.end());
  BOOST_REQUIRE_EQUAL(collected_vals.size(), 2);
  BOOST_REQUIRE_EQUAL(collected_vals[0], 1u);
  BOOST_REQUIRE_EQUAL(collected_vals[1], 2u);

  // For full key (1,10) there should be exactly one row after deduplication.
  auto n1 = root.prefix(1);
  BOOST_REQUIRE(n1.valid());

  auto n1_10 = n1.prefix(10);
  BOOST_REQUIRE(n1_10.valid());
  BOOST_REQUIRE(n1_10.is_leaf());

  auto rows_1_10 = n1_10.rows();
  BOOST_REQUIRE_EQUAL(rows_1_10.size(), 1);
}

BOOST_AUTO_TEST_CASE(test_prefix_lookup) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;
  IBT idx(default_memory_resource());

  std::array<Vector<uint32_t>, 2> env;
  env[0] = {1, 2, 3, 4, 5};
  env[1] = {10, 20, 30, 40, 50};

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // Test prefix lookup
  auto prefix = Prefix<int, int>{2, 20};
  auto node = idx.prefix_lookup(prefix.encoded());
  BOOST_REQUIRE(node.valid());
  BOOST_REQUIRE(node.is_leaf());
  BOOST_REQUIRE_EQUAL(node.rows().size(), 1);
}

BOOST_AUTO_TEST_CASE(test_merge_empty_into_empty) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;

  IBT idx1(default_memory_resource());
  IBT idx2(default_memory_resource());

  idx1.merge(idx2, 0);
  BOOST_REQUIRE_EQUAL(idx1.size(), 0);
  BOOST_REQUIRE(idx1.empty());
}

BOOST_AUTO_TEST_CASE(test_merge_empty_into_nonempty) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;

  IBT idx1(default_memory_resource());
  IBT idx2(default_memory_resource());

  // Build idx1 with some data
  std::array<Vector<uint32_t>, 2> env1;
  env1[0] = {1, 2};
  env1[1] = {10, 20};
  const auto enc_span1 = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env1[0].data(), env1[0].size()),
      std::span<const uint32_t>(env1[1].data(), env1[1].size())};

  IndexSpec spec{{0, 1}};
  idx1.build_from_encoded(spec, enc_span1, default_memory_resource());

  std::size_t size_before = idx1.size();
  idx1.merge(idx2, 0);
  BOOST_REQUIRE_EQUAL(idx1.size(), size_before);
}

BOOST_AUTO_TEST_CASE(test_merge_nonempty_into_empty) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;

  IBT idx1(default_memory_resource());
  IBT idx2(default_memory_resource());

  // Build idx2 with some data
  std::array<Vector<uint32_t>, 2> env2;
  env2[0] = {1, 2};
  env2[1] = {10, 20};
  const auto enc_span2 = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env2[0].data(), env2[0].size()),
      std::span<const uint32_t>(env2[1].data(), env2[1].size())};

  IndexSpec spec{{0, 1}};
  idx2.build_from_encoded(spec, enc_span2, default_memory_resource());

  idx1.merge(idx2, 100);
  BOOST_REQUIRE_EQUAL(idx1.size(), 2);

  // Verify row IDs are offset
  auto entries = collect_index_entries(idx1);
  BOOST_REQUIRE_EQUAL(entries.size(), 2);
  for (const auto& [key, row_id] : entries) {
    BOOST_REQUIRE_GE(row_id, 100u);
  }
}

BOOST_AUTO_TEST_CASE(test_merge_disjoint_keys) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;

  IBT idx1(default_memory_resource());
  IBT idx2(default_memory_resource());

  // idx1: (1,10), (2,20)
  std::array<Vector<uint32_t>, 2> env1;
  env1[0] = {1, 2};
  env1[1] = {10, 20};
  const auto enc_span1 = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env1[0].data(), env1[0].size()),
      std::span<const uint32_t>(env1[1].data(), env1[1].size())};

  // idx2: (3,30), (4,40)
  std::array<Vector<uint32_t>, 2> env2;
  env2[0] = {3, 4};
  env2[1] = {30, 40};
  const auto enc_span2 = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env2[0].data(), env2[0].size()),
      std::span<const uint32_t>(env2[1].data(), env2[1].size())};

  IndexSpec spec{{0, 1}};
  idx1.build_from_encoded(spec, enc_span1, default_memory_resource());
  idx2.build_from_encoded(spec, enc_span2, default_memory_resource());

  idx1.merge(idx2, 100);
  BOOST_REQUIRE_EQUAL(idx1.size(), 4);

  // Verify all entries are present
  auto entries = collect_index_entries(idx1);
  BOOST_REQUIRE_EQUAL(entries.size(), 4);
  BOOST_REQUIRE(entries.count({{1, 10}, 0}));
  BOOST_REQUIRE(entries.count({{2, 20}, 1}));
  BOOST_REQUIRE(entries.count({{3, 30}, 100}));
  BOOST_REQUIRE(entries.count({{4, 40}, 101}));
}

BOOST_AUTO_TEST_CASE(test_merge_overlapping_keys) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;

  IBT idx1(default_memory_resource());
  IBT idx2(default_memory_resource());

  // idx1: (1,10), (2,20)
  std::array<Vector<uint32_t>, 2> env1;
  env1[0] = {1, 2};
  env1[1] = {10, 20};
  const auto enc_span1 = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env1[0].data(), env1[0].size()),
      std::span<const uint32_t>(env1[1].data(), env1[1].size())};

  // idx2: (2,25), (3,30) - overlaps with idx1 on key (2,20) vs (2,25)
  std::array<Vector<uint32_t>, 2> env2;
  env2[0] = {2, 3};
  env2[1] = {25, 30};
  const auto enc_span2 = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env2[0].data(), env2[0].size()),
      std::span<const uint32_t>(env2[1].data(), env2[1].size())};

  IndexSpec spec{{0, 1}};
  idx1.build_from_encoded(spec, enc_span1, default_memory_resource());
  idx2.build_from_encoded(spec, enc_span2, default_memory_resource());

  idx1.merge(idx2, 100);
  BOOST_REQUIRE_EQUAL(idx1.size(), 4);

  // Verify sorted order: values() returns unique values (1, 2, 3)
  auto root = idx1.root();
  BOOST_REQUIRE_EQUAL(root.unique_count(), 3);
  auto root_vals = root.values();
  std::vector<uint32_t> collected_vals(root_vals.begin(), root_vals.end());
  BOOST_REQUIRE_EQUAL(collected_vals.size(), 3);
  BOOST_REQUIRE_EQUAL(collected_vals[0], 1u);
  BOOST_REQUIRE_EQUAL(collected_vals[1], 2u);
  BOOST_REQUIRE_EQUAL(collected_vals[2], 3u);
}

BOOST_AUTO_TEST_CASE(test_merge_equal_keys_different_rowids) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;

  IBT idx1(default_memory_resource());
  IBT idx2(default_memory_resource());

  // idx1: (1,10) with row_id 0
  std::array<Vector<uint32_t>, 2> env1;
  env1[0] = {1};
  env1[1] = {10};
  const auto enc_span1 = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env1[0].data(), env1[0].size()),
      std::span<const uint32_t>(env1[1].data(), env1[1].size())};

  // idx2: (1,10) with row_id 0 (will become 100 after offset)
  std::array<Vector<uint32_t>, 2> env2;
  env2[0] = {1};
  env2[1] = {10};
  const auto enc_span2 = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env2[0].data(), env2[0].size()),
      std::span<const uint32_t>(env2[1].data(), env2[1].size())};

  IndexSpec spec{{0, 1}};
  idx1.build_from_encoded(spec, enc_span1, default_memory_resource());
  idx2.build_from_encoded(spec, enc_span2, default_memory_resource());

  // Both should have size 1 before merge
  BOOST_REQUIRE_EQUAL(idx1.size(), 1);
  BOOST_REQUIRE_EQUAL(idx2.size(), 1);

  // After merge, both entries are kept (merge does not deduplicate)
  idx1.merge(idx2, 100);
  // Merge keeps both entries when keys are equal
  BOOST_REQUIRE_EQUAL(idx1.size(), 2);

  // Verify both entries exist
  auto entries = collect_index_entries(idx1);
  BOOST_REQUIRE_EQUAL(entries.size(), 2);
  BOOST_REQUIRE(entries.count({{1, 10}, 0}));    // Entry from idx1
  BOOST_REQUIRE(entries.count({{1, 10}, 100}));  // Entry from idx2 (with offset)
}

BOOST_AUTO_TEST_CASE(test_merge_preserves_sorted_order) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;

  IBT idx1(default_memory_resource());
  IBT idx2(default_memory_resource());

  // idx1: (1,10), (3,30)
  std::array<Vector<uint32_t>, 2> env1;
  env1[0] = {1, 3};
  env1[1] = {10, 30};
  const auto enc_span1 = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env1[0].data(), env1[0].size()),
      std::span<const uint32_t>(env1[1].data(), env1[1].size())};

  // idx2: (2,20), (4,40)
  std::array<Vector<uint32_t>, 2> env2;
  env2[0] = {2, 4};
  env2[1] = {20, 40};
  const auto enc_span2 = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env2[0].data(), env2[0].size()),
      std::span<const uint32_t>(env2[1].data(), env2[1].size())};

  IndexSpec spec{{0, 1}};
  idx1.build_from_encoded(spec, enc_span1, default_memory_resource());
  idx2.build_from_encoded(spec, enc_span2, default_memory_resource());

  idx1.merge(idx2, 100);
  BOOST_REQUIRE_EQUAL(idx1.size(), 4);

  // Verify sorted order: values() returns unique values (1, 2, 3, 4)
  auto root = idx1.root();
  BOOST_REQUIRE_EQUAL(root.unique_count(), 4);
  auto root_vals = root.values();
  std::vector<uint32_t> collected_vals(root_vals.begin(), root_vals.end());
  BOOST_REQUIRE_EQUAL(collected_vals.size(), 4);
  BOOST_REQUIRE_EQUAL(collected_vals[0], 1u);
  BOOST_REQUIRE_EQUAL(collected_vals[1], 2u);
  BOOST_REQUIRE_EQUAL(collected_vals[2], 3u);
  BOOST_REQUIRE_EQUAL(collected_vals[3], 4u);
}

BOOST_AUTO_TEST_CASE(test_node_handle_interface) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;
  IBT idx(default_memory_resource());

  std::array<Vector<uint32_t>, 2> env;
  env[0] = {1, 2, 3};
  env[1] = {10, 20, 30};

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  auto root = idx.root();
  BOOST_REQUIRE(root.valid());
  BOOST_REQUIRE(!root.is_leaf());
  BOOST_REQUIRE_EQUAL(root.depth(), 0);
  BOOST_REQUIRE_EQUAL(root.degree(), 3);

  // Test values() returns a range (DedupRange, which is iterable)
  auto vals = root.values();
  std::vector<uint32_t> collected_vals(vals.begin(), vals.end());
  BOOST_REQUIRE_EQUAL(collected_vals.size(), 3);

  // Test prefix navigation
  auto n1 = root.prefix(1);
  BOOST_REQUIRE(n1.valid());
  BOOST_REQUIRE(!n1.is_leaf());
  BOOST_REQUIRE_EQUAL(n1.depth(), 1);

  auto leaf = n1.prefix(10);
  BOOST_REQUIRE(leaf.valid());
  BOOST_REQUIRE(leaf.is_leaf());
  BOOST_REQUIRE_EQUAL(leaf.depth(), 2);

  auto rows = leaf.rows();
  BOOST_REQUIRE_EQUAL(rows.size(), 1);
}

BOOST_AUTO_TEST_CASE(test_contains_value) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;
  IBT idx(default_memory_resource());

  std::array<Vector<uint32_t>, 2> env;
  env[0] = {1, 2, 3, 4, 5};
  env[1] = {10, 20, 30, 40, 50};

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  auto root = idx.root();
  BOOST_REQUIRE(root.contains_value(1));
  BOOST_REQUIRE(root.contains_value(3));
  BOOST_REQUIRE(root.contains_value(5));
  BOOST_REQUIRE(!root.contains_value(0));
  BOOST_REQUIRE(!root.contains_value(6));
}

BOOST_AUTO_TEST_CASE(test_unique_count) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;
  IBT idx(default_memory_resource());

  std::array<Vector<uint32_t>, 2> env;
  env[0] = {1, 1, 2, 2, 3};  // Duplicates in first column
  env[1] = {10, 20, 30, 40, 50};

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  auto root = idx.root();
  // After deduplication, should have 3 unique values in first column
  BOOST_REQUIRE_EQUAL(root.unique_count(), 3);
}

BOOST_AUTO_TEST_CASE(test_bytes_used) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;
  IBT idx(default_memory_resource());

  BOOST_REQUIRE_EQUAL(idx.bytes_used(), 0);

  std::array<Vector<uint32_t>, 2> env;
  env[0] = {1, 2, 3};
  env[1] = {10, 20, 30};

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  BOOST_REQUIRE_GT(idx.bytes_used(), 0);
}

BOOST_AUTO_TEST_CASE(test_clear) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;
  IBT idx(default_memory_resource());

  std::array<Vector<uint32_t>, 2> env;
  env[0] = {1, 2, 3};
  env[1] = {10, 20, 30};

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  BOOST_REQUIRE_EQUAL(idx.size(), 3);
  idx.clear();
  BOOST_REQUIRE(idx.empty());
  BOOST_REQUIRE_EQUAL(idx.size(), 0);
}

BOOST_AUTO_TEST_CASE(test_eytzinger_acceleration) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;
  IBT idx(default_memory_resource());

  // Build a larger dataset to trigger Eytzinger index usage
  std::array<Vector<uint32_t>, 2> env;
  const std::size_t num_rows = 200;  // Enough to create multiple blocks
  env[0].reserve(num_rows);
  env[1].reserve(num_rows);

  for (std::size_t i = 0; i < num_rows; ++i) {
    env[0].push_back(static_cast<uint32_t>(i));
    env[1].push_back(static_cast<uint32_t>(i * 10));
  }

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  BOOST_REQUIRE_EQUAL(idx.size(), num_rows);

  // Test that prefix lookup works (should use Eytzinger acceleration at root)
  auto root = idx.root();
  auto node = root.prefix(100);
  BOOST_REQUIRE(node.valid());
  BOOST_REQUIRE(!node.is_leaf());

  auto leaf = node.prefix(1000);
  BOOST_REQUIRE(leaf.valid());
  BOOST_REQUIRE(leaf.is_leaf());
  BOOST_REQUIRE_EQUAL(leaf.rows().size(), 1);
}

BOOST_AUTO_TEST_CASE(test_single_column_index) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int>>;
  IBT idx(default_memory_resource());

  std::array<Vector<uint32_t>, 1> env;
  env[0] = {1, 2, 3, 4, 5, 3, 2, 1};

  const auto enc_span = std::array<std::span<const uint32_t>, 1>{
      std::span<const uint32_t>(env[0].data(), env[0].size())};

  IndexSpec spec{0};  // Single column index
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // After deduplication, should have 5 unique values: {1, 2, 3, 4, 5}
  BOOST_REQUIRE_EQUAL(idx.size(), 5);
  BOOST_REQUIRE(!idx.empty());

  auto root = idx.root();
  BOOST_REQUIRE(root.valid());
  BOOST_REQUIRE_EQUAL(root.unique_count(), 5);

  // Test prefix lookup
  auto node = root.prefix(3);
  BOOST_REQUIRE(node.valid());
  BOOST_REQUIRE(node.is_leaf());
  BOOST_REQUIRE_EQUAL(node.rows().size(), 1);
}

BOOST_AUTO_TEST_CASE(test_three_column_index) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int, int>>;
  IBT idx(default_memory_resource());

  std::array<Vector<uint32_t>, 3> env;
  env[0] = {1, 1, 2, 2, 3};
  env[1] = {10, 20, 10, 20, 30};
  env[2] = {100, 200, 300, 400, 500};

  const auto enc_span = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size()),
      std::span<const uint32_t>(env[2].data(), env[2].size())};

  IndexSpec spec{{0, 1, 2}};  // Three column index
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  BOOST_REQUIRE_EQUAL(idx.size(), 5);
  BOOST_REQUIRE(!idx.empty());

  auto root = idx.root();
  BOOST_REQUIRE(root.valid());
  BOOST_REQUIRE_EQUAL(root.unique_count(), 3);  // Unique values in first column: {1, 2, 3}

  // Test navigation through all three levels
  auto n1 = root.prefix(1);
  BOOST_REQUIRE(n1.valid());
  BOOST_REQUIRE(!n1.is_leaf());
  BOOST_REQUIRE_EQUAL(n1.unique_count(), 2);  // Unique values in second column: {10, 20}

  auto n2 = n1.prefix(10);
  BOOST_REQUIRE(n2.valid());
  BOOST_REQUIRE(!n2.is_leaf());
  BOOST_REQUIRE_EQUAL(n2.unique_count(), 1);  // Unique values in third column: {100}

  auto leaf = n2.prefix(100);
  BOOST_REQUIRE(leaf.valid());
  BOOST_REQUIRE(leaf.is_leaf());
  BOOST_REQUIRE_EQUAL(leaf.rows().size(), 1);
}

BOOST_AUTO_TEST_CASE(test_partial_column_index) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int, int>>;
  IBT idx(default_memory_resource());

  // Create data with 3 columns, but only index 2 of them
  std::array<Vector<uint32_t>, 3> env;
  env[0] = {1, 2, 3, 1, 2};
  env[1] = {10, 20, 30, 10, 20};
  env[2] = {100, 200, 300, 400, 500};  // This column is not indexed

  const auto enc_span = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size()),
      std::span<const uint32_t>(env[2].data(), env[2].size())};

  IndexSpec spec{{0, 1}};  // Only index first 2 columns, even though tuple has 3
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // After deduplication: (1,10), (2,20), (3,30) -> 3 unique tuples
  BOOST_REQUIRE_EQUAL(idx.size(), 3);
  BOOST_REQUIRE(!idx.empty());

  auto root = idx.root();
  BOOST_REQUIRE(root.valid());
  BOOST_REQUIRE_EQUAL(root.unique_count(), 3);  // Unique values: {1, 2, 3}
}

BOOST_AUTO_TEST_CASE(test_values_iteration_at_all_depths) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int, int>>;
  IBT idx(default_memory_resource());

  // Create data with duplicates to test deduplication
  std::array<Vector<uint32_t>, 3> env;
  env[0] = {1, 1, 2, 2, 3, 3, 3};                // First column with duplicates
  env[1] = {10, 20, 10, 20, 30, 40, 50};         // Second column
  env[2] = {100, 200, 300, 400, 500, 600, 700};  // Third column

  const auto enc_span = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size()),
      std::span<const uint32_t>(env[2].data(), env[2].size())};

  IndexSpec spec{{0, 1, 2}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // Test root level (depth 0) - should have 3 unique values: {1, 2, 3}
  auto root = idx.root();
  BOOST_REQUIRE(root.valid());
  BOOST_REQUIRE(!root.is_leaf());

  std::vector<uint32_t> root_values;
  for (auto val : root.values()) {
    root_values.push_back(val);
  }
  BOOST_REQUIRE_EQUAL(root_values.size(), 3);
  BOOST_REQUIRE_EQUAL(root_values[0], 1u);
  BOOST_REQUIRE_EQUAL(root_values[1], 2u);
  BOOST_REQUIRE_EQUAL(root_values[2], 3u);

  // Test second level (depth 1) - navigate to node with key 1
  auto n1 = root.prefix(1);
  BOOST_REQUIRE(n1.valid());
  BOOST_REQUIRE(!n1.is_leaf());

  std::vector<uint32_t> n1_values;
  for (auto val : n1.values()) {
    n1_values.push_back(val);
  }
  // Should have unique values from second column where first column = 1
  // Original data: (1,10), (1,20) -> unique second column values: {10, 20}
  BOOST_REQUIRE_EQUAL(n1_values.size(), 2);
  BOOST_REQUIRE_EQUAL(n1_values[0], 10u);
  BOOST_REQUIRE_EQUAL(n1_values[1], 20u);

  // Test third level (depth 2) - navigate to node with key (1, 10)
  auto n2 = n1.prefix(10);
  BOOST_REQUIRE(n2.valid());
  BOOST_REQUIRE(!n2.is_leaf());

  std::vector<uint32_t> n2_values;
  for (auto val : n2.values()) {
    n2_values.push_back(val);
  }
  // Should have unique values from third column where (first, second) = (1, 10)
  // Original data: (1,10,100) -> unique third column values: {100}
  BOOST_REQUIRE_EQUAL(n2_values.size(), 1);
  BOOST_REQUIRE_EQUAL(n2_values[0], 100u);

  // Test leaf level - navigate to leaf with key (1, 10, 100)
  auto leaf = n2.prefix(100);
  BOOST_REQUIRE(leaf.valid());
  BOOST_REQUIRE(leaf.is_leaf());

  // Leaf should not have values() (or should return empty range)
  auto leaf_values = leaf.values();
  std::vector<uint32_t> leaf_vals_collected(leaf_values.begin(), leaf_values.end());
  BOOST_REQUIRE_EQUAL(leaf_vals_collected.size(), 0);
}

BOOST_AUTO_TEST_CASE(test_values_with_seek_operation) {
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;
  IBT idx(default_memory_resource());

  // Build a larger dataset to potentially trigger Eytzinger index
  std::array<Vector<uint32_t>, 2> env;
  const std::size_t num_rows = 100;
  for (std::size_t i = 0; i < num_rows; ++i) {
    env[0].push_back(static_cast<uint32_t>(i % 10));  // Values 0-9 with duplicates
    env[1].push_back(static_cast<uint32_t>(i * 10));
  }

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  auto root = idx.root();
  BOOST_REQUIRE(root.valid());

  // Test that we can iterate all unique values
  std::vector<uint32_t> all_values;
  for (auto val : root.values()) {
    all_values.push_back(val);
  }

  // Should have 10 unique values: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  BOOST_REQUIRE_EQUAL(all_values.size(), 10);
  for (uint32_t i = 0; i < 10; ++i) {
    BOOST_REQUIRE_EQUAL(all_values[i], i);
  }

  // Test that values() iterator supports seek (if used by LeapfrogView)
  auto vals = root.values();
  auto it = vals.begin();
  BOOST_REQUIRE(it != vals.end());

  // Test seeking to a specific value
  it.seek(5);
  BOOST_REQUIRE(it != vals.end());
  BOOST_REQUIRE_GE(*it, 5u);

  // Test seeking beyond the range
  it.seek(100);
  // Iterator should be at or past end, or pointing to a value >= 100
  if (it != vals.end()) {
    BOOST_REQUIRE_GE(*it, 100u);
  }
}

BOOST_AUTO_TEST_CASE(test_values_completeness_arity_1) {
  // Test that values() returns ALL unique values, not fewer, at arity 1
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int>>;
  IBT idx(default_memory_resource());

  // Create data with many duplicates
  std::array<Vector<uint32_t>, 1> env;
  env[0] = {1, 1, 1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7, 7, 8, 9, 9};

  const auto enc_span = std::array<std::span<const uint32_t>, 1>{
      std::span<const uint32_t>(env[0].data(), env[0].size())};

  IndexSpec spec{0};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  auto root = idx.root();
  BOOST_REQUIRE(root.valid());

  // Collect all values
  std::vector<uint32_t> collected;
  for (auto val : root.values()) {
    collected.push_back(val);
  }

  // Should have exactly 9 unique values: {1, 2, 3, 4, 5, 6, 7, 8, 9}
  BOOST_REQUIRE_EQUAL(collected.size(), 9);
  std::vector<uint32_t> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  BOOST_REQUIRE_EQUAL_COLLECTIONS(collected.begin(), collected.end(), expected.begin(),
                                  expected.end());

  // Verify unique_count matches
  BOOST_REQUIRE_EQUAL(root.unique_count(), 9);
}

BOOST_AUTO_TEST_CASE(test_values_completeness_arity_2) {
  // Test that values() returns ALL unique values at each depth for arity 2
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;
  IBT idx(default_memory_resource());

  // Create data: (1,10), (1,20), (2,30), (2,40), (3,50)
  std::array<Vector<uint32_t>, 2> env;
  env[0] = {1, 1, 2, 2, 3};
  env[1] = {10, 20, 30, 40, 50};

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // Test root level (depth 0)
  auto root = idx.root();
  std::vector<uint32_t> root_vals;
  for (auto val : root.values()) {
    root_vals.push_back(val);
  }
  BOOST_REQUIRE_EQUAL(root_vals.size(), 3);
  BOOST_REQUIRE_EQUAL(root_vals[0], 1u);
  BOOST_REQUIRE_EQUAL(root_vals[1], 2u);
  BOOST_REQUIRE_EQUAL(root_vals[2], 3u);

  // Test depth 1 - navigate to key=1
  auto n1 = root.prefix(1);
  std::vector<uint32_t> n1_vals;
  for (auto val : n1.values()) {
    n1_vals.push_back(val);
  }
  BOOST_REQUIRE_EQUAL(n1_vals.size(), 2);
  BOOST_REQUIRE_EQUAL(n1_vals[0], 10u);
  BOOST_REQUIRE_EQUAL(n1_vals[1], 20u);

  // Test depth 1 - navigate to key=2
  auto n2 = root.prefix(2);
  std::vector<uint32_t> n2_vals;
  for (auto val : n2.values()) {
    n2_vals.push_back(val);
  }
  BOOST_REQUIRE_EQUAL(n2_vals.size(), 2);
  BOOST_REQUIRE_EQUAL(n2_vals[0], 30u);
  BOOST_REQUIRE_EQUAL(n2_vals[1], 40u);
}

BOOST_AUTO_TEST_CASE(test_values_completeness_arity_3) {
  // Test that values() returns ALL unique values at each depth for arity 3
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int, int>>;
  IBT idx(default_memory_resource());

  // Create data with duplicates at each level
  // Tuples: (1,10,100), (1,10,200), (1,20,300), (2,30,400), (2,40,500), (3,50,600)
  std::array<Vector<uint32_t>, 3> env;
  env[0] = {1, 1, 1, 2, 2, 3};              // First column: 1, 1, 1, 2, 2, 3
  env[1] = {10, 10, 20, 30, 40, 50};        // Second column
  env[2] = {100, 200, 300, 400, 500, 600};  // Third column

  const auto enc_span = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size()),
      std::span<const uint32_t>(env[2].data(), env[2].size())};

  IndexSpec spec{{0, 1, 2}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // Test root level - should have 3 unique values: {1, 2, 3}
  auto root = idx.root();
  std::vector<uint32_t> root_vals;
  for (auto val : root.values()) {
    root_vals.push_back(val);
  }
  BOOST_REQUIRE_EQUAL(root_vals.size(), 3);
  BOOST_REQUIRE_EQUAL(root_vals[0], 1u);
  BOOST_REQUIRE_EQUAL(root_vals[1], 2u);
  BOOST_REQUIRE_EQUAL(root_vals[2], 3u);

  // Test depth 1 - navigate to key=1, should have 2 unique values: {10, 20}
  auto n1 = root.prefix(1);
  std::vector<uint32_t> n1_vals;
  for (auto val : n1.values()) {
    n1_vals.push_back(val);
  }
  BOOST_REQUIRE_EQUAL(n1_vals.size(), 2);
  BOOST_REQUIRE_EQUAL(n1_vals[0], 10u);
  BOOST_REQUIRE_EQUAL(n1_vals[1], 20u);

  // Test depth 2 - navigate to key=(1,10), should have 2 unique values: {100, 200}
  // Because we have tuples (1,10,100) and (1,10,200)
  auto n2 = n1.prefix(10);
  std::vector<uint32_t> n2_vals;
  for (auto val : n2.values()) {
    n2_vals.push_back(val);
  }
  BOOST_REQUIRE_EQUAL(n2_vals.size(), 2);
  BOOST_REQUIRE_EQUAL(n2_vals[0], 100u);
  BOOST_REQUIRE_EQUAL(n2_vals[1], 200u);
}

BOOST_AUTO_TEST_CASE(test_seek_doesnt_skip_values) {
  // Critical test: ensure seek() doesn't skip values, especially with Eytzinger acceleration
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;
  IBT idx(default_memory_resource());

  // Build a larger dataset that will trigger Eytzinger index (depth 0)
  std::array<Vector<uint32_t>, 2> env;
  const std::size_t num_rows = 200;  // Enough to create multiple blocks
  for (std::size_t i = 0; i < num_rows; ++i) {
    env[0].push_back(static_cast<uint32_t>(i / 10));  // Values 0-19 with duplicates
    env[1].push_back(static_cast<uint32_t>(i * 10));
  }

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  auto root = idx.root();
  BOOST_REQUIRE(root.valid());

  // Method 1: Collect all values by iteration
  std::vector<uint32_t> all_by_iteration;
  for (auto val : root.values()) {
    all_by_iteration.push_back(val);
  }

  // Method 2: Collect all values by seeking to each expected value
  // This tests that seek() correctly finds each value when starting from beginning
  std::vector<uint32_t> all_by_seek;
  auto vals = root.values();

  // Reset iterator and seek to each value from the start
  for (uint32_t target = 0; target < 20; ++target) {
    auto it = vals.begin();  // Start from beginning each time
    it.seek(target);
    if (it != vals.end() && *it == target) {
      all_by_seek.push_back(*it);
    }
  }

  // Both methods should produce the same set of values
  BOOST_REQUIRE_EQUAL(all_by_iteration.size(), all_by_seek.size());
  BOOST_REQUIRE_EQUAL_COLLECTIONS(all_by_iteration.begin(), all_by_iteration.end(),
                                  all_by_seek.begin(), all_by_seek.end());

  // Verify we got all expected values
  BOOST_REQUIRE_EQUAL(all_by_iteration.size(), 20);
  for (uint32_t i = 0; i < 20; ++i) {
    BOOST_REQUIRE_EQUAL(all_by_iteration[i], i);
  }
}

BOOST_AUTO_TEST_CASE(test_seek_at_different_depths) {
  // Test that seek() works correctly at different depths (non-root nodes don't use Eytzinger)
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int, int>>;
  IBT idx(default_memory_resource());

  std::array<Vector<uint32_t>, 3> env;
  env[0] = {1, 1, 2, 2, 3, 3, 3};
  env[1] = {10, 20, 30, 40, 50, 60, 70};
  env[2] = {100, 200, 300, 400, 500, 600, 700};

  const auto enc_span = std::array<std::span<const uint32_t>, 3>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size()),
      std::span<const uint32_t>(env[2].data(), env[2].size())};

  IndexSpec spec{{0, 1, 2}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // Test seek at depth 1 (no Eytzinger)
  auto root = idx.root();
  auto n1 = root.prefix(1);
  BOOST_REQUIRE(n1.valid());

  // Collect by iteration
  std::vector<uint32_t> by_iteration;
  for (auto val : n1.values()) {
    by_iteration.push_back(val);
  }

  // Collect by seek
  std::vector<uint32_t> by_seek;
  auto vals = n1.values();
  auto it = vals.begin();
  it.seek(10);
  if (it != vals.end() && *it == 10) {
    by_seek.push_back(*it);
    ++it;
  }
  it.seek(20);
  if (it != vals.end() && *it == 20) {
    by_seek.push_back(*it);
  }

  // Should match
  BOOST_REQUIRE_EQUAL(by_iteration.size(), by_seek.size());
  BOOST_REQUIRE_EQUAL_COLLECTIONS(by_iteration.begin(), by_iteration.end(), by_seek.begin(),
                                  by_seek.end());
}

BOOST_AUTO_TEST_CASE(test_values_vs_unique_count) {
  // Ensure values() returns the same count as unique_count()
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;
  IBT idx(default_memory_resource());

  std::array<Vector<uint32_t>, 2> env;
  env[0] = {1, 1, 1, 2, 2, 3, 4, 4, 4, 5};
  env[1] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  auto root = idx.root();
  std::size_t count_by_iteration = 0;
  for ([[maybe_unused]] auto val : root.values()) {
    count_by_iteration++;
  }

  BOOST_REQUIRE_EQUAL(count_by_iteration, root.unique_count());
  BOOST_REQUIRE_EQUAL(count_by_iteration, 5);  // Should have 5 unique values: {1, 2, 3, 4, 5}
}

BOOST_AUTO_TEST_CASE(test_join_pattern_with_intersect) {
  // Test the exact pattern used in benchmarks: intersect() with values()
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;
  IBT idx(default_memory_resource());

  // Create data: (1,10), (1,20), (1,30), (2,40), (2,50), (3,60)
  std::array<Vector<uint32_t>, 2> env;
  env[0] = {1, 1, 1, 2, 2, 3};
  env[1] = {10, 20, 30, 40, 50, 60};

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // Pattern: for (auto x : intersect(root)) { for (auto y : intersect(root.prefix(x))) { ... } }
  auto root = idx.root();
  using IndexType = IBT;

  std::set<std::pair<uint32_t, uint32_t>> collected_pairs;
  for (auto x : IndexType::intersect(root)) {
    auto node_x = root.prefix(x);
    BOOST_REQUIRE(node_x.valid());
    for (auto y : IndexType::intersect(node_x)) {
      collected_pairs.insert({x, y});
    }
  }

  // Should collect all pairs: {(1,10), (1,20), (1,30), (2,40), (2,50), (3,60)}
  BOOST_REQUIRE_EQUAL(collected_pairs.size(), 6);
  BOOST_REQUIRE(collected_pairs.count({1, 10}));
  BOOST_REQUIRE(collected_pairs.count({1, 20}));
  BOOST_REQUIRE(collected_pairs.count({1, 30}));
  BOOST_REQUIRE(collected_pairs.count({2, 40}));
  BOOST_REQUIRE(collected_pairs.count({2, 50}));
  BOOST_REQUIRE(collected_pairs.count({3, 60}));
}

BOOST_AUTO_TEST_CASE(test_values_after_prefix_arity_2) {
  // Critical test: After prefix() on first column, iterate all values in second column
  // This is the common join pattern: find first column match, then iterate all matching second
  // column values
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;
  IBT idx(default_memory_resource());

  // Create data: (1,10), (1,20), (1,30), (2,40), (2,50), (3,60)
  // After prefix(1), we should get all second column values: {10, 20, 30}
  std::array<Vector<uint32_t>, 2> env;
  env[0] = {1, 1, 1, 2, 2, 3};
  env[1] = {10, 20, 30, 40, 50, 60};

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // Step 1: Get root and find first column match using prefix
  auto root = idx.root();
  BOOST_REQUIRE(root.valid());

  // Step 2: Use prefix to find all rows where first column = 1
  auto n1 = root.prefix(1);
  BOOST_REQUIRE(n1.valid());
  BOOST_REQUIRE(!n1.is_leaf());

  // Step 3: Iterate all values in second column that share prefix 1
  std::vector<uint32_t> second_col_vals;
  for (auto val : n1.values()) {
    second_col_vals.push_back(val);
  }

  // Should have ALL 3 values: {10, 20, 30}
  BOOST_REQUIRE_EQUAL(second_col_vals.size(), 3);
  BOOST_REQUIRE_EQUAL(second_col_vals[0], 10u);
  BOOST_REQUIRE_EQUAL(second_col_vals[1], 20u);
  BOOST_REQUIRE_EQUAL(second_col_vals[2], 30u);

  // Verify we can navigate to each leaf
  for (uint32_t expected_val : {10u, 20u, 30u}) {
    auto leaf = n1.prefix(expected_val);
    BOOST_REQUIRE(leaf.valid());
    BOOST_REQUIRE(leaf.is_leaf());
    BOOST_REQUIRE_EQUAL(leaf.rows().size(), 1);
  }

  // Test with another prefix: first column = 2
  auto n2 = root.prefix(2);
  BOOST_REQUIRE(n2.valid());
  std::vector<uint32_t> n2_vals;
  for (auto val : n2.values()) {
    n2_vals.push_back(val);
  }
  BOOST_REQUIRE_EQUAL(n2_vals.size(), 2);
  BOOST_REQUIRE_EQUAL(n2_vals[0], 40u);
  BOOST_REQUIRE_EQUAL(n2_vals[1], 50u);
}

BOOST_AUTO_TEST_CASE(test_values_after_prefix_with_duplicates) {
  // Test the join pattern with duplicates in second column
  // Data: (1,10), (1,10), (1,20), (2,30), (2,30), (2,40)
  // After prefix(1), values() should return {10, 20} (deduplicated)
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;
  IBT idx(default_memory_resource());

  std::array<Vector<uint32_t>, 2> env;
  env[0] = {1, 1, 1, 2, 2, 2};
  env[1] = {10, 10, 20, 30, 30, 40};

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // After prefix(1), should get deduplicated values: {10, 20}
  auto root = idx.root();
  auto n1 = root.prefix(1);
  BOOST_REQUIRE(n1.valid());

  std::vector<uint32_t> vals;
  for (auto val : n1.values()) {
    vals.push_back(val);
  }

  // Should have 2 unique values, not 3 (even though there are 3 rows)
  BOOST_REQUIRE_EQUAL(vals.size(), 2);
  BOOST_REQUIRE_EQUAL(vals[0], 10u);
  BOOST_REQUIRE_EQUAL(vals[1], 20u);

  // Verify unique_count matches
  BOOST_REQUIRE_EQUAL(n1.unique_count(), 2);
}

BOOST_AUTO_TEST_CASE(test_join_pattern_large_dataset) {
  // Test the join pattern with a larger dataset that might trigger Eytzinger
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;
  IBT idx(default_memory_resource());

  // Create data where first column has many duplicates
  // First column: 0,0,0,...,0, 1,1,1,...,1, 2,2,2,...,2 (each value appears 10 times)
  // Second column: unique values 0-99
  std::array<Vector<uint32_t>, 2> env;
  const std::size_t rows_per_key = 10;
  const std::size_t num_keys = 10;
  for (std::size_t key = 0; key < num_keys; ++key) {
    for (std::size_t i = 0; i < rows_per_key; ++i) {
      env[0].push_back(static_cast<uint32_t>(key));
      env[1].push_back(static_cast<uint32_t>(key * rows_per_key + i));
    }
  }

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // Test join pattern for key = 5
  auto root = idx.root();
  auto n1 = root.prefix(5);
  BOOST_REQUIRE(n1.valid());

  // Should get all 10 second column values: {50, 51, 52, ..., 59}
  std::vector<uint32_t> vals;
  for (auto val : n1.values()) {
    vals.push_back(val);
  }

  BOOST_REQUIRE_EQUAL(vals.size(), 10);
  for (std::size_t i = 0; i < 10; ++i) {
    BOOST_REQUIRE_EQUAL(vals[i], 50u + i);
  }
}

BOOST_AUTO_TEST_CASE(test_iteration_completeness_after_seek) {
  // Critical test: After seek(), ensure we can iterate all remaining values
  // This tests the exact pattern that might be failing in benchmarks
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;
  IBT idx(default_memory_resource());

  // Create data with many values
  std::array<Vector<uint32_t>, 2> env;
  for (uint32_t i = 0; i < 100; ++i) {
    env[0].push_back(i / 10);  // First column: 0,0,...,0, 1,1,...,1, ..., 9,9,...,9
    env[1].push_back(i);       // Second column: 0,1,2,...,99
  }

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  // Test: After prefix(5), should get all 10 values: {50, 51, ..., 59}
  auto root = idx.root();
  auto n1 = root.prefix(5);
  BOOST_REQUIRE(n1.valid());

  // Method 1: Direct iteration
  std::vector<uint32_t> by_iteration;
  for (auto val : n1.values()) {
    by_iteration.push_back(val);
  }

  // Method 2: Seek to start, then iterate
  auto vals = n1.values();
  auto it = vals.begin();
  it.seek(50);  // Seek to first value
  std::vector<uint32_t> by_seek_and_iterate;
  while (it != vals.end()) {
    by_seek_and_iterate.push_back(*it);
    ++it;
  }

  // Both should produce the same results
  BOOST_REQUIRE_EQUAL(by_iteration.size(), by_seek_and_iterate.size());
  BOOST_REQUIRE_EQUAL_COLLECTIONS(by_iteration.begin(), by_iteration.end(),
                                  by_seek_and_iterate.begin(), by_seek_and_iterate.end());

  // Should have exactly 10 values
  BOOST_REQUIRE_EQUAL(by_iteration.size(), 10);
  for (uint32_t i = 0; i < 10; ++i) {
    BOOST_REQUIRE_EQUAL(by_iteration[i], 50u + i);
  }
}

BOOST_AUTO_TEST_CASE(test_skip_duplicates_doesnt_skip_too_far) {
  // Test that skip_duplicates doesn't skip past the end or skip valid values
  using SR = BooleanSR;
  using IBT = ImplicitBTreeIndex<SR, std::tuple<int, int>>;
  IBT idx(default_memory_resource());

  // Create data: (1,10), (1,20), (1,30), (2,40), (2,50)
  std::array<Vector<uint32_t>, 2> env;
  env[0] = {1, 1, 1, 2, 2};
  env[1] = {10, 20, 30, 40, 50};

  const auto enc_span = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env[0].data(), env[0].size()),
      std::span<const uint32_t>(env[1].data(), env[1].size())};

  IndexSpec spec{{0, 1}};
  idx.build_from_encoded(spec, enc_span, default_memory_resource());

  auto root = idx.root();
  auto n1 = root.prefix(1);
  BOOST_REQUIRE(n1.valid());

  // Manually iterate and verify we get all 3 values
  auto vals = n1.values();
  auto it = vals.begin();
  std::vector<uint32_t> collected;

  while (it != vals.end()) {
    collected.push_back(*it);
    ++it;
  }

  BOOST_REQUIRE_EQUAL(collected.size(), 3);
  BOOST_REQUIRE_EQUAL(collected[0], 10u);
  BOOST_REQUIRE_EQUAL(collected[1], 20u);
  BOOST_REQUIRE_EQUAL(collected[2], 30u);
}

}  // namespace
