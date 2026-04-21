// Test-driven development for SortedArrayIndex merge function
#define BOOST_TEST_MODULE test_sorted_array_merge
#include "column.h"
#include "sorted_array_index.h"
#include <boost/test/included/unit_test.hpp>
#include <set>

using namespace SRDatalog;

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

BOOST_AUTO_TEST_CASE(test_merge_empty_into_empty) {
  using SR = BooleanSR;
  using SAI = SortedArrayIndex<SR, std::tuple<int, int>>;

  SAI idx1(default_memory_resource());
  SAI idx2(default_memory_resource());

  idx1.merge(idx2, 0);
  BOOST_REQUIRE_EQUAL(idx1.size(), 0);
  BOOST_REQUIRE(idx1.empty());
  BOOST_REQUIRE(idx1.verify_sorted());
}

BOOST_AUTO_TEST_CASE(test_merge_empty_into_nonempty) {
  using SR = BooleanSR;
  using SAI = SortedArrayIndex<SR, std::tuple<int, int>>;

  SAI idx1(default_memory_resource());
  SAI idx2(default_memory_resource());

  // Build idx1 with some data
  std::array<Vector<uint32_t>, 2> env1;
  env1[0] = {1, 2};
  env1[1] = {10, 20};
  const auto enc_span1 = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env1[0].data(), env1[0].size()),
      std::span<const uint32_t>(env1[1].data(), env1[1].size())};

  IndexSpec spec{{0, 1}};
  idx1.build_from_encoded(spec, enc_span1, default_memory_resource());
  BOOST_REQUIRE(idx1.verify_sorted());

  std::size_t size_before = idx1.size();
  idx1.merge(idx2, 0);
  BOOST_REQUIRE_EQUAL(idx1.size(), size_before);
  BOOST_REQUIRE(idx1.verify_sorted());
}

BOOST_AUTO_TEST_CASE(test_merge_nonempty_into_empty) {
  using SR = BooleanSR;
  using SAI = SortedArrayIndex<SR, std::tuple<int, int>>;

  SAI idx1(default_memory_resource());
  SAI idx2(default_memory_resource());

  // Build idx2 with some data
  std::array<Vector<uint32_t>, 2> env2;
  env2[0] = {1, 2};
  env2[1] = {10, 20};
  const auto enc_span2 = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env2[0].data(), env2[0].size()),
      std::span<const uint32_t>(env2[1].data(), env2[1].size())};

  IndexSpec spec{{0, 1}};
  idx2.build_from_encoded(spec, enc_span2, default_memory_resource());
  BOOST_REQUIRE(idx2.verify_sorted());

  idx1.merge(idx2, 100);
  BOOST_REQUIRE_EQUAL(idx1.size(), 2);
  BOOST_REQUIRE(idx1.verify_sorted());

  // Verify row IDs are offset
  auto entries = collect_index_entries(idx1);
  BOOST_REQUIRE_EQUAL(entries.size(), 2);
  for (const auto& [key, row_id] : entries) {
    BOOST_REQUIRE_GE(row_id, 100u);
  }
}

BOOST_AUTO_TEST_CASE(test_merge_disjoint_keys) {
  using SR = BooleanSR;
  using SAI = SortedArrayIndex<SR, std::tuple<int, int>>;

  SAI idx1(default_memory_resource());
  SAI idx2(default_memory_resource());

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
  BOOST_REQUIRE(idx1.verify_sorted());

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
  using SAI = SortedArrayIndex<SR, std::tuple<int, int>>;

  SAI idx1(default_memory_resource());
  SAI idx2(default_memory_resource());

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
  BOOST_REQUIRE(idx1.verify_sorted());
  idx2.build_from_encoded(spec, enc_span2, default_memory_resource());
  BOOST_REQUIRE(idx2.verify_sorted());

  idx1.merge(idx2, 100);
  BOOST_REQUIRE_EQUAL(idx1.size(), 4);
  BOOST_REQUIRE(idx1.verify_sorted());

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
  using SAI = SortedArrayIndex<SR, std::tuple<int, int>>;

  SAI idx1(default_memory_resource());
  SAI idx2(default_memory_resource());

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
  BOOST_REQUIRE(idx1.verify_sorted());
  idx2.build_from_encoded(spec, enc_span2, default_memory_resource());
  BOOST_REQUIRE(idx2.verify_sorted());

  // Both should have size 1 before merge
  BOOST_REQUIRE_EQUAL(idx1.size(), 1);
  BOOST_REQUIRE_EQUAL(idx2.size(), 1);

  // After merge, both entries are kept (merge does not deduplicate)
  idx1.merge(idx2, 100);
  BOOST_REQUIRE(idx1.verify_sorted());
  // Merge keeps both entries when keys are equal
  BOOST_REQUIRE_EQUAL(idx1.size(), 2);

  // Verify both entries exist
  auto entries = collect_index_entries(idx1);
  BOOST_REQUIRE_EQUAL(entries.size(), 2);
  BOOST_REQUIRE(entries.count({{1, 10}, 0}));   // Entry from idx1
  BOOST_REQUIRE(entries.count({{1, 10}, 100}));  // Entry from idx2 (with offset)
}

BOOST_AUTO_TEST_CASE(test_merge_preserves_sorted_order) {
  using SR = BooleanSR;
  using SAI = SortedArrayIndex<SR, std::tuple<int, int>>;

  SAI idx1(default_memory_resource());
  SAI idx2(default_memory_resource());

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
  BOOST_REQUIRE(idx1.verify_sorted());
  idx2.build_from_encoded(spec, enc_span2, default_memory_resource());
  BOOST_REQUIRE(idx2.verify_sorted());

  idx1.merge(idx2, 100);
  BOOST_REQUIRE_EQUAL(idx1.size(), 4);
  BOOST_REQUIRE(idx1.verify_sorted());

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

BOOST_AUTO_TEST_CASE(test_merge_with_large_offset) {
  using SR = BooleanSR;
  using SAI = SortedArrayIndex<SR, std::tuple<int, int>>;

  SAI idx1(default_memory_resource());
  SAI idx2(default_memory_resource());

  // idx1: (1,10)
  std::array<Vector<uint32_t>, 2> env1;
  env1[0] = {1};
  env1[1] = {10};
  const auto enc_span1 = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env1[0].data(), env1[0].size()),
      std::span<const uint32_t>(env1[1].data(), env1[1].size())};

  // idx2: (2,20)
  std::array<Vector<uint32_t>, 2> env2;
  env2[0] = {2};
  env2[1] = {20};
  const auto enc_span2 = std::array<std::span<const uint32_t>, 2>{
      std::span<const uint32_t>(env2[0].data(), env2[0].size()),
      std::span<const uint32_t>(env2[1].data(), env2[1].size())};

  IndexSpec spec{{0, 1}};
  idx1.build_from_encoded(spec, enc_span1, default_memory_resource());
  BOOST_REQUIRE(idx1.verify_sorted());
  idx2.build_from_encoded(spec, enc_span2, default_memory_resource());
  BOOST_REQUIRE(idx2.verify_sorted());

  std::size_t large_offset = 1000000;
  idx1.merge(idx2, large_offset);
  BOOST_REQUIRE_EQUAL(idx1.size(), 2);
  BOOST_REQUIRE(idx1.verify_sorted());

  // Verify row IDs are correctly offset
  auto entries = collect_index_entries(idx1);
  BOOST_REQUIRE_EQUAL(entries.size(), 2);
  for (const auto& [key, row_id] : entries) {
    if (key[0] == 2) {
      BOOST_REQUIRE_EQUAL(row_id, large_offset);
    }
  }
}
