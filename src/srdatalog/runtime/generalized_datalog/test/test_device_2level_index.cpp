#define BOOST_TEST_MODULE device_2level_index_test
#include "gpu/gpu_api.h"
#include <boost/test/included/unit_test.hpp>

#include <tuple>
#include <vector>

#include "gpu/device_2level_index.h"
#include "gpu/device_array.h"
#include "gpu/index_concepts.h"
#include "gpu/index_ops.h"
#include "gpu/nd_device_array.h"
#include "semiring.h"
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

using SRDatalog::IndexSpec;
using SRDatalog::GPU::Device2LevelIndex;
using SRDatalog::GPU::DeviceArray;
using SRDatalog::GPU::DeviceSortedArrayIndex;
using SRDatalog::GPU::NDDeviceArray;

// Type aliases for common test configurations
using BinaryIndex = Device2LevelIndex<BooleanSR, std::tuple<uint32_t, uint32_t>>;
using BinaryDSAI = DeviceSortedArrayIndex<BooleanSR, std::tuple<uint32_t, uint32_t>>;

// ============================================================================
// Concept Checks (compile-time)
// ============================================================================

static_assert(SRDatalog::GPU::IndexReader<BinaryIndex>,
              "Device2LevelIndex must satisfy IndexReader concept");
static_assert(SRDatalog::GPU::IndexWriter<BinaryIndex>,
              "Device2LevelIndex must satisfy IndexWriter concept");

// ============================================================================
// Helpers
// ============================================================================

template <typename T>
DeviceArray<T> host_to_device(const std::vector<T>& host) {
  DeviceArray<T> dev(host.size());
  thrust::copy(host.begin(), host.end(), dev.begin());
  return dev;
}

// Build a 2-level index from host edge data
static BinaryIndex build_2level(const std::vector<uint32_t>& col0,
                                const std::vector<uint32_t>& col1, const IndexSpec& spec) {
  NDDeviceArray<uint32_t, 2> cols(col0.size());
  auto d0 = host_to_device(col0);
  auto d1 = host_to_device(col1);
  thrust::copy(d0.begin(), d0.end(), cols.column_ptr(0));
  thrust::copy(d1.begin(), d1.end(), cols.column_ptr(1));

  std::monostate ms;
  BinaryIndex idx;
  idx.build_from_encoded_device(spec, cols, ms);
  return idx;
}

// ============================================================================
// Tests
// ============================================================================

BOOST_AUTO_TEST_CASE(test_empty_index) {
  BinaryIndex idx;
  BOOST_CHECK(idx.empty());
  BOOST_CHECK_EQUAL(idx.size(), 0);
}

BOOST_AUTO_TEST_CASE(test_build_and_query) {
  IndexSpec spec{{0, 1}};
  auto idx = build_2level({1, 2, 3}, {2, 3, 4}, spec);

  BOOST_CHECK(!idx.empty());
  BOOST_CHECK_EQUAL(idx.size(), 3);

  // After build, data is in full_ segment
  BOOST_CHECK_EQUAL(idx.full().size(), 3);
  BOOST_CHECK(idx.head().empty());
}

BOOST_AUTO_TEST_CASE(test_merge_into_head) {
  IndexSpec spec{{0, 1}};

  auto idx = build_2level({1, 2, 3}, {2, 3, 4}, spec);
  BOOST_CHECK_EQUAL(idx.size(), 3);

  auto delta = build_2level({4, 5}, {5, 6}, spec);
  idx.merge(delta, 0);

  // HEAD should now have the delta content
  BOOST_CHECK(!idx.head().empty());
  BOOST_CHECK_EQUAL(idx.size(), 5);
}

BOOST_AUTO_TEST_CASE(test_compaction) {
  IndexSpec spec{{0, 1}};

  auto idx = build_2level({1, 2, 3}, {2, 3, 4}, spec);
  auto delta = build_2level({4, 5}, {5, 6}, spec);
  idx.merge(delta, 0);

  idx.compact();

  BOOST_CHECK(idx.head().empty());
  BOOST_CHECK_EQUAL(idx.full().size(), 5);
  BOOST_CHECK_EQUAL(idx.size(), 5);
}

BOOST_AUTO_TEST_CASE(test_view_returns_full_only) {
  IndexSpec spec{{0, 1}};

  auto idx = build_2level({1, 2}, {2, 3}, spec);
  auto delta = build_2level({3, 4}, {4, 5}, spec);
  idx.merge(delta, 0);

  BOOST_CHECK(!idx.head().empty());

  // view() returns FULL segment only — no compaction on read.
  // HEAD stays intact; only full_ is returned as the view.
  auto v = idx.view();
  (void)v;

  BOOST_CHECK(!idx.head().empty());
  BOOST_CHECK_EQUAL(idx.full().size(), 2);
  BOOST_CHECK_EQUAL(idx.size(), 4);
}

BOOST_AUTO_TEST_CASE(test_clear) {
  IndexSpec spec{{0, 1}};

  auto idx = build_2level({1, 2, 3}, {2, 3, 4}, spec);
  BOOST_CHECK(!idx.empty());

  idx.clear();
  BOOST_CHECK(idx.empty());
  BOOST_CHECK_EQUAL(idx.size(), 0);
}

BOOST_AUTO_TEST_CASE(test_set_difference_update) {
  IndexSpec spec{{0, 1}};

  // "newt" = {(1,2), (2,3), (3,4), (4,5)}
  auto newt = build_2level({1, 2, 3, 4}, {2, 3, 4, 5}, spec);

  // "full" = {(1,2), (2,3)}
  auto full = build_2level({1, 2}, {2, 3}, spec);

  // delta = newt - full = {(3,4), (4,5)}
  BinaryIndex delta;

  newt.set_difference_update(full, delta);

  BOOST_CHECK_EQUAL(delta.size(), 2);
}

BOOST_AUTO_TEST_CASE(test_adl_merge_index) {
  IndexSpec spec{{0, 1}};

  auto full_idx = build_2level({1, 2, 3}, {2, 3, 4}, spec);
  auto delta_idx = build_2level({4, 5}, {5, 6}, spec);

  // Use the CPO dispatch — should pick up our merge_index_impl via ADL
  struct DummyRel {};
  DummyRel full_rel, delta_rel;

  SRDatalog::GPU::index_ops::merge_index(full_idx, delta_idx, full_rel, delta_rel);

  // Total should be 5. merge_index_impl compacts when HEAD > 10% of FULL.
  // Here: FULL=3, delta=2 → HEAD would be 2 > 0.3 → compaction fires.
  // After compaction: HEAD empty, FULL=5.
  BOOST_CHECK_EQUAL(full_idx.size(), 5);
  BOOST_CHECK(full_idx.head().empty());
  BOOST_CHECK_EQUAL(full_idx.full().size(), 5);
}

BOOST_AUTO_TEST_CASE(test_multiple_merges_then_compact) {
  IndexSpec spec{{0, 1}};

  auto idx = build_2level({1}, {2}, spec);

  for (uint32_t i = 2; i <= 10; ++i) {
    auto delta = build_2level({i}, {i + 1}, spec);
    idx.merge(delta, 0);
  }

  BOOST_CHECK_EQUAL(idx.size(), 10);

  idx.compact();
  BOOST_CHECK(idx.head().empty());
  BOOST_CHECK_EQUAL(idx.full().size(), 10);
}

BOOST_AUTO_TEST_CASE(test_merge_with_duplicates) {
  IndexSpec spec{{0, 1}};

  auto idx = build_2level({1, 2, 3}, {2, 3, 4}, spec);

  // Merge with overlapping edges — (2,3) is duplicate
  auto delta = build_2level({2, 4}, {3, 5}, spec);
  idx.merge(delta, 0);

  // DSAI merge doesn't dedup — set_difference_update handles that
  idx.compact();
  BOOST_CHECK(idx.full().size() >= 4);
}

// ============================================================================
// Multi-View Access Tests (full_view/head_view for segment iteration)
// ============================================================================

BOOST_AUTO_TEST_CASE(test_full_view_after_build) {
  IndexSpec spec{{0, 1}};

  auto idx = build_2level({1, 2, 3}, {10, 20, 30}, spec);

  // After build, all data is in FULL segment
  auto fv = idx.full_view();
  auto fr = idx.full_root();

  BOOST_CHECK(fr.valid());
  BOOST_CHECK_EQUAL(fr.degree(), 3);

  // HEAD should be empty
  auto hv = idx.head_view();
  auto hr = idx.head_root();
  BOOST_CHECK(!hr.valid());
  BOOST_CHECK_EQUAL(hr.degree(), 0);
}

BOOST_AUTO_TEST_CASE(test_both_views_after_merge) {
  IndexSpec spec{{0, 1}};

  // Build initial data into FULL
  auto idx = build_2level({1, 2, 3}, {10, 20, 30}, spec);

  // Merge delta into HEAD (no compaction)
  auto delta = build_2level({4, 5}, {40, 50}, spec);
  idx.merge(delta, 0);

  // FULL still has 3 entries
  auto fr = idx.full_root();
  BOOST_CHECK(fr.valid());
  BOOST_CHECK_EQUAL(fr.degree(), 3);

  // HEAD now has 2 entries
  auto hr = idx.head_root();
  BOOST_CHECK(hr.valid());
  BOOST_CHECK_EQUAL(hr.degree(), 2);

  // Total size = FULL + HEAD
  BOOST_CHECK_EQUAL(idx.size(), 5);
}

BOOST_AUTO_TEST_CASE(test_views_independent_iteration) {
  IndexSpec spec{{0, 1}};

  auto idx = build_2level({10, 20}, {100, 200}, spec);
  auto delta = build_2level({30}, {300}, spec);
  idx.merge(delta, 0);

  // Iterate FULL segment independently
  auto fv = idx.full_view();
  auto fr = idx.full_root();
  BOOST_REQUIRE(fr.valid());
  BOOST_CHECK_EQUAL(fr.degree(), 2);

  // Iterate HEAD segment independently
  auto hv = idx.head_view();
  auto hr = idx.head_root();
  BOOST_REQUIRE(hr.valid());
  BOOST_CHECK_EQUAL(hr.degree(), 1);

  // Both views share the same NodeView type (can be stored in same array)
  static_assert(std::is_same_v<decltype(fv), decltype(hv)>,
                "full_view() and head_view() must return the same type");
}

BOOST_AUTO_TEST_CASE(test_views_after_multiple_merges) {
  IndexSpec spec{{0, 1}};

  auto idx = build_2level({1}, {10}, spec);

  // Multiple merges accumulate in HEAD
  for (uint32_t i = 2; i <= 5; ++i) {
    auto delta = build_2level({i}, {i * 10}, spec);
    idx.merge(delta, 0);
  }

  // FULL has 1 entry, HEAD has accumulated 4 merges
  BOOST_CHECK_EQUAL(idx.full().size(), 1);
  BOOST_CHECK(!idx.head().empty());
  BOOST_CHECK_EQUAL(idx.size(), 5);

  // Both views are independently valid
  auto fr = idx.full_root();
  auto hr = idx.head_root();
  BOOST_CHECK(fr.valid());
  BOOST_CHECK(hr.valid());
  BOOST_CHECK_EQUAL(fr.degree() + hr.degree(), 5);
}
