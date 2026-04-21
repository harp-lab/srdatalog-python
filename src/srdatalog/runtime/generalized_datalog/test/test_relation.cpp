// test_relation.cpp (revised & comprehensive)
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <ranges>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

// ===== Include your relation header (it pulls in the .ipp internally) =====
#include "query.h"
#include "relation_col.h"
#include "sorted_array_index.h"

using namespace SRDatalog;

// -----------------------------
// A tiny test semiring (ℕ,+,×)
// -----------------------------
struct NatSemiring {
  using value_type = std::uint64_t;
  static constexpr value_type add(const value_type& a, const value_type& b) {
    return a + b;
  }
  static constexpr value_type mul(const value_type& a, const value_type& b) {
    return a * b;
  }
  static constexpr value_type zero() {
    return 0ull;
  }
  static constexpr value_type one() {
    return 1ull;
  }
};

// -----------------------------
// Small helpers for assertions
// -----------------------------
template <class R>
static void expect_size(const R& r, std::size_t n) {
  assert(r.size() == n && "size() mismatch");
}

template <class A, class B>
static void expect_eq(const A& a, const B& b, const char* msg) {
  if (!(a == b)) {
    std::cerr << "EXPECT_EQ failed: " << msg << "\n";
    std::abort();
  }
}

template <class A, class B>
static void expect_ne(const A& a, const B& b, const char* msg) {
  if (!(a != b)) {
    std::cerr << "EXPECT_NE failed: " << msg << "\n";
    std::abort();
  }
}

int main() {
  using SR = NatSemiring;
  // Explicitly use std::size_t as ValueType to maintain backward compatibility with
  // std::size_t-based test data for HashTrieIndex tests
  using Rel = SRDatalog::Relation<SR, std::tuple<int, std::string, int>, SRDatalog::HashTrieIndex,
                                  SRDatalog::HostRelationPolicy,
                                  std::size_t>;  // columns: c0=int, c1=string, c2=int;
                                                 // ann: uint64_t

  std::cout << "\n==== 1) Construct & append rows ====\n";
  Rel R;
  R.set_column_names({"id", "name", "val"});

  R.reserve(8);
  R.push_row({1, std::string("alice"), 10}, SR::one());  // row 0

  R.push_row({2, std::string("bob"), 20}, SR::one());  // row 1
  R.push_row({3, std::string("carol"), 30}, 5);        // row 2, ann=5
  R.push_row({2, std::string("bob"), 40}, 7);          // row 3

  expect_size(R, 4);
  std::cout << "== head(10) ==\n";
  R.head(10, std::cout);

  // Underlying columns & ann
  auto& c0 = R.column<0>();
  auto& c1 = R.column<1>();
  auto& c2 = R.column<2>();
  auto& A = R.provenance();

  expect_eq<std::size_t>(c0.size(), 4, "c0.size");
  expect_eq(c0[0], 1, "c0[0]");
  expect_eq(c1[1], std::string("bob"), "c1[1]");
  expect_eq(c2[2], 30, "c2[2]");
  expect_eq(A[2], 5ull, "ann[2]");

  std::cout << "\n==== 3) Lazy projection & collect ====\n";
  {
    R.ensure_index({{0, 1, 2}});
    auto root_cnt = R.get_index({{0, 1, 2}}).root().unique_count();
    expect_eq(root_cnt, 3, "root_cnt");
  }

  // std::cout << "\n==== 4) Projection with row-filter & ann_at ====\n";
  // {
  //   std::vector<Rel::RowId> rows = {1, 3};
  //   auto pv = R.project<1, 2>(rows);

  //   expect_eq<std::size_t>(std::ranges::size(pv), 2, "pv(size with filter)");

  //   std::cout << "--- pv.head(10) (filtered) ---\n";
  //   for (auto&& [name, val, ann] : pv | std::ranges::views::take(10)) {
  //     std::cout << "  ( " << name << ", " << val << " ), " << ann << "\n";
  //   }
  //   std::cout << "------------------------------\n";

  //   expect_eq(std::get<0>(pv[0]), std::string("bob"), "pv[0].name");

  //   expect_eq(std::get<1>(pv[1]), 40, "pv[1].val");

  //   expect_eq(std::get<2>(pv[1]), 7ull, "pv.ann_at(1)");
  // }

  // std::cout << "\n==== 6) Subset copy (materialize selected row ids) ====\n";
  // {
  //   std::vector<Rel::RowId> take = {0, 2};
  //   auto S = R.project<0, 1>(take);
  //   // collect the result into a vector
  //   auto S_vec = S | SRDatalog::to<std::vector>();
  //   expect_size(S, 2);
  //   expect_eq(std::get<0>(S_vec[0]), 1, "S.c0[0]");
  //   expect_eq(std::get<1>(S_vec[1]), std::string("carol"), "S.c1[1]");
  //   expect_eq(std::get<2>(S_vec[1]), 5ull, "S.ann[1]");
  // }

  // --------------------------------------------------------------------
  //  HashTrie: build_from_encoded, NodeHandle API, hash-probe intersection
  // --------------------------------------------------------------------
  std::cout << "\n==== 8) HashTrieIndex::build with encoder (over Relation) ====\n";
  {
    // Build index on (id, val) from relation R via encoder enc(i,row)
    // Here, for demonstration we simply fetch encoded as (id,val) directly
    IndexSpec spec{{0, 2}};
    const auto& idx = R.ensure_index(spec);  // lazy build uses idx.build(owner, spec, enc)
    assert(R.is_dirty(spec) == false);

    // Root and a couple of lookups
    auto root = idx.root();
    expect_eq(root.valid(), true, "root(valid)");

    // Values at root should include {1,2,3} for our current R rows:
    // rows: (1,10), (2,20), (3,30), (2,40)
    auto rv = root.values();
    std::unordered_set<std::size_t> rset(rv.begin(), rv.end());
    expect_eq(rset.count(1u) == 1, true, "root has 1");
    expect_eq(rset.count(2u) == 1, true, "root has 2");
    expect_eq(rset.count(3u) == 1, true, "root has 3");

    // (2,20) leaf should contain some row id(s). We don't assert exact rid
    // mapping here, but we do check the API surface.
    const auto key_220 = Prefix<int, int>{2, 20}.encoded();
    auto h = idx.prefix_lookup(key_220);
    if (h.valid() && h.is_leaf()) {
      auto rr = h.rows();
      std::cout << "(2,20) postings size = " << rr.size() << "\n";
    }

    // run contains_value/degree smoke at depth 1
    const auto key_2 = Prefix<int>{2}.encoded();
    auto n2 = idx.prefix_lookup(key_2);
    if (n2.valid() && !n2.is_leaf()) {
      auto v2 = n2.values();
      expect_eq(n2.degree(), v2.size(), "degree==values.size at n2");
      for (auto x : v2) {
        expect_eq(n2.contains_value(x), true, "n2.contains_value(x)");
      }
    }
  }

  std::cout << "\n==== 9) HashProbeIntersectView (small-side probe) ====\n";
  {
    // Build a manual 2-level index with overlapping children to exercise
    // intersection.
    using HT = typename Rel::IndexTypeInst;
    IndexSpec spec{{0, 1}};
    // tree at root has children {1,2,3}
    // at key=2, children {20,40}
    // at key=3, children {30,50}
    // SRDatalog::Vector<SRDatalog::Vector<std::size_t>> enc{
    //     {1, 10}, {2, 20}, {2, 40}, {3, 30}, {3, 50}};
    std::array<Vector<std::size_t>, 3> env;
    env[0] = {1, 2, 2, 3, 3};
    env[1] = {10, 20, 40, 30, 50};
    env[2] = {encode_to_size_t(std::string("alice")), encode_to_size_t(std::string("bob")),
              encode_to_size_t(std::string("carol")), encode_to_size_t(std::string("dave")),
              encode_to_size_t(std::string("erin"))};
    HT idx(default_memory_resource());
    const auto enc_span = std::array<std::span<const std::size_t>, 3>{
        std::span(env[0].data(), env[0].size()), std::span(env[1].data(), env[1].size()),
        std::span(env[2].data(), env[2].size())};
    idx.build_from_encoded(spec, enc_span, default_memory_resource());

    auto root = idx.root();
    auto n2 = root.prefix(std::size_t(2));  // children {20,40}
    auto n3 = root.prefix(std::size_t(3));  // children {30,50}

    // Intersect children(n2) ∩ children(n3) = {}
    {
      auto view = HT::intersect(n2, n3);
      auto count = std::ranges::distance(view);
      expect_eq<std::size_t>(count, 0, "intersection at depth1 (2 vs 3) is empty");
    }

    // Intersect n2 with another node having overlap:
    // make node nX with children {40,99}
    HT idx2(default_memory_resource());
    // idx2.build_from_encoded(spec, {{2, 40}, {9, 99}, {2, 99}},
    //                         default_memory_resource());
    std::array<Vector<std::size_t>, 3> env2;
    env2[0] = {2, 2, 2};
    env2[1] = {40, 99, 99};
    env2[2] = {encode_to_size_t(std::string("alice")), encode_to_size_t(std::string("bob")),
               encode_to_size_t(std::string("carol"))};
    const auto enc_span2 = std::array<std::span<const std::size_t>, 3>{
        std::span(env2[0].data(), env2[0].size()), std::span(env2[1].data(), env2[1].size()),
        std::span(env2[2].data(), env2[2].size())};
    idx2.build_from_encoded(spec, enc_span2, default_memory_resource());
    auto n2x = idx2.root().prefix(std::size_t(2));  // children {40,99}
    {
      // Overlap is {40}
      auto view = HT::intersect(n2, n2x);
      auto got = view | SRDatalog::to<SRDatalog::Vector>();
      expect_eq<std::size_t>(got.size(), 1, "probe size==1");
      expect_eq<std::size_t>(got[0], 40, "probe overlap==40");
    }

    // Also test that the view is lazy and zero-allocation (cannot assert
    // exactly), but we can at least iterate twice freshly:
    {
      auto view1 = HT::intersect(n2, n2x);
      auto count1 = std::ranges::distance(view1);
      expect_eq<std::size_t>(count1, 1, "probe iter pass1");

      auto view2 = HT::intersect(n2, n2x);
      auto count2 = std::ranges::distance(view2);
      expect_eq<std::size_t>(count2, 1, "probe iter pass2");
    }
    // -------------------------------------------------------
    // 10) Lazy ensure_index freshness across writes (version bump)
    // -------------------------------------------------------
    std::cout << "\n==== 10) ensure_index freshness across writes ====\n";
    {
      IndexSpec spec{{0, 2}};
      const auto& idx1 = R.ensure_index(spec);
      auto bytes1 = idx1.bytes_used();

      // mutate relation (append one row) -> should mark indexes dirty
      R.push_row({4, std::string("dave"), 50}, SR::one());

      // ensure_index should rebuild (fresh index)
      const auto& idx2 = R.ensure_index(spec);
      auto bytes2 = idx2.bytes_used();

      (void)bytes1;
      (void)bytes2;
    }

    // -------------------------------------------------------
    // 11) Spans basic invariants (valid until rebuild)
    // -------------------------------------------------------
    std::cout << "\n==== 11) Span invariants ====\n";
    {
      IndexSpec spec{{0, 2}};
      const auto& idx = R.ensure_index(spec);
      auto hroot = idx.root();
      auto vals_before = hroot.values();  // span view

      // use the span
      std::size_t s1 = vals_before.size();
      (void)s1;

      // cause a write -> mark dirty, then re-ensure (rebuild)
      R.push_row({5, std::string("erin"), 60}, SR::one());
      const auto& idx_new = R.ensure_index(spec);
      auto hroot2 = idx_new.root();
      auto vals_after = hroot2.values();

      // The old span 'vals_before' is now potentially invalid (by contract).
      // We DO NOT dereference it; we only assert the new one is usable.
      std::size_t s2 = vals_after.size();
      expect_eq(s2 >= 1, true, "new span usable after rebuild");
    }

    std::cout << "\n==== 12) HashTrieIndex::merge ====\n";
    {
      using HT = typename Rel::IndexTypeInst;
      IndexSpec spec{{0, 1}};
      memory_resource* resource = default_memory_resource();

      // Build first index with data: (1,10), (2,20), (2,40)
      HT idx1(resource);
      std::array<Vector<std::size_t>, 3> env1;
      env1[0] = {1, 2, 2};
      env1[1] = {10, 20, 40};
      env1[2] = {encode_to_size_t(std::string("a")), encode_to_size_t(std::string("b")),
                 encode_to_size_t(std::string("c"))};
      const auto enc_span1 = std::array<std::span<const std::size_t>, 3>{
          std::span(env1[0].data(), env1[0].size()), std::span(env1[1].data(), env1[1].size()),
          std::span(env1[2].data(), env1[2].size())};
      idx1.build_from_encoded(spec, enc_span1, resource);
      expect_eq(idx1.size(), 3, "idx1.size before merge");

      // Build second index with data: (3,30), (4,40)
      HT idx2(resource);
      std::array<Vector<std::size_t>, 3> env2;
      env2[0] = {3, 4};
      env2[1] = {30, 40};
      env2[2] = {encode_to_size_t(std::string("d")), encode_to_size_t(std::string("e"))};
      const auto enc_span2 = std::array<std::span<const std::size_t>, 3>{
          std::span(env2[0].data(), env2[0].size()), std::span(env2[1].data(), env2[1].size()),
          std::span(env2[2].data(), env2[2].size())};
      idx2.build_from_encoded(spec, enc_span2, resource);
      expect_eq(idx2.size(), 2, "idx2.size before merge");

      // Test 1: Merge idx2 into idx1 with offset 100
      std::size_t offset = 100;
      idx1.merge(idx2, offset);
      expect_eq(idx1.size(), 5, "idx1.size after merge");

      // Verify merged structure: root should have {1,2,3,4}
      auto root1 = idx1.root();
      auto root_vals = root1.values();
      std::unordered_set<std::size_t> root_set(root_vals.begin(), root_vals.end());
      expect_eq(root_set.count(1u) == 1, true, "merged root has 1");
      expect_eq(root_set.count(2u) == 1, true, "merged root has 2");
      expect_eq(root_set.count(3u) == 1, true, "merged root has 3");
      expect_eq(root_set.count(4u) == 1, true, "merged root has 4");
      expect_eq(root_set.size(), 4, "merged root has 4 distinct keys");

      // Verify (3,30) exists in merged index
      const auto key_330 = Prefix<int, int>{3, 30}.encoded();
      auto h330 = idx1.prefix_lookup(key_330);
      expect_eq(h330.valid(), true, "merged index has (3,30)");
      expect_eq(h330.is_leaf(), true, "(3,30) is leaf");
      if (h330.valid() && h330.is_leaf()) {
        auto rows_330 = h330.rows();
        expect_eq(rows_330.size() >= 1, true, "(3,30) has at least one row");
        // Row IDs from idx2 should be offset by 100
        bool found_offset_row = false;
        for (auto rid : rows_330) {
          if (rid >= offset && rid < offset + idx2.size()) {
            found_offset_row = true;
            break;
          }
        }
        expect_eq(found_offset_row, true, "found offset row ID in (3,30)");
      }

      // Test 2: Merge empty index (should be no-op)
      HT idx_empty(resource);
      std::size_t size_before = idx1.size();
      idx1.merge(idx_empty, 0);
      expect_eq(idx1.size(), size_before, "merge empty index doesn't change size");

      // Test 3: Merge into empty index
      HT idx_empty_dst(resource);
      HT idx_src(resource);
      std::array<Vector<std::size_t>, 3> env_src;
      env_src[0] = {5, 6};
      env_src[1] = {50, 60};
      env_src[2] = {encode_to_size_t(std::string("f")), encode_to_size_t(std::string("g"))};
      const auto enc_span_src = std::array<std::span<const std::size_t>, 3>{
          std::span(env_src[0].data(), env_src[0].size()),
          std::span(env_src[1].data(), env_src[1].size()),
          std::span(env_src[2].data(), env_src[2].size())};
      idx_src.build_from_encoded(spec, enc_span_src, resource);
      idx_empty_dst.merge(idx_src, 0);
      expect_eq(idx_empty_dst.size(), 2, "merge into empty index sets correct size");
      auto root_dst = idx_empty_dst.root();
      auto dst_vals = root_dst.values();
      std::unordered_set<std::size_t> dst_set(dst_vals.begin(), dst_vals.end());
      expect_eq(dst_set.count(5u) == 1, true, "empty dst merge has key 5");
      expect_eq(dst_set.count(6u) == 1, true, "empty dst merge has key 6");

      // Test 4: Merge with overlapping keys (should merge postings)
      HT idx_overlap1(resource);
      std::array<Vector<std::size_t>, 3> env_o1;
      env_o1[0] = {7, 7};
      env_o1[1] = {70, 71};
      env_o1[2] = {encode_to_size_t(std::string("h")), encode_to_size_t(std::string("i"))};
      const auto enc_span_o1 = std::array<std::span<const std::size_t>, 3>{
          std::span(env_o1[0].data(), env_o1[0].size()),
          std::span(env_o1[1].data(), env_o1[1].size()),
          std::span(env_o1[2].data(), env_o1[2].size())};
      idx_overlap1.build_from_encoded(spec, enc_span_o1, resource);

      HT idx_overlap2(resource);
      std::array<Vector<std::size_t>, 3> env_o2;
      env_o2[0] = {7};
      env_o2[1] = {70};
      env_o2[2] = {encode_to_size_t(std::string("j"))};
      const auto enc_span_o2 = std::array<std::span<const std::size_t>, 3>{
          std::span(env_o2[0].data(), env_o2[0].size()),
          std::span(env_o2[1].data(), env_o2[1].size()),
          std::span(env_o2[2].data(), env_o2[2].size())};
      idx_overlap2.build_from_encoded(spec, enc_span_o2, resource);

      idx_overlap1.merge(idx_overlap2, 200);
      expect_eq(idx_overlap1.size(), 3, "overlapping merge has correct size");
      const auto key_770 = Prefix<int, int>{7, 70}.encoded();
      auto h770 = idx_overlap1.prefix_lookup(key_770);
      expect_eq(h770.valid(), true, "overlapping merge has (7,70)");
      if (h770.valid() && h770.is_leaf()) {
        auto rows_770 = h770.rows();
        // Should have 2 original rows + 1 merged row = 3 total
        expect_eq(rows_770.size() >= 2, true, "(7,70) has merged postings");
      }

      // Test 5: Merge with zero offset
      HT idx_zero1(resource);
      HT idx_zero2(resource);
      std::array<Vector<std::size_t>, 3> env_z1, env_z2;
      env_z1[0] = {8};
      env_z1[1] = {80};
      env_z1[2] = {encode_to_size_t(std::string("k"))};
      env_z2[0] = {9};
      env_z2[1] = {90};
      env_z2[2] = {encode_to_size_t(std::string("l"))};
      const auto enc_span_z1 = std::array<std::span<const std::size_t>, 3>{
          std::span(env_z1[0].data(), env_z1[0].size()),
          std::span(env_z1[1].data(), env_z1[1].size()),
          std::span(env_z1[2].data(), env_z1[2].size())};
      const auto enc_span_z2 = std::array<std::span<const std::size_t>, 3>{
          std::span(env_z2[0].data(), env_z2[0].size()),
          std::span(env_z2[1].data(), env_z2[1].size()),
          std::span(env_z2[2].data(), env_z2[2].size())};
      idx_zero1.build_from_encoded(spec, enc_span_z1, resource);
      idx_zero2.build_from_encoded(spec, enc_span_z2, resource);
      idx_zero1.merge(idx_zero2, 0);
      expect_eq(idx_zero1.size(), 2, "zero offset merge has correct size");
      auto root_zero = idx_zero1.root();
      auto zero_vals = root_zero.values();
      std::unordered_set<std::size_t> zero_set(zero_vals.begin(), zero_vals.end());
      expect_eq(zero_set.count(8u) == 1, true, "zero offset merge has key 8");
      expect_eq(zero_set.count(9u) == 1, true, "zero offset merge has key 9");

      std::cout << "  ✓ All merge tests passed\n";
    }

    std::cout << "=== test triangle query (R,S,T exact schemas) ===\n";

    // Schemas:
    //   R(x,y)            → Relation<BooleanSR, std::tuple<int, int>>
    //   S(y,z,h)          → Relation<BooleanSR, std::tuple<int, int, int>>
    //   T(z,x,f)          → Relation<BooleanSR, std::tuple<int, int, int>>
    DEFINE_RELATION(r, BooleanSR, int, int);
    DEFINE_RELATION(s, BooleanSR, int, int, int);
    DEFINE_RELATION(t, BooleanSR, int, int, int);

    // ----------------------
    // EDB from the table:
    // ----------------------
    // R(x,y)
    FACT(r, BooleanSR::one(), 1, 10);
    FACT(r, BooleanSR::one(), 2, 10);
    FACT(r, BooleanSR::one(), 3, 10);
    FACT(r, BooleanSR::one(), 4, 10);
    FACT(r, BooleanSR::one(), 5, 20);
    FACT(r, BooleanSR::one(), 6, 30);
    FACT(r, BooleanSR::one(), 7, 40);
    FACT(r, BooleanSR::one(), 8, 50);

    // S(y,z,h)   (h is ignored for the triangle join)
    FACT(s, BooleanSR::one(), 10, 7, 5);
    FACT(s, BooleanSR::one(), 10, 8, 9);
    FACT(s, BooleanSR::one(), 10, 9, 4);
    FACT(s, BooleanSR::one(), 10, 100, 3);
    FACT(s, BooleanSR::one(), 20, 7, 6);
    FACT(s, BooleanSR::one(), 30, 6, 12);
    FACT(s, BooleanSR::one(), 40, 7, 57);
    FACT(s, BooleanSR::one(), 50, 7, 34);

    // T(z,x,f)   (f is ignored for the triangle join)
    // note: schema is (z,x,f). We’ll index with {1,0} so root enumerates x
    // first.
    FACT(t, BooleanSR::one(), 7, 1, 4);
    FACT(t, BooleanSR::one(), 8, 2, 3);
    FACT(t, BooleanSR::one(), 9, 3, 3);
    FACT(t, BooleanSR::one(), 7, 5, 4);
    FACT(t, BooleanSR::one(), 6, 7, 6);
    FACT(t, BooleanSR::one(), 7, 8, 2);
    FACT(t, BooleanSR::one(), 123, 4, 1);

    // ----------------------
    // Indexes used by the triangle plan:
    //   R: (x,y)     → {0,1}
    //   S: (y,z)     → {0,1}   (ignore h)
    //   T: (x,z) via building over (z,x, f) with spec {1,0}
    // ----------------------
    DEFINE_INDEX(r, 0, 1);
    DEFINE_INDEX(s, 0, 1);
    DEFINE_INDEX(t, 1, 0);

    // Root cursors
    auto INDEX_ROOT(r, 0, 1) = INDEX(r, 0, 1).root();
    auto INDEX_ROOT(t, 1, 0) = INDEX(t, 1, 0).root();
    auto INDEX_ROOT(s, 0, 1) = INDEX(s, 0, 1).root();
    // Intersect candidates for x
    // for x in R.x ∩ T.x
    using IndexType = typename decltype(r)::IndexTypeInst;
    for (auto x : IndexType::intersect(INDEX_ROOT(r, 0, 1), INDEX_ROOT(t, 1, 0))) {
      std::cout << "x=" << static_cast<int>(x) << "\n";
      // r_x = r[x]; t_x = t[x];
      auto INDEX_AT(r, x, 0, 1) = INDEX_ROOT(r, 0, 1).prefix(x);
      auto INDEX_AT(t, x, 1, 0) = INDEX_ROOT(t, 1, 0).prefix(x);
      // for y in r_x[y] ∩ s[y]
      for (auto y : IndexType::intersect(INDEX_AT(r, x, 0, 1), INDEX_ROOT(s, 0, 1))) {
        std::cout << "  (x,y)=(" << static_cast<int>(x) << "," << static_cast<int>(y) << ")\n";
        // s_y = s[y];
        const auto INDEX_AT(s, y, 0, 1) = INDEX_ROOT(s, 0, 1).prefix(y);  // depth-1 over z
        // for z in s_y[z] ∩ t_x[z]
        for (auto z : IndexType::intersect(INDEX_AT(s, y, 0, 1), INDEX_AT(t, x, 1, 0))) {
          std::cout << "    (x,y,z)=(" << static_cast<int>(x) << "," << static_cast<int>(y) << ","
                    << static_cast<int>(z) << ")\n";
        }
      }
    }

    std::cout << "\n==== 10) Test push_row encoding and push_intern_row ====\n";
    {
      using TestRel =
          SRDatalog::Relation<SR, std::tuple<int, std::string, int>, SRDatalog::HashTrieIndex,
                              SRDatalog::HostRelationPolicy, std::size_t>;
      TestRel R1, R2;

      // Test 1: push_row should encode values to interned columns
      R1.push_row({100, std::string("test"), 200}, SR::one());
      R1.push_row({101, std::string("test2"), 201}, SR::one());

      expect_size(R1, 2);
      expect_eq(R1.interned_size(), 2, "interned_size matches size after push_row");

      // Check that interned columns are populated
      expect_eq(R1.interned_column<0>().size(), 2, "interned column 0 size");
      expect_eq(R1.interned_column<1>().size(), 2, "interned column 1 size");
      expect_eq(R1.interned_column<2>().size(), 2, "interned column 2 size");

      // Check that encoded values match
      expect_eq(R1.interned_column<0>()[0], encode_to_size_t(100), "interned col0[0]");
      expect_eq(R1.interned_column<0>()[1], encode_to_size_t(101), "interned col0[1]");
      expect_eq(R1.interned_column<2>()[0], encode_to_size_t(200), "interned col2[0]");
      expect_eq(R1.interned_column<2>()[1], encode_to_size_t(201), "interned col2[1]");

      // Test 2: push_intern_row with pre-encoded values
      std::array<std::size_t, 3> encoded_row1 = {
          encode_to_size_t(300), encode_to_size_t(std::string("encoded1")), encode_to_size_t(400)};
      std::array<std::size_t, 3> encoded_row2 = {
          encode_to_size_t(301), encode_to_size_t(std::string("encoded2")), encode_to_size_t(401)};

      R2.push_intern_row(encoded_row1, SR::one());
      R2.push_intern_row(encoded_row2, SR::one());

      expect_size(R2, 2);
      expect_eq(R2.interned_size(), 2, "interned_size matches size after push_intern_row");

      // Check that columns are decoded correctly
      // expect_eq(R2.column<0>()[0], 300, "decoded col0[0]");
      // expect_eq(R2.column<0>()[1], 301, "decoded col0[1]");
      // expect_eq(R2.column<1>()[0], std::string("encoded1"), "decoded col1[0]");
      // expect_eq(R2.column<1>()[1], std::string("encoded2"), "decoded col1[1]");
      // expect_eq(R2.column<2>()[0], 400, "decoded col2[0]");
      // expect_eq(R2.column<2>()[1], 401, "decoded col2[1]");

      // Check that interned columns match
      expect_eq(R2.interned_column<0>()[0], encoded_row1[0], "interned col0[0] matches");
      expect_eq(R2.interned_column<0>()[1], encoded_row2[0], "interned col0[1] matches");
      expect_eq(R2.interned_column<1>()[0], encoded_row1[1], "interned col1[0] matches");
      expect_eq(R2.interned_column<1>()[1], encoded_row2[1], "interned col1[1] matches");

      std::cout << "  ✓ push_row encoding test passed\n";
      std::cout << "  ✓ push_intern_row test passed\n";
    }

    std::cout << "\n==== 11) Test reconstruct_columns_from_interned ====\n";
    {
      using TestRel =
          SRDatalog::Relation<SR, std::tuple<int, std::string, int>, SRDatalog::HashTrieIndex,
                              SRDatalog::HostRelationPolicy, std::size_t>;
      TestRel R;

      // First, push some rows normally
      R.push_row({500, std::string("original"), 600}, SR::one());
      R.push_row({501, std::string("original2"), 601}, SR::one());

      expect_size(R, 2);
      expect_eq(R.column<0>()[0], 500, "original col0[0]");
      expect_eq(R.column<1>()[0], std::string("original"), "original col1[0]");

      // Modify interned columns directly (simulating loading from serialized data)
      R.interned_column<0>()[0] = encode_to_size_t(700);
      R.interned_column<0>()[1] = encode_to_size_t(701);
      R.interned_column<1>()[0] = encode_to_size_t(std::string("reconstructed"));
      R.interned_column<1>()[1] = encode_to_size_t(std::string("reconstructed2"));
      R.interned_column<2>()[0] = encode_to_size_t(800);
      R.interned_column<2>()[1] = encode_to_size_t(801);

      // Reconstruct columns from interned values
      R.reconstruct_columns_from_interned();

      // Check that columns are now reconstructed from interned values
      expect_eq(R.column<0>()[0], 700, "reconstructed col0[0]");
      expect_eq(R.column<0>()[1], 701, "reconstructed col0[1]");
      expect_eq(R.column<1>()[0], std::string("reconstructed"), "reconstructed col1[0]");
      expect_eq(R.column<1>()[1], std::string("reconstructed2"), "reconstructed col1[1]");
      expect_eq(R.column<2>()[0], 800, "reconstructed col2[0]");
      expect_eq(R.column<2>()[1], 801, "reconstructed col2[1]");

      std::cout << "  ✓ reconstruct_columns_from_interned test passed\n";
    }

    std::cout << "\n==== 12) Test SortedArrayIndex with all features ====\n";
    {
      using SortedRel =
          SRDatalog::Relation<SR, std::tuple<int, std::string, int>, SRDatalog::SortedArrayIndex,
                              SRDatalog::HostRelationPolicy, std::size_t>;
      SortedRel R;

      std::cout << "  --- Test 12.1: Basic push_row and encoding ---\n";
      R.push_row({1, std::string("alice"), 10}, SR::one());
      R.push_row({2, std::string("bob"), 20}, SR::one());
      R.push_row({3, std::string("carol"), 30}, SR::one());
      R.push_row({1, std::string("alice"), 10}, SR::one());  // duplicate

      expect_size(R, 4);
      expect_eq(R.interned_size(), 4, "interned_size matches size");
      expect_eq(R.interned_column<0>().size(), 4, "interned column 0 size");
      expect_eq(R.interned_column<1>().size(), 4, "interned column 1 size");
      expect_eq(R.interned_column<2>().size(), 4, "interned column 2 size");

      // Check encoding
      expect_eq(R.interned_column<0>()[0], encode_to_size_t(1), "interned col0[0]");
      expect_eq(R.interned_column<0>()[1], encode_to_size_t(2), "interned col0[1]");
      expect_eq(R.interned_column<2>()[0], encode_to_size_t(10), "interned col2[0]");

      std::cout << "  ✓ Basic push_row and encoding test passed\n";

      std::cout << "  --- Test 12.2: push_intern_row with SortedArrayIndex ---\n";
      SortedRel R2;
      std::array<std::size_t, 3> encoded_row1 = {
          encode_to_size_t(100), encode_to_size_t(std::string("encoded1")), encode_to_size_t(200)};
      std::array<std::size_t, 3> encoded_row2 = {
          encode_to_size_t(101), encode_to_size_t(std::string("encoded2")), encode_to_size_t(201)};

      R2.push_intern_row(encoded_row1, SR::one());
      R2.push_intern_row(encoded_row2, SR::one());

      expect_size(R2, 2);
      // expect_eq(R2.column<0>()[0], 100, "decoded col0[0]");
      // expect_eq(R2.column<1>()[0], std::string("encoded1"), "decoded col1[0]");
      // expect_eq(R2.column<2>()[0], 200, "decoded col2[0]");
      expect_eq(R2.interned_column<0>()[0], encoded_row1[0], "interned col0[0] matches");

      std::cout << "  ✓ push_intern_row test passed\n";

      std::cout << "  --- Test 12.3: Index building and querying ---\n";
      // Build index on first two columns
      IndexSpec spec;
      spec.cols = {0, 1};
      const auto& idx = R.ensure_index(spec);

      expect_eq(idx.size() > 0, true, "index size > 0");
      expect_eq(idx.empty(), false, "index not empty");

      // Query using prefix lookup
      auto root = idx.root();
      expect_eq(root.valid(), true, "root is valid");
      expect_eq(root.is_leaf(), false, "root is not leaf");

      // Test prefix lookup
      auto key1 = encode_to_size_t(1);
      auto node1 = root.prefix(key1);
      expect_eq(node1.valid(), true, "prefix(1) is valid");

      std::cout << "  ✓ Index building and querying test passed\n";

      std::cout << "  --- Test 12.4: Index search operations ---\n";
      // Test contains_value
      auto values = root.values();
      expect_eq(root.unique_count() > 0, true, "root has values");

      // Test prefix lookup with full key
      std::array<std::size_t, 2> full_key = {encode_to_size_t(1),
                                             encode_to_size_t(std::string("alice"))};
      auto full_node = idx.prefix_lookup(full_key);
      expect_eq(full_node.valid(), true, "full key lookup is valid");
      if (full_node.valid() && full_node.is_leaf()) {
        auto rows = full_node.rows();
        expect_eq(rows.size() > 0, true, "full key has rows");
      }

      std::cout << "  ✓ Index search operations test passed\n";

      std::cout << "  --- Test 12.5: reconstruct_columns_from_interned ---\n";
      SortedRel R3;
      R3.push_row({500, std::string("original"), 600}, SR::one());
      R3.push_row({501, std::string("original2"), 601}, SR::one());

      // Modify interned columns directly
      R3.interned_column<0>()[0] = encode_to_size_t(700);
      R3.interned_column<1>()[0] = encode_to_size_t(std::string("reconstructed"));
      R3.interned_column<2>()[0] = encode_to_size_t(800);

      // Reconstruct columns
      R3.reconstruct_columns_from_interned();

      expect_eq(R3.column<0>()[0], 700, "reconstructed col0[0]");
      expect_eq(R3.column<1>()[0], std::string("reconstructed"), "reconstructed col1[0]");
      expect_eq(R3.column<2>()[0], 800, "reconstructed col2[0]");

      std::cout << "  ✓ reconstruct_columns_from_interned test passed\n";

      std::cout << "  --- Test 12.6: Index merge with SortedArrayIndex ---\n";
      SortedRel R4, R5;
      R4.push_row({1, std::string("a"), 10}, SR::one());
      R4.push_row({2, std::string("b"), 20}, SR::one());
      R5.push_row({3, std::string("c"), 30}, SR::one());
      R5.push_row({4, std::string("d"), 40}, SR::one());

      IndexSpec spec_merge;
      spec_merge.cols = {0, 1};
      const auto& idx4 = R4.ensure_index(spec_merge);
      const auto& idx5 = R5.ensure_index(spec_merge);

      // Clone idx4 and merge idx5 into it
      using IndexType = SortedRel::IndexTypeInst;
      IndexType idx4_copy(default_memory_resource());
      idx4_copy.clone_from(idx4, default_memory_resource());
      idx4_copy.merge(idx5, static_cast<uint32_t>(R4.size()));

      expect_eq(idx4_copy.size() >= idx4.size(), true, "merged index size >= original");

      std::cout << "  ✓ Index merge test passed\n";

      std::cout << "  --- Test 12.7: Multiple indexes with SortedArrayIndex ---\n";
      SortedRel R6;
      R6.push_row({1, std::string("x"), 100}, SR::one());
      R6.push_row({2, std::string("y"), 200}, SR::one());
      R6.push_row({1, std::string("z"), 300}, SR::one());

      IndexSpec spec1;
      spec1.cols = {0};
      IndexSpec spec2;
      spec2.cols = {0, 1};
      IndexSpec spec3;
      spec3.cols = {0, 1, 2};

      const auto& idx1 = R6.ensure_index(spec1);
      const auto& idx2 = R6.ensure_index(spec2);
      const auto& idx3 = R6.ensure_index(spec3);

      expect_eq(idx1.size() > 0, true, "index1 size > 0");
      expect_eq(idx2.size() > 0, true, "index2 size > 0");
      expect_eq(idx3.size() > 0, true, "index3 size > 0");

      // Test queries on different indexes
      auto root1 = idx1.root();
      auto root2 = idx2.root();
      auto root3 = idx3.root();

      expect_eq(root1.valid(), true, "root1 is valid");
      expect_eq(root2.valid(), true, "root2 is valid");
      expect_eq(root3.valid(), true, "root3 is valid");

      std::cout << "  ✓ Multiple indexes test passed\n";

      std::cout << "  --- Test 12.8: Index with different ValueType and RowIdType ---\n";
      // Note: Using std::size_t for ValueType to match interned columns
      using SortedRelCustom =
          SRDatalog::Relation<SR, std::tuple<int, int>, SRDatalog::SortedArrayIndex,
                              SRDatalog::HostRelationPolicy, std::size_t, uint32_t>;
      SortedRelCustom R7;
      R7.push_row({1, 10}, SR::one());
      R7.push_row({2, 20}, SR::one());
      R7.push_row({3, 30}, SR::one());

      IndexSpec spec_custom;
      spec_custom.cols = {0, 1};
      const auto& idx_custom = R7.ensure_index(spec_custom);

      expect_eq(idx_custom.size(), 3, "custom index size");
      auto root_custom = idx_custom.root();
      expect_eq(root_custom.valid(), true, "custom root is valid");

      std::cout << "  ✓ Custom ValueType/RowIdType test passed\n";

      std::cout << "  ✓ All SortedArrayIndex tests passed\n";
    }

    std::cout << "\nAll tests passed.\n";
    return 0;
  }
}