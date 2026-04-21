#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>
#include <vector>

#include "gpu/runtime/custom_set_op.h"

// Mock boolean semiring needed for MergeMatchOp
struct MockBooleanSR {
  using value_type = bool;
  static __host__ __device__ bool add(bool a, bool b) {
    return a || b;
  }
  static __host__ __device__ bool zero() {
    return false;
  }
  static __host__ __device__ bool one() {
    return true;
  }
};

// Copy of MergeMatchOp from device_sorted_array_index.ipp adaptation
template <typename SR>
struct MergeMatchOp {
  using ValueType = typename SR::value_type;
  bool emit_matches;

  __device__ __host__ explicit MergeMatchOp(bool emit) : emit_matches(emit) {}

  __device__ thrust::pair<bool, ValueType> operator()(const auto& /*key*/, ValueType val_new,
                                                      ValueType val_full) {
    ValueType merged = SR::add(val_new, val_full);
    // Explicitly update full via side-effect if value changes
    if (merged != val_full) {
      return {true, merged};
    }
    return {emit_matches, merged};
  }
};

int main() {
  using T = uint32_t;

  // Partial Overlap: Full 0-60, New 40-100.
  size_t full_size = 100000;
  size_t new_size = 20000;

  std::vector<T> h_full_k1(full_size), h_full_k2(full_size);
  std::vector<T> h_new_k1(new_size), h_new_k2(new_size);

  std::mt19937 rng(42);
  std::uniform_int_distribution<T> dist_full(0, 60);
  std::uniform_int_distribution<T> dist_new(40, 100);

  for (size_t i = 0; i < full_size; ++i) {
    h_full_k1[i] = dist_full(rng);
    h_full_k2[i] = dist_full(rng);
  }
  for (size_t i = 0; i < new_size; ++i) {
    h_new_k1[i] = dist_new(rng);
    h_new_k2[i] = dist_new(rng);
  }

  // Move to device
  thrust::device_vector<T> d_full_k1 = h_full_k1;
  thrust::device_vector<T> d_full_k2 = h_full_k2;
  thrust::device_vector<T> d_new_k1 = h_new_k1;
  thrust::device_vector<T> d_new_k2 = h_new_k2;

  // Create Zip Iterators
  auto full_zip =
      thrust::make_zip_iterator(thrust::make_tuple(d_full_k1.begin(), d_full_k2.begin()));
  auto new_zip = thrust::make_zip_iterator(thrust::make_tuple(d_new_k1.begin(), d_new_k2.begin()));

  // Sort and unique
  thrust::sort(full_zip, full_zip + full_size);
  auto full_end = thrust::unique(full_zip, full_zip + full_size);
  full_size = full_end - full_zip;
  d_full_k1.resize(full_size);
  d_full_k2.resize(full_size);

  thrust::sort(new_zip, new_zip + new_size);
  auto new_end = thrust::unique(new_zip, new_zip + new_size);
  new_size = new_end - new_zip;
  d_new_k1.resize(new_size);
  d_new_k2.resize(new_size);

  std::cout << "Data sizes after unique: Full=" << full_size << ", New=" << new_size << std::endl;

  // Provenance (bools)
  thrust::device_vector<bool> d_full_prov(full_size, true);
  thrust::device_vector<bool> d_new_prov(new_size, true);

  // GOLD STANDARD: thrust::set_difference
  thrust::device_vector<T> gold_k1(new_size), gold_k2(new_size);
  auto gold_zip_out =
      thrust::make_zip_iterator(thrust::make_tuple(gold_k1.begin(), gold_k2.begin()));

  // Using thrust::less for tuple comparison
  auto gold_end = thrust::set_difference(new_zip, new_zip + new_size, full_zip,
                                         full_zip + full_size, gold_zip_out);
  size_t gold_count = gold_end - gold_zip_out;
  std::cout << "Gold Diff Count: " << gold_count << std::endl;

  // CUSTOM OP: set_difference
  thrust::device_vector<T> custom_k1(new_size), custom_k2(new_size);
  thrust::device_vector<bool> custom_prov(new_size);
  auto custom_zip_out =
      thrust::make_zip_iterator(thrust::make_tuple(custom_k1.begin(), custom_k2.begin()));

  MergeMatchOp<MockBooleanSR> op(false);

  auto result_pair = thrust::cuda_cub::set_difference_by_key_with_match_custom(
      thrust::device, new_zip, new_zip + new_size, full_zip, full_zip + full_size,
      d_new_prov.begin(), d_full_prov.begin(), custom_zip_out, custom_prov.begin(),
      thrust::less<thrust::tuple<T, T>>(), op);

  size_t custom_count = result_pair.first - custom_zip_out;
  std::cout << "Custom Diff Count: " << custom_count << std::endl;

  if (custom_count != gold_count) {
    std::cerr << "DIFF Mismatch! Gold: " << gold_count << ", Custom: " << custom_count << std::endl;
    return 1;
  }

  // MERGE TEST
  // Replicate merge correctness check with thrust::merge
  // We want to merge Full and Result(Delta).
  // Result(Delta) is in custom_zip_out (size custom_count).
  // Note: custom_zip_out is NOT necessarily sorted?
  // set_difference output IS sorted if inputs are sorted. (Standard behavior).

  size_t merged_size_gold = full_size + custom_count;
  std::cout << "Expected Merged Size: " << merged_size_gold << std::endl;

  // Use thrust::merge_by_key (as per implementation)
  thrust::device_vector<T> m_k1(merged_size_gold), m_k2(merged_size_gold);
  thrust::device_vector<bool> m_prov(merged_size_gold);
  auto m_zip = thrust::make_zip_iterator(thrust::make_tuple(m_k1.begin(), m_k2.begin()));

  auto merge_end = thrust::merge_by_key(
      thrust::device, full_zip, full_zip + full_size, custom_zip_out, custom_zip_out + custom_count,
      d_full_prov.begin(), custom_prov.begin(), m_zip, m_prov.begin());

  size_t real_merged_size = merge_end.first - m_zip;
  std::cout << "Thrust Merge Size: " << real_merged_size << std::endl;

  if (real_merged_size != merged_size_gold) {
    std::cerr << "MERGE Mismatch! Expected: " << merged_size_gold << ", Got: " << real_merged_size
              << std::endl;
    return 2;
  }

  std::cout << "SUCCESS! All counts match." << std::endl;
  return 0;
}
