/**
 * @file aos_device_array.h
 * @brief N-dimensional device array (AoS) using flat row-major storage.
 */

#pragma once

#include "./device_array.h"
#include "./macro.h"
#include "gpu/gpu_api.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <rmm/device_vector.hpp>
#include <rmm/exec_policy.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace SRDatalog::GPU {

/**
 * @brief Lightweight view for compile-time fixed dimensions (N) in AoS layout.
 * @note Trivially copyable. Pass by value to CUDA kernels.
 */
template <typename T, std::size_t N>
struct AoSView {
  T* data;
  std::size_t num_rows;
  // Stride is implicitly N (consecutive elements)

  [[nodiscard]] GPU_HD constexpr std::size_t arity() const noexcept {
    return N;
  }

  // Get pointer to the start of a row (tuple)
  [[nodiscard]] GPU_HD T* row_ptr(std::size_t row) const noexcept {
    return data + row * N;
  }

  template <std::size_t Dim>
  [[nodiscard]] GPU_HD T& get(std::size_t row) const noexcept {
    static_assert(Dim < N, "Dimension out of bounds");
    return data[row * N + Dim];
  }

  [[nodiscard]] GPU_HD T& get(std::size_t row, std::size_t dim) const noexcept {
    return data[row * N + dim];
  }

  [[nodiscard]] GPU_HD T& operator()(std::size_t row, std::size_t dim) const noexcept {
    return data[row * N + dim];
  }

  [[nodiscard]] GPU_HD T* raw() const noexcept {
    return data;
  }
};

/**
 * @brief N-dimensional device array with flat row-major (AoS) storage.
 *
 * Memory layout:
 * [row0_col0, row0_col1...][row1_col0, row1_col1...]
 */
template <typename T, std::size_t N>
class AoSDeviceArray {
 public:
  using value_type = T;
  using size_type = std::size_t;
  using view_type = AoSView<T, N>;
  using const_view_type = AoSView<const T, N>;

  AoSDeviceArray() = default;

  explicit AoSDeviceArray(size_type num_rows) : storage_(num_rows * N), num_rows_(num_rows) {
    // Initialize storage? DeviceArray constructor allocates but doesn't initialize
    // unless we assume T is POD and we don't care.
    // DeviceArray uses rmm::device_uvector without initialization.
  }

  // Non-copyable
  AoSDeviceArray(const AoSDeviceArray&) = delete;
  AoSDeviceArray& operator=(const AoSDeviceArray&) = delete;

  // Move constructor
  AoSDeviceArray(AoSDeviceArray&&) noexcept = default;
  AoSDeviceArray& operator=(AoSDeviceArray&&) noexcept = default;

  ~AoSDeviceArray() = default;

  [[nodiscard]] view_type view() noexcept CLANG_LIFETIMEBOUND {
    return view_type{storage_.data().get(), num_rows_};
  }

  [[nodiscard]] const_view_type view() const noexcept CLANG_LIFETIMEBOUND {
    return const_view_type{storage_.data().get(), num_rows_};
  }

  [[nodiscard]] const_view_type cview() const noexcept {
    return const_view_type{storage_.data().get(), num_rows_};
  }

  [[nodiscard]] static constexpr std::size_t arity() noexcept {
    return N;
  }

  [[nodiscard]] size_type num_rows() const noexcept {
    return num_rows_;
  }

  [[nodiscard]] size_type capacity() const noexcept {
    // Capacity in rows
    return storage_.capacity() / N;
  }

  [[nodiscard]] bool empty() const noexcept {
    return num_rows_ == 0;
  }

  void resize(size_type new_num_rows) {
    storage_.resize(new_num_rows * N);
    num_rows_ = new_num_rows;
  }

  void reserve(size_type new_capacity_rows) {
    storage_.reserve(new_capacity_rows * N);
  }

  void shrink_to_fit() {
    storage_.shrink_to_fit();
  }

  /**
   * @brief Add a new row to the array (copies from host)
   * @param values Array of N values to append (host-accessible)
   */
  void push_back(const T* values) {
    ensure_capacity_for_next();
    GPU_CHECK_ERROR(GPU_MEMCPY(storage_.data().get() + num_rows_ * N, values, N * sizeof(T),
                               GPU_HOST_TO_DEVICE));
    ++num_rows_;
  }

  // Helper for single element (arity 1)
  template <typename U = T>
  void push_back(const U& value)
    requires(N == 1)
  {
    push_back(&value);
  }

  /**
   * @brief Add a new row from a tuple-like structure
   */
  template <typename TupleLike>
  void push_back(const TupleLike& values) {
    ensure_capacity_for_next();

    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      T host_values[N] = {static_cast<T>(std::get<Is>(values))...};
      GPU_CHECK_ERROR(GPU_MEMCPY(storage_.data().get() + num_rows_ * N, host_values, N * sizeof(T),
                                 GPU_HOST_TO_DEVICE));
    }(std::make_index_sequence<N>{});

    ++num_rows_;
  }

  void concat(const AoSDeviceArray& other) {
    if (other.empty()) {
      return;
    }
    const size_type old_num_rows = num_rows_;
    const size_type new_num_rows = old_num_rows + other.num_rows_;

    storage_.resize(new_num_rows * N);

    // Contiguous copy of all rows
    GPU_CHECK_ERROR(GPU_MEMCPY(storage_.data().get() + old_num_rows * N,
                               other.storage_.data().get(), other.num_rows_ * N * sizeof(T),
                               GPU_DEVICE_TO_DEVICE));

    num_rows_ = new_num_rows;
  }

  void clone_into(AoSDeviceArray& other) const {
    other.clear();
    if (empty())
      return;

    other.resize(num_rows_);

    GPU_CHECK_ERROR(GPU_MEMCPY(other.storage_.data().get(), storage_.data().get(),
                               num_rows_ * N * sizeof(T), GPU_DEVICE_TO_DEVICE));
    other.num_rows_ = num_rows_;
  }

  /**
   * @brief Batch copy data from host SoA columns, interleaving into AoS format
   * @param host_arrays Array of N host arrays, each containing num_rows elements (SoA)
   * @param num_rows Number of rows to copy
   * @note Interleaves the columns on host then copies to device
   */
  template <typename HostArray>
  void assign_from_host(const std::array<HostArray, N>& host_arrays, size_type num_rows) {
    if (num_rows == 0) {
      clear();
      return;
    }

    // Resize to fit new data
    resize(num_rows);

    // Create interleaved buffer on host
    std::vector<T> interleaved(num_rows * N);
    for (size_type row = 0; row < num_rows; ++row) {
      for (size_type col = 0; col < N; ++col) {
        interleaved[row * N + col] = static_cast<T>(host_arrays[col][row]);
      }
    }

    // Copy interleaved data to device
    GPU_CHECK_ERROR(GPU_MEMCPY(storage_.data().get(), interleaved.data(), num_rows * N * sizeof(T),
                               GPU_HOST_TO_DEVICE));
  }

  void clear() noexcept {
    num_rows_ = 0;
    storage_.clear();
  }

  // Raw data access
  [[nodiscard]] T* data() noexcept {
    return storage_.data().get();
  }

  [[nodiscard]] const T* data() const noexcept {
    return storage_.data().get();
  }

  // Memory stats
  [[nodiscard]] std::size_t bytes_used() const noexcept {
    return num_rows_ * N * sizeof(T);
  }

  [[nodiscard]] std::size_t bytes_allocated() const noexcept {
    return storage_.capacity() * sizeof(T);
  }

  /**
   * @brief Expose internal storage for zero-copy swap with TVJoin
   */
  rmm::device_vector<T>& internal_vector() {
    return storage_;
  }

  void swap(rmm::device_vector<T>& other_vec) {
    storage_.swap(other_vec);
    num_rows_ = storage_.size() / N;  // Update logical size after swap
  }

 private:
  void ensure_capacity_for_next() {
    const size_type cap = storage_.capacity() / N;
    if (num_rows_ >= cap) {
      size_type new_cap = cap == 0 ? 1 : cap * 2;
      storage_.reserve(new_cap * N);
    }

    // Explicitly resize if size is too small, as vector::reserve doesn't change size
    if ((num_rows_ + 1) * N > storage_.size()) {
      storage_.resize((num_rows_ + 1) * N);
    }
  }

  rmm::device_vector<T> storage_;
  size_type num_rows_{0};
};

}  // namespace SRDatalog::GPU
