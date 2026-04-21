/**
 * @file nd_device_array.h
 * @brief N-dimensional device array (SoA) using flat column-major storage.
 */

#pragma once

#include "./device_array.h"
#include "./macro.h"
#include "gpu/gpu_api.h"  // GPU API abstraction (replaces cuda_runtime.h)
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>    // for std::get, std::make_index_sequence
#include <utility>  // for std::forward, std::index_sequence
#include <vector>

namespace SRDatalog::GPU {

/**
 * @brief Lightweight view for compile-time fixed dimensions (N).
 * @note Trivially copyable. Pass by value to CUDA kernels.
 */
template <typename T, std::size_t N>
struct NDArrayView {
  T* data;
  std::size_t num_rows;
  std::size_t stride;

  [[nodiscard]] GPU_HD constexpr std::size_t arity() const noexcept {
    return N;
  }

  template <std::size_t Dim>
  [[nodiscard]] GPU_HD T* column_ptr() const noexcept {
    static_assert(Dim < N, "Dimension out of bounds");
    return data + Dim * stride;
  }

  template <std::size_t Dim>
  [[nodiscard]] GPU_HD cuda_std::span<T> column() const noexcept {
    static_assert(Dim < N, "Dimension out of bounds");
    return cuda_std::span<T>{data + Dim * stride, num_rows};
  }

  template <std::size_t Dim>
  [[nodiscard]] GPU_HD T& get(std::size_t row) const noexcept {
    static_assert(Dim < N, "Dimension out of bounds");
    return data[(Dim * stride) + row];
  }

  [[nodiscard]] GPU_HD T* column_ptr(std::size_t dim) const noexcept {
    return (dim < N) ? data + dim * stride : nullptr;
  }

  [[nodiscard]] GPU_HD T& get(std::size_t row, std::size_t dim) const noexcept {
    return data[(dim * stride) + row];
  }

  [[nodiscard]] GPU_HD T& operator()(std::size_t row, std::size_t dim) const noexcept {
    return data[(dim * stride) + row];
  }

  [[nodiscard]] GPU_HD T* raw() const noexcept {
    return data;
  }

  [[nodiscard]] GPU_HD std::size_t get_stride() const noexcept {
    return stride;
  }
};

/**
 * @brief N-dimensional device array with flat column-major storage.
 *
 * Memory layout:
 * [col0: 0..stride][col1: 0..stride]...[colN-1: 0..stride]
 *
 * Where stride >= num_rows. This allows resize() to shrink without
 * reallocating or moving data.
 */
template <typename T, std::size_t N>
class NDDeviceArray {
 public:
  using value_type = T;
  using size_type = std::size_t;
  using view_type = NDArrayView<T, N>;
  using const_view_type = NDArrayView<const T, N>;

  NDDeviceArray() = default;

  explicit NDDeviceArray(size_type num_rows)
      : storage_(N * num_rows), num_rows_(num_rows), stride_(num_rows) {}

  // Non-copyable (DeviceArray is non-copyable)
  NDDeviceArray(const NDDeviceArray&) = delete;
  NDDeviceArray& operator=(const NDDeviceArray&) = delete;

  // Movable (DeviceArray is movable)
  NDDeviceArray(NDDeviceArray&&) noexcept = default;
  NDDeviceArray& operator=(NDDeviceArray&&) noexcept = default;

  // Destructor (default is fine - will call storage_.~DeviceArray() which deallocates)
  ~NDDeviceArray() = default;

  /**
   * @brief Get a view of the array.
   * @warning The returned view contains pointers that become invalid if resize() reallocates.
   *          Use array.column_ptr<I>() directly if you need pointers that remain valid after
   * resize().
   * @return View containing pointers to current storage (lifetime bound to this array)
   * @note The view's lifetime is bound to this NDDeviceArray instance. It becomes invalid if
   *       resize() reallocates or if the array is destroyed.
   */
  [[nodiscard]] view_type view() noexcept CLANG_LIFETIMEBOUND {
    return view_type{storage_.data(), num_rows_, stride_};
  }

  /**
   * @brief Get a const view of the array.
   * @warning The returned view contains pointers that become invalid if resize() reallocates.
   *          Use array.column_ptr<I>() directly if you need pointers that remain valid after
   * resize().
   * @return Const view containing pointers to current storage (lifetime bound to this array)
   * @note The view's lifetime is bound to this NDDeviceArray instance. It becomes invalid if
   *       resize() reallocates or if the array is destroyed.
   */
  [[nodiscard]] const_view_type view() const noexcept CLANG_LIFETIMEBOUND {
    return const_view_type{storage_.data(), num_rows_, stride_};
  }

  /**
   * @brief Get a const view of the array.
   * @warning The returned view contains pointers that become invalid if resize() reallocates.
   *          Use array.column_ptr<I>() directly if you need pointers that remain valid after
   * resize().
   * @return Const view containing pointers to current storage (lifetime bound to this array)
   * @note The view's lifetime is bound to this NDDeviceArray instance. It becomes invalid if
   *       resize() reallocates or if the array is destroyed.
   */
  [[nodiscard]] const_view_type cview() const noexcept {
    return const_view_type{storage_.data(), num_rows_, stride_};
  }

  [[nodiscard]] static constexpr std::size_t arity() noexcept {
    return N;
  }

  [[nodiscard]] size_type num_rows() const noexcept {
    return num_rows_;
  }

  [[nodiscard]] size_type stride() const noexcept {
    return stride_;
  }

  [[nodiscard]] size_type capacity() const noexcept {
    return stride_;
  }

  [[nodiscard]] bool empty() const noexcept {
    return num_rows_ == 0;
  }

  /**
   * @brief Resize array. Only reallocates if growing beyond stride.
   * @note Allocates exactly new_num_rows (no extra capacity) to minimize memory usage
   */
  /// @brief Set logical row count without reallocating (for padding)
  void set_num_rows(size_type n) {
    num_rows_ = n;
  }

  void resize(size_type new_num_rows) {
    // CRITICAL: Check BOTH stride_ AND that storage is actually allocated
    // After clear(), stride_ may still be old value but storage_.data() is null
    if (new_num_rows <= stride_ && storage_.data() != nullptr) {
      num_rows_ = new_num_rows;
      return;
    }

    // Allocate with padding per column for cooperative warp-level access.
    size_type new_stride = new_num_rows;
    DeviceArray<T> new_storage(N * (new_stride + 32));

    // Copy existing columns (each column is contiguous but spaced by old stride)
    // Only copy the actual number of rows (num_rows_), not stride_ elements
    // This ensures we don't read beyond valid data boundaries
    if (num_rows_ > 0 && storage_.data() != nullptr) {
      const size_type copy_count = num_rows_;  // Copy actual data, not stride
      for (std::size_t dim = 0; dim < N; ++dim) {
        thrust::copy_n(thrust::device, storage_.data() + dim * stride_, copy_count,
                       new_storage.data() + dim * new_stride);
      }
    }

    storage_ = std::move(new_storage);
    num_rows_ = new_num_rows;
    stride_ = new_stride;
  }

  /**
   * @brief Resize array on a specific stream (stream-ordered allocation and copy)
   * @param new_num_rows New number of rows
   * @param stream CUDA/HIP stream for stream-ordered operations
   */
  void resize(size_type new_num_rows, GPU_STREAM_T stream) {
    if (new_num_rows <= stride_ && storage_.data() != nullptr) {
      num_rows_ = new_num_rows;
      return;
    }

    size_type new_stride = new_num_rows;
    DeviceArray<T> new_storage(N * (new_stride + 32), stream);

    if (num_rows_ > 0 && storage_.data() != nullptr) {
      const size_type copy_count = num_rows_;
      for (std::size_t dim = 0; dim < N; ++dim) {
        thrust::copy_n(rmm::exec_policy(rmm::cuda_stream_view{stream}),
                       storage_.data() + dim * stride_, copy_count,
                       new_storage.data() + dim * new_stride);
      }
    }

    storage_ = std::move(new_storage);
    num_rows_ = new_num_rows;
    stride_ = new_stride;
  }

  /**
   * @brief Reserve capacity without changing size.
   */
  void reserve(size_type new_capacity) {
    if (new_capacity <= stride_) {
      return;
    }

    // Allocate new storage - this will allocate N * new_capacity elements
    // Note: DeviceArray constructor allocates memory, which requires CUDA to be initialized
    const size_type total_elements = N * new_capacity;
    DeviceArray<T> new_storage(total_elements);

    // Verify allocation succeeded (should not be null for non-zero size)
    if (total_elements > 0 && new_storage.data() == nullptr) {
      throw std::runtime_error("NDDeviceArray::reserve: Failed to allocate device memory for " +
                               std::to_string(total_elements) + " elements");
    }

    // Copy existing data if any (only if we have existing valid data)
    // Note: stride_ might be 0 for default-constructed arrays, but that's OK - we skip copy
    if (num_rows_ > 0 && storage_.data() != nullptr) {
      // Only copy if we have a valid stride (non-zero means we had previous allocation)
      if (stride_ > 0) {
        for (std::size_t dim = 0; dim < N; ++dim) {
          thrust::copy_n(thrust::device, storage_.data() + dim * stride_, num_rows_,
                         new_storage.data() + dim * new_capacity);
        }
      }
    }

    storage_ = std::move(new_storage);
    stride_ = new_capacity;
  }

  /**
   * @brief Compact storage so stride equals num_rows.
   */
  void shrink_to_fit() {
    if (stride_ == num_rows_) {
      return;
    }

    if (num_rows_ == 0) {
      storage_.reset();
      stride_ = 0;
      return;
    }

    DeviceArray<T> new_storage(N * num_rows_);

    for (std::size_t dim = 0; dim < N; ++dim) {
      thrust::copy_n(thrust::device, storage_.data() + dim * stride_, num_rows_,
                     new_storage.data() + dim * num_rows_);
    }

    storage_ = std::move(new_storage);
    stride_ = num_rows_;
  }

  /**
   * @brief Add a new row to the array (copies from host)
   * @param values Array of N values to append (host-accessible)
   * @note This is equivalent to appending one row across all columns
   */
  void push_back(const T* values) {
    if (num_rows_ >= stride_) {
      reserve(stride_ == 0 ? 1 : stride_ * 2);
    }
    // Copy one element per column from host to device
    for (std::size_t dim = 0; dim < N; ++dim) {
      thrust::copy_n(thrust::host, values + dim, 1, storage_.data() + dim * stride_ + num_rows_);
    }
    ++num_rows_;
  }

  /**
   * @brief Add a new row from a tuple-like structure
   * @param values Tuple or array of N values
   */
  template <typename TupleLike>
  void push_back(const TupleLike& values) {
    if (num_rows_ >= stride_) {
      reserve(stride_ == 0 ? 1 : stride_ * 2);
    }
    // Extract values from tuple-like structure
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      T host_values[N] = {static_cast<T>(std::get<Is>(values))...};
      for (std::size_t dim = 0; dim < N; ++dim) {
        thrust::copy_n(thrust::host, host_values + dim, 1,
                       storage_.data() + dim * stride_ + num_rows_);
      }
    }(std::make_index_sequence<N>{});
    ++num_rows_;
  }

  /**
   * @brief Construct a new row in-place
   * @param args Arguments to construct each column element
   * @note This constructs N elements (one per column)
   */
  template <typename... Args>
  void emplace_back(Args&&... args) {
    static_assert(sizeof...(Args) == N,
                  "emplace_back requires exactly N arguments (one per column)");
    if (num_rows_ >= stride_) {
      reserve(stride_ == 0 ? 1 : stride_ * 2);
    }
    // Construct on host, then copy to device
    T host_values[N] = {T(std::forward<Args>(args))...};
    for (std::size_t dim = 0; dim < N; ++dim) {
      thrust::copy_n(thrust::host, host_values + dim, 1,
                     storage_.data() + dim * stride_ + num_rows_);
    }
    ++num_rows_;
  }

  /**
   * @brief Append all rows from another array
   * @param other Source array to append from
   */
  void concat(const NDDeviceArray& other) {
    if (other.empty()) {
      return;
    }
    const size_type old_num_rows = num_rows_;
    reserve(num_rows_ + other.num_rows_);
    // Copy each column
    for (std::size_t dim = 0; dim < N; ++dim) {
      thrust::copy_n(thrust::device, other.storage_.data() + dim * other.stride_, other.num_rows_,
                     storage_.data() + dim * stride_ + old_num_rows);
    }
    num_rows_ = old_num_rows + other.num_rows_;
  }

  /**
   * @brief Clone contents into another array
   * @param other Target array to clone into (will be cleared first)
   */
  void clone_into(NDDeviceArray& other) const {
    other.clear();
    if (empty()) {
      return;
    }
    other.reserve(num_rows_);
    // Copy each column
    for (std::size_t dim = 0; dim < N; ++dim) {
      thrust::copy_n(thrust::device, storage_.data() + dim * stride_, num_rows_,
                     other.storage_.data() + dim * other.stride_);
    }
    other.num_rows_ = num_rows_;
  }

  /**
   * @brief Batch copy data from host arrays (one per column)
   * @param host_arrays Array of N host arrays, each containing num_rows elements
   * @param num_rows Number of rows to copy
   * @note This is more efficient than calling push_back in a loop
   */
  template <typename HostArray>
  void assign_from_host(const std::array<HostArray, N>& host_arrays, size_type num_rows) {
    if (num_rows == 0) {
      clear();
      return;
    }

    if (num_rows > stride_) {
      reserve(num_rows);
    }
    num_rows_ = num_rows;

    for (std::size_t dim = 0; dim < N; ++dim) {
      GPU_MEMCPY(column_ptr(dim), host_arrays[dim].data(), num_rows * sizeof(T),
                 GPU_HOST_TO_DEVICE);
    }
  }

  /**
   * @brief Clear contents (size to 0, keep capacity/stride).
   */
  void clear() noexcept {
    num_rows_ = 0;
  }

  /**
   * @brief Deallocate all memory.
   */
  void reset() {
    storage_.reset();
    num_rows_ = 0;
    stride_ = 0;
  }

  /**
   * @brief Swap with another array.
   */
  void swap(NDDeviceArray& other) noexcept {
    storage_.swap(other.storage_);
    std::swap(num_rows_, other.num_rows_);
    std::swap(stride_, other.stride_);
  }

  /**
   * @brief Swap internal storage with external buffer
   * @param other_vec External device_uvector buffer to swap with
   * @param new_num_rows New number of rows after swap
   * @note This allows reusing external buffers (like the persistent buffer in Impl)
   * @warning The caller must ensure other_vec has enough capacity for N * new_num_rows elements
   */
  void swap_internal_storage(typename DeviceArray<T>::internal_storage_type& other_vec,
                             size_type new_num_rows) noexcept {
    storage_.swap_internal_data(other_vec, N * new_num_rows);
    num_rows_ = new_num_rows;
    stride_ = new_num_rows;  // After swap, stride equals num_rows
  }

  // Compile-time column access
  template <std::size_t Dim>
  [[nodiscard]] T* column_ptr() noexcept {
    // TODO: use device_ptr instead
    static_assert(Dim < N, "Dimension out of bounds");
    return storage_.data() + Dim * stride_;
  }

  template <std::size_t Dim>
  [[nodiscard]] const T* column_ptr() const noexcept {
    // TODO: use device_ptr instead
    static_assert(Dim < N, "Dimension out of bounds");
    return storage_.data() + Dim * stride_;
  }

  // Runtime column access
  [[nodiscard]] T* column_ptr(std::size_t dim) noexcept {
    // TODO: use device_ptr instead
    return (dim < N) ? storage_.data() + dim * stride_ : nullptr;
  }

  [[nodiscard]] const T* column_ptr(std::size_t dim) const noexcept {
    // TODO: use device_ptr instead
    return (dim < N) ? storage_.data() + dim * stride_ : nullptr;
  }

  // Raw data access
  [[nodiscard]] T* data() noexcept {
    return storage_.data();
  }

  [[nodiscard]] const T* data() const noexcept {
    return storage_.data();
  }

  // Memory stats
  [[nodiscard]] std::size_t bytes_used() const noexcept {
    return num_rows_ * N * sizeof(T);
  }

  [[nodiscard]] std::size_t bytes_allocated() const noexcept {
    return stride_ * N * sizeof(T);
  }

  /**
   * @brief Print the first N rows to an output stream
   * @param n Number of rows to print (default: all rows)
   * @param os Output stream (default: std::cout)
   */
  void print_first(size_type n = std::numeric_limits<size_type>::max(),
                   std::ostream& os = std::cout) const {
    if (empty() || storage_.data() == nullptr) {
      os << "NDDeviceArray<" << N << "> (empty)\n";
      return;
    }

    const size_type print_count = std::min(n, num_rows_);
    std::vector<T> host_data(N * print_count);

    // Copy each column separately
    for (std::size_t dim = 0; dim < N; ++dim) {
      const T* col_ptr = column_ptr(dim);
      if (col_ptr == nullptr) {
        os << "NDDeviceArray: column_ptr(" << dim << ") returned null\n";
        return;
      }
      GPU_ERROR_T err = GPU_MEMCPY(host_data.data() + dim * print_count, col_ptr,
                                   print_count * sizeof(T), GPU_DEVICE_TO_HOST);
      if (err != GPU_SUCCESS) {
        os << "NDDeviceArray: Failed to copy column " << dim
           << " from device: " << GPU_GET_ERROR_STRING(err) << "\n";
        return;
      }
    }

    os << "NDDeviceArray<" << N << "> (num_rows=" << num_rows_ << ", stride=" << stride_
       << ", showing first " << print_count << " rows):\n";
    for (size_type i = 0; i < print_count; ++i) {
      os << "  [" << i << "] = (";
      for (std::size_t dim = 0; dim < N; ++dim) {
        if (dim > 0) {
          os << ", ";
        }
        os << host_data[dim * print_count + i];
        if constexpr (std::is_integral_v<T>) {
          os << " (0x" << std::hex << static_cast<std::uint64_t>(host_data[dim * print_count + i])
             << std::dec << ")";
        }
      }
      os << ")\n";
    }
    if (print_count < num_rows_) {
      os << "  ... (" << (num_rows_ - print_count) << " more rows)\n";
    }
  }

 private:
  DeviceArray<T> storage_;
  size_type num_rows_{0};
  size_type stride_{0};
};

}  // namespace SRDatalog::GPU