/**
 * @file device_array.h
 * @brief GPU device array wrapper using RMM (RAPIDS Memory Manager) allocator
 */

#pragma once

// Note: Requires LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE to be defined
// (defined in xmake.lua when nvidia or rocm config is enabled)
// This enables CCCL's cuda::mr namespace which RMM v24.12.01 and hipMM depend on

#include <cstddef>
// Use GPU API abstraction instead of direct CUDA includes
#include "gpu/gpu_api.h"

// Workaround for spdlog/fmt consteval issue with clang-cuda
// Use std::format instead of fmt to avoid consteval issues with fmt::basic_format_string
// This avoids the fmt::basic_format_string consteval problem entirely
// See:
// https://forums.developer.nvidia.com/t/spdlog-doesnt-work-with-nvcc-12-6-and-c-20-bug-in-nvcc/332729/3
#ifndef SPDLOG_USE_STD_FORMAT
#define SPDLOG_USE_STD_FORMAT
#endif

// RMM is host-only API - include headers unconditionally since DeviceArray is host-only
#include <rmm/mr/device/per_device_resource.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <vector>

// RMM execution policies for Thrust (ensure Thrust temporary allocations use RMM pool)
// Include RMM headers unconditionally - if they fail in device code, that's fine since
// DeviceArray is host-only anyway
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace SRDatalog::GPU {

/**
 * @brief Thread-safe singleton that provides a global GPU pool memory resource
 * @note GPU (CUDA or HIP) must be initialized before this is called (call init_cuda() first)
 * @note This function is host-only (allocation happens on host)
 * @note Implementation is in device_array_rmm_impl.h (included below for host code)
 */
__host__ rmm::mr::device_memory_resource* get_gpu_pool_memory_resource();

/**
 * @brief Print RMM pool memory usage report
 * @note This function is host-only and intended for debugging/monitoring
 * @note Implementation is in device_array_rmm_impl.h (included below for host code)
 */
__host__ void print_rmm_memory_report();

/**
 * @brief GPU device array wrapper using RMM device_uvector
 * @tparam T Element type
 *
 * Similar to std::vector but for device memory. Uses rmm::device_uvector<uint32_t> internally
 * to store raw data, with default pooling memory resource.
 * @note This class is host-only (all methods are __host__)
 */
template <typename T>
class DeviceArray {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using device_ptr_type = thrust::device_ptr<T>;
  using const_device_ptr_type = thrust::device_ptr<const T>;
  using iterator = device_ptr_type;
  using const_iterator = const_device_ptr_type;

  // Internal storage type alias - implementation detail, but exposed for swap_internal_data API
  using internal_storage_type = rmm::device_uvector<uint32_t>;

  /**
   * @brief Construct an empty device array
   * @note Host-only: allocates device memory from host
   * @note Uses default pooling memory resource (set via rmm::mr::set_current_device_resource)
   */
  __host__ DeviceArray() : vec_(0, rmm::cuda_stream_per_thread), logical_size_(0) {}

  /**
   * @brief Construct a device array with specified size
   * @param size Number of elements of type T
   * @note Host-only: allocates device memory from host
   * @note Uses default pooling memory resource (set via rmm::mr::set_current_device_resource)
   * @note Internally stores data as uint32_t, so size is calculated as bytes rounded up to uint32_t
   */
  __host__ explicit DeviceArray(size_type size)
      : vec_(bytes_to_u32_size(size * sizeof(T)), rmm::cuda_stream_per_thread),
        logical_size_(size) {}

  /**
   * @brief Construct a device array with specified size on a specific stream
   * @param size Number of elements of type T
   * @param stream CUDA/HIP stream for stream-ordered allocation
   */
  __host__ explicit DeviceArray(size_type size, GPU_STREAM_T stream)
      : vec_(bytes_to_u32_size(size * sizeof(T)), rmm::cuda_stream_view{stream}),
        logical_size_(size) {}

  /**
   * @brief Destructor - deallocates device memory
   * @note Host-only: deallocates device memory from host
   */
  __host__ ~DeviceArray() = default;

  // Non-copyable
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;

  // Movable
  DeviceArray(DeviceArray&& other) noexcept
      : vec_(std::move(other.vec_)), logical_size_(other.logical_size_) {
    other.logical_size_ = 0;
  }

  DeviceArray& operator=(DeviceArray&& other) noexcept {
    if (this != &other) {
      vec_ = std::move(other.vec_);
      logical_size_ = other.logical_size_;
      other.logical_size_ = 0;
    }
    return *this;
  }

  // Data access
  [[nodiscard]] pointer data() noexcept {
    return reinterpret_cast<pointer>(vec_.data());
  }
  [[nodiscard]] const_pointer data() const noexcept {
    return reinterpret_cast<const_pointer>(vec_.data());
  }

  [[nodiscard]] device_ptr_type device_ptr() noexcept {
    return thrust::device_ptr<T>(reinterpret_cast<T*>(vec_.data()));
  }
  [[nodiscard]] const_device_ptr_type device_ptr() const noexcept {
    return thrust::device_ptr<const T>(reinterpret_cast<const T*>(vec_.data()));
  }

  // Element access
  [[nodiscard]] auto operator[](size_type idx) noexcept {
    return device_ptr()[idx];
  }
  [[nodiscard]] auto operator[](size_type idx) const noexcept {
    return device_ptr()[idx];
  }

  // Iterators
  [[nodiscard]] iterator begin() noexcept {
    return device_ptr();
  }
  [[nodiscard]] const_iterator begin() const noexcept {
    return device_ptr();
  }
  [[nodiscard]] const_iterator cbegin() const noexcept {
    return device_ptr();
  }

  [[nodiscard]] iterator end() noexcept {
    return device_ptr() + size();
  }
  [[nodiscard]] const_iterator end() const noexcept {
    return device_ptr() + size();
  }
  [[nodiscard]] const_iterator cend() const noexcept {
    return device_ptr() + size();
  }

  // Size and capacity
  [[nodiscard]] size_type size() const noexcept {
    // Return logical size (device_uvector might round up for alignment)
    return logical_size_;
  }
  [[nodiscard]] size_type capacity() const noexcept {
    // Convert from uint32_t capacity back to T capacity
    return (vec_.capacity() * sizeof(uint32_t)) / sizeof(T);
  }
  [[nodiscard]] bool empty() const noexcept {
    return vec_.size() == 0 || size() == 0;
  }

  /**
   * @brief Resize array. Only reallocates if growing beyond capacity.
   * @param new_size New logical size (number of T elements)
   * @param preserve_data If true, copy existing data when reallocating (default: false)
   * @note device_uvector handles reallocation automatically
   */
  void resize(size_type new_size, bool preserve_data = false) {
    (void)preserve_data;  // device_uvector preserves data by default
    vec_.resize(bytes_to_u32_size(new_size * sizeof(T)), rmm::cuda_stream_per_thread);
    logical_size_ = new_size;
  }

  /**
   * @brief Resize array on a specific stream (stream-ordered allocation)
   * @param new_size New logical size (number of T elements)
   * @param stream CUDA/HIP stream for stream-ordered allocation
   */
  void resize(size_type new_size, GPU_STREAM_T stream) {
    vec_.resize(bytes_to_u32_size(new_size * sizeof(T)), rmm::cuda_stream_view{stream});
    logical_size_ = new_size;
  }

  /**
   * @brief Reserve capacity without changing size
   * @param new_capacity New capacity in T elements
   */
  void reserve(size_type new_capacity) {
    vec_.reserve(bytes_to_u32_size(new_capacity * sizeof(T)), rmm::cuda_stream_per_thread);
  }

  /**
   * @brief Shrink capacity to match size
   */
  void shrink_to_fit() {
    vec_.shrink_to_fit(rmm::cuda_stream_per_thread);
  }

  /**
   * @brief Add element to end of array (copies from host)
   * @param value Value to append (must be host-accessible)
   */
  void push_back(const T& value) {
    // Need to resize to accommodate new element
    const size_type old_size = size();
    resize(old_size + 1, true);
    // Copy the value to the last position (synchronous copy)
    GPU_MEMCPY(data() + old_size, &value, sizeof(T), GPU_HOST_TO_DEVICE);
  }

  /**
   * @brief Add element to end of array (moves from host)
   * @param value Value to append (must be host-accessible)
   */
  void push_back(T&& value) {
    push_back(value);  // For device arrays, we always copy
  }

  /**
   * @brief Construct element in-place at end of array
   * @param args Arguments to construct T
   * @return Reference to newly constructed element (device pointer)
   */
  template <typename... Args>
  T& emplace_back(Args&&... args) {
    // Construct on host first, then copy to device
    T temp(std::forward<Args>(args)...);
    push_back(temp);
    return data()[size() - 1];
  }

  /**
   * @brief Append all elements from another array
   * @param other Source array to append from
   */
  void concat(const DeviceArray& other) {
    if (other.empty()) {
      return;
    }
    const size_type old_size = size();
    const size_type new_size = old_size + other.size();
    resize(new_size, true);
#ifndef __CUDA_ARCH__
    // Use RMM execution policy so Thrust temporary allocations use the RMM pool
    thrust::copy_n(rmm::exec_policy{}, other.data(), other.size(), data() + old_size);
#else
    thrust::copy_n(thrust::device, other.data(), other.size(), data() + old_size);
#endif
  }

  /**
   * @brief Clone contents into another array
   * @param other Target array to clone into (will be cleared first)
   */
  void clone_into(DeviceArray& other) const {
    other.clear();
    if (empty()) {
      return;
    }
    other.resize(size(), false);
#ifndef __CUDA_ARCH__
    // Use RMM execution policy so Thrust temporary allocations use the RMM pool
    thrust::copy_n(rmm::exec_policy{}, data(), size(), other.data());
#else
    thrust::copy_n(thrust::device, data(), size(), other.data());
#endif
  }

  /**
   * @brief Batch copy data from host array
   * @param host_data Host array containing data to copy
   * @param count Number of elements to copy
   * @note This is more efficient than calling push_back in a loop
   */
  void assign_from_host(const T* host_data, size_type count) {
    if (count == 0) {
      clear();
      return;
    }
    resize(count, false);
    GPU_MEMCPY(data(), host_data, count * sizeof(T), GPU_HOST_TO_DEVICE);
  }

  /**
   * @brief Clear contents (set size to 0, keep capacity)
   */
  void clear() noexcept {
    vec_.resize(0, rmm::cuda_stream_per_thread);
    logical_size_ = 0;
  }

  /**
   * @brief Reset (deallocate all memory)
   */
  void reset() {
    vec_.resize(0, rmm::cuda_stream_per_thread);
    vec_.shrink_to_fit(rmm::cuda_stream_per_thread);
  }

  /**
   * @brief Swap contents with another array
   */
  void swap(DeviceArray& other) noexcept {
    std::swap(vec_, other.vec_);
    std::swap(logical_size_, other.logical_size_);
  }

  /**
   * @brief Swap internal device_uvector data with external vector
   * @param other_vec External device_uvector to swap with
   * @param new_logical_size Logical size (number of T elements) after swap
   * @note This allows swapping with external buffers while maintaining correct logical size
   * @warning The caller must ensure new_logical_size matches the actual data in other_vec
   *          (i.e., new_logical_size * sizeof(T) <= other_vec.size() * sizeof(uint32_t))
   */
  void swap_internal_data(internal_storage_type& other_vec, size_type new_logical_size) noexcept {
    std::swap(vec_, other_vec);
    logical_size_ = new_logical_size;
  }

  /**
   * @brief Get reference to internal device_uvector (for advanced usage)
   * @return Reference to internal storage vector
   * @warning Modifying the internal vector directly can break size() consistency.
   *          Use swap_internal_data() or resize() instead when possible.
   */
  [[nodiscard]] internal_storage_type& internal_vector() noexcept {
    return vec_;
  }
  [[nodiscard]] const internal_storage_type& internal_vector() const noexcept {
    return vec_;
  }

  [[nodiscard]] rmm::mr::device_memory_resource* memory_resource() noexcept {
    return rmm::mr::get_current_device_resource();
  }
  [[nodiscard]] const rmm::mr::device_memory_resource* memory_resource() const noexcept {
    return rmm::mr::get_current_device_resource();
  }

  [[nodiscard]] std::size_t bytes_used() const noexcept {
    return size() * sizeof(T);
  }
  [[nodiscard]] std::size_t bytes_allocated() const noexcept {
    return capacity() * sizeof(T);
  }

  /**
   * @brief Print the first N elements to an output stream
   * @param n Number of elements to print (default: all elements)
   * @param os Output stream (default: std::cout)
   */
  void print_first(size_type n = std::numeric_limits<size_type>::max(),
                   std::ostream& os = std::cout) const {
    if (empty() || data() == nullptr) {
      os << "DeviceArray (empty)\n";
      return;
    }

    const size_type print_count = std::min(n, size());
    std::vector<T> host_data(print_count);

    GPU_ERROR_T err =
        GPU_MEMCPY(host_data.data(), data(), print_count * sizeof(T), GPU_DEVICE_TO_HOST);
    if (err != GPU_SUCCESS) {
      os << "DeviceArray: Failed to copy data from device: " << GPU_GET_ERROR_STRING(err) << "\n";
      return;
    }

    os << "DeviceArray (size=" << size() << ", capacity=" << capacity() << ", showing first "
       << print_count << "):\n";
    for (size_type i = 0; i < print_count; ++i) {
      os << "  [" << i << "] = " << host_data[i];
      if constexpr (std::is_integral_v<T>) {
        os << " (0x" << std::hex << static_cast<std::uint64_t>(host_data[i]) << std::dec << ")";
      }
      os << "\n";
    }
    if (print_count < size()) {
      os << "  ... (" << (size() - print_count) << " more elements)\n";
    }
  }

 private:
  // Helper to convert bytes to uint32_t size (round up)
  static constexpr size_type bytes_to_u32_size(size_type bytes) noexcept {
    return (bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);
  }

  // Store data as uint32_t, reinterpret as T when needed
  internal_storage_type vec_;
  // Track logical size separately since device_uvector might round up for alignment
  size_type logical_size_{0};
};

}  // namespace SRDatalog::GPU

// Include RMM implementation (host-only) after the class definition
// This provides the implementation of get_gpu_pool_memory_resource()
#ifndef __CUDA_ARCH__
#include "device_array_rmm_impl.h"
#endif  // __CUDA_ARCH__