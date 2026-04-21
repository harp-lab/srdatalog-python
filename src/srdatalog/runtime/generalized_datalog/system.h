/**
 * @file system.h
 * @brief System abstraction layer for memory management and hardware configuration.
 *
 * @details This header provides a unified interface for memory management and
 * system-level abstractions used throughout SRDatalog. Currently, it focuses
 * on memory resource management using Boost's PMR (Polymorphic Memory Resource)
 * facilities, but is designed to evolve into a more comprehensive system
 * abstraction layer.
 *
 * @section system_current Current Implementation
 *
 * The current implementation provides:
 * - **Memory Resource Abstractions**: Lightweight aliases around Boost PMR
 *   facilities, allowing the rest of the codebase to depend on a single header
 * - **PMR-Aware Containers**: Type aliases for containers that support custom
 *   memory allocators (Vector, Map)
 * - **Default Resource Access**: Helper function to obtain the default memory
 *   resource
 *
 * This indirection makes it possible to change the backend allocator in one
 * place, enabling different memory strategies (arena allocation, memory pooling,
 * custom allocators) without modifying the rest of the codebase.
 *
 * @section system_future Future Evolution
 *
 * This module is designed to evolve into a more polymorphic and configurable
 * system abstraction layer that will support:
 *
 * - **Hardware Abstraction**: Runtime selection of execution backends based on
 *   available hardware (CPU, GPU, TPU, etc.)
 * - **Parallel Framework Support**: Configurable parallel execution frameworks
 *   (OpenMP, TBB, CUDA, SYCL, etc.) selected at runtime based on configuration
 *   and hardware capabilities
 * - **Memory Strategy Selection**: Runtime configuration of memory allocation
 *   strategies optimized for different hardware and workloads
 * - **Performance Tuning**: Runtime configuration of performance parameters
 *   (thread counts, batch sizes, cache sizes) based on hardware characteristics
 *
 * The current simple aliases will be extended with polymorphic interfaces and
 * factory patterns that allow runtime selection of implementations based on
 * configuration files, environment variables, or programmatic configuration.
 *
 * @section system_usage Usage
 *
 * @code{.cpp}
 * using namespace SRDatalog;
 *
 * // Use default memory resource
 * Vector<int> vec(default_memory_resource());
 *
 * // Use custom memory resource (e.g., arena allocator)
 * Arena arena;
 * Vector<int> vec2(&arena);
 *
 * // PMR-aware map
 * Map<std::string, int> my_map(default_memory_resource());
 * @endcode
 *
 * @note The current implementation is intentionally minimal to provide a stable
 *       foundation. Future enhancements will maintain backward compatibility
 *       while adding polymorphic capabilities.
 *
 * @see memory_resource for the memory resource type
 * @see Vector for PMR-aware vector type
 * @see Map for PMR-aware map type
 */
#pragma once

#include <boost/container/container_fwd.hpp>
#include <boost/container/pmr/deque.hpp>
#include <boost/container/pmr/memory_resource.hpp>
#include <boost/container/pmr/polymorphic_allocator.hpp>
#include <boost/container/pmr/vector.hpp>
#include <boost/functional/hash.hpp>
#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered/unordered_flat_set.hpp>
#include <hwy/aligned_allocator.h>

namespace SRDatalog {

struct IdentityHash {
  // This typedef tells Boost: "Trust me, this hash is high quality."
  // It disables Boost's automatic re-hashing/mixing of the result.
  using is_avalanching = void;

  // Simply return the key itself. Zero CPU cost.
  std::size_t operator()(std::size_t k) const noexcept {
    return k;
  }
};

struct FibonacciHash {
  using is_avalanching = void;

  // The Golden Ratio constant for 64-bit (2^64 / phi)
  static constexpr std::size_t kGoldenRatio = 0x9e3779b97f4a7c15ULL;

  inline std::size_t operator()(std::size_t k) const noexcept {
    return k * kGoldenRatio;
  }
};

/**
 * @brief Polymorphic memory resource type used throughout SRDatalog.
 * @details This is an alias to Boost's PMR memory_resource, which provides
 *          a polymorphic interface for memory allocation. Different implementations
 *          (arena allocators, pool allocators, etc.) can be used interchangeably
 *          through this interface.
 * @see boost::container::pmr::memory_resource for the underlying type
 */
using memory_resource = boost::container::pmr::memory_resource;

/**
 * @brief Project-wide default polymorphic allocator type.
 * @details This allocator uses std::byte as the value type and can be used
 *          with any memory_resource. It provides type-erased allocation
 *          capabilities.
 * @see boost::container::pmr::polymorphic_allocator for the underlying type
 */
using polymorphic_allocator = boost::container::pmr::polymorphic_allocator<std::byte>;

/**
 * @brief PMR-aware vector type alias.
 * @tparam T Element type stored in the vector
 * @details This is a drop-in replacement for std::vector that supports custom
 *          memory resources. All containers using this type can benefit from
 *          custom allocation strategies (arena allocation, pooling, etc.).
 * @note This type will be extended in the future to support hardware-specific
 *       optimizations and parallel execution frameworks.
 * @see boost::container::pmr::vector for the underlying type
 */
template <typename T>
using Vector = boost::container::pmr::vector<T>;

template <typename T>
using Deque = boost::container::pmr::deque<T>;

/**
 * @brief PMR-aware unordered flat map type alias.
 * @tparam K Key type
 * @tparam V Value type
 * @details This is a high-performance hash map that supports custom memory
 *          resources. The "flat" variant provides better cache locality than
 *          traditional node-based maps.
 * @note This type will be extended in the future to support hardware-specific
 *       optimizations (e.g., GPU-friendly hash tables).
 * @see boost::unordered::pmr::unordered_flat_map for the underlying type
 */
template <typename K, typename V>
using Map = boost::unordered::unordered_flat_map<
    K, V, boost::hash<K>, std::equal_to<>,
    boost::container::pmr::polymorphic_allocator<std::pair<const K, V>>>;

/**
 * @brief PMR-aware unordered flat set type alias.
 * @tparam T Element type stored in the set
 * @details This is a high-performance hash set that supports custom memory
 *          resources. The "flat" variant provides better cache locality than
 *          traditional node-based sets. Ensures uniqueness of elements.
 * @note This type will be extended in the future to support hardware-specific
 *       optimizations (e.g., GPU-friendly hash tables).
 * @see boost::unordered::unordered_flat_set for the underlying type
 */
template <typename T>
using Set = boost::unordered::unordered_flat_set<T, boost::hash<T>, std::equal_to<>,
                                                 boost::container::pmr::polymorphic_allocator<T>>;

/**
 * @brief Tag type for uninitialized resize operations.
 * @details This tag can be passed to vector::resize() to avoid zero-initialization
 *          of new elements, which is useful when elements will be immediately
 *          overwritten. This avoids unnecessary page faults and improves performance.
 * @note This abstraction allows the codebase to work even if the underlying
 *       container implementation changes (e.g., from Boost to std::vector in C++20).
 * @see boost::container::default_init for the underlying tag
 * @example
 * @code{.cpp}
 * Vector<int> vec;
 * vec.resize(1000, default_init);  // Resize without zero-initialization
 * // ... immediately overwrite elements ...
 * @endcode
 */
constexpr auto default_init = boost::container::default_init;
/**
 * @brief Obtain the default memory resource used for PMR-aware containers.
 * @return Pointer to the process-wide default PMR resource supplied by Boost
 * @details Returns the default memory resource that will be used by containers
 *          when no explicit resource is provided. This is typically the system
 *          allocator, but can be changed globally via Boost's PMR API.
 * @note In the future, this function may be extended to return hardware-specific
 *       memory resources based on runtime configuration.
 * @see boost::container::pmr::get_default_resource for the underlying function
 */
inline auto default_memory_resource() -> memory_resource* {
  return boost::container::pmr::get_default_resource();
}

// ============================================================================
// Highway Aligned Memory Resource (for SIMD-optimized allocations)
// ============================================================================
/**
 * @brief PMR memory resource that uses Highway's aligned allocation functions.
 * @details This memory resource provides 64-byte aligned allocations compatible
 *          with SIMD operations. It uses Highway's AllocateAlignedBytes and
 *          FreeAlignedBytes functions, which are target-independent and work
 *          across all SIMD targets.
 * @note This is particularly useful for Highway VQSort and other SIMD operations
 *       that require aligned memory.
 */
class HighwayAlignedMemoryResource : public memory_resource {
 public:
  void* do_allocate(size_t bytes, size_t /*alignment*/) override {
    // Highway's AllocateAlignedBytes provides proper alignment (typically 64 bytes)
    // It's target-independent and works across all SIMD targets
    void* ptr = hwy::AllocateAlignedBytes(bytes, nullptr, nullptr);
    // if (ptr == nullptr) {
    //   throw std::bad_alloc();
    // }
    return ptr;
  }

  void do_deallocate(void* p, size_t /*bytes*/, size_t /*alignment*/) override {
    if (p != nullptr) {
      hwy::FreeAlignedBytes(p, nullptr, nullptr);
    }
  }

  bool do_is_equal(const memory_resource& other) const noexcept override {
    return this == &other;
  }
};

/**
 * @brief Get a singleton instance of HighwayAlignedMemoryResource.
 * @return Reference to the global HighwayAlignedMemoryResource instance
 * @details This provides a single instance that can be reused across the
 *          application for aligned allocations.
 */
inline HighwayAlignedMemoryResource& highway_aligned_memory_resource() {
  static HighwayAlignedMemoryResource instance;
  return instance;
}

/**
 * @brief Type alias for 64-byte aligned Boost PMR vectors using Highway's allocator.
 * @tparam T Element type stored in the vector
 * @details This is a PMR vector that uses Highway's aligned allocator, ensuring
 *          proper alignment for SIMD operations.
 */
template <typename T>
using AlignedVector = Vector<T>;

/**
 * @brief Create a 64-byte aligned Boost PMR vector using Highway's allocator.
 * @tparam T Element type
 * @param n Number of elements
 * @return AlignedVector<T> with n elements, using Highway's aligned allocator
 * @details This helper function creates a vector with proper alignment for SIMD
 *          operations. The vector uses Highway's allocation functions which provide
 *          optimal alignment for Highway VQSort and other SIMD operations.
 */
template <typename T>
AlignedVector<T> make_aligned_vector(size_t n) {
  AlignedVector<T> vec(&highway_aligned_memory_resource());
  vec.resize(n);
  return vec;
}

#ifdef __linux__
#include <sys/mman.h>
#endif

inline void optimize_vector_memory(void* ptr, size_t size_bytes) {
  // 1. Align pointer to page boundary (downwards)
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t aligned_addr = addr & ~(4095ULL);

#ifdef __linux__
  // 2. Tell Kernel: "I will access this sequentially, please pre-load."
  // MADV_HUGEPAGE: Use 2MB pages (Reduces TLB misses and Faults)
  // MADV_SEQUENTIAL: Aggressive read-ahead
  madvise(reinterpret_cast<void*>(aligned_addr), size_bytes, MADV_HUGEPAGE);
  madvise(reinterpret_cast<void*>(aligned_addr), size_bytes, MADV_SEQUENTIAL);
#endif
}

}  // namespace SRDatalog