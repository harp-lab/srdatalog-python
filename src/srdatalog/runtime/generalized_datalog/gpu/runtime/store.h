#pragma once

/**
 * @file store.h
 * @brief State management stores for GPU pipeline execution
 *
 * @details Provides VarStore, HandleStore, and ProvStore for managing
 * runtime state during GPU join execution.
 *
 * @note Compared to the CPU runtime, the GPU version uses a simplified state
 * management strategy. Variables are stored in fixed-size arrays within
 * wrapped structs (e.g., VarStore, HandleStore) rather than using std::tuple.
 * This simplifies indexing in GPU kernels but may not allow the compiler to
 * perform register allocation as effectively as it would for individual
 * stack variables or tuple fields. On PTX seems compiler inlines it well now,
 * but need be careful about future behavior.
 */

#include "../semiring.h"
#include <boost/hana.hpp>
//
#include "gpu/gpu_api.h"  // GPU API abstraction (replaces cuda_runtime.h)

namespace SRDatalog::GPU::state {

/**
 * @brief Variable store - maps Var<'x'> to values via compile-time position lookup
 */
template <typename VarPosMap, typename ValueType,
          std::size_t NumVars = decltype(boost::hana::size(VarPosMap{}))::value>
struct VarStore {
  static constexpr std::size_t num_vars = NumVars;

  // on registers
  ValueType values_[NumVars];

  __device__ VarStore() {
#pragma unroll
    for (std::size_t i = 0; i < NumVars; ++i) {
      values_[i] = 0;
    }
  }

  template <typename Var>
  __device__ void set(ValueType val) {
    constexpr std::size_t idx =
        std::decay_t<decltype(*boost::hana::find(VarPosMap{}, boost::hana::type_c<Var>))>::value;
    static_assert(idx < NumVars, "Variable index out of bounds");
    values_[idx] = val;
  }

  template <typename Var>
  __device__ ValueType get() const {
    constexpr std::size_t idx =
        std::decay_t<decltype(*boost::hana::find(VarPosMap{}, boost::hana::type_c<Var>))>::value;
    static_assert(idx < NumVars, "Variable index out of bounds");
    return values_[idx];
  }

  __device__ ValueType& operator[](std::size_t idx) {
    return values_[idx];
  }
  __device__ const ValueType& operator[](std::size_t idx) const {
    return values_[idx];
  }
};

/**
 * @brief Handle store - maps IndexSpec to NodeHandle via compile-time position lookup
 */
/**
 * @brief ViewStore - Stores immutable views corresponding to handles.
 */
template <typename ViewType, std::size_t NumHandles>
struct ViewStore {
  static constexpr std::size_t num_handles = NumHandles;

  // on registers (pointers to global memory)
  const ViewType* views_[NumHandles];

  __device__ ViewStore() = default;

  // Initialize from global arrays
  // Initialize from global arrays
  // We only need source views here. Full views are for OutputContext.
  __device__ explicit ViewStore(const ViewType* source_views) {
#pragma unroll
    for (std::size_t i = 0; i < NumHandles; ++i) {
      views_[i] = &source_views[i];
    }
  }

  // Access by index
  __device__ const ViewType& operator[](std::size_t idx) const {
    return *views_[idx];
  }
};

/**
 * @brief HandleStore - Stores iterator handles in a linear array.
 * @details accessing handles is done by index (0 to NumHandles-1).
 */
template <typename HandleType, std::size_t NumHandles>
struct HandleStore {
  static constexpr std::size_t num_handles = NumHandles;
  using ViewType = typename HandleType::View;
  using ViewStoreType = ViewStore<ViewType, NumHandles>;

  // on registers
  HandleType handles_[NumHandles];
  const ViewStoreType* view_store_{nullptr};

  __device__ HandleStore() = default;

  // Set reference to view store
  __device__ void set_view_store(const ViewStoreType* vs) {
    view_store_ = vs;
  }

  // Get view for a handle index
  __device__ const ViewType& get_view(std::size_t idx) const {
    // We assume view_store_ is set properly in root executor
    return (*view_store_)[idx];
  }

  // Access by index
  __device__ HandleType& operator[](std::size_t idx) {
    return handles_[idx];
  }
  __device__ const HandleType& operator[](std::size_t idx) const {
    return handles_[idx];
  }

  __device__ void copy_from(const HandleStore& other) {
#pragma unroll
    for (std::size_t i = 0; i < NumHandles; ++i) {
      handles_[i] = other.handles_[i];
    }
    // Preserve view store pointer
    view_store_ = other.view_store_;
  }
};

/**
 * @brief Provenance store - maps Schema to provenance value
 * @details When SR is NoProvenance (has_provenance_v<SR> == false), all operations
 * become no-ops and storage is minimized to a single monostate.
 */
template <typename RelationPosMap, Semiring SR,
          std::size_t NumRelations = decltype(boost::hana::size(RelationPosMap{}))::value>
struct ProvStore {
  static constexpr std::size_t num_relations = NumRelations;
  static constexpr bool enabled = has_provenance_v<SR>;
  using ProvType = semiring_value_t<SR>;

  // Conditional storage: full array when provenance enabled, empty when disabled
  // Use array of size 1 when disabled to avoid zero-size array issues
  ProvType provs_[enabled ? NumRelations : 1];

  __device__ ProvStore() {
    if constexpr (enabled) {
#pragma unroll
      for (std::size_t i = 0; i < NumRelations; ++i) {
        provs_[i] = sr_one<SR>();
      }
    }
    // No initialization needed when disabled
  }

  template <typename Schema>
  __device__ void set(ProvType prov) {
    if constexpr (enabled) {
      constexpr std::size_t idx = std::decay_t<decltype(*boost::hana::find(
          RelationPosMap{}, boost::hana::type_c<Schema>))>::value;
      static_assert(idx < NumRelations, "Schema index out of bounds");
      provs_[idx] = prov;
    }
    // No-op when disabled
  }

  template <typename Schema>
  __device__ ProvType get() const {
    if constexpr (enabled) {
      constexpr std::size_t idx = std::decay_t<decltype(*boost::hana::find(
          RelationPosMap{}, boost::hana::type_c<Schema>))>::value;
      static_assert(idx < NumRelations, "Schema index out of bounds");
      return provs_[idx];
    } else {
      return {};  // Return default monostate
    }
  }

  __device__ ProvType combine_all() const {
    if constexpr (enabled) {
      ProvType result = sr_one<SR>();
#pragma unroll
      for (std::size_t i = 0; i < NumRelations; ++i) {
        result = sr_mul<SR>(result, provs_[i]);
      }
      return result;
    } else {
      return {};  // Return default monostate
    }
  }

  __device__ void reset() {
    if constexpr (enabled) {
#pragma unroll
      for (std::size_t i = 0; i < NumRelations; ++i) {
        provs_[i] = sr_one<SR>();
      }
    }
    // No-op when disabled
  }
};

}  // namespace SRDatalog::GPU::state
