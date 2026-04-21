#pragma once

/**
 * @file output_context.h
 * @brief Output context for two-phase GPU execution (count + materialize)
 */

#include "../index_concepts.h"  // For StorageLayout
#include "./store.h"
#include <boost/hana.hpp>

namespace SRDatalog::GPU {

/**
 * @brief Output context for two-phase execution (count + materialize)
 *
 * @tparam Layout Storage layout for output (SoA or AoS)
 * @tparam Arity Number of columns (needed for AoS row-major indexing)
 */
template <typename ValueType, Semiring SR, bool SizeOnly, StorageLayout Layout = StorageLayout::SoA,
          std::size_t Arity = 2, typename FullHandleMap = void, typename HandleType = void>
struct OutputContext {
  static constexpr bool has_full_handles = !std::is_same_v<FullHandleMap, void>;

  // Helper to lazily compute hana::size only when FullHandleMap is not void
  template <typename T, bool Enable>
  struct get_handle_map_size {
    static constexpr std::size_t value = 0;
  };
  template <typename T>
  struct get_handle_map_size<T, true> {
    static constexpr std::size_t value = decltype(boost::hana::size(T{}))::value;
  };

  static constexpr std::size_t num_full_handles =
      get_handle_map_size<FullHandleMap, has_full_handles>::value;

  ValueType* output_data_;
  // Conditional provenance output: nullptr_t when disabled, pointer when enabled
  [[no_unique_address]]
  std::conditional_t<has_provenance_v<SR>, semiring_value_t<SR>*, std::nullptr_t> output_prov_;
  std::size_t
      output_stride_;  // For SoA: stride between columns. For AoS: arity (redundant but simpler)
  uint32_t local_count_;
  uint32_t write_base_;

  // Full handles for deduplication - use monostate as placeholder when disabled
  // Helper to avoid evaluating HandleType::View when HandleType is void
  template <typename T, bool Enable>
  struct get_view_type {
    using type = std::monostate;
  };
  template <typename T>
  struct get_view_type<T, true> {
    using type = typename T::View;
  };

  using ConditionalHandleType = std::conditional_t<has_full_handles, HandleType, std::monostate>;
  using ConditionalViewType = typename get_view_type<HandleType, has_full_handles>::type;
  ConditionalHandleType full_handles_[has_full_handles ? num_full_handles : 1];
  ConditionalViewType full_views_[has_full_handles ? num_full_handles : 1];

  // Constructor accepts conditional provenance pointer type
  template <typename ProvPtrType>
  __device__ OutputContext(ValueType* output_data, ProvPtrType output_prov,
                           std::size_t output_stride, uint32_t write_base)
      : output_data_(output_data), output_prov_(), output_stride_(output_stride), local_count_(0),
        write_base_(write_base) {
    // Only assign provenance pointer if semiring has provenance
    if constexpr (has_provenance_v<SR>) {
      output_prov_ = output_prov;
    }
  }

  /// @brief Set full handle for a schema
  template <typename Schema>
  __device__ void set_full_handle(const ConditionalHandleType& handle,
                                  const ConditionalViewType& view) {
    if constexpr (has_full_handles) {
      constexpr std::size_t idx = std::decay_t<decltype(*boost::hana::find(
          FullHandleMap{}, boost::hana::type_c<Schema>))>::value;
      full_handles_[idx] = handle;
      full_views_[idx] = view;
    }
  }

  /// @brief Get full handle for a schema
  template <typename Schema>
  __device__ ConditionalHandleType get_full_handle() const {
    if constexpr (has_full_handles) {
      constexpr std::size_t idx = std::decay_t<decltype(*boost::hana::find(
          FullHandleMap{}, boost::hana::type_c<Schema>))>::value;
      return full_handles_[idx];
    } else {
      return ConditionalHandleType{};  // Invalid handle
    }
  }

  /// @brief Get full view for a schema
  template <typename Schema>
  __device__ const ConditionalViewType& get_full_view() const {
    if constexpr (has_full_handles) {
      constexpr std::size_t idx = std::decay_t<decltype(*boost::hana::find(
          FullHandleMap{}, boost::hana::type_c<Schema>))>::value;
      return full_views_[idx];
    } else {
      return full_views_[0];
    }
  }

  /// @brief Emit a tuple (count or materialize based on SizeOnly)
  template <typename Schema, typename Terms, typename VarPosMapT, std::size_t NumVars>
  __device__ void emit(const state::VarStore<VarPosMapT, ValueType, NumVars>& vars,
                       semiring_value_t<SR> prov) {
    if constexpr (SizeOnly) {
      local_count_++;
    } else {
      uint32_t output_pos = write_base_ + local_count_;
      write_columns<Terms, 0, VarPosMapT, NumVars>(vars, output_pos);
      if constexpr (has_provenance_v<SR>) {
        output_prov_[output_pos] = prov;
      }
      local_count_++;
    }
  }

  /// @brief Emit a tuple directly from column values (for JIT kernels)
  /// Usage: output.emit_direct(col0, col1, col2, ...);
  template <typename... Cols>
  __device__ void emit_direct(Cols... cols) {
    if constexpr (SizeOnly) {
      local_count_++;
    } else {
      uint32_t output_pos = write_base_ + local_count_;
      write_values_variadic<0>(output_pos, cols...);
      local_count_++;
    }
  }

  /// @brief Add n to the count (for bulk counting without per-element loop)
  __device__ void add_count(uint32_t n) {
    local_count_ += n;
  }

  __device__ uint32_t count() const {
    return local_count_;
  }

 private:
  template <typename Terms, std::size_t Col, typename VarPosMapT, std::size_t NumVars>
  __device__ void write_columns(const state::VarStore<VarPosMapT, ValueType, NumVars>& vars,
                                uint32_t output_pos) {
    if constexpr (Col < std::tuple_size_v<Terms>) {
      using Term = std::tuple_element_t<Col, Terms>;
      ValueType val = vars.template get<Term>();

      if constexpr (Layout == StorageLayout::AoS) {
        // Row-major: output_data_[(row * arity) + col]
        output_data_[(output_pos * Arity) + Col] = val;
      } else {
        // Column-major (SoA): output_data_[(col * stride) + row]
        output_data_[(Col * output_stride_) + output_pos] = val;
      }

      write_columns<Terms, Col + 1, VarPosMapT, NumVars>(vars, output_pos);
    }
  }

  /// @brief Write variadic column values (for JIT emit_direct)
  template <std::size_t Col, typename T, typename... Rest>
  __device__ void write_values_variadic(uint32_t output_pos, T val, Rest... rest) {
    if constexpr (Layout == StorageLayout::AoS) {
      output_data_[(output_pos * Arity) + Col] = static_cast<ValueType>(val);
    } else {
      output_data_[(Col * output_stride_) + output_pos] = static_cast<ValueType>(val);
    }
    if constexpr (sizeof...(Rest) > 0) {
      write_values_variadic<Col + 1>(output_pos, rest...);
    }
  }
};

}  // namespace SRDatalog::GPU
