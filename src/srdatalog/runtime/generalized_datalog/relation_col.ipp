#pragma once
// ========================= relation_impl.ipp =========================
// Implementation for: Relation<SR, AttrTuple> declared in relation_col.hpp
// - Header-only (all templates/inline)
// - Semiring-aware dedup (SR::add)
// - Lazy HashTrie indexes with freshness via version_/indexes_dirty_
// ====================================================================
#include "gpu/gpu_api.h"  // GPU API abstraction for CUDA/HIP compatibility
#include "helper.h"
#include "logging.h"
#include "relation_col.h"
#include "semiring.h"
#include "sorted_array_index.h"  // For SortedArrayIndex type check
#include <algorithm>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/tuple/tuple.hpp>
#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <utility>

namespace SRDatalog {

namespace mp = boost::mp11;
// ----------------------------- Tiny helpers -----------------------------

// Helper to get the actual ValueType for build_from_encoded
// For HashmapIndex, we need ValueType (template parameter), not IndexType
// For other index types, we use ValueRange::value_type
template <typename IndexTypeInst, typename IndexValueType>
struct get_actual_value_type {
  using type = IndexValueType;
};
template <typename IndexTypeInst, typename IndexValueType>
  requires requires { typename IndexTypeInst::ValueTypeAlias; }
struct get_actual_value_type<IndexTypeInst, IndexValueType> {
  using type = typename IndexTypeInst::ValueTypeAlias;
};

/// @brief Convert version number to string representation
inline std::string version_to_string(std::size_t version) {
  switch (version) {
    case 0:
      return "FULL_VER";
    case 1:
      return "DELTA_VER";
    case 2:
      return "NEW_VER";
    case 10:
      return "UNKNOWN_VER";
    default:
      return "VER_" + std::to_string(version);
  }
}

/// @brief Get a reference to the I-th attribute column of a relation r.
template <std::size_t I, Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType, typename Policy,
          typename ValueType, typename RowIdType>
static auto& col_at(Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>& r) {
  return r.template column<I>();
}
template <std::size_t I, Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType, typename Policy,
          typename ValueType, typename RowIdType>
static auto& interned_col_at(Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>& r) {
  return r.template interned_column<I>();
}
// const
template <std::size_t I, Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType, typename Policy,
          typename ValueType, typename RowIdType>
static const auto& col_at(
    const Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>& r) {
  return r.template column<I>();
}
template <std::size_t I, Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType, typename Policy,
          typename ValueType, typename RowIdType>
static const auto& interned_col_at(
    const Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>& r) {
  return r.template interned_column<I>();
}

// template helper to dispatch by column index
// It is very common pattern when you construct lexical order on tuple type
// and you want to custom the order of comparison.
template <class Rel, std::size_t Arity = Rel::arity>
inline void dispatch_by_column(std::size_t k, auto&& fn) {
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    bool found = false;
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"
#endif
    ((found || (k == I ? (found = true, fn.template operator()<I>(), false) : false)), ...);
#ifdef __clang__
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    (void)found;  // suppress unused warning if all branches handled
  }(std::make_index_sequence<Arity>{});
}

// // Template meta programming to unroll the tuple construction
// // This technique is very useful when you switch to cuda's thrust library
// template <Semiring SR, ColumnElementTuple AttrTuple>
// template <std::size_t... I>
// inline auto Relation<SR, AttrTuple>::row_tuple_impl_(RowId r,
// std::index_sequence<I...>) const {
//   return std::make_tuple(col_at<I>(*this)[r]...);
// }

// ============================================================
// Relation<SR, AttrTuple>: basic utilities & storage control
// ============================================================

// Clear the entire relation (attributes + annotations), bump version and mark
// indexes dirty.
template <Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType, typename Policy,
          typename ValueType, typename RowIdType>
inline void Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>::clear() {
  if constexpr (StorageTraits::is_device) {
    // Device: Only clear encoded columns and annotations
    device_storage().device_interned_cols_.clear();
    if constexpr (has_provenance_v<SR>) {
      device_storage().device_ann_.clear();
    }
  } else {
    // Host: Clear both original and encoded columns
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      ((col_at<I>(*this).clear()), ...);
    }(std::make_index_sequence<arity>{});
    for (auto& col : host_storage().interned_cols_) {
      col.clear();
    }
    // Clear annotations
    host_storage().ann_.clear();
  }
  // Clear the indexes
  for (auto& [_, idx] : indexes_) {
    idx.clear();
  }
}

// Reserve capacity on all attribute columns and the annotation column.
template <Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType, typename Policy,
          typename ValueType, typename RowIdType>
template <std::size_t... Is>
inline void Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>::reserve_impl_(
    std::size_t n, std::index_sequence<Is...>) {
  if constexpr (StorageTraits::is_device) {
    // Device: reserve handled by main reserve() method
    // This method shouldn't be called for device relations
  } else {
    (col_at<Is>(*this).reserve(n), ...);  // assumes Column<T>::reserve
    host_storage().ann_.reserve(n);
  }
}

template <Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType, typename Policy,
          typename ValueType, typename RowIdType>
template <template <typename, typename, typename...> class DeviceIndexType, typename DevicePolicy,
          typename DeviceValueType, typename DeviceRowIdType>
inline Relation<SR, AttrTuple, IndexType, HostRelationPolicy, ValueType, RowIdType>
Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>::to_host(
    const Relation<SR, AttrTuple, DeviceIndexType, DevicePolicy, DeviceValueType, DeviceRowIdType>&
        device_rel,
    memory_resource* resource) {
  static_assert(detail::IsDevicePolicy<DevicePolicy>,
                "to_host() requires a device relation as input");

  Relation<SR, AttrTuple, IndexType, HostRelationPolicy, ValueType, RowIdType> host_rel(
      resource ? resource : default_memory_resource(), device_rel.version());
  host_rel.set_column_names(device_rel.column_names());
  host_rel.set_name(device_rel.name());
  host_rel.index_specs() = device_rel.index_specs();

#if SRDATALOG_GPU_AVAILABLE == 1
  std::size_t n = device_rel.interned_size();

  // Device relation's Layout determines copy strategy
  using DeviceRelType =
      Relation<SR, AttrTuple, DeviceIndexType, DevicePolicy, DeviceValueType, DeviceRowIdType>;
  constexpr auto device_layout = DeviceRelType::Layout;

  if constexpr (device_layout == GPU::StorageLayout::AoS) {
    // AoS layout: Copy entire buffer, de-interleave on host
    constexpr std::size_t arity_val = DeviceRelType::arity;
    std::vector<DeviceValueType> aos_data(n * arity_val);
    if (n > 0) {
      GPU_MEMCPY(aos_data.data(), device_rel.unsafe_interned_columns().data(),
                 n * arity_val * sizeof(DeviceValueType), GPU_DEVICE_TO_HOST);
    }
    // De-interleave AoS to SoA host columns
    for (std::size_t col = 0; col < arity_val; ++col) {
      host_rel.host_storage().interned_cols_[col].reserve(n);
      for (std::size_t r = 0; r < n; ++r) {
        host_rel.host_storage().interned_cols_[col].push_back(
            static_cast<ValueType>(aos_data[r * arity_val + col]));
      }
    }
  } else {
    // SoA layout: Copy column by column (existing approach)
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      (
          [&]<std::size_t I>() {
            // Copy from device column to host vector
            std::vector<DeviceValueType> device_data(n);
            auto device_col_ptr = device_rel.unsafe_interned_columns().template column_ptr<I>();
            if (n > 0) {
              GPU_MEMCPY(device_data.data(), device_col_ptr, n * sizeof(DeviceValueType),
                         GPU_DEVICE_TO_HOST);
            }

            // Convert DeviceValueType to ValueType and store in host interned column
            host_rel.host_storage().interned_cols_[I].reserve(n);
            for (std::size_t r = 0; r < n; ++r) {
              host_rel.host_storage().interned_cols_[I].push_back(
                  static_cast<ValueType>(device_data[r]));
            }
          }.template operator()<Is>(),
          ...);
    }(std::make_index_sequence<
        Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>::arity>{});
  }

  // Copy annotations from device to host (only if semiring has provenance)
  if constexpr (has_provenance_v<SR>) {
    if (n > 0) {
      if constexpr (std::is_same_v<typename SR::value_type, bool>) {
        std::vector<uint8_t> temp_ann(n);
        GPU_MEMCPY(temp_ann.data(), device_rel.provenance().data(), n * sizeof(bool),
                   GPU_DEVICE_TO_HOST);
        host_rel.host_storage().ann_.reserve(n);
        for (std::size_t r = 0; r < n; ++r) {
          host_rel.host_storage().ann_.push_back(static_cast<bool>(temp_ann[r]));
        }
      } else {
        std::vector<typename SR::value_type> device_ann(n);
        GPU_MEMCPY(device_ann.data(), device_rel.provenance().data(),
                   n * sizeof(typename SR::value_type), GPU_DEVICE_TO_HOST);
        host_rel.host_storage().ann_.reserve(n);
        for (std::size_t r = 0; r < n; ++r) {
          host_rel.host_storage().ann_.push_back(device_ann[r]);
        }
      }
    }
  }
#else
  static_assert(
      Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>::StorageTraits::is_device ==
          false,
      "to_host() requires GPU support to copy from device");
#endif

  return host_rel;
}

template <Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType, typename Policy,
          typename ValueType, typename RowIdType>
template <template <typename, typename, typename...> class HostIndexType, typename HostPolicy,
          typename HostValueType, typename HostRowIdType>
inline Relation<SR, AttrTuple, IndexType, DeviceRelationPolicy, ValueType, RowIdType>
Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>::to_device(
    const Relation<SR, AttrTuple, HostIndexType, HostPolicy, HostValueType, HostRowIdType>&
        host_rel) {
  static_assert(!detail::IsDevicePolicy<HostPolicy>,
                "to_device() requires a host relation as input");

  static_assert(
      Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>::StorageTraits::is_device,
      "to_device() must be called on a Relation with DeviceRelationPolicy");

  Relation<SR, AttrTuple, IndexType, DeviceRelationPolicy, ValueType, RowIdType> device_rel(
      host_rel.version());
  device_rel.set_column_names(host_rel.column_names());
  device_rel.set_name(host_rel.name());
  device_rel.index_specs() = host_rel.index_specs();  // Now uses const version

#if SRDATALOG_GPU_AVAILABLE == 1
  std::size_t n = host_rel.interned_size();

  constexpr std::size_t arity =
      Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>::arity;

  if (n > 0) {
    // Step 1: Prepare all interned column data on host (convert ValueType to ValueType)
    std::array<std::vector<ValueType>, arity> host_cols;
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      ((host_cols[Is].resize(n)), ...);
    }(std::make_index_sequence<arity>{});

    // Convert all rows from ValueType to ValueType on host (batch conversion)
    for (std::size_t r = 0; r < n; ++r) {
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        ((host_cols[Is][r] = static_cast<ValueType>(host_rel.template interned_column<Is>()[r])),
         ...);
      }(std::make_index_sequence<arity>{});
    }

    // Step 2: Batch copy all interned columns to device
    device_rel.unsafe_interned_columns().assign_from_host(host_cols, n);

    // Step 3: Copy annotations from host to device (only if semiring has provenance)
    if constexpr (has_provenance_v<SR>) {
      const auto& host_ann = host_rel.provenance();
      using AnnType = typename SR::value_type;

      if constexpr (std::is_same_v<AnnType, bool>) {
        // std::vector<bool> doesn't have .data(), convert to regular vector (uint8_t)
        std::vector<uint8_t> host_ann_vec(host_ann.begin(), host_ann.end());
        device_rel.provenance().resize(n, false);

        GPU_MEMCPY(device_rel.provenance().data(), host_ann_vec.data(), n * sizeof(bool),
                   GPU_HOST_TO_DEVICE);
      } else {
        std::vector<AnnType> host_ann_vec(host_ann.begin(), host_ann.end());
        device_rel.provenance().assign_from_host(host_ann_vec.data(), n);
      }
    }
  }
#else
  static_assert(
      Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>::StorageTraits::is_device ==
          false,
      "to_device() requires GPU support");
#endif

  return device_rel;
}

// ============================================================
// Debug printing: head(n)
// Prints first n rows; primarily for debugging.
// ============================================================
template <Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType, typename Policy,
          typename ValueType, typename RowIdType>
inline void Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>::head(
    std::size_t n, std::ostream& os) const {
  if constexpr (StorageTraits::is_device) {
    // Device: Can't print original columns (not stored)
    os << "Device relation - original columns not available for printing.\n";
    os << "Size: " << size() << ", Interned size: " << interned_size() << "\n";
    return;
  }

  const std::size_t m = std::min(n, size());

  // Print header: column names if available; otherwise c0..c{k-1}
  if (!col_names_.empty()) {
    for (const auto& s : col_names_)
      os << s << '\t';
  } else {
    mp::mp_for_each<mp::mp_iota_c<arity>>([&](auto I) { os << 'c' << I << '\t'; });
  }
  os << "ann\n";

  // Print m rows
  for (std::size_t r = 0; r < m; ++r) {
    mp::mp_for_each<mp::mp_iota_c<arity>>([&](auto I) { os << col_at<I>(*this)[r] << '\t'; });
    os << host_storage().ann_[r] << '\n';
  }
}

// ============================================================
// Debug printing: head_interned(n)
// Prints first n rows of interned (encoded) values; primarily for debugging.
// For device relations, copies data from device to host before printing.
// ============================================================
template <Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType, typename Policy,
          typename ValueType, typename RowIdType>
inline void Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>::head_interned(
    std::size_t n, std::ostream& os) const {
  const std::size_t m = std::min(n, interned_size());

  // Print header: column names if available; otherwise c0..c{k-1}
  if (!col_names_.empty()) {
    for (const auto& s : col_names_)
      os << s << '\t';
  } else {
    mp::mp_for_each<mp::mp_iota_c<arity>>([&](auto I) { os << 'c' << I << '\t'; });
  }
  os << "ann\n";

  if constexpr (StorageTraits::is_device) {
#if SRDATALOG_GPU_AVAILABLE == 1
    // Device: Copy interned columns from device to host before printing
    constexpr auto device_layout = Layout;

    // Prepare host buffers for interned columns - use template ValueType, not hardcoded uint32_t
    std::array<std::vector<ValueType>, arity> host_cols;
    for (std::size_t c = 0; c < arity; ++c) {
      host_cols[c].resize(m);
    }

    if (m > 0) {
      if constexpr (device_layout == GPU::StorageLayout::AoS) {
        // AoS layout: Copy entire buffer, de-interleave on host
        std::vector<ValueType> aos_data(m * arity);
        GPU_MEMCPY(aos_data.data(), device_storage().device_interned_cols_.data(),
                   m * arity * sizeof(ValueType), GPU_DEVICE_TO_HOST);
        // De-interleave AoS to SoA host columns
        for (std::size_t r = 0; r < m; ++r) {
          for (std::size_t c = 0; c < arity; ++c) {
            host_cols[c][r] = aos_data[r * arity + c];
          }
        }
      } else {
        // SoA layout: Copy column by column
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          (
              [&]<std::size_t I>() {
                auto device_col_ptr =
                    device_storage().device_interned_cols_.template column_ptr<I>();
                GPU_MEMCPY(host_cols[I].data(), device_col_ptr, m * sizeof(ValueType),
                           GPU_DEVICE_TO_HOST);
              }.template operator()<Is>(),
              ...);
        }(std::make_index_sequence<arity>{});
      }
    }

    // Copy annotations from device to host
    std::vector<semiring_value_t<SR>> host_ann(m);
    if constexpr (has_provenance_v<SR>) {
      if (m > 0) {
        if constexpr (std::is_same_v<semiring_value_t<SR>, bool>) {
          std::vector<uint8_t> temp_ann(m);
          GPU_MEMCPY(temp_ann.data(), device_storage().device_ann_.data(), m * sizeof(bool),
                     GPU_DEVICE_TO_HOST);
          for (std::size_t r = 0; r < m; ++r) {
            host_ann[r] = static_cast<bool>(temp_ann[r]);
          }
        } else {
          GPU_MEMCPY(host_ann.data(), device_storage().device_ann_.data(),
                     m * sizeof(semiring_value_t<SR>), GPU_DEVICE_TO_HOST);
        }
      }
    }

    // Print m rows
    for (std::size_t r = 0; r < m; ++r) {
      for (std::size_t c = 0; c < arity; ++c) {
        os << host_cols[c][r] << '\t';
      }
      if constexpr (has_provenance_v<SR>) {
        os << host_ann[r] << '\n';
      } else {
        os << "N/A\n";
      }
    }
#else
    os << "GPU not available - cannot print device relation interned values.\n";
#endif
  } else {
    // Host: Print interned columns directly
    for (std::size_t r = 0; r < m; ++r) {
      mp::mp_for_each<mp::mp_iota_c<arity>>(
          [&](auto I) { os << interned_col_at<I>(*this)[r] << '\t'; });
      if constexpr (has_provenance_v<SR>) {
        os << host_storage().ann_[r] << '\n';
      } else {
        os << "N/A\n";
      }
    }
  }
}

// Public const ensure: delegates to non-const impl so we can rebuild lazily
template <Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType, typename Policy,
          typename ValueType, typename RowIdType>
inline auto Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>::ensure_index(
    const IndexSpec& spec, bool build) const -> const IndexTypeInst& {
  // Cast away const to reuse a single implementation (safe: internal rebuild
  // only)
  return const_cast<Relation&>(*this).ensure_index_nc_(spec, build);
}

// Non-const ensure implementation: creates/rebuilds as needed
template <Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType, typename Policy,
          typename ValueType, typename RowIdType>
inline auto Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>::ensure_index_nc_(
    const IndexSpec& spec, bool build) -> const IndexTypeInst& {

  // Find or create index entry
  auto it = indexes_.find(spec);
  if (it == indexes_.end()) {
    // If not registered, register lazily (idempotent)
    index_specs_.push_back(spec);
    if constexpr (StorageTraits::is_device) {
      // Device: Indexes don't take memory_resource
      it = indexes_.emplace(spec, IndexTypeInst{}).first;
    } else {
      // Host: Pass relation's resource to ensure indexes use the same memory resource
      // Use SFINAE to check if index accepts memory_resource in constructor
      if constexpr (requires { IndexTypeInst{host_storage().resource_}; }) {
        it = indexes_.emplace(spec, IndexTypeInst{host_storage().resource_}).first;
      } else {
        it = indexes_.emplace(spec, IndexTypeInst{}).first;
      }
    }
  }

  IndexTypeInst& idx = it->second;

  // Freshness rule: rebuild if any writes since last build or never built
  if (build && is_dirty(spec)) {
    // DELTA_VER should never need a rebuild from intern cols — the delta index
    // is produced by set_difference_update and owned by the index itself.
    // If we reach here for DELTA_VER, it means something called ensure_index()
    // on a DELTA relation whose intern cols are out of sync with the index.
    if (version() == 3 /* DELTA_VER */) {
      std::cerr << "[WARNING] Rebuilding DELTA_VER index from intern cols for relation '" << name()
                << "' spec " << spec.to_string() << " (interned_size=" << interned_size()
                << ", rows_processed=" << idx.rows_processed()
                << "). This is likely a bug — DELTA indexes should come from "
                << "set_difference_update, not ensure_index()." << std::endl;
    }

    if constexpr (StorageTraits::is_device) {
      // Device: Index building uses encoded data directly (no original columns)
      // Check if index is already up to date
      if (idx.rows_processed() == interned_size()) {
        // Already up to date - no rebuild needed
        return idx;
      }

      // Check if index is dirty by comparing tuple count in index vs relation
      // If mismatch, we need to determine if it's a full rebuild or incremental merge
      const std::size_t relation_size = interned_size();
      const std::size_t index_rows_processed = idx.rows_processed();

      /**
       * @brief Device Index Dirty Checking and Rebuild Logic
       *
       * The index is considered dirty if idx.rows_processed() != relation_size.
       * We determine whether a full rebuild or incremental merge is needed:
       *
       * Full Rebuild Scenarios (high performance cost):
       * 1. Index doesn't exist: rows_processed == 0 && size == 0
       * 2. Index is completely out of sync: rows_processed > relation_size (error state)
       * 3. Index is empty but relation has data: relation_size > 0 && size == 0
       *
       * @warning Full rebuild has significant performance cost. This usually happens when:
       * - Building index on NEW_VER relation for the first time (common in semi-naive evaluation)
       * - Relation was cleared and repopulated
       * - Index was never built before
       *
       * In semi-naive evaluation, full rebuild risk is highest when building indexes on NEW_VER
       * relations, as they start empty and get populated during iteration. Each iteration may
       * trigger a full rebuild if the index doesn't exist yet.
       *
       * @note For incremental updates, use merge() on the index if you only want to add new rows
       * incrementally. However, merge() requires building a temporary index for the new rows first,
       * then merging it into the existing index. The current implementation does full rebuild for
       * incremental updates as well (see TODO below).
       */
      const bool needs_full_rebuild =
          (index_rows_processed == 0 && idx.size() == 0) ||  // Index doesn't exist
          (index_rows_processed > relation_size) ||  // Index ahead of relation (error state)
          (relation_size > 0 && idx.size() == 0);    // Relation has data but index is empty

      if (needs_full_rebuild) {
        // Full rebuild: Build index from all relation data
        // LOG_DEBUG << "Device index full rebuild: relation_size=" << relation_size
        //           << ", index_rows_processed=" << index_rows_processed
        //           << ", index_size=" << idx.size();

        // Pass device arrays directly to build_from_encoded_device
        auto& device_interned_cols = device_storage().device_interned_cols_;
        auto& device_ann = device_storage().device_ann_;

        // Check if index supports build_from_encoded_device
        if constexpr (requires {
                        idx.build_from_encoded_device(spec, device_interned_cols, device_ann);
                      }) {
          idx.build_from_encoded_device(spec, device_interned_cols, device_ann);
        } else {
          throw std::runtime_error(
              "Device index type does not support build_from_encoded_device(). "
              "Use DeviceSortedArrayIndex for device relations.");
        }
      } else {
        // Incremental update: Use merge if available, otherwise fall back to full rebuild
        // This happens when index has processed some rows but relation has more
        const std::size_t new_rows_start = index_rows_processed;
        const std::size_t new_rows_end = relation_size;

        if (new_rows_start < new_rows_end) {
          // LOG_DEBUG << "Device index incremental update: merging rows " << new_rows_start << " to
          // "
          //           << new_rows_end;

          // For incremental updates, we could use merge() if the index supports it
          // However, merge() requires another index to merge from, which we don't have here
          // So for now, we do a full rebuild for incremental updates too
          // TODO: Implement proper incremental merge by creating a temporary index for new rows
          // and merging it into the existing index
          auto& device_interned_cols = device_storage().device_interned_cols_;
          auto& device_ann = device_storage().device_ann_;

          if constexpr (requires {
                          idx.build_from_encoded_device(spec, device_interned_cols, device_ann);
                        }) {
            // Full rebuild for now - incremental merge would require building a temporary
            // index for new rows and then merging
            idx.build_from_encoded_device(spec, device_interned_cols, device_ann);
          } else {
            throw std::runtime_error(
                "Device index type does not support build_from_encoded_device(). "
                "Use DeviceSortedArrayIndex for device relations.");
          }
        }
      }
    } else {
      // Host: Build index from encoded data
      // Encode if needed (multiple indices may share encoded columns)
      if (interned_size() != size()) {
        auto old_size = interned_size();
        resize_interned_columns(size());
        // Encode the new rows
        [this, old_size]<std::size_t... Is>(std::index_sequence<Is...>) {
          (([]<std::size_t I>(auto* self, std::size_t old_sz) {
             auto& interned_col = self->template interned_column<I>();
             for (std::size_t r = old_sz; r < self->size(); ++r) {
               interned_col[r] = encode_to_size_t(self->template column<I>()[r]);
             }
           }.template operator()<Is>(this, old_size)),
           ...);
        }(std::make_index_sequence<arity>{});
      }
      // Check if index needs incremental build
      // Use rows_processed() instead of size() to handle deduplication correctly.
      // For HashTrieIndex: rows_processed() == size() (no deduplication)
      // For SortedArrayIndex: rows_processed() >= size() (after deduplication)
      if (idx.rows_processed() == interned_size()) {
        // Already up to date
        return idx;
      }
      // If index has processed more rows than relation has, the index is out of sync
      // This can happen if relation was cleared and repopulated. Clear and rebuild from scratch.
      if (idx.rows_processed() > interned_size()) {
        // Index is out of sync - clear and rebuild from all relation data
        idx.clear();
      }
      // Get encoded rows for incremental build (from rows_processed to interned_size)
      // After clearing above, rows_processed() will be 0, so this will build from all data
      auto const encoded = encoded_rows(idx.rows_processed(), interned_size());
      // Build index from encoded data
      using IndexValueType = typename IndexTypeInst::ValueRange::value_type;
      using ActualValueType = typename get_actual_value_type<IndexTypeInst, IndexValueType>::type;
      if constexpr (std::is_same_v<std::size_t, ActualValueType>) {
        // Index uses std::size_t, use directly
        // Check if build_from_encoded accepts resource parameter
        if constexpr (requires {
                        idx.build_from_encoded(spec, encoded, host_storage().resource_);
                      }) {
          idx.build_from_encoded(spec, encoded, host_storage().resource_);
        } else {
          idx.build_from_encoded(spec, encoded);
        }
      } else {
        // Convert std::size_t spans to ActualValueType spans
        std::array<std::span<const ActualValueType>, arity> converted_encoded{};
        static thread_local std::array<Vector<ActualValueType>, arity> temp_buffers;
        for (std::size_t c = 0; c < arity; ++c) {
          std::size_t size = encoded[c].size();
          temp_buffers[c] = Vector<ActualValueType>(host_storage().resource_);
          temp_buffers[c].resize(size);
          for (std::size_t i = 0; i < size; ++i) {
            temp_buffers[c][i] = static_cast<ActualValueType>(encoded[c][i]);
          }
          converted_encoded[c] = std::span(temp_buffers[c].data(), temp_buffers[c].size());
        }
        // Check if build_from_encoded accepts resource parameter
        if constexpr (requires {
                        idx.build_from_encoded(spec, converted_encoded, host_storage().resource_);
                      }) {
          idx.build_from_encoded(spec, converted_encoded, host_storage().resource_);
        } else {
          idx.build_from_encoded(spec, converted_encoded);
        }
      }
    }
  }
  // assert(is_dirty(spec) == false);

  return idx;
}

// Build index by taking ownership of intern columns.
// After this call: index has sorted/deduped data, intern cols are empty.
template <Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType, typename Policy,
          typename ValueType, typename RowIdType>
inline auto Relation<SR, AttrTuple, IndexType, Policy, ValueType,
                     RowIdType>::build_index_take_ownership(const IndexSpec& spec)
    -> const IndexTypeInst& {
  static_assert(StorageTraits::is_device,
                "build_index_take_ownership only supported for device relations");

  auto& idx = indexes_[spec];  // get or create

  auto& device_interned_cols = device_storage().device_interned_cols_;
  auto& device_ann = device_storage().device_ann_;

  if constexpr (requires { idx.build_take_ownership(spec, device_interned_cols, device_ann); }) {
    idx.build_take_ownership(spec, device_interned_cols, device_ann);
  } else {
    throw std::runtime_error("Index type does not support build_take_ownership(). "
                             "Use DeviceSortedArrayIndex for device relations.");
  }

  // Clear intern cols — index is now the sole owner of sorted/deduped data.
  // After this: interned_size() == 0, rows_processed_ == 0 → is_dirty() == false.
  resize_interned_columns(0);
  if constexpr (has_provenance_v<SR>) {
    device_ann.resize(0);
  }

  return idx;
}

template <Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType, typename Policy,
          typename ValueType, typename RowIdType>
inline auto Relation<SR, AttrTuple, IndexType, Policy, ValueType,
                     RowIdType>::rebuild_index_from_existing(const IndexSpec& target_spec)
    -> const IndexTypeInst& {
  static_assert(StorageTraits::is_device,
                "rebuild_index_from_existing only supported for device relations");

  // Find the first non-empty existing index (the canonical one from compute-delta)
  const IndexTypeInst* source_idx = nullptr;
  const IndexSpec* source_spec = nullptr;
  for (const auto& [spec, idx] : indexes_) {
    std::cerr << "[rebuild_from_existing] " << name_ << " spec=" << spec.to_string()
              << " size=" << idx.size() << " empty=" << idx.empty() << std::endl;
    if (!idx.empty()) {
      source_idx = &idx;
      source_spec = &spec;
      break;
    }
  }

  if (!source_idx || !source_spec) {
    // No existing index — target is empty too
    std::cerr << "[rebuild_from_existing] " << name_
              << " NO source found for target=" << target_spec.to_string() << std::endl;
    auto& target_idx = indexes_[target_spec];
    target_idx.clear();
    return target_idx;
  }

  std::cerr << "[rebuild_from_existing] " << name_ << " source=" << source_spec->to_string()
            << " size=" << source_idx->size() << " -> target=" << target_spec.to_string()
            << std::endl;

  auto& target_idx = indexes_[target_spec];

  if constexpr (requires { target_idx.build_from_index(*source_spec, target_spec, *source_idx); }) {
    target_idx.build_from_index(*source_spec, target_spec, *source_idx);
  } else {
    throw std::runtime_error("Index type does not support build_from_index(). "
                             "Use DeviceSortedArrayIndex for device relations.");
  }

  std::cerr << "[rebuild_from_existing] " << name_ << " target=" << target_spec.to_string()
            << " built size=" << target_idx.size() << std::endl;

  return target_idx;
}

// Single-argument overload: just return values (for backward compatibility)
template <CNodeHandle NodeHandleT>
auto column_intersect(const NodeHandleT& a) noexcept {
  return a.values();
}

template <CNodeHandle NodeHandleT, CNodeHandle... OtherNodeHandleTs>
auto column_anti_intersect(const NodeHandleT& a, const OtherNodeHandleTs&... others) noexcept {
  auto other_handles = std::make_tuple(others...);
  auto predicate = [other_handles](const auto& key) {
    bool any_not_contain = true;

    std::apply(
        [&any_not_contain, &key](const auto&... handles) {
          ((any_not_contain = any_not_contain || !handles.contains_value(key)), ...);
        },
        other_handles);

    return any_not_contain;
  };

  // Return a lazy view using std::ranges::filter_view (C++20)
  // Note: For SortedArrayIndex, values() now returns unique values (deduplicated at the handle
  // level).
  return std::ranges::views::filter(a.values(), predicate);
}

template <CIndex IndexT, typename... Ts>
auto lookup_prefix(const IndexT& index, const Prefix<Ts...>& prefix) noexcept ->
    typename IndexT::NodeHandle {
  return index.prefix_lookup(prefix.encoded());
}

// Overload that takes Index type as template parameter and calls its static intersect method
// This is the new preferred way to call column_intersect
template <CIndex IndexType, CNodeHandle... NodeHandleTs>
auto column_intersect(const NodeHandleTs&... handles) noexcept {
  // Delegate to the Index's static strategy
  return IndexType::intersect(handles...);
}

// Overload for tuple of handles with Index type - delegates to Index's static intersect
template <CIndex IndexType, tmp::CTuple TupleOfHandles>
auto column_intersect(const TupleOfHandles& handles) noexcept {
  // Unpack tuple and delegate to Index's static intersect method
  return std::apply([](const auto&... handles) { return IndexType::intersect(handles...); },
                    handles);
}

template <Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType, typename Policy,
          typename ValueType, typename RowIdType>
void Relation<SR, AttrTuple, IndexType, Policy, ValueType, RowIdType>::reconstruct_from_index(
    const IndexSpec& spec) {
  // Debug: Print what index is being requested
  // if constexpr (StorageTraits::is_device) {
  //   std::cout << "reconstruct_from_index: Relation " << name() << " version " << version()
  //             << " requesting index spec: " << spec.to_string() << std::endl;
  //   std::cout << "  Available index entries: " << index_specs_.size() << std::endl;
  //   for (const auto& available_spec : index_specs_) {
  //     std::cout << "    " << available_spec.to_string() << std::endl;
  //   }
  // }
  auto& idx = get_index(spec);
  // Only device indices have reconstruct_to_relation
  // CPU indices are built FROM relation intern values (via ensure_index), not the other way around
  if constexpr (StorageTraits::is_device) {
    auto& device_cols = device_storage().device_interned_cols_;
    if constexpr (has_provenance_v<SR>) {
      auto& device_prov = device_storage().device_ann_;
      idx.reconstruct_to_relation(spec, device_cols, device_prov);
    } else {
      // NoProvenance - pass placeholder (reconstruct_to_relation handles this internally)
      GPU::DeviceArray<semiring_value_t<SR>> dummy_prov;
      idx.reconstruct_to_relation(spec, device_cols, dummy_prov);
    }
  } else {
    // CPU: Index is already built from relation intern values (via ensure_index)
    // No reconstruction needed - relation intern values are the source of truth for CPU
    // This function should not be called for CPU relations - it's GPU-only
    static_assert(StorageTraits::is_device,
                  "reconstruct_from_index is GPU-only, CPU indices don't support it");
  }
}

}  // namespace SRDatalog
