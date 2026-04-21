#pragma once
/**
 * @file index_traits.h
 * @brief Traits for mapping host indices to device indices.
 *
 * This file provides an extensible mechanism for declaring which device index
 * type should be used for a given host index type.
 */

#include <tuple>

namespace SRDatalog::GPU {

/**
 * @brief Marker type for TVJoin indices.
 *
 * @details Use this as the IndexType in RelationSchema to indicate the relation
 * should use DeviceTVJoinIndex on device. On host, the system will use HashTrieIndex
 * for loading, then convert to TVJoin format during H2D transfer.
 *
 * Example:
 * @code
 * using PathSchema = relation<"Path"_s, BooleanSR, int, int, TVJoinMarker>;
 * @endcode
 */
template <typename SR, typename AttrTuple, typename... IndexCols>
struct TVJoinMarker {};

/**
 * @brief Type trait to check if an index type is TVJoinMarker.
 */
template <typename T>
struct is_tvjoin_marker : std::false_type {};

template <typename SR, typename AttrTuple, typename... Args>
struct is_tvjoin_marker<TVJoinMarker<SR, AttrTuple, Args...>> : std::true_type {};

template <typename T>
inline constexpr bool is_tvjoin_marker_v = is_tvjoin_marker<T>::value;

}  // namespace SRDatalog::GPU
