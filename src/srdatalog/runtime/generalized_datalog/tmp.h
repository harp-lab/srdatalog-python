/**
 * @file tmp.h
 * @brief Template metaprogramming utilities for SRDatalog.
 *
 * @details This file provides template metaprogramming utilities for working
 * with tuples and type lists. It simplifies common operations using C++20
 * features.
 *
 * @note **C++20/23 Features Used**: This file uses C++20 concepts (CTuple) for
 *       type constraints. The utilities are simplified with C++20 features
 *       including fold expressions, `if constexpr`, and `consteval` where
 *       applicable.
 */
#pragma once
#include <boost/mp11.hpp>
#include <tuple>

namespace SRDatalog::tmp {

namespace mp = boost::mp11;

// Type predicates
template <typename T>
using is_tuple = mp::mp_similar<T, std::tuple<>>;

template <typename T>
inline constexpr bool is_tuple_v = is_tuple<T>::value;

/// @brief Concept for tuple types.
/// @note **C++20 feature**: Uses C++20 concepts for type constraints.
/// @tparam T The type to check
template <typename T>
concept CTuple = is_tuple_v<T>;

// Tuple operations (direct aliases to Mp11)
template <typename... Ts>
using ToTuples = std::tuple<Ts...>;

template <typename Tuple>
struct TupleTail;

template <typename Head, typename... Tail>
struct TupleTail<std::tuple<Head, Tail...>> {
  using type = std::tuple<Tail...>;
};

template <typename Tuple>
using TupleTail_t = typename TupleTail<Tuple>::type;

template <typename T, typename U>
using TupleConcat = mp::mp_append<T, U>;

template <typename T, typename Tuple>
using Contains = mp::mp_contains<Tuple, T>;

template <typename T, typename Tuple>
inline constexpr bool contains_v = Contains<T, Tuple>::value;

template <typename T, typename Tuple>
using PushBack_t = mp::mp_push_back<Tuple, T>;

template <typename Tuple>
using TupleUnique = mp::mp_unique<Tuple>;

template <typename SetA, typename SetB>
using TupleSetIntersection = mp::mp_set_intersection<SetA, SetB>;

template <typename SetA, typename SetB>
using TupleSetDifference = mp::mp_set_difference<SetA, SetB>;

template <std::size_t N, typename T>
using take_t = mp::mp_take_c<T, N>;

// Runtime helpers (keep these)
template <typename... Ts>
constexpr auto tuple_head(std::tuple<Ts...> t) {
  return std::get<0>(t);
}

template <typename T, typename... Ts>
auto tuple_tail(const std::tuple<T, Ts...>& t) {
  return std::apply([](auto _ /*first*/, auto... rest) { return std::make_tuple(rest...); }, t);
}

}  // namespace SRDatalog::tmp
