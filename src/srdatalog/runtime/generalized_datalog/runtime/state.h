/**
 * @file runtime/state.h
 * @brief Runtime state management for variable bindings.
 *
 * @details This file provides concepts and utilities for managing runtime state
 * (variable bindings) during MIR execution. The state is represented as a
 * Boost.Hana map where keys are compile-time variable types and values are
 * runtime values.
 */

#pragma once

#ifndef BOOST_HANA_CONFIG_ENABLE_STRING_UDL
#define BOOST_HANA_CONFIG_ENABLE_STRING_UDL
#endif
#include <boost/hana.hpp>
#include <boost/type_index.hpp>
#include <concepts>
#include <type_traits>

#include "ast.h"
#include "func_type_helper.h"
#include "mir.h"  // For CIndexSpec
// Note: query.h is NOT included here to avoid circular dependency
#include "type_name.h"
#include <iostream>
#include <string>

namespace SRDatalog {
using namespace AST;  // Bring AST concepts into scope

namespace hana = boost::hana;
namespace mp = boost::mp11;
using namespace boost::hana::literals;

#ifndef SRD_HOST_DEVICE
#include "gpu/macro.h"  // Provides GPU_HD macro
#define SRD_HOST_DEVICE GPU_HD
#endif

// -----------------------------------------------------------------
// State Management Helpers (Hana-based)
// -----------------------------------------------------------------
namespace state {

/**
 * @brief Helper functions for extracting values from Hana state map
 *
 * @details These helpers use tag dispatching to handle different tuple types
 * (std::tuple vs mir::BoundVars) without template specialization conflicts.
 */
template <typename StateMap, typename... Vars>
  requires(CVar<Vars> && ...)
inline auto extract_values(const StateMap& state, std::tuple<Vars...>*) {
  // Look up each Var in the state map and return as std::tuple directly
  return std::make_tuple(state[hana::type_c<Vars>]...);
}

template <typename StateMap>
inline auto extract_values(const StateMap&, std::tuple<>*) {
  return std::make_tuple();
}

// -----------------------------------------------------------------
// State Management Helpers (Tuple-based with compile-time position map)
// -----------------------------------------------------------------

/**
 * @brief Get a single value from tuple-based state for a given variable
 *
 * @details Uses compile-time position map to look up variable position,
 * then accesses the tuple at runtime. All Hana operations are compile-time.
 *
 * @tparam VarPosMap Compile-time hana::map mapping variable types to positions
 * @tparam StateTuple Runtime state as std::tuple<Value1, Value2, ...>
 * @tparam Var The variable type to extract
 * @return The value of the variable from the state tuple
 */
template <typename VarPosMap, typename StateTuple, CVar Var>
inline auto get_value_from_tuple(const StateTuple& state) {
  // Get position from compile-time map, then access tuple at runtime
  // All Hana operations (at_key, type_c) are evaluated at compile-time
  constexpr auto pos = std::decay_t<decltype(hana::at_key(VarPosMap{}, hana::type_c<Var>))>::value;
  return std::get<pos>(state);
}

/**
 * @brief Primary template for ValueExtractor.
 * It remains undefined to force a compile error if a non-tuple is passed.
 */
template <typename VarPosMap, typename StateTuple, typename Terms>
struct ValueExtractor;

/**
 * @brief Partial specialization for std::tuple<Vars...>.
 * This "cracks open" the Terms type to access Vars... directly.
 */
template <typename VarPosMap, typename StateTuple, typename... Vars>
struct ValueExtractor<VarPosMap, StateTuple, std::tuple<Vars...>> {

  // static inline ensures aggressive optimization
  static inline auto extract(const StateTuple& state) {
    if constexpr (sizeof...(Vars) == 0) {
      return std::make_tuple();
    } else {
      // Direct expansion without helper functions
      return std::make_tuple(get_value_from_tuple<VarPosMap, StateTuple, Vars>(state)...);
    }
  }
};

/**
 * @brief Helper function to make the call site syntax nicer.
 * This is just a thin wrapper around the struct.
 */
template <typename VarPosMap, typename StateTuple, typename Terms>
inline auto extract_values_tuple(const StateTuple& state) {
  return ValueExtractor<VarPosMap, StateTuple, Terms>::extract(state);
}

/**
 * @brief Helper functions for extracting values from Hana state map,
 * return a std::array of the values
 * @tparam StateMap The state map type
 * @tparam Vars The variable types to extract
 * @param state The state map
 * @return A std::array of the values
 */
template <typename StateMap, typename... Vars>
inline auto extract_value_array(const StateMap& state, std::array<Vars...>*) {
  return std::array<Vars...>{state[hana::type_c<Vars>]...};
}

/**
 * @brief Primary template for ArrayValueExtractor.
 * It remains undefined to force a compile error if a non-array is passed.
 */
template <typename VarPosMap, typename StateTuple, typename ArrayType>
struct ArrayValueExtractor;

/**
 * @brief Partial specialization for std::array<Vars...>.
 * This "cracks open" the ArrayType to access Vars... directly.
 */
template <typename VarPosMap, typename StateTuple, typename... Vars>
struct ArrayValueExtractor<VarPosMap, StateTuple, std::array<Vars...>> {
  // static inline ensures aggressive optimization
  static inline auto extract(const StateTuple& state) {
    if constexpr (sizeof...(Vars) == 0) {
      return std::array<Vars...>{};
    } else {
      // Direct expansion without helper functions
      return std::array<Vars...>{get_value_from_tuple<VarPosMap, StateTuple, Vars>(state)...};
    }
  }
};

/**
 * @brief Helper function to make the call site syntax nicer.
 * This is just a thin wrapper around the struct.
 */
template <typename VarPosMap, typename StateTuple, typename ArrayType>
inline auto extract_value_array_tuple(const StateTuple& state) {
  return ArrayValueExtractor<VarPosMap, StateTuple, ArrayType>::extract(state);
}

/**
 * @brief Primary template for DecodedValueExtractor.
 * It remains undefined to force a compile error if non-tuples are passed.
 */
template <typename VarPosMap, typename StateTuple, typename VarsTuple, typename TypesTuple>
struct DecodedValueExtractor;

/**
 * @brief Partial specialization for std::tuple<Vars...> and std::tuple<Types...>.
 * This "cracks open" both tuple types to access Vars... and Types... directly.
 */
template <typename VarPosMap, typename StateTuple, typename... Vars, typename... Types>
  requires(sizeof...(Types) == sizeof...(Vars) && (Decodable<Types> && ...))
struct DecodedValueExtractor<VarPosMap, StateTuple, std::tuple<Vars...>, std::tuple<Types...>> {
  // static inline ensures aggressive optimization
  static inline auto extract(const StateTuple& state) {
    if constexpr (sizeof...(Vars) == 0) {
      return std::make_tuple();
    } else {
      // Extract and decode each value directly using the single-value helper
      return std::make_tuple(
          decode_from_size_t<Types>(get_value_from_tuple<VarPosMap, StateTuple, Vars>(state))...);
    }
  }
};

/**
 * @brief Helper function to make the call site syntax nicer.
 * This is just a thin wrapper around the struct.
 */
template <typename VarPosMap, typename StateTuple, typename VarsTuple, typename TypesTuple>
inline auto extract_values_tuple_decoded(const StateTuple& state) {
  return DecodedValueExtractor<VarPosMap, StateTuple, VarsTuple, TypesTuple>::extract(state);
}

/**
 * @brief Get value for a single term (Var, Const, or CppExpr) from tuple-based state
 *
 * @details Handles different term types:
 * - Var: extracts from StateTuple using VarPosMap
 * - Const: returns compile-time constant value
 * - CppExpr: evaluates expression with captured variables (not yet fully supported)
 *
 * @tparam VarPosMap Compile-time hana::map mapping variable types to positions
 * @tparam StateTuple Runtime state as std::tuple<Value1, Value2, ...>
 * @tparam Term The term type (Var, Const, or CppExpr)
 */
template <typename VarPosMap, typename StateTuple, typename Term>
constexpr auto get_term_value_tuple(const StateTuple& state) {
  if constexpr (CVar<Term>) {
    // Case 1: Variable - get value from state tuple
    return get_value_from_tuple<VarPosMap, StateTuple, Term>(state);
  } else if constexpr (CConst<Term>) {
    // Case 2: Constant - use compile-time constant value
    return Term::value;
  } else if constexpr (CCppExpr<Term>) {
    // Case 3: C++ Expression - would evaluate expression
    static_assert(!std::is_same_v<Term, Term>, "CppExpr not yet supported in head position");
    return std::size_t(0);  // unreachable
  } else {
    // Case 4: Unknown term type - compile error
    static_assert(!std::is_same_v<Term, Term>, "Unknown term type");
    return std::size_t(0);  // unreachable
  }
}

/**
 * @brief Extract values from Terms tuple (Vars, Consts, CppExpr) from tuple-based state
 *
 * @details Extracts values for each term in the Terms tuple. Handles Vars (from state),
 * Consts (compile-time values), and CppExpr (not yet supported).
 *
 * @tparam VarPosMap Compile-time hana::map mapping variable types to positions
 * @tparam StateTuple Runtime state as std::tuple<Value1, Value2, ...>
 * @tparam Terms The tuple of term types to extract
 */
template <typename VarPosMap, typename StateTuple, tmp::CTuple Terms>
inline auto extract_term_values_tuple(const StateTuple& state) {
  return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    return std::make_tuple(
        get_term_value_tuple<VarPosMap, StateTuple, std::tuple_element_t<Is, Terms>>(state)...);
  }(std::make_index_sequence<std::tuple_size_v<Terms>>{});
}

/**
 * @brief Set a value at a specific position in the tuple-based state
 *
 * @details Sets the value for a variable at its compile-time determined position.
 * The state tuple must be pre-allocated with the correct size based on the join plan.
 * All variable positions are known at compile time, so the tuple type is fixed.
 *
 * @tparam VarPosMap Compile-time hana::map mapping variable types to positions
 * @tparam StateTuple Runtime state as std::tuple<Value1, Value2, ...> (fixed size)
 * @tparam Var The variable type to set
 * @tparam Value The value type
 * @param state The runtime state tuple (passed by reference, will be modified)
 * @param value The value to set
 */
template <typename VarPosMap, typename StateTuple, CVar Var, typename Value>
inline void set_value_in_tuple(StateTuple& state, const Value& value) {
  // Get position from compile-time map
  constexpr auto pos = std::decay_t<decltype(hana::at_key(VarPosMap{}, hana::type_c<Var>))>::value;
  static_assert(pos < std::tuple_size_v<StateTuple>,
                "Variable position must be within tuple bounds. Tuple size must match join plan.");
  // Set value at the position
  std::get<pos>(state) = value;
}

/**
 * @brief Print the variable-to-position mapping (VarPosMap)
 *
 * @details Debug helper to print the compile-time position map showing which
 * variable is stored at which position in the state tuple.
 *
 * @tparam VarPosMap Compile-time hana::map mapping variable types to positions
 */
template <typename VarPosMap>
void print_var_pos_map() {
  std::cout << "VarPosMap: ";
  hana::for_each(hana::keys(VarPosMap{}), [](auto key) {
    using VarType = typename decltype(key)::type;
    constexpr auto pos = decltype(hana::at_key(VarPosMap{}, key))::value;
    std::cout << var_to_string<VarType>() << " -> pos[" << pos << "], ";
  });
  std::cout << std::endl;
}

/**
 * @brief Print the tuple-based state with variable names
 *
 * @details Debug helper to print the runtime state tuple, showing variable names
 * and their values. Uses the VarPosMap to map positions back to variable names.
 *
 * @tparam VarPosMap Compile-time hana::map mapping variable types to positions
 * @tparam StateTuple Runtime state as std::tuple<Value1, Value2, ...>
 * @param state The runtime state tuple
 */
template <typename VarPosMap, typename StateTuple>
void print_state_tuple(const StateTuple& state) {
  std::cout << "state: ";
  hana::for_each(hana::keys(VarPosMap{}), [&](auto key) {
    using VarType = typename decltype(key)::type;
    constexpr auto pos = decltype(hana::at_key(VarPosMap{}, key))::value;
    if constexpr (pos < std::tuple_size_v<StateTuple>) {
      std::cout << var_to_string<VarType>() << " = " << std::get<pos>(state) << ", ";
    }
  });
  std::cout << ";" << std::endl;
}

/**
 * @brief Extracts a type that maps to a specific position in a position map.
 *
 * @details Finds the type that maps to the given position by iterating
 * through all keys in the map and checking their position values.
 * Works with any compile-time hana::map that maps types to positions
 * (e.g., VarPosMap, IteratorPosMap).
 *
 * @tparam PosMap Compile-time hana::map mapping types to positions
 * @tparam Pos The position to find
 */
template <typename PosMap, std::size_t Pos>
struct FindTypeAtPos {
  // Helper to check if a key maps to position Pos
  template <typename Key>
  struct KeyMatchesPos {
    using AtKeyResult = decltype(hana::at_key(PosMap{}, Key{}));
    static constexpr bool value = std::remove_reference_t<AtKeyResult>::value == Pos;
  };

  // Recursive search through keys
  // template <typename... Keys>
  // struct FindInKeys;

  // template <typename FirstKey, typename... RestKeys>
  // struct FindInKeys<FirstKey, RestKeys...> {
  //   using type = std::conditional_t<KeyMatchesPos<FirstKey>::value, FirstKey,
  //                                   typename FindInKeys<RestKeys...>::type>;
  // };

  // template <>
  // struct FindInKeys<> {
  //   // Should never reach here if Pos is valid
  //   using type = void;
  // };

  // GCC does not support partial specialization in class scope, so we use a Dummy parameter to
  // allow partial specialization. In clang, you can use above code instead.
  template <typename Dummy, typename... Keys>
  struct FindInKeysImpl;

  template <typename Dummy, typename FirstKey, typename... RestKeys>
  struct FindInKeysImpl<Dummy, FirstKey, RestKeys...> {
    using type = std::conditional_t<KeyMatchesPos<FirstKey>::value, FirstKey,
                                    typename FindInKeysImpl<Dummy, RestKeys...>::type>;
  };

  template <typename Dummy>
  struct FindInKeysImpl<Dummy> {
    using type = void;
  };

  template <typename... Keys>
  using FindInKeys = FindInKeysImpl<void, Keys...>;

  using KeysTuple = decltype(hana::keys(PosMap{}));
  using type = typename mp::mp_apply<FindInKeys, KeysTuple>::type;
};

/**
 * @brief Extracts types from a position map in position order (0, 1, 2, ...).
 *
 * @details Iterates through positions 0 to N-1 and finds the type that maps
 * to each position. This ensures the order matches exactly what's in the map.
 * Works with any compile-time hana::map that maps types to positions.
 *
 * @tparam PosMap Compile-time hana::map mapping types to positions
 * @tparam Size The number of entries in the map
 */
template <typename PosMap, std::size_t Size>
struct ExtractTypesFromPosMap {
  // Build tuple for positions 0..Size-1
  template <std::size_t... Is>
  static constexpr auto make_tuple_impl(std::index_sequence<Is...>) {
    return std::tuple<typename FindTypeAtPos<PosMap, Is>::type...>{};
  }

  using type = decltype(make_tuple_impl(std::make_index_sequence<Size>{}));
};

/**
 * @brief Gets a mutable reference to an iterator from IteratorTuple using IteratorPosMap.
 *
 * @details Looks up the iterator position from IteratorPosMap at compile-time,
 * then returns a mutable reference to the iterator at that position in IteratorTuple.
 * This allows the caller to modify the iterator in-place (e.g., apply prefixes).
 *
 * @tparam IteratorPosMap Compile-time hana::map mapping IndexSpecT types to iterator positions
 * @tparam IteratorTuple Runtime tuple of iterators
 * @tparam IndexSpecT The IndexSpecT type to look up
 * @param iterators The runtime iterator tuple (passed by reference)
 * @return Mutable reference to the iterator at the position specified by IteratorPosMap
 */
template <typename IteratorPosMap, typename IteratorTuple, typename IndexSpecT>
inline auto& get_iterator_ref(IteratorTuple& iterators) {
  // decay for rvalue reference &&
  constexpr auto iter_pos =
      std::decay_t<decltype(hana::at_key(IteratorPosMap{}, hana::type_c<IndexSpecT>))>::value;
  return std::get<iter_pos>(iterators);
}

/**
 * @brief Updates an iterator in IteratorTuple by replacing it with itself prefixed with a value.
 *
 * @details Looks up the iterator mapped to the given Index and replaces it with itself prefixed
 * with the provided value. This is useful for updating iterators when variable bindings change.
 * Handles both single values and tuples of values.
 *
 * @tparam IteratorPosMap Compile-time hana::map mapping IndexSpecT types to iterator positions
 * @tparam IteratorTuple Runtime tuple of iterators
 * @tparam Index The IndexSpecT type whose mapped iterator will be updated
 * @param iterators The runtime iterator tuple (passed by reference, will be modified)
 * @param prefix_val The prefix value(s) to apply (can be a single value or a tuple)
 */
template <typename IteratorPosMap, typename IteratorTuple, mir::CIndexSpec Index>
inline void update_iterator_with_prefixed(IteratorTuple& iterators, const auto& prefix_val) {
  // Get mutable reference to the iterator mapped to Index
  auto& iter = get_iterator_ref<IteratorPosMap, IteratorTuple, Index>(iterators);

  // Check if prefix_val is a tuple type and handle accordingly
  if constexpr (tmp::is_tuple_v<std::decay_t<decltype(prefix_val)>>) {
    // Apply prefix with multiple values from tuple
    std::apply([&](const auto&... vals) { iter = iter.prefix(vals...); }, prefix_val);
  } else {
    // Single value case
    iter = iter.prefix(prefix_val);
  }
}

// -----------------------------------------------------------------
// Tuple-based Save/Restore optimization
// -----------------------------------------------------------------

/**
 * @brief Helper to save specific handles to a tuple using compile-time indices.
 *
 * @tparam HandleStore Type of the handle store (must support operator[])
 * @tparam Indices boost::mp11::mp_list of compile-time indices (integral_constant)
 */
template <typename HandleStore, typename... Indices>
inline SRD_HOST_DEVICE auto save_handles_impl(const HandleStore& handles,
                                              boost::mp11::mp_list<Indices...>) {
  return std::make_tuple(handles[Indices::value]...);
}

template <typename Indices, typename HandleStore>
inline SRD_HOST_DEVICE auto save_handles_to_tuple(const HandleStore& handles) {
  return save_handles_impl(handles, Indices{});
}

/**
 * @brief Helper to restore handles from a tuple using compile-time indices.
 *
 * @tparam HandleStore Type of the handle store
 * @tparam Tuple Type of the saved tuple
 * @tparam Indices boost::mp11::mp_list of compile-time indices
 * @tparam I Index sequence for tuple access
 */
template <typename HandleStore, typename Tuple, typename... Indices, std::size_t... I>
inline SRD_HOST_DEVICE void restore_handles_impl(HandleStore& handles, const Tuple& saved,
                                                 boost::mp11::mp_list<Indices...>,
                                                 std::index_sequence<I...>) {
  // Fold expression to restore each handle
  ((handles[Indices::value] = std::get<I>(saved)), ...);
}

template <typename Indices, typename HandleStore, typename Tuple>
inline SRD_HOST_DEVICE void restore_handles_from_tuple(HandleStore& handles, const Tuple& saved) {
  restore_handles_impl(handles, saved, Indices{},
                       std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

}  // namespace state

}  // namespace SRDatalog
