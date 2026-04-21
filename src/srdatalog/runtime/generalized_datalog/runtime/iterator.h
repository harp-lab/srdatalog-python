/**
 * @file runtime/iterator.h
 * @brief Iterator generation from ColumnSource MIR operations.
 *
 * @details This file provides template metaprogramming utilities to generate
 * runtime iterators from compile-time ColumnSource MIR operations. These
 * iterators are used for join operations in the MIR executor.
 */

#pragma once

#include "ast.h"  // For CSemiNaiveDatabase, CRelationSchema
#include "mir.h"  // For MIR types (mir::ColumnSource, etc.) and IndexSpecT
#include "runtime/state.h"

namespace SRDatalog {
using namespace AST;  // Bring AST concepts into scope

// -----------------------------------------------------------------
// Part 2: Iterator Generation (Interpreting ColumnSource)
// -----------------------------------------------------------------
namespace detail {
/**
 * @brief Template metaprogramming helper to get an iterator from a ColumnSource
 * using pre-created iterator tuple.
 *
 * @details Uses the IteratorPosMap to look up the iterator from a pre-created
 * iterator tuple. The iterators are updated in-place during pipeline execution
 * as variables are bound, so no prefix computation is needed here.
 *
 * @tparam Source The ColumnSource MIR operation
 * @tparam IteratorPosMap Compile-time hana::map mapping IndexSpecT types to iterator positions
 * @tparam IteratorTuple Runtime tuple of iterators (one per unique IndexSpecT, updated in-place)
 */
template <mir::CColumnSource Source, typename IteratorPosMap, typename IteratorTuple>
struct GenerateIterator;

/**
 * @brief Specialization for ColumnSource: gets iterator from tuple.
 *
 * @details Looks up the iterator from the iterator tuple using the IndexSpecT.
 * The iterator is already updated (prefixed) during pipeline execution, so we
 * just return it directly.
 *
 * @tparam Schema The relation schema
 * @tparam Cols The column indexes for the index
 * @tparam Ver The version number
 * @tparam PrefixVars The tuple of prefix variables (unused here, kept for compatibility)
 * @tparam IteratorPosMap Compile-time hana::map mapping IndexSpecT types to iterator positions
 * @tparam IteratorTuple Runtime tuple of iterators
 */
template <CRelationSchema Schema, int... Cols, std::size_t Ver, tmp::CTuple PrefixVars,
          typename IteratorPosMap, typename IteratorTuple>
struct GenerateIterator<
    mir::ColumnSource<mir::IndexSpecT<Schema, std::integer_sequence<int, Cols...>, Ver>,
                      PrefixVars>,
    IteratorPosMap, IteratorTuple> {
  static auto get(const IteratorTuple& iterators) {
    // 1. Get the IndexSpecT type for this ColumnSource
    using Idx = mir::IndexSpecT<Schema, std::integer_sequence<int, Cols...>, Ver>;

    // 2. Look up the iterator position from the compile-time map
    // decltype preserves the reference qualifiers (in this case, an rvalue reference &&). In C++,
    // you cannot access a static member (like ::value) directly on a reference type (e.g.,
    // T&&::value is invalid syntax). std::decay_t removes the reference and/or cv-qualifiers,
    // so the result is a plain type.
    constexpr auto iter_pos =
        std::decay_t<decltype(hana::at_key(IteratorPosMap{}, hana::type_c<Idx>))>::value;

    // 3. Get and return the iterator from the iterator tuple
    // The iterator is already updated (prefixed) during pipeline execution
    return std::get<iter_pos>(iterators);
  }
};

}  // namespace detail

}  // namespace SRDatalog
