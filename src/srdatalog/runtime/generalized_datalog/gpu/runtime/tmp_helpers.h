#pragma once

/**
 * @file tmp_helpers.h
 * @brief Template metaprogramming helpers for GPU pipeline execution, this is used for testing
 * convinence
 */

// Enable string UDL before including any boost::hana headers
#ifndef BOOST_HANA_CONFIG_ENABLE_STRING_UDL
#define BOOST_HANA_CONFIG_ENABLE_STRING_UDL
#endif

#include "../mir.h"
#include "../tmp.h"
#include <boost/hana/string.hpp>

namespace SRDatalog::GPU {

using tmp::TupleTail;
using tmp::TupleTail_t;

using mir::ComputePositionMap;
using mir::ExtractColumnSourcesWithVarInPrefix;
using mir::ExtractDestinationSchemas;
using mir::ExtractRelationSchemas;
using mir::HasMoreColumnJoins;

/// @brief Extract values from VarStore based on a tuple of variables
template <typename VarPosMap, typename ValueType, typename Terms>
struct ExtractTermValues;

template <typename VarPosMap, typename ValueType, typename... Vars>
struct ExtractTermValues<VarPosMap, ValueType, std::tuple<Vars...>> {
  template <typename Store>
  __device__ static void get_array(const Store& vars, ValueType* output) {
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      ((output[Is] = vars.template get<std::tuple_element_t<Is, std::tuple<Vars...>>>()), ...);
    }(std::make_index_sequence<sizeof...(Vars)>{});
  }
};

}  // namespace SRDatalog::GPU
