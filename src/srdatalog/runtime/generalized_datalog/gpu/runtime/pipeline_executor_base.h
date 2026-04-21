#pragma once

#include "../../runtime/aggregation.h"
#include "../../runtime/state.h"
#include "../device_sorted_array_index.h"
#include "../func_type_helper.h"
#include "../mir.h"
#include "../semiring.h"
#include "./deduplication.h"
#include "./output_context.h"
#include "./store.h"
#include "./tmp_helpers.h"
#include <type_traits>
#include <utility>

#ifdef USE_ROCm
// On HIP/ROCm, cooperative groups are provided by hip_cooperative_groups
#include <hip/hip_cooperative_groups.h>
namespace cg = cooperative_groups;
#else
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif

namespace SRDatalog::GPU {
namespace mp = boost::mp11;

// Helper: Device Decoded Value Extractor
template <typename T>
__device__ T device_decode_from_size_t(std::size_t encoded) {
  if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
    return static_cast<T>(encoded);
  } else if constexpr (std::is_floating_point_v<T>) {
    if constexpr (sizeof(T) == sizeof(std::size_t)) {
      return *reinterpret_cast<const T*>(&encoded);
    } else {
      using UInt = std::conditional_t<sizeof(T) == 4, uint32_t, uint16_t>;
      UInt val_uint = static_cast<UInt>(encoded);
      return *reinterpret_cast<const T*>(&val_uint);
    }
  } else {
    return static_cast<T>(encoded);
  }
}

namespace detail {

template <typename OpsTuple, typename Var>
struct GetIndicesToUpdateSimple {
  template <typename Seq>
  struct seq_to_mp_list_impl;

  template <typename T, T... Is>
  struct seq_to_mp_list_impl<std::integer_sequence<T, Is...>> {
    using type = boost::mp11::mp_list<boost::mp11::mp_size_t<static_cast<std::size_t>(Is)>...>;
  };

  template <typename Op>
  struct handles_for_op_impl {
    using type = boost::mp11::mp_list<>;
  };

  template <typename Op>
    requires requires { Op::template handles_for_var<Var>(); }
  struct handles_for_op_impl<Op> {
    using type = typename seq_to_mp_list_impl<
        std::decay_t<decltype(Op::template handles_for_var<Var>())>>::type;
  };

  template <typename Op>
  using handles_for_op_as_mp_list = typename handles_for_op_impl<Op>::type;

  using OpsList = boost::mp11::mp_rename<OpsTuple, boost::mp11::mp_list>;
  using HandlesLists = boost::mp11::mp_transform<handles_for_op_as_mp_list, OpsList>;
  using Flattened = boost::mp11::mp_apply<boost::mp11::mp_append, HandlesLists>;
  using type = boost::mp11::mp_unique<Flattened>;
};

template <typename OpsTuple, typename Var, typename Offset>
struct GetIndicesToUpdate {
  using type = typename GetIndicesToUpdateSimple<OpsTuple, Var>::type;
};

template <typename OpsTuple, typename Offset>
struct GetIndicesToUpdateBound {
  template <typename Var>
  using fn = typename GetIndicesToUpdate<OpsTuple, Var, Offset>::type;
};

}  // namespace detail

/**
 * @brief GPU Pipeline Executor - handles inner MIR ops after root binding
 */
template <typename MIROps, typename VarPosMap, std::size_t IteratorOffset, typename RelationPosMap,
          int GroupSize,
          bool IsLeafLevel = !mir::HasMoreColumnJoins<SRDatalog::tmp::TupleTail_t<MIROps>>::value>
struct GPUPipelineExecutor;

// Base case: no more operations
template <typename VarPosMap, std::size_t IteratorOffset, typename RelationPosMap, int GroupSize,
          bool IsLeafLevel>
struct GPUPipelineExecutor<std::tuple<>, VarPosMap, IteratorOffset, RelationPosMap, GroupSize,
                           IsLeafLevel> {

  template <Semiring SR, typename ValueType, typename HandleType, std::size_t NumVars,
            std::size_t NumHandles, std::size_t NumRelations, typename Group, typename OutputCtx>
  __device__ static void execute(Group /*tile*/,
                                 state::VarStore<VarPosMap, ValueType, NumVars>& /*vars*/,
                                 state::HandleStore<HandleType, NumHandles>& /*handles*/,
                                 state::ProvStore<RelationPosMap, SR, NumRelations>& /*provs*/,
                                 OutputCtx& /*output_ctx*/) {}
};

// Device Decoded Value Extractor
template <typename VarPosMapExtractor, typename StateTuple, typename VarsTupleExtractor,
          typename TypesTuple>
struct DeviceDecodedValueExtractor;

template <typename VarPosMapExtractor, typename ValueType, std::size_t NumVars, typename... Vars,
          typename... Types>
struct DeviceDecodedValueExtractor<VarPosMapExtractor,
                                   state::VarStore<VarPosMapExtractor, ValueType, NumVars>,
                                   std::tuple<Vars...>, std::tuple<Types...>> {
  __device__ static auto extract(
      const state::VarStore<VarPosMapExtractor, ValueType, NumVars>& vars) {
    if constexpr (sizeof...(Vars) == 0) {
      return std::make_tuple();
    } else {
      return std::make_tuple(device_decode_from_size_t<Types>(vars.template get<Vars>())...);
    }
  }
};

}  // namespace SRDatalog::GPU
