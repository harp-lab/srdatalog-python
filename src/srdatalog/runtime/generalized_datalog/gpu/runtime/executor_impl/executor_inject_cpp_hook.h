#pragma once

#include "../gpu_fixpoint_executor_common.h"

namespace SRDatalog::GPU {

/**
 * @brief GPU Fixpoint Executor specialization for InjectCppHook instruction.
 *
 * @details This executor invokes a user-provided functor with the database.
 * The functor can access any relation using get_relation_by_schema<Schema, Ver>(db).
 *
 * In C++20, stateless lambdas are default-constructible, so Functor{} works.
 * The functor is invoked with the database reference.
 *
 * @tparam Functor The callable type that takes (auto& db) and returns void
 */
template <typename Functor, typename... TailOps, typename DB>
struct GPUFixpointExecutor<std::tuple<mir::InjectCppHook<Functor>, TailOps...>, DB> {
  static bool execute(DB& db, std::size_t iteration, std::size_t max_output) {
    // C++20: stateless lambdas are default-constructible
    // Create an instance and invoke it with the database
    constexpr Functor functor{};
    functor(db);

    // Continue to remaining operations
    return GPUFixpointExecutor<std::tuple<TailOps...>, DB>::execute(db, iteration, max_output);
  }
};

}  // namespace SRDatalog::GPU
