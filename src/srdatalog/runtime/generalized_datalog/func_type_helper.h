#pragma once

#include <tuple>

namespace func_type_detail {
template <typename T>
struct function_traits;

// Specialization for function pointers
template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> {
  using return_type = R;
  using arg_types = std::tuple<Args...>;
  static constexpr std::size_t arity = sizeof...(Args);

  template <std::size_t N>
  using arg_type = std::tuple_element_t<N, arg_types>;
};

// Specialization for const member function (lambdas)
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) const> {
  using return_type = R;
  using arg_types = std::tuple<Args...>;
  static constexpr std::size_t arity = sizeof...(Args);

  template <std::size_t N>
  using arg_type = std::tuple_element_t<N, arg_types>;
};

// Specialization for non-const member function
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...)> {
  using return_type = R;
  using arg_types = std::tuple<Args...>;
  static constexpr std::size_t arity = sizeof...(Args);

  template <std::size_t N>
  using arg_type = std::tuple_element_t<N, arg_types>;
};

// Specialization for callable objects (including lambdas)
template <typename T>
struct function_traits : function_traits<decltype(&T::operator())> {};

}  // namespace func_type_detail

// Public API
template <typename F>
using function_arg_types_t = typename func_type_detail::function_traits<F>::arg_types;

template <typename F>
using function_return_type_t = typename func_type_detail::function_traits<F>::return_type;

template <typename F, std::size_t N>
using function_arg_t = typename func_type_detail::function_traits<F>::template arg_type<N>;

template <typename F>
inline constexpr std::size_t function_arity_v = func_type_detail::function_traits<F>::arity;
