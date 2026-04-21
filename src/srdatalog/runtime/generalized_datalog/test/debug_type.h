#pragma once
#include <iostream>
#include <string_view>

namespace debug {
template <typename T>
constexpr std::string_view type_name() {
#if defined(__clang__)
  return __PRETTY_FUNCTION__;
#elif defined(__GNUC__)
  return __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
  return __FUNCSIG__;
#else
  return "Unsupported compiler";
#endif
}

template <typename T>
void print_type() {
  std::cout << type_name<T>() << std::endl;
}
}  // namespace debug
