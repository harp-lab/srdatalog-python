/**
 * @file helper.h
 * @brief Helper utilities for range operations.
 *
 * @details This file provides helper utilities for working with C++20/23 ranges,
 * implementing functionality that may not be available in older standard
 * libraries.
 *
 * @note **C++20/23 Features Used**: This file uses `std::ranges` extensively
 *       for range operations, `std::ranges::input_range` for type constraints,
 *       `if constexpr` for compile-time conditionals, and `std::ranges::sized_range`
 *       for sized range optimization. It provides a compatibility layer for
 *       `std::ranges::to` (C++23) in older standard libraries.
 */
#pragma once

#include <iterator>
#include <ranges>
#include <utility>
#include <algorithm>

// ----------------------------- Range helpers -----------------------------
namespace SRDatalog {
/// @brief Helper to convert a range to a container using the pipe operator
/// @details Usage: auto vec = range | to<std::vector>();
///          Compatible with std::ranges::to but works with older standard libraries
/// @note **C++20 features**: Uses `std::ranges::input_range` concept,
///       `std::ranges::sized_range` for optimization, `if constexpr` for
///       compile-time conditionals, and `std::ranges::copy` for range copying.
template<template<typename...> class Container>
struct to_fn {
    template<std::ranges::input_range R>
    constexpr auto operator()(R&& r) const {
        using value_type = std::ranges::range_value_t<R>;
        Container<value_type> result;
        if constexpr (std::ranges::sized_range<R>) {
            result.reserve(std::ranges::size(r));
        }
        std::ranges::copy(r, std::back_inserter(result));
        return result;
    }
};

/// @brief Function template for converting ranges to containers
/// @details Usage: auto vec = range | to<std::vector>();
///          Supports both: range | to<std::vector>() and range | to<std::vector>
template<template<typename...> class Container>
struct to_impl {
    using type = to_fn<Container>;
    constexpr to_fn<Container> operator()() const {
        return {};
    }
};

template<template<typename...> class Container>
inline constexpr to_impl<Container> to{};

/// @brief Pipe operator for range-to-container conversion
/// @note **C++20 feature**: Uses `std::ranges::input_range` concept for type
///       constraints and supports the pipe operator syntax (range | to<Container>()).
///       This provides a C++23-compatible interface for range conversion.
// Allow pipe operator usage: range | to<std::vector>()
template<template<typename...> class Container, std::ranges::input_range R>
constexpr auto operator|(R&& r, const to_impl<Container>& impl) {
    return impl()(std::forward<R>(r));
}

// Also support: range | to<std::vector>() (with explicit call)
template<template<typename...> class Container, std::ranges::input_range R>
constexpr auto operator|(R&& r, const to_fn<Container>& fn) {
    return fn(std::forward<R>(r));
}
}