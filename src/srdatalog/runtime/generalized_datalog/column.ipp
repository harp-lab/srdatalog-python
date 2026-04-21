#pragma once

// column.ipp: implementation for template<class T> class Column
// This file is meant to be included by column.hpp (not compiled on its own).

#include "column.h"
#include <utility>   // std::forward

namespace SRDatalog {
// ----- Capacity / size -----
template<class T>
inline void Column<T>::reserve(std::size_t n) { raw.reserve(n); }

template<class T>
inline void Column<T>::clear() noexcept { raw.clear(); }

template<class T>
inline std::size_t Column<T>::size() const noexcept { return raw.size(); }

// ----- Appends -----
template<class T>
inline void Column<T>::push_back(const T& v) { raw.push_back(v); }

template<class T>
inline void Column<T>::push_back(T&& v) { raw.push_back(std::move(v)); }

template<class T>
template<class... Args>
inline T& Column<T>::emplace_back(Args&&... args) {
  return raw.emplace_back(std::forward<Args>(args)...);
}

// ----- Raw buffer access -----
template<class T>
inline T* Column<T>::data() noexcept { return raw.data(); }

template<class T>
inline const T* Column<T>::data() const noexcept { return raw.data(); }

// ----- Element access -----
template<class T>
inline T& Column<T>::operator[](std::size_t i) { return raw[i]; }

template<class T>
inline const T& Column<T>::operator[](std::size_t i) const { return raw[i]; }

// ----- Iterators -----
template<class T>
inline auto Column<T>::begin() noexcept -> Vector<T>::iterator {
  return raw.begin();
}

template<class T>
inline auto Column<T>::end() noexcept -> Vector<T>::iterator {
  return raw.end();
}

template<class T>
inline auto Column<T>::begin() const noexcept -> Vector<T>::const_iterator {
  return raw.begin();
}

template<class T>
inline auto Column<T>::end() const noexcept -> Vector<T>::const_iterator {
  return raw.end();
}
} // namespace SRDatalog