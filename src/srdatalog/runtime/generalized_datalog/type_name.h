#pragma once

#include <boost/type_index.hpp>
#include <string>

namespace SRDatalog {

// debug print the type name
template <typename T>
std::string get_type_name() {
  boost::typeindex::type_index ti = boost::typeindex::type_id<T>();
  return ti.pretty_name();
}

}  // namespace SRDatalog
