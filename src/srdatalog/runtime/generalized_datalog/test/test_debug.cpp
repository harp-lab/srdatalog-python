#include "ast.h"
#include "mir.h"
#include <iostream>
#include <tuple>
#include <type_traits>

// Inline stripped down mir_printer.h content

namespace SRDatalog {
namespace detail {

// Forward declaration
template <typename T>
void print_mir_impl(std::ostream& os, int indent);

template <typename T>
void print_type_name(std::ostream& os) {
  if constexpr (AST::is_var_v<T>) {
    os << AST::var_to_string<T>();
  } else {
    os << typeid(T).name();
  }
}

template <typename Tuple>
void print_var_tuple(std::ostream& os) {
  os << "()";
}

template <typename T>
void print_column_source(std::ostream& os, int indent) {
  os << "(column-source)";
}

template <typename T>
void print_mir_node_fallback(std::ostream& os, int indent) {
  os << "(unknown)";
}

template <typename T>
void print_mir_impl(std::ostream& os, int indent) {
  if constexpr (mir::is_column_source_v<T>)
    print_column_source<T>(os, indent);
  else
    print_mir_node_fallback<T>(os, indent);
}

}  // namespace detail

template <typename T>
void print_mir(std::ostream& os) {
  detail::print_mir_impl<T>(os, 0);
}

}  // namespace SRDatalog

// Test Main
using namespace SRDatalog;
using namespace SRDatalog::AST::Literals;

using SR = BooleanSR;
using RSchema = AST::RelationSchema<decltype("R"_s), SR, std::tuple<int, int>>;
using IndexR_AB = mir::IndexSpecT<RSchema, std::integer_sequence<int, 0, 1>, FULL_VER>;
using SourceR = mir::ColumnSource<IndexR_AB, std::tuple<>>;

int main() {
  print_mir<SourceR>(std::cout);
  return 0;
}
