#pragma once

#include "ast.h"
#include "mir.h"
#include <iostream>
#include <tuple>
#include <type_traits>

namespace SRDatalog {

namespace detail {

struct MirPrinter {

  // Helper to print indentation
  static void print_indent(std::ostream& os, int indent) {
    for (int i = 0; i < indent; ++i)
      os << "  ";
  }

  // NOTE: In-class definition allows bodies to reference other members defined later.
  // We need print_impl to be a template.

  // Print Tuple of MIR ops
  template <typename... Os>
  static void print_mir_tuple(std::ostream& os, const std::tuple<Os...>&, int indent) {
    // We call print_impl<Os>. Since print_impl is a member template defined in this class,
    // we can call it.
    ((print_impl<Os>(os, indent), os << "\n"), ...);
  }

  // Helper to print integer sequence column indexes
  template <int... Cols>
  static void print_integer_sequence(std::ostream& os, std::integer_sequence<int, Cols...>) {
    const char* sep = "";
    ((os << sep << Cols, sep = " "), ...);
  }

  // Helper to convert std::index_sequence to std::integer_sequence<int, ...>
  // Helper to convert std::index_sequence to std::integer_sequence<int, ...>
  template <std::size_t... Is>
  static auto to_int_seq(std::index_sequence<Is...>)
      -> std::integer_sequence<int, static_cast<int>(Is)...>;

  // Helper to print version
  static void print_version(std::ostream& os, std::size_t ver) {
    if (ver == NEW_VER)
      os << "NEW";
    else if (ver == DELTA_VER)
      os << "DELTA";
    else if (ver == FULL_VER)
      os << "FULL";
    else
      os << ver;
  }

  // Type Printers
  template <typename T>
  static void print_type_name(std::ostream& os) {
    if constexpr (AST::is_var_v<T>) {
      os << AST::var_to_string<T>();
    } else if constexpr (SRDatalog::mir::is_index_spec_v<T>) {
      os << "(index-spec :schema ";
      os << AST::tstring_to_string<typename T::schema_type::name_type>();

      os << " :index (";
      using ColSeq = typename T::column_indexes_type;
      using Schema = typename T::schema_type;

      // If column_indexes_type is empty (placeholder), use canonical index
      using CanonicalSeq = typename Schema::canonical_index_type;
      using CanonicalIntSeq = decltype(to_int_seq(CanonicalSeq{}));
      using FinalColSeq = std::conditional_t<std::is_same_v<ColSeq, std::integer_sequence<int>>,
                                             CanonicalIntSeq, ColSeq>;
      print_integer_sequence(os, FinalColSeq{});
      os << ")";  // End :index

      os << " :ver ";
      print_version(os, T::kVersion);
      os << ")";  // End index-spec
    } else {
      os << typeid(T).name();
    }
  }

  template <typename Tuple>
  static void print_type_tuple(std::ostream& os) {
    os << "(tuple";
    if constexpr (std::tuple_size_v<Tuple> > 0) {
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        auto print_one = [&](auto idx) {
          os << " ";
          print_type_name<std::tuple_element_t<idx, Tuple>>(os);
        };
        (print_one(std::integral_constant<std::size_t, Is>{}), ...);
      }(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
    }
    os << ")";
  }

  template <typename Tuple>
  static void print_var_tuple(std::ostream& os) {
    os << "(";
    if constexpr (std::tuple_size_v<Tuple> > 0) {
      [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        bool first = true;
        auto print_one = [&](auto idx) {
          if (!first)
            os << " ";
          first = false;
          print_type_name<std::tuple_element_t<idx, Tuple>>(os);
        };
        (print_one(std::integral_constant<std::size_t, Is>{}), ...);
      }(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
    }
    os << ")";
  }

  // MIR Node Printers

  template <typename T>
  static void print_column_source(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(column-source";
    os << " :index ";
    print_type_name<typename T::index_spec_type>(os);
    os << " :ver ";
    using IndexSpec = typename T::index_spec_type;
    constexpr std::size_t ver = IndexSpec::kVersion;
    if (ver == NEW_VER)
      os << "NEW";
    else if (ver == DELTA_VER)
      os << "DELTA";
    else if (ver == FULL_VER)
      os << "FULL";
    else
      os << ver;
    os << " :prefix ";
    print_var_tuple<typename T::prefix_vars_type>(os);
    os << ")";
  }

  template <typename T>
  static void print_scan(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(scan";
    os << " :vars ";
    print_var_tuple<typename T::vars_type>(os);
    os << " :index ";
    print_type_name<typename T::index_spec_type>(os);
    os << " :ver ";
    using IndexSpec = typename T::index_spec_type;
    constexpr std::size_t ver = IndexSpec::kVersion;
    if (ver == NEW_VER)
      os << "NEW";
    else if (ver == DELTA_VER)
      os << "DELTA";
    else if (ver == FULL_VER)
      os << "FULL";
    else
      os << ver;
    os << " :prefix ";
    print_var_tuple<typename T::prefix_vars_type>(os);
    if constexpr (T::has_handles) {
      os << " :handle-start " << T::handle_start;
    }
    os << ")";
  }

  template <typename T>
  static void print_column_join(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(column-join";
    os << " :var " << AST::var_to_string<typename T::var_type>();
    if constexpr (T::has_handles) {
      os << " :handle-start " << T::handle_start;
      // TODO: Print var_to_handles_map if needed (it's a type map, hard to print at runtime without
      // Hana magic) For now, presence of handle-start implies optimized handled join.
    }
    os << "\n";
    print_indent(os, indent + 1);
    os << ":sources (\n";
    print_mir_tuple(os, typename T::sources_type{}, indent + 2);
    print_indent(os, indent + 1);
    os << "))";  // Close sources and column-join
  }

  template <typename T>
  static void print_cartesian_join(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(cartesian-join";
    os << " :vars ";
    print_var_tuple<typename T::vars_type>(os);
    if constexpr (T::has_handles) {
      os << " :handle-start " << T::handle_start;
    }
    os << "\n";
    print_indent(os, indent + 1);
    os << ":sources (\n";
    print_mir_tuple(os, typename T::sources_type{}, indent + 2);
    print_indent(os, indent + 1);
    os << "))";  // Close sources and cartesian-join
  }

  template <typename T>
  static void print_aggregate(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(aggregate";
    os << " :bind " << AST::var_to_string<typename T::var_type>();
    os << " :index ";
    print_type_name<typename T::index_spec_type>(os);
    os << " :prefix ";
    print_var_tuple<typename T::prefix_vars_type>(os);
    if constexpr (T::has_handles) {
      os << " :handle-start " << T::handle_start;
    }
    os << ")";
  }

  template <typename T>
  static void print_destination_relation(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(insert-into";
    os << " :schema " << AST::tstring_to_string<typename T::schema_type::name_type>();
    os << " :ver ";
    if (T::Version == NEW_VER)
      os << "NEW";
    else if (T::Version == DELTA_VER)
      os << "DELTA";
    else if (T::Version == FULL_VER)
      os << "FULL";
    else
      os << T::Version;

    os << " :terms ";
    print_var_tuple<typename T::terms_type>(os);
    os << ")";
  }

  template <typename T>
  static void print_check_size(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(check-size";
    os << " :schema " << AST::tstring_to_string<typename T::schema_type::name_type>();
    os << " :ver ";
    if (T::Version == NEW_VER)
      os << "NEW";
    else if (T::Version == DELTA_VER)
      os << "DELTA";
    else if (T::Version == FULL_VER)
      os << "FULL";
    else
      os << T::Version;
    os << ")";
  }

  template <typename T>
  static void print_clear_relation(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(clear-relation";
    os << " :schema " << AST::tstring_to_string<typename T::schema_type::name_type>();
    os << " :ver ";
    if (T::Version == NEW_VER)
      os << "NEW";
    else if (T::Version == DELTA_VER)
      os << "DELTA";
    else if (T::Version == FULL_VER)
      os << "FULL";
    else
      os << T::Version;
    os << ")";
  }

  template <typename T>
  static void print_insert_from_relation(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(insert-from-relation";
    os << " :schema " << AST::tstring_to_string<typename T::schema_type::name_type>();
    os << " :from-ver " << T::FromVersion;
    os << " :to-ver " << T::ToVersion;
    os << " :index ";
    print_type_name<typename T::index_spec_type>(os);
    os << ")";
  }

  template <typename T>
  static void print_swap_relations(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(swap-relations";
    os << " :schema " << AST::tstring_to_string<typename T::schema_type::name_type>();
    os << ")";
  }

  template <typename T>
  static void print_compute_delta(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(compute-delta";
    os << " :schema " << AST::tstring_to_string<typename T::schema_type::name_type>();
    os << ")";
  }

  template <typename T>
  static void print_merge_index(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(merge-index";
    os << " :index ";
    print_type_name<typename T::index_spec_type>(os);
    os << ")";
  }

  template <typename T>
  static void print_compute_delta_and_merge(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(compute-delta-and-merge";
    os << " :schema " << AST::tstring_to_string<typename T::schema_type::name_type>();
    os << ")";
  }

  template <typename T>
  static void print_rebuild_index(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(rebuild-index";
    os << " :index ";
    print_type_name<typename T::index_spec_type>(os);
    os << " :ver ";
    using IndexSpecType = typename T::index_spec_type;
    constexpr std::size_t ver = IndexSpecType::kVersion;
    if (ver == NEW_VER)
      os << "NEW";
    else if (ver == DELTA_VER)
      os << "DELTA";
    else if (ver == FULL_VER)
      os << "FULL";
    else
      os << ver;
    os << ")";
  }

  template <typename T>
  static void print_set_version(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(set-version";
    os << " :schema " << AST::tstring_to_string<typename T::schema_type::name_type>();
    os << " :ver ";
    if (T::Version == NEW_VER)
      os << "NEW";
    else if (T::Version == DELTA_VER)
      os << "DELTA";
    else if (T::Version == FULL_VER)
      os << "FULL";
    else
      os << T::Version;
    os << ")";
  }

  template <typename T>
  static void print_pipeline(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(pipeline";

    using Srcs = typename T::source_specs_type;
    using Dests = typename T::destination_specs_type;

    os << " :sources ";
    print_type_tuple<Srcs>(os);
    os << " :dests ";
    print_type_tuple<Dests>(os);
    os << "\n";

    print_mir_tuple(os, typename T::mir_ops_type{}, indent + 1);
    print_indent(os, indent);
    os << ")";
  }

  template <typename T>
  static void print_execute_pipeline(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(execute-pipeline";

    using Srcs = typename T::pipeline_type::source_specs_type;
    using Dests = typename T::pipeline_type::destination_specs_type;

    os << " :sources ";
    print_type_tuple<Srcs>(os);
    os << " :dests ";
    print_type_tuple<Dests>(os);

    os << "\n";
    print_impl<typename T::pipeline_type>(os, indent + 1);
    os << "\n";
    print_indent(os, indent);
    os << ")";
  }

  template <typename T>
  static void print_fixpoint_plan(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(fixpoint-plan\n";
    print_mir_tuple(os, typename T::instructions_type{}, indent + 1);
    print_indent(os, indent);
    os << ")";
  }

  template <typename T>
  static void print_if_clause(std::ostream& os, int indent) {
    print_indent(os, indent);
    os << "(if-clause";
    os << " :captures ";
    using ExprDeps = typename AST::GetExprDependencies<typename T::expr_type>::type;
    print_var_tuple<ExprDeps>(os);
    os << ")";
  }

  // Helper SFINAE check for instructions_type
  template <typename T, typename = void>
  struct has_instructions_type : std::false_type {};
  template <typename T>
  struct has_instructions_type<T, std::void_t<typename T::instructions_type>> : std::true_type {};
  template <typename T>
  static constexpr bool has_instructions_type_v = has_instructions_type<T>::value;

  // Dispatcher - Defined Inline
  template <typename T>
  static void print_impl(std::ostream& os, int indent) {
    // Check FixpointPlan first using the type trait
    if constexpr (mir::is_fixpoint_plan_v<T>) {
      print_fixpoint_plan<T>(os, indent);
    }
    // Check RebuildIndex and SetVersion early (before other relation ops that might have similar
    // members) Use explicit predicates only - no fallback logic
    else if constexpr (mir::is_rebuild_index_v<T>)
      print_rebuild_index<T>(os, indent);
    else if constexpr (mir::is_set_version_v<T>)
      print_set_version<T>(os, indent);
    else if constexpr (mir::is_column_source_v<T>)
      print_column_source<T>(os, indent);
    else if constexpr (mir::is_column_join_v<T>)
      print_column_join<T>(os, indent);
    else if constexpr (mir::is_cartesian_join_v<T>)
      print_cartesian_join<T>(os, indent);
    else if constexpr (mir::is_aggregate_v<T>)
      print_aggregate<T>(os, indent);
    else if constexpr (mir::is_destination_relation_v<T>)
      print_destination_relation<T>(os, indent);
    else if constexpr (mir::is_check_size_v<T>)
      print_check_size<T>(os, indent);
    else if constexpr (mir::is_clear_relation_v<T>)
      print_clear_relation<T>(os, indent);
    else if constexpr (mir::is_insert_from_relation_v<T>)
      print_insert_from_relation<T>(os, indent);
    else if constexpr (mir::is_swap_relations_v<T>)
      print_swap_relations<T>(os, indent);
    else if constexpr (mir::is_compute_delta_v<T>)
      print_compute_delta<T>(os, indent);
    else if constexpr (mir::is_merge_index_v<T>)
      print_merge_index<T>(os, indent);
    else if constexpr (mir::is_compute_delta_and_merge_v<T>)
      print_compute_delta_and_merge<T>(os, indent);
    else if constexpr (mir::is_pipeline_v<T>)
      print_pipeline<T>(os, indent);

    else if constexpr (AST::is_if_clause_v<T>)
      print_if_clause<T>(os, indent);
    else if constexpr (mir::is_execute_pipeline_v<T>)
      print_execute_pipeline<T>(os, indent);
    else if constexpr (mir::is_scan_v<T>)
      print_scan<T>(os, indent);

    else {
      // No fallback - fail explicitly if type is not handled
      static_assert(!std::is_same_v<T, T>, "Unhandled MIR instruction type in printer");
    }
  }
};

}  // namespace detail

/**
 * @brief Pretty print a MIR type as an S-expression.
 *
 * @tparam T The MIR type to print (Pipeline, ColumnJoin, etc.)
 * @param os The output stream
 */
template <typename T>
void print_mir(std::ostream& os) {
  detail::MirPrinter::print_impl<T>(os, 0);
}

}  // namespace SRDatalog
