/**
 * @file runtime/io.h
 * @brief File I/O utilities for loading data into relations.
 *
 * @details This file provides functions for loading data from files (CSV, TSV,
 * space-separated) into relations, with automatic delimiter detection.
 */

#pragma once

#include "relation_col.h"  // For Relation, Semiring, ColumnElementTuple
#include "runtime/query.h"
#include <algorithm>
#include <charconv>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace SRDatalog {

struct DelimiterInfo {
  char single_char = ' ';
  bool is_multi_space = false;
  std::size_t min_spaces = 1;
};

inline DelimiterInfo detect_delimiter(std::ifstream& file, std::string& first_line) {
  std::string line;
  DelimiterInfo info;

  while (std::getline(file, line)) {
    if (line.empty() || (line.find_first_not_of(" \t\r\n") == std::string::npos)) {
      continue;
    }
    first_line = line;

    std::size_t tab_count = std::count(line.begin(), line.end(), '\t');
    std::size_t comma_count = std::count(line.begin(), line.end(), ',');

    if (tab_count > 0) {
      info.single_char = '\t';
      info.is_multi_space = false;
      return info;
    }
    if (comma_count > 0) {
      info.single_char = ',';
      info.is_multi_space = false;
      return info;
    }

    std::size_t max_consecutive_spaces = 0;
    std::size_t current_consecutive = 0;
    for (char c : line) {
      if (c == ' ') {
        current_consecutive++;
        max_consecutive_spaces = std::max(max_consecutive_spaces, current_consecutive);
      } else {
        current_consecutive = 0;
      }
    }

    if (max_consecutive_spaces >= 2) {
      info.single_char = ' ';
      info.is_multi_space = true;
      info.min_spaces = max_consecutive_spaces;
      return info;
    }

    info.single_char = ' ';
    info.is_multi_space = false;
    return info;
  }
  return info;
}

namespace detail {

// Fast integer parser
template <typename T>
inline const char* parse_int(const char* p, const char* end, T& value) {
  value = 0;
  bool negative = false;
  // Skip optional whitespace before number (if not in multi-space mode, handled by caller?)
  // Actually, caller handles delimiters. Leading whitespace in field?
  // Original IO trimmed whitespace.
  while (p < end && (*p == ' ' || *p == '\t'))
    p++;

  if (p < end && *p == '-') {
    negative = true;
    p++;
  }
  while (p < end && *p >= '0' && *p <= '9') {
    value = value * 10 + (*p - '0');
    p++;
  }
  if (negative)
    value = -value;
  return p;
}

// Fast float parser
template <typename T>
inline const char* parse_float(const char* p, const char* end, T& value) {
  // Try std::from_chars (C++17) for floats if available, or strtod fallback
  // Since we don't know compiler support for float from_chars perfectly,
  // and we want robust speed, we use strtod but need null-termination.
  // mmap is not null-terminated.
  // We scan to end of token, copy to buffer, use strtod/from_chars.

  const char* token_start = p;
  // Scan until whitespace or delimiter (caller logic knows delimiter)
  // But here we rely on caller passing correct "end" of token?
  // No, caller passes "end" of file.
  // We need to parse valid float chars.
  // [+|-][digit]*[.][digit]*[e[+|-]digit*]
  // This is complex to scan perfectly without logic.
  // Simpler: Copy token until delimiter.
  // But we don't know delimiter here!
  // It's better if `parse_float` takes the delimiter.
  return p;  // Placeholder, see logic below
}

// Helper to determine next delimiter position
inline const char* find_delimiter(const char* p, const char* end, const DelimiterInfo& info) {
  if (info.is_multi_space) {
    // Skip spaces until next non-space is a delimiter?
    // Multi-space seps are sequences of >= min_spaces spaces.
    // This is tricky.
    // Original IO: regex " {min,}".
    // New logic: scan for sequence of spaces.
    // Scan until we find >= min_spaces spaces.
    // Or we scan until next token start?
    // Multi-space usually means "whitespace separated".
    // We scan forward for `min_spaces` spaces.
    const char* curr = p;
    while (curr < end) {
      // Search for space
      if (*curr == ' ') {
        // Check if it's a separator sequence
        const char* seek = curr;
        size_t count = 0;
        while (seek < end && *seek == ' ') {
          count++;
          seek++;
        }
        if (count >= info.min_spaces)
          return curr;
        curr = seek;  // Skip these spaces as they are part of content?
                      // Wait, " {2,}" regex means 2 or more spaces.
                      // If we found 1 space, it's content?
                      // Usually multi-space file implies alignement, so 1 space might be content?
                      // Original IO regex logic was robust.
                      // Scan logic: if we find `min_spaces` spaces, that's a delimiter.
      } else if (*curr == '\t' || *curr == '\n' || *curr == '\r') {
        // Newline is always row delimiter
        return curr;
      }
      curr++;
    }
    return curr;
  } else {
    // Single char
    while (p < end && *p != info.single_char && *p != '\n' && *p != '\r')
      p++;
    return p;
  }
}

template <typename AttrTuple, typename Relation>
void fast_load_file_impl(Relation& rel, const std::string& filename) {
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1)
    throw std::runtime_error("Could not open file: " + filename);

  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    close(fd);
    throw std::runtime_error("Could not stat: " + filename);
  }

  size_t file_size = sb.st_size;
  if (file_size == 0) {
    close(fd);
    return;
  }

  char* mapped = (char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (mapped == MAP_FAILED) {
    close(fd);
    throw std::runtime_error("Could not mmap: " + filename);
  }

  // Hint OS
  madvise(mapped, file_size, MADV_SEQUENTIAL);

  const char* p = mapped;
  const char* end = mapped + file_size;

  // Detect delimiter using first line logic (reusing implementation helper?)
  // We can construct a stream from first few bytes or reimplement detection on buffer.
  // Reimplementing on buffer is faster.
  // Simple detection: check matches of \t, ,, or multi-space in first line.

  DelimiterInfo delim_info;
  const char* line_end = p;
  while (line_end < end && *line_end != '\n')
    line_end++;

  // Analyze first line p -> line_end
  size_t tabs = 0, commas = 0, max_spaces = 0, curr_spaces = 0;
  for (const char* c = p; c < line_end; c++) {
    if (*c == '\t')
      tabs++;
    if (*c == ',')
      commas++;
    if (*c == ' ') {
      curr_spaces++;
      if (curr_spaces > max_spaces)
        max_spaces = curr_spaces;
    } else {
      curr_spaces = 0;
    }
  }

  if (tabs > 0) {
    delim_info.single_char = '\t';
  } else if (commas > 0) {
    delim_info.single_char = ',';
  } else if (max_spaces >= 2) {
    delim_info.is_multi_space = true;
    delim_info.min_spaces = max_spaces;
  } else {
    delim_info.single_char = ' ';
  }  // Default single space?

  // Estimate rows
  size_t est_rows = file_size / 20;  // Rough guess

  // Reserve
  static constexpr size_t Arity = Relation::arity;

  [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    ((rel.template column<Is>().reserve(est_rows)), ...);
    ((rel.template interned_column<Is>().reserve(est_rows)), ...);
  }(std::make_index_sequence<Arity>{});
  if constexpr (has_provenance_v<typename Relation::semiring_type>) {
    rel.provenance().reserve(est_rows);
  }

  // Parsing Loop
  while (p < end) {
    // Skip leading whitespace/empty lines if at start of line
    while (p < end && (*p == '\n' || *p == '\r'))
      p++;
    // Do NOT skip spaces if space is delimiter!
    if (p >= end)
      break;

    // Parse columns
    bool row_error = false;

    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      (([&]() {
         if (row_error)
           return;  // Skip remaining cols

         using ColType = typename std::tuple_element<Is, AttrTuple>::type;
         ColType val;

         // Trim leading whitespace ?? Original trimmed inside token.
         // Here we are at start of token.
         while (p < end && (*p == ' ' || *p == '\t') && !delim_info.is_multi_space &&
                delim_info.single_char != ' ' && delim_info.single_char != '\t')
           p++;

         if constexpr (std::is_integral_v<ColType>) {
           p = parse_int(p, end, val);
         } else if constexpr (std::is_floating_point_v<ColType>) {
           // Parse float
           char buf[64];
           size_t i = 0;
           while (p < end && i < 63 &&
                  (*p == '.' || *p == '-' || *p == '+' || *p == 'e' || *p == 'E' ||
                   (*p >= '0' && *p <= '9'))) {
             buf[i++] = *p++;
           }
           buf[i] = '\0';
// From chars or strtod
#if __cpp_lib_to_chars >= 201611L
           std::from_chars(buf, buf + i, val);
#else
           val = static_cast<ColType>(std::strtod(buf, nullptr));
#endif
         } else {
           // String
           // Find next delimiter position
           const char* sep;
           if (delim_info.is_multi_space) {
             // Scan for min_spaces
             sep = p;
             while (sep < end && *sep != '\n' && *sep != '\r') {
               if (*sep == ' ') {
                 size_t cnt = 0;
                 const char* s = sep;
                 while (s < end && *s == ' ') {
                   cnt++;
                   s++;
                 }
                 if (cnt >= delim_info.min_spaces)
                   break;
               }
               sep++;
             }
           } else {
             sep = p;
             while (sep < end && *sep != delim_info.single_char && *sep != '\n' && *sep != '\r')
               sep++;
           }

           // Extract string
           // Trim trailing whitespace? Original did trim.
           const char* val_end = sep;
           while (val_end > p && (*(val_end - 1) == ' ' || *(val_end - 1) == '\t'))
             val_end--;
           val = std::string(p, val_end - p);

           p = sep;
         }

         // Advance past delimiter
         if (delim_info.is_multi_space) {
           while (p < end && *p == ' ')
             p++;
         } else {
           if (p < end && *p == delim_info.single_char)
             p++;
         }

         // Push
         rel.template column<Is>().push_back(val);
         // Encode using proper value_type from relation
         using ValType = typename Relation::value_type;
         rel.template interned_column<Is>().push_back(static_cast<ValType>(encode_to_size_t(val)));
       }()),
       ...);
    }(std::make_index_sequence<Arity>{});

    if (!row_error) {
      if constexpr (has_provenance_v<typename Relation::semiring_type>) {
        rel.provenance().push_back(Relation::semiring_type::one());
      }
    }
    // SR::one() might not be boolean true.
    // Relation stores 'Annotation' which is SR::value_type.
    // If SR=BooleanSR, value_type is bool.
    // If SR=CountingSR, value_type is int.
    // I should use SR::one().
    // But fast_load_file_impl doesn't have SR template param!
    // It has Schema.
    // Schema::semiring_type::one().

    // Check rest of line
    while (p < end && *p != '\n')
      p++;
    if (p < end)
      p++;  // Skip \n
  }

  munmap(mapped, file_size);
  close(fd);
}

}  // namespace detail

template <CRelationSchema Schema, typename DB>
void load_from_file(DB& runtime_db, const std::string& file_path) {
  using AttrTuple = typename Schema::attr_ts_type;

  // Build all indexes later, but we need relation now
  auto& rel = get_relation_by_schema<Schema, FULL_VER>(runtime_db);

  // Load all data first without building indexes
  detail::fast_load_file_impl<AttrTuple>(rel, file_path);

  // Build all indexes after loading is complete
  rel.build_all_indexes();
}

/**
 * @brief Load data from a file directly into a relation.
 *
 * @details This function loads data from a file (CSV, TSV, or space-separated)
 *          directly into a relation object. It automatically detects the delimiter
 *          and parses the file accordingly. Each line in the file becomes a row
 *          in the relation with the semiring's one() value as the annotation.
 *          After loading, it builds all registered indexes.
 *
 * @tparam SR The semiring type used for annotations
 * @tparam AttrTuple The tuple type containing the attribute types
 * @tparam IndexType The index type to use (default: HashTrieIndex)
 * @param relation The relation to load data into
 * @param file_path The path to the file to load
 *
 * @throw std::runtime_error if the file cannot be opened or has invalid format
 *
 * @example
 * ```cpp
 * using namespace SRDatalog;
 * using SR = BooleanSR;
 * Relation<SR, std::tuple<int, int>> rel;
 * load_file(rel, "data.csv");
 * ```
 */
template <Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType = HashTrieIndex,
          typename Policy = HostRelationPolicy>
void load_file(Relation<SR, AttrTuple, IndexType, Policy>& relation, const std::string& file_path) {
  // Load all data first without building indexes
  detail::fast_load_file_impl<AttrTuple>(relation, file_path);

  // Build all registered indexes after loading is complete
  relation.build_all_indexes();
}

/**
 * @brief Load data from a file directly into a relation and build specific indexes.
 *
 * @details This function loads data from a file (CSV, TSV, or space-separated)
 *          directly into a relation object. It automatically detects the delimiter
 *          and parses the file accordingly. Each line in the file becomes a row
 *          in the relation with the semiring's one() value as the annotation.
 *          After loading, it builds only the specified indexes.
 *
 * @tparam SR The semiring type used for annotations
 * @tparam AttrTuple The tuple type containing the attribute types
 * @tparam IndexType The index type to use (default: HashTrieIndex)
 * @param relation The relation to load data into
 * @param file_path The path to the file to load
 * @param index_specs Vector of IndexSpec objects specifying which indexes to build
 *
 * @throw std::runtime_error if the file cannot be opened or has invalid format
 *
 * @example
 * ```cpp
 * using namespace SRDatalog;
 * using SR = BooleanSR;
 * Relation<SR, std::tuple<int, int>> rel;
 * std::vector<IndexSpec> specs = {{{0, 1}}, {{1, 0}}};
 * load_file(rel, "data.csv", specs);
 * ```
 */
template <Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType = HashTrieIndex,
          typename Policy = HostRelationPolicy>
void load_file(Relation<SR, AttrTuple, IndexType, Policy>& relation, const std::string& file_path,
               const std::vector<IndexSpec>& index_specs) {
  // Load all data first without building indexes
  detail::fast_load_file_impl<AttrTuple>(relation, file_path);

  // Build only the specified indexes after loading is complete
  for (const auto& spec : index_specs) {
    relation.ensure_index(spec);
  }
}

/**
 * @brief Load data from a file directly into a relation and build specific indexes.
 *
 * @details This function loads data from a file (CSV, TSV, or space-separated)
 *          directly into a relation object. It automatically detects the delimiter
 *          and parses the file accordingly. Each line in the file becomes a row
 *          in the relation with the semiring's one() value as the annotation.
 *          After loading, it builds only the specified indexes.
 *
 * @tparam SR The semiring type used for annotations
 * @tparam AttrTuple The tuple type containing the attribute types
 * @tparam IndexType The index type to use (default: HashTrieIndex)
 * @param relation The relation to load data into
 * @param file_path The path to the file to load
 * @param index_specs Initializer list of IndexSpec objects specifying which indexes to build
 *
 * @throw std::runtime_error if the file cannot be opened or has invalid format
 *
 * @example
 * ```cpp
 * using namespace SRDatalog;
 * using SR = BooleanSR;
 * Relation<SR, std::tuple<int, int>> rel;
 * load_file(rel, "data.csv", {{{0, 1}}, {{1, 0}}});
 * ```
 */
template <Semiring SR, ColumnElementTuple AttrTuple,
          template <Semiring, ColumnElementTuple, typename...> class IndexType = HashTrieIndex,
          typename Policy = HostRelationPolicy>
void load_file(Relation<SR, AttrTuple, IndexType, Policy>& relation, const std::string& file_path,
               std::initializer_list<IndexSpec> index_specs) {
  // Load all data first without building indexes
  detail::fast_load_file_impl<AttrTuple>(relation, file_path);

  // Build only the specified indexes after loading is complete
  for (const auto& spec : index_specs) {
    relation.ensure_index(spec);
  }
}

}  // namespace SRDatalog
