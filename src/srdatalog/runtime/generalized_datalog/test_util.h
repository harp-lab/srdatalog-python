/**
 * @file test_util.h
 * @brief Utility functions for tests and benchmarks
 */

#pragma once

#include <filesystem>

namespace SRDatalog::TestUtil {

/**
 * @brief Find the project root directory by searching for xmake.lua
 *
 * This function searches up the directory tree from the current working
 * directory to find the project root (identified by the presence of xmake.lua).
 *
 * @return The path to the project root directory
 */
inline std::filesystem::path find_project_root() {
  std::filesystem::path current = std::filesystem::current_path();

  // Try to find xmake.lua by going up the directory tree
  std::filesystem::path search_dir = current;
  for (int i = 0; i < 10; ++i) {
    if (std::filesystem::exists(search_dir / "xmake.lua")) {
      return search_dir;
    }
    if (search_dir == search_dir.root_path()) {
      break;
    }
    search_dir = search_dir.parent_path();
  }

  // Fallback: return current directory
  return current;
}

}  // namespace SRDatalog::TestUtil
