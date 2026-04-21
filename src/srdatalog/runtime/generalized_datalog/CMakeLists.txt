cmake_minimum_required(VERSION 3.24)

# ---- CPM Package Manager Setup ----
# CPM automatically downloads and manages C++ dependencies
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake)

project(DatalogColumnStore
  VERSION 0.1.0
  DESCRIPTION "Column-oriented Datalog core + semiring + relation + small tests"
  LANGUAGES CXX)

# ---- 让 clangd 工作：生成 compile_commands.json ----
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Warnings & 限制错误数量（锁定首错）----
if (MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-ferror-limit=1)
  elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fmax-errors=1)
  endif()
endif()

# set build type, default to RelWithDebInfo
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ---- Header search paths ----
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}   # for the shim headers we generate below
)

# ---- Header shims (compat for includes in your sources) ----
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/semiring.hpp "#pragma once\n#include \"semiring.h\"\n")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/mpi_type.h   "#pragma once\n#include \"mpi_types.h\"\n")

# ---- ccache (optional) ----
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
  message(STATUS "ccache found: ${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# ---- CPM: Boost 1.88 with container support ----
# Boost is used throughout the codebase for:
# - Boost.Hana: compile-time metaprogramming
# - Boost.TypeIndex: type introspection
# - Boost.Container: container libraries
# CPMAddPackage(
#   NAME Boost
#   VERSION 1.88.0
#   GITHUB_REPOSITORY boostorg/boost
#   GIT_TAG boost-1.88.0
#   OPTIONS
#     "BOOST_INCLUDE_LIBRARIES container"
#     "BOOST_BUILD_SHARED_LIBS ON"
#     "BOOST_ENABLE_CMAKE ON"
# )
find_package(Boost REQUIRED COMPONENTS container)
message(STATUS "Boost 1.88.0 with container support added via CPM")

# ---- CPM: OpenMPI ----
# MPI support for distributed computation
find_package(MPI REQUIRED)
message(STATUS "MPI found: ${MPI_VERSION}")

# ---- Header-only interface target ----
add_library(datalog_core INTERFACE)
target_include_directories(datalog_core INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
target_compile_features(datalog_core INTERFACE cxx_std_20)
# Link Boost to datalog_core so all targets inherit it
target_link_libraries(datalog_core INTERFACE Boost::container)

# ---- Build Options ----
option(BUILD_TEST_TMP "Build test_tmp executable" ON)
if(BUILD_TEST_TMP)
  message(STATUS "Building test_tmp executable")
endif()

# =========================
# Executables (tests/tools)
# =========================

# 1) Column tests
add_executable(test_column
  test/test_column.cpp
  column.h
  column.ipp
)
target_link_libraries(test_column PRIVATE datalog_core)

# 2) Semiring tests  ← 修正文件名
add_executable(test_semiring
  test/test_simiring.cpp   # <-- 如果你的文件实际叫这个
  semiring.h
)
target_link_libraries(test_semiring PRIVATE datalog_core)

# 3) TMP tests (optional, uses Boost Hana metaprogramming)
# Note: tmp.h is a header-only template library
if(BUILD_TEST_TMP)
  add_executable(test_tmp
    test/test_tmp.cpp
    ast.h
    query.h
    runtime.h
    tmp.h
  )
  target_link_libraries(test_tmp PRIVATE datalog_core Boost::container)
endif()

# 3) Relation tests
add_executable(test_relation
  test/test_relation.cpp
  relation_col.h
  relation_col.ipp
  column.h
  column.ipp
  semiring.h
)
target_link_libraries(test_relation PRIVATE datalog_core  Boost::container)

# 4) Distributed test
# add_executable(test_dist
#   test/test_dist.cpp
#   dist_relation_io.h
#   dist_relation_io.ipp
# )

# 4) MPI datatype tests
add_executable(test_mpi_type
  test/test_mpi_type.cpp
  mpi_types.h
)
target_link_libraries(test_mpi_type PRIVATE datalog_core MPI::MPI_CXX)

# 5) IO tests
add_executable(test_io
  test/test_io.cpp
  runtime.h
  ast.h
  semiring.h
)
target_link_libraries(test_io PRIVATE datalog_core)

# 6) TC tests (optional)
if(BUILD_TEST_TMP)
  add_executable(test_tc
    test/test_tc.cpp
    ast.h
    query.h
    runtime.h
    semiring.h
  )
  target_link_libraries(test_tc PRIVATE datalog_core Boost::container)

  add_executable(test_mir_aggregation
    test/test_mir_aggregation.cpp
    ast.h
    query.h
    runtime.h
    semiring.h
    runtime/aggregation.h
  )
  target_link_libraries(test_mir_aggregation PRIVATE datalog_core Boost::container)
endif()

# Build-all target
set(ALL_TARGETS test_column test_semiring test_relation test_mpi_type test_io test_hana_state)
if(BUILD_TEST_TMP)
  list(APPEND ALL_TARGETS test_tmp test_tc)
endif()
add_custom_target(tests ALL
  DEPENDS ${ALL_TARGETS})
