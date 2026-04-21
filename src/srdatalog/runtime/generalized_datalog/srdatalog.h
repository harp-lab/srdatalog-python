/**
 * @file srdatalog.h
 * @brief Main include header for SRDatalog C++ library.
 *
 * @details This header includes all public API headers needed to use SRDatalog.
 * Simply include this header to get access to the entire SRDatalog library:
 *
 * @code{.cpp}
 * #include "srdatalog.h"
 * // Now you have access to all SRDatalog functionality
 * @endcode
 *
 * This header includes:
 * - Semiring concepts and implementations
 * - Template metaprogramming utilities
 * - System abstractions (memory management)
 * - Column and relation data structures
 * - AST definitions for Datalog programs
 * - Query compilation utilities
 * - Runtime execution engine
 * - Helper utilities
 *
 * @note This header is designed to be a convenience include. Individual headers
 *       can still be included separately if needed for fine-grained control.
 */

#pragma once
// for mimalloc new/delete, please make sure your target link mimalloc library.
// #ifdef DISABLE_MIMALLOC
// #else
// #include <mimalloc-new-delete.h>
// #endif

// Core concepts and utilities (no dependencies on other project headers)
#include "func_type_helper.h"
#include "helper.h"
#include "semiring.h"
#include "system.h"
#include "tmp.h"

// Data structures (depend on core utilities)
#include "column.h"
#include "relation_col.h"

// AST and compilation (depend on data structures)
#include "ast.h"
#include "mir_dsl.h"
#include "query.h"

// Runtime execution (depends on AST and compilation)
#include "runtime.h"

// Optional: MPI support (uncomment if needed)
// #include "mpi_types.h"
// #include "dist_relation_io.h"
