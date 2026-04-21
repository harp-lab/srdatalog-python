/**
 * @file runtime/runtime.h
 * @brief Main runtime header - includes all runtime components.
 *
 * @details This header includes all runtime execution components:
 * - State management (variable bindings)
 * - Iterator generation (from ColumnSource operations)
 * - MIR executor (core template metaprogramming interpreter)
 * - Query execution (fixpoint, semi-naive evaluation)
 * - I/O utilities (file loading)
 *
 * @note This is the main entry point for runtime functionality. Individual
 *       components can be included separately if needed for fine-grained control.
 */

#pragma once

#include "query.h"

// Include all runtime components in dependency order
#include "runtime/executor.h"
#include "runtime/io.h"
#include "runtime/iterator.h"
#include "runtime/query.h"
#include "runtime/state.h"
