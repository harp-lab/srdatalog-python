/**
 * @file runtime.h
 * @brief Runtime execution engine for MIR (Mid-level Intermediate
 * Representation).
 *
 * @details This file is now a compatibility header that includes the modular
 * runtime components from the runtime/ directory. The runtime engine has been
 * split into separate files for better organization:
 *
 * - runtime/state.h - State management (variable bindings)
 * - runtime/iterator.h - Iterator generation from ColumnSource operations
 * - runtime/executor.h - MIR executor (core template metaprogramming interpreter)
 * - runtime/query.h - Query execution (fixpoint, semi-naive evaluation)
 * - runtime/io.h - I/O utilities (file loading)
 *
 * @note For new code, consider including runtime/runtime.h directly or
 *       individual components as needed. This header is maintained for
 *       backward compatibility.
 *
 * @section design_overview Design Overview
 *
 * SRDatalog is a Datalog engine built with C++ template metaprogramming that
 * provides compile-time query optimization and efficient runtime execution.
 * The system follows a multi-stage compilation and execution pipeline:
 *
 * @subsection compilation_pipeline Compilation Pipeline
 *
 * 1. **AST Definition**: Rules are defined using high-level AST types
 *    (Rule, Clause, Var, etc.) that represent Datalog syntax
 *
 * 2. **MIR Compilation**: AST rules are compiled to MIR (Mid-level IR)
 *    operations using CompileRuleToMIR, which generates:
 *    - ColumnSource operations for relation access
 *    - ColumnJoin operations for multi-way joins
 *    - DestinationRelation operations for fact insertion
 *
 * 3. **Index Planning**: Required indexes are computed at compile-time
 *    based on join plans and clause access patterns
 *
 * @subsection execution_model Execution Model
 *
 * The execution uses semi-naive evaluation for recursive rules:
 *
 * - **Full Relations**: Complete set of derived facts
 * - **Delta Relations**: Facts derived in the previous iteration
 * - **New Relations**: Facts derived in the current iteration
 *
 * Each iteration:
 * 1. Executes rules using delta relations (semi-naive optimization)
 * 2. Stores new facts in new relations
 * 3. Merges new into full and swaps new/delta for next iteration
 * 4. Stops when no new facts are derived (fixpoint reached)
 *
 * @subsection join_algorithm Join Algorithm
 *
 * The system implements Worst-Case Optimal Joins (WCOJ) using hash
 * intersection:
 *
 * - Each relation column is indexed using HashTrieIndex
 * - Joins are performed by intersecting hash sets at each level
 * - This provides optimal worst-case complexity for multi-way joins
 *
 * @section memory_management Memory Management
 *
 * The system uses polymorphic memory resources (PMR) for efficient memory
 * allocation:
 *
 * - Relations can be allocated on custom memory resources
 * - This enables arena allocation, memory pooling, and custom allocators
 * - Indexes share memory resources with their parent relations
 *
 * @section performance_considerations Performance Considerations
 *
 * - **Compile-time Optimization**: All query planning happens at compile-time
 * - **Lazy Index Building**: Indexes are built on-demand when accessed
 * - **Semi-naive Evaluation**: Only processes new facts in each iteration
 * - **WCOJ Algorithm**: Optimal worst-case join complexity
 * - **Column-oriented Storage**: Better cache locality for analytical workloads
 */

#pragma once

// Backward compatibility: include the modular runtime components
#include "runtime/runtime.h"
