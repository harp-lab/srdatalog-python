#pragma once

/**
 * @file runtime.h
 * @brief Complete GPU runtime - includes all components for easy usage
 *
 * @details This is the main include file that aggregates all GPU runtime components.
 * Include this file to get access to the complete GPU pipeline execution system.
 *
 * Components included:
 * - store.h: VarStore, HandleStore, ProvStore (state management)
 * - output_context.h: OutputContext (two-phase output)
 * - deduplication.h: FilterAgainstFull (deduplication logic)
 * - tmp_helpers.h: Template metaprogramming helpers
 * - pipeline_executor.h: GPUPipelineExecutor (inner executor)
 * - root_executor.h: GPURootExecutor (root-level executor)
 * - launcher.h: GPUPipelineLauncher (host-side launcher)
 */

// Host-side launcher
#include "./launcher.h"
//
#include "./deduplication.h"
#include "./output_context.h"
#include "./pipeline_executor.h"
#include "./root_executor.h"
#include "./store.h"
#include "./tmp_helpers.h"
