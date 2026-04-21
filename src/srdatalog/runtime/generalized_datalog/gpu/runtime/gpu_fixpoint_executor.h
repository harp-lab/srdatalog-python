#pragma once

#include "gpu_fixpoint_executor_common.h"

// Include instruction implementation specializations
#include "executor_impl/executor_check_size.h"
#include "executor_impl/executor_compute_delta.h"
#include "executor_impl/executor_create_index.h"
#include "executor_impl/executor_execute_pipeline.h"
#include "executor_impl/executor_inject_cpp_hook.h"
#include "executor_impl/executor_insert_merge.h"
#include "executor_impl/executor_jit_execute_pipeline.h"
#include "executor_impl/executor_materialized_pipeline.h"
#include "executor_impl/executor_merge_index.h"
#include "executor_impl/executor_nested_plan.h"
#include "executor_impl/executor_post_stratum_reconstruct.h"
#include "executor_impl/executor_rebuild_index.h"
#include "executor_impl/executor_relation_ops.h"
