#pragma once

/**
 * @file pipeline_executor.h
 * @brief Inner pipeline executor - recursive TMP dispatcher for MIR operations
 */

#include "pipeline_executor_base.h"

// Instruction-specific specializations
#include "instructions/instruction_aggregate.h"
#include "instructions/instruction_cartesian_join.h"
#include "instructions/instruction_column_join.h"
#include "instructions/instruction_destination_relation.h"
#include "instructions/instruction_filter.h"
#include "instructions/instruction_if_clause.h"
#include "instructions/instruction_let_clause.h"
#include "instructions/instruction_negation.h"
#include "instructions/instruction_scan.h"
