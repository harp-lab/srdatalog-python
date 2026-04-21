/**
 * @file mir.h
 * @brief Mid-level Intermediate Representation (MIR) definitions and compilation.
 *
 * @details This file now serves as an aggregate header for the modularized MIR system:
 * - mir_def.h: Core MIR AST structures and types
 * - mir_plan.h: Fixpoint execution plan definitions
 * - mir_gpu.h: GPU optimization passes
 * - mir_compiler.h: AST to MIR compilation logic
 *
 * MIR is generated from the high-level AST during compilation and is closer
 * to the actual execution model, enabling column-oriented join operations.
 */

#pragma once

#include "mir_compiler.h"
#include "mir_def.h"
#include "mir_gpu.h"
#include "mir_plan.h"