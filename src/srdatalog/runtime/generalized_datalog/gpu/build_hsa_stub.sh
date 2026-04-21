#!/bin/bash
# Build script for HSA stub library to work around ROCm 7.1.1 library version mismatch
# This provides the missing hsa_amd_memory_get_preferred_copy_engine symbol

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
STUB_SRC="$SCRIPT_DIR/hsa_stub.c"
STUB_SO="$PROJECT_DIR/build/libhsa_stub.so"

# Find ROCm include path
ROCM_INCLUDE="/opt/rocm-7.1.1/include"
if [ ! -d "$ROCM_INCLUDE" ]; then
    ROCM_INCLUDE="/opt/rocm/include"
fi

# Find ROCm lib path
ROCM_LIB="/opt/rocm-7.1.1/lib"
if [ ! -d "$ROCM_LIB" ]; then
    ROCM_LIB="/opt/rocm/lib"
fi

# Build the stub library
if [ -f "$STUB_SRC" ]; then
    mkdir -p "$(dirname "$STUB_SO")"
    gcc -shared -fPIC -o "$STUB_SO" "$STUB_SRC" \
        -I"$ROCM_INCLUDE" \
        -L"$ROCM_LIB" \
        -lhsa-runtime64 \
        -Wl,-rpath,"$ROCM_LIB" 2>&1
    if [ $? -eq 0 ] && [ -f "$STUB_SO" ]; then
        echo "Built HSA stub library: $STUB_SO"
        exit 0
    else
        echo "Failed to build HSA stub library" >&2
        exit 1
    fi
else
    echo "HSA stub source not found: $STUB_SRC" >&2
    exit 1
fi

