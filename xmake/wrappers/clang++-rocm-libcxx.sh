#!/bin/bash
# Wrapper script for ROCm clang++ to automatically use libc++
# This script intercepts clang++ calls and adds libc++ flags

# Find libc++ headers
LIBCXX_INCLUDE=""
for path in "/usr/lib/llvm-14/include/c++/v1" \
            "/usr/lib/llvm-15/include/c++/v1" \
            "/usr/lib/llvm-16/include/c++/v1" \
            "/usr/include/c++/v1"; do
    if [ -d "$path" ] && [ -f "$path/format" ]; then
        LIBCXX_INCLUDE="$path"
        break
    fi
done

# Use ROCm clang++ as the base compiler
ROCM_CLANG="/opt/rocm/llvm/bin/clang++"
if [ ! -f "$ROCM_CLANG" ]; then
    # Fallback to any clang++ in PATH
    ROCM_CLANG="clang++"
fi

# Build arguments array
ARGS=()

# Check if -stdlib is already specified
HAS_STDLIB=false
for arg in "$@"; do
    if [[ "$arg" == *"-stdlib"* ]]; then
        HAS_STDLIB=true
        break
    fi
done

# Add libc++ flags if not already present and if we found libc++
if [ -n "$LIBCXX_INCLUDE" ] && [ "$HAS_STDLIB" = false ]; then
    # Check if this is a C++ compilation (not just linking)
    IS_COMPILE=false
    for arg in "$@"; do
        if [[ "$arg" == *.cpp ]] || [[ "$arg" == *.cxx ]] || [[ "$arg" == *.cc ]] || \
           [[ "$arg" == *.C ]] || [[ "$arg" == -x ]] || [[ "$arg" == -c ]]; then
            IS_COMPILE=true
            break
        fi
    done
    
    if [ "$IS_COMPILE" = true ] || [ "$#" -eq 0 ]; then
        ARGS+=("-I$LIBCXX_INCLUDE")
        ARGS+=("-stdlib=libc++")
    fi
    
    # Add linker flags if linking
    IS_LINK=false
    for arg in "$@"; do
        if [[ "$arg" != -c ]] && [[ "$arg" != -E ]] && [[ "$arg" != -S ]]; then
            IS_LINK=true
            break
        fi
    done
    
    if [ "$IS_LINK" = true ]; then
        ARGS+=("-lc++")
        ARGS+=("-lc++abi")
        ARGS+=("-L/usr/lib/x86_64-linux-gnu")
    fi
fi

# Execute the compiler with all arguments
exec "$ROCM_CLANG" "${ARGS[@]}" "$@"


