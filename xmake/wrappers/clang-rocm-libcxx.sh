#!/bin/bash
# Wrapper script for ROCm clang (C compiler) - forwards to clang++ wrapper for consistency
# This ensures both C and C++ compilers have the same libc++ configuration

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/clang++-rocm-libcxx.sh" "$@"


