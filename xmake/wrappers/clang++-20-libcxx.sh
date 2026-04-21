#!/bin/bash
# Wrapper for clang++-20 that always uses libc++ for C++23 <format> support
# This ensures package checking uses libc++ even when toolchain flags aren't applied

exec /usr/bin/clang++-20 -stdlib=libc++ "$@"
