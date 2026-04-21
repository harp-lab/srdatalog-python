# ROCm Clang Wrapper Scripts

## Overview

These wrapper scripts automatically add libc++ flags to ROCm clang compiler invocations, enabling C++23 support (including `<format>` header).

## Files

- `clang++-rocm-libcxx.sh` - Wrapper for C++ compiler (clang++)
- `clang-rocm-libcxx.sh` - Wrapper for C compiler (clang) - forwards to clang++ wrapper

## How It Works

1. **Detects libc++ installation**: Automatically finds libc++ headers in common locations:
   - `/usr/lib/llvm-14/include/c++/v1`
   - `/usr/lib/llvm-15/include/c++/v1`
   - `/usr/lib/llvm-16/include/c++/v1`
   - `/usr/include/c++/v1`

2. **Adds libc++ flags automatically**:
   - `-I<libcxx_include_path>` - Include path for libc++ headers
   - `-stdlib=libc++` - Use libc++ instead of libstdc++
   - `-lc++ -lc++abi` - Link against libc++ libraries
   - `-L/usr/lib/x86_64-linux-gnu` - Library search path

3. **Smart detection**: Only adds flags when:
   - Compiling C++ code (detects by file extension or `-x c++` flag)
   - Not already specified (checks for existing `-stdlib` flag)
   - libc++ is found on the system

## Usage

### Automatic (via xmake)

When `--rocm=y` is set, xmake automatically uses these wrappers:

```bash
xmake config --rocm=y --toolchain=clang
xmake
```

### Manual

You can use the wrapper directly:

```bash
# Compile a file
/root/SRDatalog/xmake/wrappers/clang++-rocm-libcxx.sh -std=c++23 -c file.cpp

# Or create a symlink in your PATH
ln -s /root/SRDatalog/xmake/wrappers/clang++-rocm-libcxx.sh /usr/local/bin/clang++-rocm
export CC=clang++-rocm
export CXX=clang++-rocm
```

### System-wide Installation (Optional)

To make the wrapper available system-wide:

```bash
# Create symlinks
sudo ln -s /root/SRDatalog/xmake/wrappers/clang++-rocm-libcxx.sh /usr/local/bin/clang++-rocm
sudo ln -s /root/SRDatalog/xmake/wrappers/clang-rocm-libcxx.sh /usr/local/bin/clang-rocm

# Or replace system clang (more aggressive)
# WARNING: This affects all users and may break other builds
sudo mv /usr/bin/clang++ /usr/bin/clang++.orig  # Backup
sudo ln -s /root/SRDatalog/xmake/wrappers/clang++-rocm-libcxx.sh /usr/bin/clang++
```

## Testing

Test if the wrapper works:

```bash
# Test format header
echo '#include <format>' | \
  /root/SRDatalog/xmake/wrappers/clang++-rocm-libcxx.sh \
  -x c++ -std=c++23 - -fsyntax-only

# Should succeed without errors
```

## Troubleshooting

### Wrapper not being used

If packages still fail, check:
1. Is the wrapper executable? `chmod +x clang++-rocm-libcxx.sh`
2. Is xmake using it? Check `xmake show` output
3. Are CMake packages detecting it? Check CMake cache files

### libc++ not found

Install libc++:
```bash
apt-get install libc++-dev libc++abi-dev
```

### Still getting format errors

The wrapper might not be detecting C++ compilation correctly. Check:
- Is `-x c++` flag present?
- Is the file extension `.cpp`, `.cxx`, `.cc`, or `.C`?
- Try adding `-v` flag to see what compiler is actually being called

## How xmake Uses It

When `--rocm=y` is set, xmake.lua:
1. Detects the wrapper scripts
2. Sets `set_config("cc", clang_wrapper)` and `set_config("cxx", clangxx_wrapper)`
3. All compiler invocations (including package builds) use the wrapper
4. The wrapper automatically adds libc++ flags

This ensures that:
- Your project code gets libc++ support
- Package builds (spdlog, benchmark, etc.) get libc++ support
- No manual flag specification needed


