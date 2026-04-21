-- clang-20 toolchain for xmake
-- Uses system clang-20 with libc++ for C++23 support (including <format> header)

toolchain("clang-20")
    set_kind("clang")
    
    on_check(function (toolchain)
        -- Check if clang-20 is available
        return os.isfile("/usr/bin/clang++-20")
    end)
    
    on_load(function (toolchain)
        local clang20 = "/usr/bin/clang-20"
        local clangxx20 = "/usr/bin/clang++-20"
        
        if not os.isfile(clangxx20) then
            raise("clang-20 not found! Please install: apt-get install clang-20 libc++-20-dev")
        end
        
        -- Use wrapper script for C++ compiler to ensure libc++ is always used
        -- This is critical for package checking which doesn't use toolchain flags
        local wrapper_dir = path.join(os.projectdir(), "xmake", "wrappers")
        local clangxx_wrapper = path.join(wrapper_dir, "clang++-20-libcxx.sh")
        
        -- Always use wrapper if it exists (it should exist - we create it)
        if os.isfile(clangxx_wrapper) then
            -- Use wrapper that always adds -stdlib=libc++
            toolchain:set("toolset", "cc", clang20)
            toolchain:set("toolset", "cxx", clangxx_wrapper)
            toolchain:set("toolset", "ld", clangxx_wrapper)
            toolchain:set("toolset", "sh", clangxx_wrapper)
            print("clang-20 toolchain: Using wrapper script for libc++ support")
            print("  C++ compiler: " .. clangxx_wrapper)
        else
            -- Fallback: use clang-20 directly (package checking may fail without -stdlib=libc++)
            toolchain:set("toolset", "cc", clang20)
            toolchain:set("toolset", "cxx", clangxx20)
            toolchain:set("toolset", "ld", clangxx20)
            toolchain:set("toolset", "sh", clangxx20)
            print("clang-20 toolchain: WARNING - wrapper script not found at " .. clangxx_wrapper)
            print("  Falling back to direct clang-20 (package checking may fail)")
        end
        
        -- Add libc++ flags automatically (clang-20 uses libc++ by default with -stdlib=libc++)
        -- This ensures C++23 <format> header is available
        toolchain:add("cxflags", "-stdlib=libc++")
        toolchain:add("ldflags", "-lc++")
        toolchain:add("ldflags", "-lc++abi")
        
        -- Use lld linker
        toolchain:add("ldflags", "-fuse-ld=lld")
        toolchain:add("shflags", "-fuse-ld=lld")
        
        -- Add libc++ library path if available
        local libcxx_lib = "/usr/lib/llvm-20/lib"
        if os.isdir(libcxx_lib) then
            toolchain:add("ldflags", "-L" .. libcxx_lib)
        end
        
        print("clang-20 toolchain: Using system clang-20 with libc++ (C++23 <format> support)")
        print("  C compiler: " .. clang20)
        print("  C++ compiler: " .. clangxx20)
    end)
toolchain_end()
