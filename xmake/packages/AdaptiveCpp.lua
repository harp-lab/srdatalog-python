-- AdaptiveCpp package definition for XMake
-- Equivalent to CMake's:
--   set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${ACPP_SOURCE_ROOT}/cmake)
--   find_package(AdaptiveCpp CONFIG REQUIRED)
--
-- This package definition locates AdaptiveCpp (formerly hipSYCL) installation
-- and extracts include directories, libraries, and compile flags from the CMake config.
-- If not found and clang toolchain is used, it will automatically clone and build from source.

package("AdaptiveCpp")
    set_homepage("https://github.com/AdaptiveCpp/AdaptiveCpp")
    set_description("AdaptiveCpp (formerly hipSYCL) - implementation of SYCL for CPUs and GPUs")
    
    add_urls("https://github.com/AdaptiveCpp/AdaptiveCpp.git")
    add_versions("develop", "develop")
    
    -- Check if clang is available for building
    local function can_build_from_source()
        import("lib.detect.find_tool")
        
        -- Check if clang++ is available
        local clang = find_tool("clang++")
        if not clang then
            return false
        end
        
        -- Also check for cmake and ninja
        local cmake = find_tool("cmake")
        local ninja = find_tool("ninja")
        
        return cmake ~= nil and ninja ~= nil
    end
    
    -- Use find_package to locate the CMake config and extract information
    on_fetch(function (package)
        import("lib.detect.find_package")
        import("lib.detect.find_tool")
        import("core.base.option")
        
        -- Standard installation prefixes to check
        local prefixes = {"/usr/local", "/opt/homebrew", "/usr", "/opt/sycl"}
        local acpp_source_root = os.getenv("ACPP_SOURCE_ROOT")
        
        -- If ACPP_SOURCE_ROOT is set, use it (for source builds)
        if acpp_source_root and os.isdir(acpp_source_root) then
            table.insert(prefixes, 1, acpp_source_root)  -- Check this first
        end
        
        -- Try to find AdaptiveCpp in standard locations
        for _, prefix in ipairs(prefixes) do
            local cmake_dir = path.join(prefix, "lib", "cmake", "AdaptiveCpp")
            local config_file = path.join(cmake_dir, "adaptivecpp-config.cmake")
            
            -- Check if CMake config exists
            if os.isfile(config_file) then
                local include_dir = path.join(prefix, "include")
                local include_acpp_dir = path.join(include_dir, "AdaptiveCpp")
                local lib_dir = path.join(prefix, "lib")
                
                -- Verify directories exist
                if os.isdir(include_dir) then
                    -- Return package info manually extracted from installation
                    local result = {
                        includedirs = {include_dir, include_acpp_dir},
                        linkdirs = {lib_dir},
                        -- AdaptiveCpp uses imported targets, so we link against the common library
                        links = {"acpp-common"},
                        -- Also add the cmake directory for any CMake modules that might be needed
                        cmake_dir = cmake_dir
                    }
                    
                    -- Verify at least one include directory exists
                    if os.isdir(include_dir) or os.isdir(include_acpp_dir) then
                        print("AdaptiveCpp found at: " .. prefix)
                        return result
                    end
                end
            end
        end
        
        -- Try using find_package as fallback
        local search_paths = {}
        for _, prefix in ipairs(prefixes) do
            table.insert(search_paths, prefix)
            table.insert(search_paths, path.join(prefix, "lib", "cmake"))
        end
        
        local result = find_package("AdaptiveCpp", {
            paths = search_paths,
            check = function(pkg)
                return pkg and (pkg.includedirs or pkg.libfiles)
            end
        })
        
        if result then
            return result
        end
        
        -- If not found and clang/cmake/ninja are available, return nil to trigger on_install
        if can_build_from_source() then
            print("AdaptiveCpp not found in system. Will build from source (clang++ detected).")
            return nil  -- This will trigger on_install
        end
        
        -- If not found and cannot build, raise an error
        local error_msg = "AdaptiveCpp not found!"
        error_msg = error_msg .. "\n  Searched in: " .. table.concat(prefixes, ", ")
        if acpp_source_root then
            error_msg = error_msg .. "\n  ACPP_SOURCE_ROOT is set to: " .. acpp_source_root
        end
        error_msg = error_msg .. "\n  Automatic build from source requires: clang++, cmake, and ninja."
        error_msg = error_msg .. "\n  Please install AdaptiveCpp manually or ensure clang++/cmake/ninja are available."
        raise(error_msg)
    end)
    
    -- Build AdaptiveCpp from source when using clang toolchain
    on_install(function (package)
        import("lib.detect.find_tool")
        import("core.tool.compiler")
        
        -- Check if clang is available
        local clang = find_tool("clang")
        if not clang then
            raise("clang not found! AdaptiveCpp requires clang to build.")
        end
        
        -- Get source directory
        local sourcedir = package:sourcehdir()
        if not os.isdir(sourcedir) then
            -- Clone the repository
            print("Cloning AdaptiveCpp repository...")
            os.vrunv("git", {"clone", "--depth", "1", "--branch", "develop", 
                            "https://github.com/AdaptiveCpp/AdaptiveCpp.git", sourcedir})
        end
        
        -- Create build directory
        local builddir = path.join(package:installdir(), "build")
        os.mkdir(builddir)
        
        -- Configure with CMake
        print("Configuring AdaptiveCpp with CMake...")
        local configs = {
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_INSTALL_PREFIX=" .. package:installdir(),
            "-DCMAKE_CXX_COMPILER=" .. (clang.program or "clang++"),
            "-DACPP_COMPILER_FEATURE_PROFILE=full",
            "-DACPP_ENABLE_OPENCL_BACKEND=OFF",  -- Disable OpenCL for faster build
        }
        
        -- Add CUDA support if available
        local nvcc = find_tool("nvcc")
        if nvcc then
            table.insert(configs, "-DACPP_ENABLE_CUDA_BACKEND=ON")
        end
        
        local cmake_args = {"-S", sourcedir, "-B", builddir, "-GNinja"}
        for _, config in ipairs(configs) do
            table.insert(cmake_args, config)
        end
        os.vrunv("cmake", cmake_args)
        
        -- Build and install
        print("Building AdaptiveCpp (this may take a while)...")
        os.vrunv("cmake", {"--build", builddir, "--target", "install"})
        
        -- Return package info
        local include_dir = path.join(package:installdir(), "include")
        local include_acpp_dir = path.join(include_dir, "AdaptiveCpp")
        local lib_dir = path.join(package:installdir(), "lib")
        
        package:addenv("PATH", path.join(package:installdir(), "bin"))
        
        package:set("includedirs", {include_dir, include_acpp_dir})
        package:set("linkdirs", {lib_dir})
        package:set("links", {"acpp-common"})
    end)
package_end()

