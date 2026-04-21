-- IPS²Ra package definition for XMake
-- IPS²Ra is an in-place parallel super scalar radix sort algorithm
-- Repository: https://github.com/ips4o/ips2ra.git

package("ips2ra")
    set_homepage("https://github.com/ips4o/ips2ra")
    set_description("In-place Parallel Super Scalar Radix Sort (IPS²Ra) - Header-only library")
    set_license("BSD-2-Clause")
    
    add_urls("https://github.com/ips4o/ips2ra.git", {submodules = true})
    add_versions("master", "master")
    
    -- IPS²Ra is header-only, just needs TBB (which the target already has configured)
    -- No need to build - just clone and set include directory
    
    -- Ensure package metadata is set when package is loaded (after installation)
    on_load(function (package)
        local install_dir = package:installdir()
        if install_dir and os.isdir(install_dir) then
            local include_dir = path.join(install_dir, "include")
            
            -- Set include directory (headers are in include/ips2ra.hpp)
            if os.isdir(include_dir) then
                package:set("includedirs", {include_dir})
            end
        end
        
    end)
    
    on_install(function (package)
        -- IPS²Ra is header-only - manually copy headers and create package structure
        -- Since target already has TBB, we don't need CMake build
        
        -- Use cachedir for source (xmake will handle cleanup)
        local source_dir = path.join(package:cachedir(), "ips2ra-source")
        
        -- Clone the repository if it doesn't exist
        if not os.isdir(source_dir) or not os.isdir(path.join(source_dir, ".git")) then
            print("Cloning IPS²Ra repository...")
            os.mkdir(path.directory(source_dir))
            local repo_url = "https://github.com/ips4o/ips2ra.git"
            local branch = package:version() or "master"
            os.vrunv("git", {"clone", "--depth", "1", "--branch", branch,
                            "--recurse-submodules", repo_url, source_dir})
        end
        
        -- Copy include directory to install directory
        local source_include = path.join(source_dir, "include")
        local install_include = path.join(package:installdir(), "include")
        
        if os.isdir(source_include) then
            os.mkdir(install_include)
            os.cp(source_include .. "/*", install_include)
            print("IPS²Ra: Copied include directory to " .. install_include)
        else
            raise("IPS²Ra include directory not found in source: " .. source_include)
        end
        
        -- Set include directory only - don't set links for header-only packages
        package:set("includedirs", {install_include})
    end)
    
    on_test(function (package)
        -- Test is optional - check if header exists
        local include_dir = path.join(package:installdir(), "include")
        if os.isfile(path.join(include_dir, "ips2ra.hpp")) then
            assert(package:has_cxxincludes("ips2ra.hpp"))
        end
    end)
package_end()

