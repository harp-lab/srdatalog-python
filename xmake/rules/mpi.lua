-- xmake/rules/mpi.lua
-- MPI rule: automatically configures MPI compiler/linker flags for a target
-- This rule detects MPI installation and configures compile/link flags
-- similar to CMake's FindMPI module
-- Usage: add_rules("mpi") in your target definition

rule("mpi")
    on_load(function (target)
        local find_tool = nil
        if type(import) == "function" then
            find_tool = import("lib.detect.find_tool")
        end
        
        local os_module = os
        if (not os_module or type(os_module.iorunv) ~= "function") and type(import) == "function" then
            local core_os = import("core.base.os")
            if core_os then
                os_module = core_os
            end
        end
        
        if not os_module or type(os_module.iorunv) ~= "function" then
            target:add("syslinks", "mpi", "mpi_cxx", {optional = true})
            return
        end
        
        local function apply_compile_flags(flag_str)
            if not flag_str or flag_str == "" then
                return false
            end
            local applied = false
            for flag in flag_str:gmatch("[^%s]+") do
                if flag:match("^-I") then
                    target:add("includedirs", flag:sub(3))
                    applied = true
                elseif flag:match("^-L") then
                    target:add("linkdirs", flag:sub(3))
                    applied = true
                elseif flag:match("^-l") then
                    target:add("syslinks", flag:sub(3))
                    applied = true
                elseif flag:match("^%-") then
                    target:add("cxflags", flag)
                    applied = true
                end
            end
            return applied
        end
        
        local function apply_link_flags(flag_str)
            if not flag_str or flag_str == "" then
                return false
            end
            local applied = false
            for flag in flag_str:gmatch("[^%s]+") do
                if flag:match("^-L") then
                    target:add("linkdirs", flag:sub(3))
                    applied = true
                elseif flag:match("^-l") then
                    target:add("syslinks", flag:sub(3))
                    applied = true
                elseif flag:match("^%-") then
                    target:add("ldflags", flag)
                    applied = true
                end
            end
            return applied
        end
        
        if not find_tool then
            target:add("syslinks", "mpi", "mpi_cxx", {optional = true})
            return
        end
        
        local mpicxx = find_tool("mpicxx") or find_tool("mpic++")
        if not mpicxx then
            target:add("syslinks", "mpi", "mpi_cxx", {optional = true})
            return
        end
        
        local compile_flags = os_module.iorunv(mpicxx.program, {"-showme:compile"})
        local link_flags = os_module.iorunv(mpicxx.program, {"-showme:link"})
        local compile_applied = apply_compile_flags(compile_flags)
        local link_applied = apply_link_flags(link_flags)
        
        -- Also ensure MPI include paths are added with force flag (for toolchains that might override)
        if compile_flags and compile_flags ~= "" then
            for flag in compile_flags:gmatch("[^%s]+") do
                if flag:match("^-I") then
                    target:add("includedirs", flag:sub(3), {force = true})
                end
            end
        end
        
        if compile_applied or link_applied then
            return
        end
        
        -- Fallback: try to link with standard MPI libraries
        target:add("syslinks", "mpi", "mpi_cxx", {optional = true})
    end)
rule_end()

