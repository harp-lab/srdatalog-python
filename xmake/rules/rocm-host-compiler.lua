-- ROCm host compiler rule
-- Ensures host libraries use system clang, not ROCm's clang
-- This rule runs on configuration to check and warn if ROCm clang is being used

rule("rocm.host.compiler")
    on_config(function (target)
        if not has_config("rocm") then
            return
        end
        
        import("lib.detect.find_tool")
        
        -- Check what compiler is being used
        local cc = target:get("cc") or get_config("cc")
        local cxx = target:get("cxx") or get_config("cxx")
        
        -- If compiler paths contain /rocm, warn user
        if cc and cc:match("/rocm") then
            print("Warning: ROCm clang detected for host compilation: " .. cc)
            print("  This may cause build failures for host libraries.")
            print("  Recommended: xmake config --rocm=y --cc=/usr/bin/clang --cxx=/usr/bin/clang++")
        elseif cxx and cxx:match("/rocm") then
            print("Warning: ROCm clang++ detected for host compilation: " .. cxx)
            print("  This may cause build failures for host libraries.")
            print("  Recommended: xmake config --rocm=y --cc=/usr/bin/clang --cxx=/usr/bin/clang++")
        end
    end)
rule_end()


