-- GPU test rule: configures GPU test targets (CUDA or HIP)
-- Usage: add_rules("gpu.test") (requires --nvidia=y or --rocm=y)

rule("gpu.test")
    on_load(function (target)
        if has_config("nvidia") then
            target:set("toolchains", "clang-cuda")
            target:add("rules", "clang-cuda.target")
        elseif has_config("rocm") then
            target:set("toolchains", "clang-hip")
            target:add("rules", "clang-hip.target")
        else
            target:set("default", false)
            return
        end
    end)
rule_end()

