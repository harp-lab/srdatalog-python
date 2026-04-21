# docker buildx bake config for the srdatalog-python RunPod / wheel-test image.
#
# Build:
#   docker buildx bake                          # default target
#   docker buildx bake --push                   # build + push to registry
#   docker buildx bake --set default.tags+=foo  # extra tag
#
# No external contexts — everything the build needs lives under docker/ or
# the repo root, so this bake file is reproducible from a fresh git clone.

variable "RELEASE" {
    default = "0.1.0-cuda12.9-clang20"
}

variable "IMAGE_NAME" {
    default = "stargazermiao/srdatalog-python-runpod"
}

variable "CUDA_VERSION"   { default = "12.9.1" }
variable "UBUNTU_VERSION" { default = "24.04" }
variable "CLANG_VERSION"  { default = "20" }

target "default" {
    dockerfile = "docker/Dockerfile"
    context    = "."
    tags = [
        "${IMAGE_NAME}:${RELEASE}",
        "${IMAGE_NAME}:latest",
    ]
    args = {
        CUDA_VERSION   = "${CUDA_VERSION}"
        UBUNTU_VERSION = "${UBUNTU_VERSION}"
        CLANG_VERSION  = "${CLANG_VERSION}"
    }
    # RunPod runs on x86_64; pin so cross-arch builders don't accidentally
    # produce arm64.
    platforms = ["linux/amd64"]
}
