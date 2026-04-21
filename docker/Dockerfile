###############################################################################
# srdatalog-python — RunPod / wheel-test image
#
# Twin purpose:
#   1. RunPod dev pod (SSH + Jupyter + nginx proxy + CUDA toolchain)
#   2. Local sandbox for verifying `uv build` produces a self-contained wheel
#      (see docker/test_wheel.sh)
#
# Image size notes:
#   - Base `nvidia/cuda:12.9.1-devel-ubuntu24.04` ships nvcc + CUDA headers
#     (~4 GB). No nvhpc, no LLVM-from-source, no AdaptiveCpp — the Python
#     frontend JIT-compiles with stock clang++ via subprocess, and that's
#     all the runtime needs.
#   - clang-20 pulled from apt.llvm.org (deb.ubuntu is still on clang-18).
###############################################################################
ARG CUDA_VERSION=12.9.1
ARG UBUNTU_VERSION=24.04
ARG CLANG_VERSION=20

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG CLANG_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore

# ---------------------------------------------------------------------------- #
#                              System packages                                 #
# ---------------------------------------------------------------------------- #
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl wget gnupg lsb-release \
        git build-essential cmake ninja-build \
        python3.12 python3.12-dev python3.12-venv python3-pip \
        libpython3-dev unzip zip vim \
        nginx openssh-server && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------- #
#                         Clang from apt.llvm.org                              #
# ---------------------------------------------------------------------------- #
RUN curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/llvm.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/noble/ llvm-toolchain-noble-${CLANG_VERSION} main" \
        > /etc/apt/sources.list.d/llvm.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        clang-${CLANG_VERSION} \
        lld-${CLANG_VERSION} \
        libomp-${CLANG_VERSION}-dev \
        libc++-${CLANG_VERSION}-dev \
        libc++abi-${CLANG_VERSION}-dev && \
    update-alternatives --install /usr/bin/clang   clang   /usr/bin/clang-${CLANG_VERSION}   100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${CLANG_VERSION} 100 && \
    rm -rf /var/lib/apt/lists/*

ENV CXX=clang++ CC=clang

# ---------------------------------------------------------------------------- #
#                             uv (Python packager)                             #
# ---------------------------------------------------------------------------- #
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx
ENV PATH="/usr/local/bin:${PATH}"

# ---------------------------------------------------------------------------- #
#                              Jupyter Lab                                     #
# ---------------------------------------------------------------------------- #
RUN pip3 install --no-cache-dir --break-system-packages jupyterlab

# ---------------------------------------------------------------------------- #
#                              CUDA / nvcc smoke                               #
# ---------------------------------------------------------------------------- #
RUN nvcc --version && clang++ --version

# ---------------------------------------------------------------------------- #
#                         RunPod infrastructure                                #
# ---------------------------------------------------------------------------- #
RUN mkdir -p /workspace

# nginx reverse proxy (mirrors containers/container-template/proxy/)
RUN wget -O /etc/init.d/nginx https://www.linode.com/docs/assets/660-init-deb.sh && \
    chmod +x /etc/init.d/nginx && \
    /usr/sbin/update-rc.d -f nginx defaults

COPY docker/nginx.conf   /etc/nginx/nginx.conf
COPY docker/readme.html  /usr/share/nginx/html/readme.html
COPY docker/start.sh     /start.sh
COPY docker/post_start.sh /post_start.sh
RUN chmod +x /start.sh /post_start.sh

# ---------------------------------------------------------------------------- #
#                         Application source                                   #
# ---------------------------------------------------------------------------- #
# Bake the repo into the pod so contributors can `uv sync && uv build` on
# first login. End users installing from PyPI never see this.
COPY . /root/srdatalog-python
WORKDIR /root/srdatalog-python

# Install dev deps + editable package so `srdatalog` CLI works on shell login
# without activating a venv. On Ubuntu 24.04 system Python is PEP-668 managed,
# so --break-system-packages is required. We skip creating a .venv — the whole
# point of the pod is interactive work, and `docker/test_wheel.sh` does its
# own fresh-venv wheel test on demand.
RUN python3 scripts/populate_vendor.py && \
    uv pip install --system --break-system-packages -e ".[dev]" && \
    srdatalog info

# Expose: 22 SSH, 8000 VSCode Server, 8001 VSCode proxy, 8888 Jupyter Lab
EXPOSE 22 8000 8001 8888

CMD ["/start.sh"]
