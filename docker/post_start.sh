#!/bin/bash
# Post-start for the srdatalog-python RunPod container.
# Runs after /start.sh has initialized nginx, SSH, and Jupyter.

set -u

echo ""
echo "==================================================================="
echo "  srdatalog-python pod ready"
echo "==================================================================="
echo "Toolchain:"
printf "  python : %s\n" "$(python3 --version 2>&1)"
printf "  uv     : %s\n" "$(uv --version 2>&1)"
printf "  clang  : %s\n" "$(clang++ --version 2>&1 | head -1)"
printf "  nvcc   : %s\n" "$(nvcc --version 2>&1 | tail -1)"
echo ""
echo "Quickstart:"
echo "  cd /root/srdatalog-python"
echo "  uv run pytest                       # run the test suite"
echo "  uv run python examples/triangle.py  # emit a JIT .cpp tree"
echo "  srdatalog info                      # package + runtime paths"
echo ""
echo "Publish pipeline:"
echo "  bash docker/test_wheel.sh           # build wheel, install in fresh venv,"
echo "                                      # verify self-containment + JIT compile"
echo ""
echo "Editable install is already set up. Edit source in /root/srdatalog-python"
echo "and changes take effect immediately."
echo "==================================================================="
