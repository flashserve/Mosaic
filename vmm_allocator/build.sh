#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build script for CUDA VMM Allocator

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=========================================="
echo "Building CUDA VMM Allocator Extension"
echo "=========================================="

# 清理旧的编译文件
echo "Cleaning old build files..."
rm -rf build dist *.egg-info *.so

# 编译扩展
echo "Building extension..."
python setup.py build_ext --inplace

# 检查是否成功
if [ -f "vmm_allocator*.so" ]; then
    echo ""
    echo "✅ Build successful!"
    echo "Extension file: $(ls vmm_allocator*.so)"
    echo ""
    echo "You can now use:"
    echo "  import vmm_allocator"
    echo ""
else
    echo ""
    echo "❌ Build failed! Extension file not found."
    exit 1
fi

