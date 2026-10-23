/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

// 中文精读：pybind11 把三个 C++ 函数注册到 PyTorch 构建出的 _C 模块。
// Python 调用 _C.rasterize_gaussians 时实际跳到 RasterizeGaussiansCUDA，
// 后者才完成张量内存分配并调用底层 CUDA 流水线；这里没有自动生成反向公式。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  // 反向函数作为独立入口注册，由 Python autograd.Function.backward 显式调用。
  // 因此扩展成功导入不代表反向已自动验证，需要结合源码不变性与实际运行检查。
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
}