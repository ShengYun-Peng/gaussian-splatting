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

// 中文精读：这一层把 PyTorch 张量转换成底层 CUDA 所需的指针，并创建输出/梯度缓冲区。
// 它不实现高斯公式；真正算法调度在 cuda_rasterizer/rasterizer_impl.cu。
// 正常调用假定真实输入在兼容的 CUDA 设备上、float32 且形状正确；本版本只做了有限的显式检查。
#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

// 生成一个「请求 N 字节显存并返回地址」的回调，底层无需依赖 PyTorch 的具体分配接口。
// lambda 按引用捕获字节张量，扩容后的 Tensor 仍由外层持有，并在前向结束后返回给 Python 保存。
// 这里的 N 是字节数，因为三个缓存张量都使用 uint8/Byte 类型。
std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        // resize_ 原地改变缓存长度；底层拿到连续内存的 char* 后再自行按类型/对齐划分。
        // 不要对已有前向缓存随意 resize，否则反向保存的几何/排序状态可能失效。
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
// 前向扩展入口，参数位置必须与 Python args 元组一致。
// 返回顺序：实例数 R、颜色、半径、几何缓存、排序缓存、像素缓存、逆深度。
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool antialiasing,
	const bool debug)
{
  // 这里只显式检查三维中心是二维张量且最后一维为 3。
  // 不要据此误认为所有输入的 dtype、设备、连续性和形状都已完整验证；调用方需要维持接口约定。
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  // 用 means3D 的设备等选项构造输出，并显式选 float32/int32。
  // int_opts 在这个版本中没有被后续使用，保留它有助于对照原源码，不代表另一路实际分配。
  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  // 颜色形状为 [NUM_CHANNELS,H,W]，逆深度形状最终为 [1,H,W]。
  // 源码先建零通道逆深度张量，再无条件重新分配单通道版本，所以本版本正常前向总会计算逆深度。
  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_invdepth = torch::full({0, H, W}, 0.0, float_opts);
  float* out_invdepthptr = nullptr;

  out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
  out_invdepthptr = out_invdepth.data<float>();

  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  // 三个缓存使用当前 CUDA 设备；调用者应确保当前设备与输入张量设备一致。
  // 颜色/半径跟随 means3D.options()，而缓存这里用当前设备，所以多 GPU 场景需要留意这一约定。
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  // P=0 时跳过底层算法，避免前缀和末项访问及零规模核启动。
  // 这一分支保留输出初始化的零颜色/零逆深度，不会自动填入非黑背景；这是固定源码的实际行为。
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  // 球谐张量非空时 M=sh.size(1)，即每高斯分配的系数数目。
	  // 若使用预计算颜色，sh 是空占位，M=0，底层不会走球谐着色分支。
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  // 把张量转连续后取原始指针；底层按 C/CUDA 的扁平数组访问，无需知道 Tensor 对象。
	  // 颜色、尺度、协方差的空张量对应可选路径；读取非空数据时须满足预期 dtype 和设备。
	  // 几何缓存的所有权仍在 Tensor，避免手写 CUDA malloc/free 与 Python 生命周期脱节。
	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_invdepthptr,
		antialiasing,
		radii.contiguous().data<int>(),
		debug);
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 // 反向扩展入口：输入包括原前向张量、图像上游梯度、三个缓存和实例数 R。
 // 它先创建全零梯度，因为像素核用 atomicAdd 累积，而被剔除/未使用的参数应保持零。
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_invdepth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool antialiasing,
	const bool debug)
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  // 梯度形状逐项对齐原参数：中心 [P,3]、颜色 [P,3]、opacity [P,1]、协方差 [P,6]、
  // 球谐 [P,M,3]、尺度 [P,3]、四元数 [P,4]。二维中心也占 [P,3]，只有 x/y 实际使用。
  // conic 梯度临时存为 [P,2,2]，传回 Python 的八个结果中不包括这个内部中间梯度。
  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  torch::Tensor dL_dinvdepths = torch::zeros({0, 1}, means3D.options());
  
  float* dL_dinvdepthsptr = nullptr;
  float* dL_dout_invdepthptr = nullptr;
  // 只有存在逆深度输出梯度时才分配每高斯逆深度梯度，避免无关分支访问空指针。
  // 这里判断的是张量是否有通道，不是检测梯度数值是否全零。
  if(dL_dout_invdepth.size(0) != 0)
  {
	dL_dinvdepths = torch::zeros({P, 1}, means3D.options());
	dL_dinvdepths = dL_dinvdepths.contiguous();
	dL_dinvdepthsptr = dL_dinvdepths.data<float>();
	dL_dout_invdepthptr = dL_dout_invdepth.data<float>();
  }

  if(P != 0)
  {  
	  // 传入的缓存是保存的 uint8 Tensor，reinterpret_cast<char*> 恢复字节地址。
	  // 底层按前向同样布局还原结构，首先逐像素累积，再逐高斯展开链式法则。
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  opacities.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_invdepthptr,
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dinvdepthsptr,
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  antialiasing,
	  debug);
  }

  // C++ 返回顺序从二维中心梯度开始，与 Python .apply 输入顺序不同。
  // Python backward 会重新排列这八个梯度并在设置对象处补 None，务必对照两个层次理解。
  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

// 可见性查询分配 [P] 布尔张量并启动粗筛核；空输入直接返回空结果。
// 它不渲染图像，也不考虑前景遮挡，具体过滤规则见 auxiliary.h 的 in_frustum。
torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}
