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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

// 中文精读：前向的两个阶段接口。preprocess 按高斯计算，render 按屏幕块/像素计算。
// preprocess 产生 means2D、深度、半径、逆协方差和块覆盖数；排序由 rasterizer_impl.cu 在中间完成。
// render 接收已排序列表，输出颜色、逆深度及反向所需的最终 T 和最后贡献者。
namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	// 输入分组：几何与外观参数 → 相机与图像参数 → 逐高斯输出 → 网格与控制开关。
	// P/D/M 是高斯数量/球谐阶数/系数容量，二维协方差单位为像素²。
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool antialiasing);

	// Main rasterization method.
	// ranges 指定每块在 point_list 中的半开区间；features 是每高斯颜色，不是逐像素图。
	// final_T 与 n_contrib 保留逆序重建所需信息；depths 是高斯中心 z，depth 输出实际是逆深度图。
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* depths,
		float* depth);
}


#endif
