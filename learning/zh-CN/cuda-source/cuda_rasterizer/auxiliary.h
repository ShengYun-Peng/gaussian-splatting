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

// 中文精读：前向与反向共享的坐标约定、球谐常数、裁剪工具和归一化导数。
// 本文件的矩阵索引比函数名字更可靠：4×4 平铺数组按列取值，平移位于 12/13/14。
// 阅读 Python 相机中的 transpose 与 GLM 构造器时，应始终把这些实际索引作为核对依据。
#ifndef CUDA_RASTERIZER_AUXILIARY_H_INCLUDED
#define CUDA_RASTERIZER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"

// BLOCK_SIZE=16×16=256，通常是 8 个 warp；一个渲染块同时服务 256 个候选像素。
// 球谐常数匹配特定的实球谐基顺序和符号，必须与前向、反向及 Python 实现一致。
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE/32)
// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

// 把归一化坐标 v 映射为像素中心坐标：v=-1 → -0.5，v=1 → S-0.5。
// 因此整数像素位置 0,...,S-1 对应位于图像边界内半个像素的中心。
// 导数为 S/2，这正是反向渲染中 ddelx_dx、ddely_dy 的来源。
__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

// 用中心与保守半径估计覆盖块的半开矩形，结果限制在 [0,grid.x]×[0,grid.y]。
// 最小/最大坐标是块下标，不是像素下标；范围为空表示不影响任何屏幕块。
// 这里直接使用整数转换与边界裁剪，是源码的具体离散规则，不能与连续几何边界等同。
__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

// 重载版本允许横纵使用不同半径；当前核心前向使用前一个圆半径版本。
// 两种版本都只负责生成候选矩形，不逐块做精确的椭圆相交测试。
__forceinline__ __device__ void getRect(const float2 p, int2 ext_rect, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - ext_rect.x) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - ext_rect.y) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + ext_rect.x + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + ext_rect.y + BLOCK_Y - 1) / BLOCK_Y)))
	};
}


// 计算仿射点变换 t=R·p+t0：点的齐次第四分量视为 1，所以必须加平移。
// 索引 0/4/8 构成输出 x 的三项；这里的 4x3 表示取 4×4 变换的前三个输出分量。
__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

// 完整齐次变换保留 w，用于后续透视除法。
// 此函数本身不除以 w，也不判断是否在可见范围；调用者明确完成这两步。
__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

// 方向向量没有位置平移，所以这里只乘线性旋转部分。
// 与点变换相比唯一但关键的差异就是去掉 matrix[12/13/14]。
__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

// 反向传播的梯度是协向量：若 t=R·p+t0，则 dL/dp=Rᵀ·dL/dt。
// 本函数专门做这个转置乘法，不是把相机点位置逆变换回世界坐标。
__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

// 对单位化 v/||v|| 做向量-雅可比积后，只取关于原始 v.z 的导数。
// 统一公式为 (I/||v|| - vvᵀ/||v||³)·dv；下面逐分量展开它。
__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

// 输入 dv 是对归一化结果的上游梯度，输出是对归一化前 v 的梯度。
// 归一化会消去径向变化，只保留切向方向变化；分母 ||v||³ 来自商法则。
// 此实现没有零长度保护，因此调用方必须保证向量非零。
__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

// 四维版本适用于四元数归一化，数学结构与三维相同。
// 当前固定版本的 computeCov3D 把 CUDA 内部四元数归一化及其反传注释掉了；
// 因此这个工具的存在不代表当前旋转梯度会在 CUDA 中通过它。
__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}

// 标准 sigmoid 可把实数变到 (0,1)；正常渲染的不透明度通常已在 Python 激活。
// 这个辅助函数并没有在当前核心前向中再次作用到 opacity。
__forceinline__ __device__ float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

// 粗筛：读取中心，求相机坐标及投影坐标，实际生效的条件仅为 p_view.z>0.2。
// 横纵视锥检查在 if 后的原注释中被禁用；画外椭圆随后由块矩形裁剪处理。
// 此函数也不判断高斯是否被其他高斯遮挡，不能等同于最终图像中的可见贡献。
__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		// prefiltered 表示调用方承诺输入已满足这层过滤；若仍落在近裁剪面后，就触发 GPU trap。
		// 这不是自动开启额外滤波，而是对前置过滤约定的一种检查。
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}

// 宏先执行 A；debug=true 时 cudaDeviceSynchronize 强制同步并检查错误。
// 常规模式避免每一步同步以提高性能，因此异步错误可能到更晚的 CUDA 调用才暴露。
// 宏中的反斜线用于续行；学习时不要把注释放在反斜线后面，避免破坏预处理。
#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
