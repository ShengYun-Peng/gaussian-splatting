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

// 中文精读：本文件把各个 CUDA 核组织成完整前向与反向流水线。
// 前向：逐高斯预处理 → 覆盖块数前缀和 → 高斯/块实例复制 → 按块和深度排序 → 逐块合成。
// 反向：复用排序与缓存 → 逐像素逆序反传 → 逐高斯把二维属性梯度传回三维参数。
// P 是原始高斯数；R/num_rendered 是复制后的高斯与块重叠实例数，两者通常不同。
#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
// 估计表示块编号所需的高位位数，供基数排序指定有效位范围。
// 排序键低 32 位始终用于深度，高位只有块编号需要的若干位有意义，无需总排序完整 64 位。
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
// 一线程一高斯，调用共享粗筛函数写 bool；只判断几何筛选，不处理遮挡。
// 此路径用于 Python markVisible，核心前向自己也会做相同近裁剪检查。
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
// 一个高斯可能覆盖多个屏幕块，为每次重叠生成一条 (排序键,高斯编号)。
// 这里只复制索引与键，不复制完整高斯参数；渲染阶段仍通过编号读取原属性数组。
// 前缀和已经给每个高斯分配独占写入区间，因此写实例数组不需要原子加法。
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		// InclusiveSum 的第 i 项是截至 i 的覆盖块总数，所以第 i 个高斯起点取 offsets[i-1]。
		// 例如覆盖数 [2,3,0,2] → 前缀和 [2,5,5,7]，四个写入区间为 [0,2)、[2,5)、[5,5)、[5,7)。
		// 不可见高斯半径为零，对应空区间，不生成任何键。
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				// 高 32 位编码行优先块编号 tile_id=y*grid.x+x，低 32 位编码中心深度的浮点位模式。
				// 正有限浮点数的 IEEE 754 位模式按无符号整数排序时与数值顺序一致；近裁剪确保有效深度为正。
				// 若改成允许负深度或特殊 NaN 值，不能继续直接使用这套位排序技巧。
				uint64_t key = y * grid.x + x;
				key <<= 32;
				// 这里读取 float 的原始 32 位表示，不是把深度数值转换成整数。
				// 排序后先聚合同块实例，再在块内按深度从小到大排，实现前向需要的近到远合成顺序。
				// 这是高斯中心深度排序，不是对每个像素分别做精确几何排序。
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
// 排序后同一块的实例已连续，只需检测相邻键的高位何时变化。
// 每个线程检查一个实例边界，写出该块在 point_list 中的半开 [start,end) 区间。
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		// 发生块编号变化时，当前位置同时是前一块的结束和当前块的开始。
		// 首项/末项需要单独设置边界；没有实例的块此前已清零，因此其区间保持 [0,0)。
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

// 把一整块字节显存按固定顺序切成多个类型化数组，每个起点按 128 字节对齐。
// 这里主要做指针布局与 CUB 临时空间查询，不逐元素初始化数据。
// 同样的布局函数在反向重新调用，必须保持字段次序、P 和对齐约定完全一致。
CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	// GeometryState 按高斯保存：深度、颜色截断标记、半径、二维中心、六项协方差、
	// 逆协方差+有效不透明度、RGB、覆盖块数、前缀和与扫描临时空间。
	// 上层训练增密会改变 P，缓冲区因而通过回调动态重新分配。
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	// CUB 的第一种调用传空临时空间，只查询需要多少工作区字节，不执行真正扫描。
	// 拿到 scan_size 后再分配空间，前向后面以非空指针再次调用才完成计算。
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

// 像素缓存按 N=宽×高分配；accum_alpha 名字容易误导，实际保存最终透射率 T。
// n_contrib 保存最后贡献者位置；ranges 语义上按屏幕块使用，源码这里按像素数预留了更大的容量。
CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

// 这里的 P 是传入的实例数量，含同一高斯对多个块的重复条目。
// 同时分配排序前/后键和值数组，再询问 CUB 基数排序所需临时空间。
CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
// 顶层前向只负责调度，计算细节在 forward.cu。
// 三个 Buffer 回调来自 PyTorch 字节张量，既允许 CUDA 动态申请空间，又使缓存生命周期由张量管理。
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* depth,
	bool antialiasing,
	int* radii,
	bool debug)
{
	// 半视场角满足 tan(FoV/2)=(图像尺寸/2)/焦距，所以 fy=height/(2*tan_fovy)。
	// 焦距单位为像素，与协方差投影和像素位置保持一致；tanfov 参数并非完整 FoV 的正切值。
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	// 先通过相同布局函数估算所需字节，再让回调扩容，最后在真实指针上划分字段。
	// 这种设计避免为每个中间数组单独向 PyTorch/驱动申请显存。
	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	// 调用者可提供半径输出；没有时使用内部缓存。
	// 正常 Python 路径需要 radii 给训练统计使用，所以通常传入单独的整数张量。
	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	// 屏幕按 BLOCK_X×BLOCK_Y 切块，宽高分别向上取整，最后一排/列由 inside 掩码处理越界。
	// block 的二维线程布局恰好对应一个屏幕块内的像素坐标。
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	// 第一步：逐高斯计算投影椭圆、颜色、深度、半径及覆盖块数。
	// 此时只知道每个高斯要复制多少条实例，还没有确定它们在总列表中的位置。
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered,
		antialiasing
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	// 第二步：对每个高斯的覆盖块数做包含自身的前缀和，得到独占写入区间。
	// 复杂度主要随高斯数 P 增长；这比对每次重叠都做全局原子分配更适合批量 GPU 处理。
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	// 前缀和最后一项等于实例总数 R，把这个单值从 GPU 拷回 CPU，用来分配排序缓冲区。
	// 这一步引入主机/设备依赖，是该流水线中的同步点；R 不等于实际有贡献的像素数或高斯数。
	// 底层假定 P>0；外层 rasterize_points.cu 用 if(P!=0) 保护这里的 P-1 访问。
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	// 第三步：按已分配的区间生成所有块/深度键与原高斯编号。
	// 后续排序只重排这些轻量实例，不修改训练中的高斯参数顺序。
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	// 第四步：基数排序按键升序，同时携带对应高斯编号进行相同重排。
	// 有效位范围为 [0,32+bit)，包含全部深度位和足够的块编号位。
	// 得到的列表可直接被各屏幕块独立读取，无需像素线程再自行排序。
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	// 所有块区间先清零；只对实际出现实例的块写 start/end。
	// 这样空块也可以正常进入渲染核，其 range 长度为零，最后直接输出背景色。
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	// R=0 时不启动零工作量的范围识别核；颜色渲染仍启动，以便为空场景块写背景。
	// 请区分「P>0 但全部被剔除」与 C++ 外层「P=0」分支，后者直接保留初始化的输出零值。
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	// 颜色来源二选一：上层预计算颜色，或预处理核已写好的球谐颜色。
	// 无论来源如何，像素合成核都把它当每高斯的 CHANNELS 个特征值来处理。
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.depths,
		depth), debug)

	// 返回 R 供 Python 保存到 autograd 上下文，反向才能重建 BinningState 的同样布局。
	// 它是缓存大小信息，而不是用于计算损失的可微输出。
	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
// 反向不重新做分块与排序，直接复用前向的三个缓存，保证对同一条离散计算路径求导。
// 可微的几何和颜色中间量可按需重算，离散决定则沿用前向结果。
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_invdepths,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dinvdepth,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool antialiasing,
	bool debug)
{
	// 这里仅把缓存字节重新解释为字段指针，并非重新跑一遍前向。
	// P、R、图像宽高必须对应同一帧，否则布局偏移和数据含义都会错位。
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	// 第一段反向从图像梯度得到每个高斯的二维位置、逆协方差、有效不透明度、颜色、逆深度梯度。
	// 一个高斯被很多像素共享，所以这一段内部使用原子累加。
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_invdepths,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dinvdepth), debug);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	// 第二段需要前向实际使用的三维协方差：要么来自上层输入，要么来自几何缓存。
	// 然后依次反传矩阵求逆、投影雅可比、齐次中心、球谐与尺度/旋转。
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		opacities,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		dL_dinvdepth,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		antialiasing), debug);
}
