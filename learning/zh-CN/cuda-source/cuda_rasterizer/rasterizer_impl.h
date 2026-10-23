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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

// 中文精读：定义字节缓存内部的内存布局；这些结构只持有指针，不负责拥有或释放显存。
// fromChunk 必须在前向和反向用同样的元素数调用，才能恢复一致的偏移。
namespace CudaRasterizer
{
	template <typename T>
	// 把当前地址向上对齐到 alignment 的整数倍，再划出 count 个 T 元素。
	// 位运算 (地址+alignment-1)&~(alignment-1) 要求 alignment 是二的幂；这里使用 128。
	// chunk 按引用更新为下一块起点，ptr 指向本字段首元素，这不是一次新显存分配。
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	// 按高斯组织的中间量：depths 是相机 z，means2D 是像素中心，cov3D 为六项对称协方差。
	// conic_opacity 为逆二维协方差三项加有效 opacity；clamped 记录球谐颜色负值截断。
	// tiles_touched 是覆盖块数量，point_offsets 是其包含自身的前缀和。
	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	// ranges 按块描述排序列表区间；n_contrib 按像素保存最后贡献者位置。
	// accum_alpha 虽含 alpha 字样，实际保存的是最终透射率 T，不是累计不透明度 1-T。
	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	// 排序前后各保存一套键和值：64 位 key=[块编号|深度位模式]，32 位 value=原高斯编号。
	// 数组长度是重叠实例数 R；一个高斯覆盖多个块就会出现在多条 value 中。
	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	// 使用相同 fromChunk 布局计算器在空基址上推进偏移，估计最终需要多少字节，再加 128 对齐余量。
	// 这是固定源码的指针算术布局技巧；学习时可理解为「模拟排布」而不是读取空地址的数据。
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};