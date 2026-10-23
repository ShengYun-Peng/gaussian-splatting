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

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

// 中文精读：这里是编译期配置，修改后必须重编译扩展。
// 默认三通道 RGB；16×16 的屏幕块对应 256 线程，也决定共享内存批次大小。
// 改变通道数会影响颜色布局、模板实例与缓冲区，不能当作运行时任意参数调整。
#define NUM_CHANNELS 3 // Default 3, RGB
#define BLOCK_X 16
#define BLOCK_Y 16

#endif