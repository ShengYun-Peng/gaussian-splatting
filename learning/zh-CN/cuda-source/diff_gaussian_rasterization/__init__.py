#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# 中文精读：Python 与 C++/CUDA 之间的自动求导桥接层。
# 调用链为 GaussianRasterizer.forward → _RasterizeGaussians.apply → _C.rasterize_gaussians。
# _C 是编译出来的扩展；PyTorch 无法自动追踪 CUDA 内部逐项运算，所以反向需要人工实现。
from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

# 把张量搬到 CPU 后克隆，可用于保留故障输入的快照；普通对象直接复用。
# 这一固定版本的 forward/backward 没有调用这个工具函数，不要误以为每次渲染都会备份。
def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

# 薄包装保留统一的参数顺序；.apply 由 PyTorch 建立自定义算子的求导节点。
# means3D 是 [P,3] 世界中心；means2D 是用于接收屏幕梯度的占位张量；
# sh 为 [P,M,3]，颜色可预先算好为 [P,3]；尺度 [P,3]、旋转 [P,4]、协方差 [P,6]。
def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

# 自定义 autograd.Function 的 forward 保存必要状态，backward 返回各输入的梯度。
# 这里不会从 Python 自动推导 CUDA 程序；反向对应 backward.cu 中手写的链式法则。
# means2D 没有传给 C++ 前向：真正的二维位置由三维中心和相机计算；
# 但 backward 会为这个占位输入返回二维梯度，让训练代码用它统计增密依据。
class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        # 下面的元组顺序必须严格匹配 rasterize_points.cu 的 C++ 函数签名。
        # 相机设置从不可变 NamedTuple 展开；它们不作为可训练张量输入返回梯度。
        # 传入的尺度、不透明度、旋转通常已经分别经过 exp、sigmoid、归一化；此桥接层不重复激活。
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        # 七个返回值中只有颜色、半径、逆深度暴露给上层；三个字节缓存留给反向恢复。
        # num_rendered 是高斯与块的重叠实例总数，一个高斯覆盖多个块会重复计数，不能当可见高斯数。
        # color=[3,H,W]，radii=[P] 为整数像素半径，invdepths=[1,H,W] 为加权逆深度。
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        # ctx 保持同一次前向的相机设置与实例数量，save_for_backward 保持张量生命周期。
        # 缓存含投影、颜色截断掩码、排序结果、最终透射率等；这些量决定本次离散剔除与合成路径。
        # 这不是把整个计算图存下来，而是保存手写反向所需要的最小一组输入与辅助状态。
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, invdepths

    @staticmethod
    # 三个入参分别对应 forward 的三个输出梯度：颜色、整数半径、逆深度。
    # 半径来自 ceil/剔除等离散决定，不做可微优化，使用占位名 _ 忽略它。
    # grad_out_depth 的名字沿用源码，但它实际对应逆深度图，并不是普通 z 深度图。
    def backward(ctx, grad_out_color, _, grad_out_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        # 将缓存与输出梯度交给 CUDA：先按像素逆序合成求二维属性梯度，再传播到三维参数。
        # 前向的排序与覆盖集合直接复用；这里没有对排序、阈值比较或块编号本身求导。
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_out_depth, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)        

        # 返回元组的槽位必须与 .apply 的九个输入逐一一致，不能按 C++ 输出顺序直接返回。
        # means2D 梯度既服务于三维位置反传的内部链路，也被上层 retain_grad 收集用于增密判断。
        # 最后 None 表示 raster_settings 这个配置对象没有梯度；本接口也不直接优化相机参数。
        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

# 一帧渲染的固定设置：宽高以像素计，tanfovx/y 是半视场角的正切值，不是弧度角。
# viewmatrix 表示世界到相机，projmatrix 通常是复合投影，campos 是世界坐标下相机中心。
# sh_degree 控制当前球谐阶数；antialiasing 控制滤波透明度补偿；debug 会引入同步检查。
class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    # 这个查询只返回布尔标记，因此放在 no_grad 下，不建立训练梯度。
    # 固定版本底层主要检测相机前方的近裁剪条件，不能将结果理解成完整遮挡可见性。
    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        # 外观表示必须二选一：让 CUDA 从球谐求颜色，或由上层提供现成 RGB。
        # 两种都给会让梯度来源不明确，两种都不给则无法着色，因此这里直接报错。
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        # 几何也必须二选一：完整的尺度+旋转，或预先计算的三维协方差。
        # 条件同时检查「缺少尺度或旋转」与「预计算协方差和尺度/旋转混用」两类问题。
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        # 未使用的输入用零元素张量占位，便于保持扩展函数固定签名。
        # 这一版本在 C++ 中依靠空张量的数据指针表示缺失路径；不要把占位张量误当零颜色或零尺度。
        # 这些空张量默认位于 CPU，但没有元素可供核访问；真实输入则须满足扩展的 CUDA/类型要求。
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

