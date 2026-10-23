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

# 中文精读：此文件是训练器与 CUDA 光栅器的桥梁，负责组织张量、选择计算路径并整理输出。
# N 表示高斯数，H/W 表示图像尺寸，K=(最大球谐阶数+1)^2 表示每通道球谐系数数目。
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

# 输入包括相机、可训练高斯集合、渲染开关与 GPU 背景色；输出既服务图像损失，也服务增密。
# 整个渲染不使用多层感知机：颜色由球谐系数与观察方向计算，几何由位置和协方差决定。
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 构造 [N,3] 占位张量以接收 CUDA 反传给出的二维均值梯度；它的零值不是高斯真实投影坐标。
    # 真正的投影在 CUDA 预处理中计算。自定义 autograd 接口将屏幕梯度返回到这个输入位置。
    # 末尾 +0 让结果成为非叶张量，因此下一段 retain_grad 很关键，否则默认不会保留其 .grad。
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        # 保留非叶张量的梯度，训练循环在 loss.backward() 后读取前两维梯度范数来决定增密。
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # 视场角 FoVx/FoVy 为弧度；tan(FoV/2) 与焦距关系为 fx=W/(2·tan(FoVx/2))。
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # 这些常量描述一次相机渲染；NamedTuple 本身不会像高斯参数那样被优化。
    raster_settings = GaussianRasterizationSettings(
        # 图像尺寸确定输出 [3,H,W] 和屏幕分块数量，不能与相机内参使用的分辨率错配。
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        # 像素所有高斯融合完后的剩余透射率乘背景色；scale_modifier 临时整体缩放高斯尺度。
        bg=bg_color,
        scale_modifier=scaling_modifier,
        # world_view_transform 是按 CUDA 接口约定存储的世界到相机矩阵。
        # full_proj_transform 已组合视图与投影；不要误当成单独的相机内参矩阵。
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        # active_sh_degree 决定本轮实际使用的球谐阶数，可能小于已经分配内存的 max_sh_degree。
        sh_degree=pc.active_sh_degree,
        # 相机中心为世界坐标，用来计算从相机指向高斯中心的观察方向。
        campos=viewpoint_camera.camera_center,
        # 未声明输入已预先完成可见性筛选，CUDA 会进行自己的近裁剪与屏幕覆盖检查。
        prefiltered=False,
        debug=pipe.debug,
        # 抗锯齿开关由当前版本支持；具体二维协方差低通与不透明度补偿在 CUDA forward 中解释。
        antialiasing=pipe.antialiasing
    )

    # 建立 nn.Module 包装器，最终调用自定义 torch.autograd.Function 与编译好的 C++/CUDA 扩展。
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 位置为世界坐标 [N,3]；get_opacity 返回 sigmoid 激活后的 [N,1]，不是原始可训练 logits。
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 两条互斥路径：传尺度+旋转让 CUDA 构造协方差，或在 Python 先计算协方差并直接传入。
    # 用 None 表示该路径不提供此参数；底层包装器会将它转换为空张量占位。
    scales = None
    rotations = None
    cov3D_precomp = None

    # Python 路径得到 [N,6] 对称矩阵上三角：xx,xy,xz,yy,yz,zz。
    # 这里“Python”仍是 GPU 上的 PyTorch 张量运算，并不意味着把模型移到 CPU。
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # get_scaling 将对数尺度 exp 成正值；get_rotation 将四元数归一化。
        # 尺度是标准差大小，协方差的特征值对应尺度的平方。
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 颜色同样二选一：直接提供 RGB，或提供球谐系数让光栅器根据视角求颜色。
    shs = None
    colors_precomp = None
    # 外部指定 override_color 可绕过球谐求值，便于调试或可视化某种标量/属性。
    if override_color is None:
        if pipe.convert_SHs_python:
            # 模型系数原为 [N,K,3]，转换为 eval_sh 要求的 [N,3,K]，最后一维按球谐基序排列。
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # 用“高斯位置−相机中心”取得观察方向；归一化后只保留方向，不让距离改变球谐颜色。
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # 按当前活动阶数计算每个高斯的 RGB 偏移；加 0.5 与 RGB2SH 中减 0.5 的编码约定配对。
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # 此处只截断负值，不截断大于 1 的高斯颜色；最终图像会在返回前统一限制到 [0,1]。
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # 可选加速光栅器支持把直流系数 [N,1,3] 与其余系数 [N,K−1,3] 分开传入。
            # 当前锁定标准子模块没有 dc 参数；通常 separate_sh=False，只有安装匹配的加速扩展才走该接口。
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # 注意原代码在此分支无条件引用 dc；若同时强制 Python 球谐或 override_color，dc 可能尚未赋值。
    # 这是原版分支组合的限制，中文注释不修改实现；学习标准路径时保留 separate_sh=False。
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        # 标准光栅化路径：先投影高斯得到二维椭圆，再按分块与深度排序，最后逐像素进行前到后透明度融合。
        # 输出半径 [N] 的零值表示未进入有效屏幕覆盖；depth_image 是累积逆深度 [1,H,W]。
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    # 曝光补偿使用该图像的 [3,4] 仿射参数：RGB 乘 3×3 矩阵，再加 3 维偏移。
    # 开关而非注释标题决定是否执行，因此离线 render.py 也可以启用已训练曝光。
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        # 先将 [3,H,W] 变为 [H,W,3] 才能右乘颜色矩阵；随后转回通道优先并广播颜色偏移。
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # 输出限制为合法图像范围；饱和区域的 clamp 梯度行为也属于原训练图的一部分。
    rendered_image = rendered_image.clamp(0, 1)
    # 字典保留计算图中的张量引用：render 用于损失，viewspace_points 用于读取梯度。
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        # nonzero 返回 [K,1] 整数索引，不能直接把它当作 [N] 布尔数组；增密代码以此索引原张量。
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        # 名字沿用原接口，但本版本值是 Σ(T_i·α_i/z_i)，没有除以总不透明度，也没有取倒数变成深度。
        "depth" : depth_image
        }
    
    return out
