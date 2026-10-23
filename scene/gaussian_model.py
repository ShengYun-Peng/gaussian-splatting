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

# 中文导读：本文件管理“场景里有哪些三维高斯”，以及这些高斯怎样被优化、增殖和删除。
# 渲染在 gaussian_renderer 与 CUDA 子模块中完成；这里负责给渲染器提供可微参数。
# 下文统一记 P 为当前高斯数量、D 为最大球谐阶数、K=(D+1)^2 为每个颜色通道的系数数。
# 稠密化会改变 P，因此不能只拼接参数值，还必须同步修改 Adam 动量和逐点统计量。
# 阅读顺序建议：参数激活 → 点云初始化 → 优化器 → 克隆/分裂/剪枝 → 保存/恢复。
import torch
import numpy as np
# 逆 sigmoid 把目标不透明度映射回可优化的实数；学习率函数返回随迭代变化的闭包。
# build_rotation 将四元数归一化后转换为三维旋转矩阵。
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
# PLY 负责持久化逐点属性；RGB2SH 把初始 RGB 变成零阶球谐系数。
# distCUDA2 是 CUDA 近邻扩展，返回每个点的三个近邻平方距离的平均值。
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

# 稀疏 Adam 来自特定版本的光栅化扩展，普通版本未必导出这个类。
# 此处保留上游的宽泛异常处理；真正构造优化器时还会尝试并回退到普通 Adam。
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

# GaussianModel 本身没有继承 nn.Module：参数仍是 nn.Parameter，
# 但优化器通过显式参数组持有它们，不能依赖 model.parameters() 自动收集。
# 以下带下划线的成员通常是“优化器存储值”，get_* 属性通常给出渲染所需值。
# 必须区分这两层：_scaling 是对数尺度，_opacity 是 logit，_rotation 是未强制单位化的四元数。
class GaussianModel:

    # 集中定义从无约束参数到几何/光学量的映射；映射参与自动微分。
    def setup_functions(self):
        # 输入 scaling 为 [P,3] 的正标准差，rotation 为 [P,4]，返回 [P,6] 对称协方差。
        # 协方差描述三维椭球的方向与宽度，并不是直接给出屏幕上最终的二维椭圆。
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            # 令 S=diag(sx,sy,sz)、R 为四元数生成的旋转矩阵，则 L=R·(modifier·S)。
            # modifier 缩放标准差，因此最终协方差会乘以 modifier 的平方。
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            # 批量计算 Σ=L Lᵀ=modifier²·R S² Rᵀ；有效旋转和非零 modifier 下，正尺度给出正定协方差。
            # transpose(1,2) 只转置每个 3×3 矩阵，不交换最前面的高斯批量维。
            actual_covariance = L @ L.transpose(1, 2)
            # 利用对称性，只存 [Σxx, Σxy, Σxz, Σyy, Σyz, Σzz] 六个数供光栅器读取。
            symm = strip_symmetric(actual_covariance)
            return symm
        
        # 存储 s_raw∈ℝ，使用 s=exp(s_raw)>0；避免优化器直接产生负标准差。
        # 反向传播时 ∂s/∂s_raw=s，所以相同原始参数步长对应相对尺度变化，而非固定长度变化。
        self.scaling_activation = torch.exp
        # 初始化或分裂得到正尺度后，用 log 写回原始参数空间；不能重复取指数。
        self.scaling_inverse_activation = torch.log

        # 把上面的组合函数保存为回调，以便 get_covariance 统一调用。
        self.covariance_activation = build_covariance_from_scaling_rotation

        # 存储 α_raw∈ℝ，使用 α=sigmoid(α_raw)∈(0,1)；α 是单个高斯的基础不透明度。
        # 像素处的贡献还需乘以二维高斯权重和此前累计透射率，不等于最终像素覆盖率。
        # 反向传播还要乘 α(1-α)，因此极接近 0/1 的原始参数会进入 sigmoid 梯度较小的区域。
        self.opacity_activation = torch.sigmoid
        # inverse_sigmoid(α)=log(α/(1-α))，用于初始化和重置不透明度。
        self.inverse_opacity_activation = inverse_sigmoid

        # 沿默认的 dim=1 归一化 [P,4] 四元数；顺序为 [w,x,y,z]。
        # 优化器更新原始四个分量，渲染读取时再满足单位四元数约束。
        self.rotation_activation = torch.nn.functional.normalize


    # 构造时只建立空容器；真实 GPU 参数由 create_from_pcd 或 load_ply 创建。
    def __init__(self, sh_degree, optimizer_type="default"):
        # 初始只启用零阶球谐，即与观察方向无关的颜色；后续逐阶增加表达能力。
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        # 最大阶数决定整个训练期间预分配的系数数量，active_sh_degree 只决定使用哪部分。
        self.max_sh_degree = sh_degree  
        # 中心 μ 的形状为 [P,3]，单位与输入点云/相机世界坐标一致，没有额外激活。
        self._xyz = torch.empty(0)
        # 零阶颜色系数形状 [P,1,3]：中间维是球谐系数，最后一维是 RGB。
        self._features_dc = torch.empty(0)
        # 其余球谐系数形状 [P,K-1,3]；与零阶分开存放，便于使用不同学习率。
        self._features_rest = torch.empty(0)
        # [P,3] 对数标准差，沿局部坐标系的三个主轴分别控制椭球宽度。
        self._scaling = torch.empty(0)
        # [P,4] 原始四元数决定椭球朝向；并不是欧拉角，也不是旋转矩阵。
        self._rotation = torch.empty(0)
        # [P,1] 不透明度 logit；保存 PLY 时同样保存这个原始值。
        self._opacity = torch.empty(0)
        # [P] 记录观察窗口内出现过的最大屏幕投影半径，用于大高斯剪枝判据。
        self.max_radii2D = torch.empty(0)
        # [P,1] 累计“屏幕位置梯度范数”；名称含 xyz，但统计的不是三维 _xyz.grad。
        self.xyz_gradient_accum = torch.empty(0)
        # [P,1] 记录参与统计的可见次数，最终用累积量/次数得到平均梯度。
        self.denom = torch.empty(0)
        # 训练前还没有优化器；training_setup 会绑定后续建立的参数对象。
        self.optimizer = None
        # 稠密化中的相对大小阈值，乘以场景半径后才具有世界坐标的长度单位。
        self.percent_dense = 0
        # 位置学习率的场景尺度因子；通常使用相机分布给出的场景半径。
        self.spatial_lr_scale = 0
        # 安装激活函数，供初始化、读取属性和协方差构造共同使用。
        self.setup_functions()

    # 训练 checkpoint 使用的快照，顺序必须与 restore 的解包顺序完全一致。
    # 与 PLY 不同，这里额外保存 Adam 状态、当前启用阶数和稠密化统计。
    # 注意此版本的元组没有保存 _exposure、曝光优化器及曝光学习率状态；
    # 所以不能把它理解为包含所有曝光训练状态的完整快照。
    def capture(self):
        # 这里返回张量/状态对象，由调用者 torch.save 写入文件；本函数本身不做磁盘 I/O。
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            # Adam 的一阶/二阶动量及 step 随主优化器状态保存，恢复后可延续主参数优化。
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    # 把模型快照接回训练流程；training_args 提供学习率与稠密化配置。
    def restore(self, model_args, training_args):
        # 暂存统计量与优化器字典，因为接下来的 training_setup 会初始化这些成员。
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        # 先用恢复出来的 nn.Parameter 重建参数组，确保优化器引用的是正确对象。
        self.training_setup(training_args)
        # 覆盖 training_setup 刚清零的统计量，保留中断前已观察的梯度和可见次数。
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        # 按参数组布局加载旧 Adam 动量；组的顺序和含义必须与保存时兼容。
        self.optimizer.load_state_dict(opt_dict)

    # 这些 @property 像属性一样读取，例如 model.get_scaling，无需加括号。
    @property
    def get_scaling(self):
        # [P,3]：将 log 标准差转为实际标准差，保留 exp 的反向传播链。
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        # [P,4]：返回单位四元数；只改变读取结果，不会原地改写 _rotation。
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        # [P,3]：直接暴露中心参数，渲染损失可以沿此张量回传到位置。
        return self._xyz
    
    @property
    def get_features(self):
        # 取出 [P,1,3] 的常数项和 [P,K-1,3] 的方向相关项。
        features_dc = self._features_dc
        features_rest = self._features_rest
        # 沿球谐系数维拼接成 [P,K,3]；不是沿颜色维拼接。
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        # 仅零阶系数，供支持分离球谐输入的光栅化路径使用。
        return self._features_dc
    
    @property
    def get_features_rest(self):
        # 仅高阶系数；D=0 时这个张量的系数维长度为 0。
        return self._features_rest
    
    @property
    def get_opacity(self):
        # 返回 [P,1] 的基础不透明度，优化器实际更新的仍是其 logit。
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        # 返回 [图像数,3,4] 的可训练颜色仿射矩阵；曝光没有 sigmoid 等约束。
        return self._exposure

    # 同一三维场景可能包含亮度/色调不同的照片，按图像名取对应校正矩阵。
    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            # 训练时用名字→下标映射，取出该图像的可训练 [3,4] 矩阵。
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            # 读取模型用于评估时，可直接采用 exposure.json 中冻结的矩阵。
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        # 尺度先 exp；四元数直接传入原始值，因为 build_rotation 内部会再次归一化。
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        # 每次只提升一阶且不超过预分配上限；train.py 定期调用，渐进拟合方向颜色。
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # 由 SfM/随机点云建立可训练高斯，一点对应一个初始高斯。
    # pcd.points、pcd.colors 为 [P,3]；颜色已归一化到 [0,1]。
    # cam_infos 虽然标注为 int，实际调用传入相机信息列表，后面会遍历并调用 len。
    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        # 位置的学习率与场景尺度绑定，避免同一数值步长在大小场景中含义差异过大。
        self.spatial_lr_scale = spatial_lr_scale
        # 把 NumPy 点坐标复制为 float32 CUDA 张量；初始中心即点云中的世界坐标。
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # 零阶基函数常数 C0≈0.282095，RGB2SH(c)=(c-0.5)/C0。
        # 渲染时用 C0·系数+0.5 恢复初始颜色，因此这些值不是直接的 RGB。
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # 先按 [P,RGB,K] 布局构建，所有高阶系数初始化为 0，即初始颜色不依赖视角。
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # 把 [P,3] 初始颜色填进第 0 个球谐系数。
        features[:, :3, 0 ] = fused_color
        # 源码细节：第 1 维只有 3 个通道，3: 是空切片，这行实际上不写入任何元素。
        # 高阶系数仍为 0，原因是上一条 torch.zeros，而不是这一行赋值。
        features[:, 3:, 1:] = 0.0

        # 打印的是初始高斯数量；之后经过克隆、分裂、剪枝，数量会改变。
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # CUDA 近邻搜索排除点自身，取最近三个点的平方距离平均值，结果为 [P]。
        # 这里的 dist2 不是单个最近邻距离，也不是三个距离先平均再平方。
        # 下限 1e-7 防止重复点/极小间距让后面的 sqrt、log 产生零尺度或负无穷。
        # 正常点云应足够大以提供三个邻居；极小玩具点集不能照搬这个初始化假设。
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # sqrt 把长度平方变成标准差；log 转入可优化空间。
        # [...,None] 把 [P] 变成 [P,1]，repeat(1,3) 得到三个主轴相同的初始球形高斯。
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # 每点创建四元数 [w,x,y,z]；先全零，再把 w 置 1，得到单位旋转。
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # 目标初始 α=0.1，实际参数存 log(0.1/0.9)≈-2.197，而不是直接存 0.1。
        # 较低初始不透明度让多个高斯仍有共同参与拟合的机会。
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 把中心包装为叶子参数；本类不继承 nn.Module，稍后仍需显式加入优化器。
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # [P,3,1]→[P,1,3]；transpose 改变步长，contiguous 保证光栅器读取时布局连续。
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # 高阶项同理变为 [P,K-1,3]；所有阶的存储一次性分配，逐阶启用不需增大张量。
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # 尺度、四元数、不透明度全部保持原始参数形式，并启用梯度。
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # 投影半径尚未观测，按每点一个标量初始化为 0。
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # 曝光参数属于照片而非高斯；名字用作训练/渲染和 JSON 保存之间的稳定索引。
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        # None 表示使用当前可训练曝光，而不是从 JSON 加载的冻结字典。
        self.pretrained_exposures = None
        # 每张图初始矩阵为 [I₃|0]，形状 [图像数,3,4]，保持输入 RGB 不变。
        # 前三列表示颜色混合，最后一列表示偏置；实际乘法方向见渲染器的行向量 RGB @ A。
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        # 曝光也参与自动微分，但使用独立优化器，不会随高斯增删而改变行数。
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    # 准备优化器及稠密化统计；应在 create_from_pcd 等参数初始化之后调用。
    def training_setup(self, training_args):
        # percent_dense×场景半径是“小高斯克隆、大高斯分裂”的分界长度。
        self.percent_dense = training_args.percent_dense
        # 累积梯度与可见次数都保留 [P,1]，方便与逐点不透明度等属性对齐。
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 每个组恰好一个“大张量参数”，该张量第 0 维枚举所有高斯。
        # name 是后面替换/裁剪/拼接参数的索引协议，不能随意改名或打乱含义。
        l = [
            # 位置学习率按场景半径缩放；中心的优化单位与世界坐标长度一致。
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # 零阶颜色承担主要外观拟合，使用完整 feature_lr。
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # 高阶球谐学习率降为 1/20，减缓方向细节在几何尚未稳定时的变化。
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            # 后三组分别更新不透明度 logit、log 标准差和原始四元数，各有独立学习率。
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            # 全局 lr=0 只是默认值，各参数组已显式给 lr，所以参数仍会更新。
            # eps=1e-15 是 Adam 分母的稳定项；实际动量在首次 step 时懒初始化。
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # 稀疏实现可只处理可见高斯；是否能用取决于已安装的光栅化扩展版本。
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                # 如果扩展缺失或构造失败，这里沿用上游行为回退到 torch.optim.Adam。
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # 曝光不是逐高斯属性，所以不加入上面的六组，也不参加后续增殖/剪枝。
        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        # 构造位置学习率函数：在对数空间从初值插值到终值，相当于指数衰减。
        # 这里没有传 lr_delay_steps，其默认值为 0；因此仅设置 delay_mult 并不会启用预热。
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        # 曝光调度另外传入 delay_steps，可先平滑升到正常速率，再按总迭代数衰减。
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        # 仅在使用可训练曝光时更新曝光组学习率；冻结的预训练曝光不走这条分支。
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                # 把本轮迭代号代入闭包，写入优化器真正用于下一次 step 的 lr。
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        # 主优化器只有 xyz 组随迭代调整，其余组保持 training_setup 给定的学习率。
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # 返回位置学习率便于记录；找到 xyz 后即可结束循环。
                return lr

    # 声明 PLY 顶点属性名，排列必须与 save_ply 拼接列的顺序严格一致。
    def construct_list_of_attributes(self):
        # 法线只是 PLY 兼容字段，本模型没有优化每点表面法线。
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        # 零阶总计 1×3=3 列，依次对应 RGB 三通道。
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        # 高阶共 (K-1)×3 列；保存时颜色维在前，同一通道的所有系数连续排列。
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        # 接着存一个不透明度 logit、三个 log 尺度和四个原始四元数分量。
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # 返回列名列表，既定义数据类型，也约定之后加载模型时如何解释每一列。
        return l

    # 导出可供渲染恢复的 PLY；它存参数值，不保存 Adam 状态或训练迭代统计。
    def save_ply(self, path):
        # 允许目标迭代目录尚不存在，先递归创建目录。
        mkdir_p(os.path.dirname(path))

        # detach 切断自动微分，cpu 才能转 NumPy；文件写入不会参与训练计算图。
        xyz = self._xyz.detach().cpu().numpy()
        # 法线填零占位，不能把保存的 nx/ny/nz 当成真实表面法线。
        normals = np.zeros_like(xyz)
        # [P,1,3]→[P,3,1]→[P,3]，使零阶颜色与 PLY 字段顺序一致。
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # [P,K-1,3]→[P,3,K-1]→[P,3(K-1)]；加载时要按相同顺序 reshape。
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # 下列三种量都保存“原始参数”：logit、log 标准差、未归一化四元数。
        # 若误把 PLY opacity 当成 [0,1] 透明度，或把 scale 当真实半径，复现会出错。
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        # 每个命名属性存为 32 位浮点 f4，NumPy 结构化数组的一行对应一个高斯。
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # 分配 P 条记录，然后按属性声明的列序横向拼接所有数据。
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        # 普通二维数组转成逐行 tuple 后才能写入具有不同字段名的结构化数组。
        elements[:] = list(map(tuple, attributes))
        # 将结构化记录声明为 PLY 的 vertex 元素，再由 plyfile 写出文件。
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # 周期性压低不透明度，让后续优化重新分配贡献，减少早期不透明高斯的遮挡。
    def reset_opacity(self):
        # 这里执行的是 α_new=min(α_old,0.01)：已有更小 α 不会被抬高。
        # 先在真实不透明度空间截断，再用 logit 回到优化器存储空间。
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        # 替换参数同时重置该组 Adam 动量；只给 self._opacity 赋新值会让优化器继续更新旧对象。
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        # 从优化器返回的新对象重新绑定成员，保持渲染器与优化器引用同一参数。
        self._opacity = optimizable_tensors["opacity"]

    # 从已训练 PLY 恢复几何、外观；这不是恢复完整训练 checkpoint 的路径。
    def load_ply(self, path, use_train_test_exp = False):
        # 读取 vertex 的命名字段，后续不依赖文件中属性原本出现的顺序。
        plydata = PlyData.read(path)
        # 仅 train_test_exp 模式尝试额外加载曝光；普通 PLY 本身没有每图曝光字段。
        if use_train_test_exp:
            # 路径一般为模型目录/point_cloud/iteration_x/point_cloud.ply，向上两级找 exposure.json。
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                # JSON 是图像名→3×4 数值列表，先按普通 Python 字典读取。
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                # 各矩阵转到 CUDA，requires_grad=False：评估时采用训练保存的固定校正。
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                # 找不到文件时记为 None；此函数本身并不创建可训练 _exposure 或 exposure_mapping。
                # 若另写恢复训练流程，需特别检查曝光初始化，不能把 load_ply 等同 create_from_pcd。
                self.pretrained_exposures = None

        # 分别读取 x/y/z 一维列，stack(axis=1) 合成 [P,3] 世界坐标。
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        # 原始 logit 列 [P] 增加末尾维，恢复为 [P,1]。
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # 先重建保存前的 [P,3,1] 布局，再分别填入 RGB 零阶系数。
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # 筛选所有高阶 SH 字段；按数字后缀排序，避免字符串排序将 10 排在 2 前面。
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        # 当前配置的最大阶数必须与 PLY 高阶系数数量吻合；相差阶数不能直接加载。
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        # 先读取为二维 [P,3(K-1)]，保持和 save_ply 的展平顺序一致。
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # 恢复 [P,RGB,K-1]；不是直接 reshape 成 [P,K-1,RGB]，那会打乱颜色与系数。
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # 按数字顺序读取 scale_0..2，值已经在 log 空间，不能再次取 log。
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 同样读取 rot_0..3，保持 [w,x,y,z] 次序；使用时再进行归一化。
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 重新创建 float32 CUDA 叶子参数；PLY 加载不含任何旧优化器引用或动量。
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        # 从文件布局 [P,3,系数数] 转回内存布局 [P,系数数,3]，并确保连续。
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        # 不透明度、尺度、旋转均原样恢复原始参数，get_* 在渲染时负责激活。
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # 加载的是已训练模型，直接启用最大阶数，不再从零阶暖启动。
        self.active_sh_degree = self.max_sh_degree

    # 替换一个参数组的整个张量，并把该组 Adam 动量清零，主要用于 reset_opacity。
    # Adam 的 state 以参数对象本身为键，因此新建 Parameter 后必须迁移键。
    # 此上游实现假设目标组已有 Adam 状态；首次优化 step 之前调用可能得到 None。
    def replace_tensor_to_optimizer(self, tensor, name):
        # 结果按组名返回新参数，调用者据此更新模型成员。
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # 只操作目标名称对应的组，其他几何/外观组及曝光优化器不变。
            if group["name"] == name:
                # 取出旧对象关联的状态，包含 step、exp_avg、exp_avg_sq 等字段。
                stored_state = self.optimizer.state.get(group['params'][0], None)
                # 一阶动量与二阶动量都重置为新张量同形状的零；step 则沿用原状态。
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                # 先删旧键，再建立新叶子参数，最后把状态绑定到新键，避免遗留旧对象引用。
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                # 把优化器实际持有的新对象交还调用者，不能返回未包装的输入 tensor。
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # 按“保留掩码”压缩六组参数和对应 Adam 动量。
    # mask 为 [P] 布尔张量，True 表示保留；与外部 prune_points 的删除语义相反。
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # 某个组可能尚未产生梯度/执行 step，因此状态允许不存在。
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # 动量与参数共享第 0 维点编号，必须使用同一掩码裁剪两种动量。
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                # 裁剪产生新参数对象，迁移状态的键；step 等非逐点状态无需裁剪。
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                # 记录此组的新参数，供外层重新绑定 _xyz/_features_* 等成员。
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 无旧动量时只裁剪参数；Adam 会在后续 step 时为它初始化状态。
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # 公开的点删除操作：mask=True 表示要删掉这个高斯。
    def prune_points(self, mask):
        # 取反变成优化器需要的“保留”掩码；这两种语义混淆会删除相反的一组点。
        valid_points_mask = ~mask
        # 先原子式地更新所有主参数组与动量，再接回模型属性。
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # 六个成员必须全部引用替换后的参数，才能维持各属性相同的点数和编号。
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 训练统计不是 Parameter，优化器不会替我们处理；这里手动按同一顺序裁剪。
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        # 可见次数与最大屏幕半径也按保留索引同步更新。
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        # 临时半径在 densify_and_prune 中设置；此版本 prune_points 依赖它已存在。
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    # 为每个参数组追加新点：旧参数/旧动量保留，新点的动量从零开始。
    # 点数变化发生在第 0 维，其他维度（坐标、颜色系数等）必须保持一致。
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # 实现以“每组一个张量”作为前提；若以后拆成多个 Parameter，这套逻辑需重写。
            assert len(group["params"]) == 1
            # 按组名取出要追加的 [新增点数,...] 张量，不能把真实尺度误放进 scaling 组。
            extension_tensor = tensors_dict[group["name"]]
            # 取得旧参数对应的 Adam 状态，支持状态尚未初始化的情况。
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                # 原点的 exp_avg/exp_avg_sq 不变，新点追加零动量；不会复制父点的动量。
                # 标量 step 仍继承此组旧值，所以这是组级时间步下的新点，而非全新优化器。
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                # 删除旧参数键，用 cat 创建更长的叶子 Parameter，然后迁移扩容后的状态。
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                # 返回的是组内新对象，外层据此让渲染数据源与优化器重新一致。
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 如果旧组没有状态，直接拼接参数，等待 Adam 第一次更新时创建状态。
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # 克隆与分裂共享的收尾步骤：追加参数、对齐临时半径，并重置统计窗口。
    # 训练循环在 torch.no_grad() 中执行这些结构变更；它们是离散管理操作，
    # 并不会通过“选点/增删点”这一步反向传播，下一轮才对新的叶子参数求梯度。
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        # 字典键与 training_setup 的六个 name 严格对应，值都是原始参数空间的数据。
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        # 先扩容优化器，再把六个成员指向扩容后的参数对象。
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 临时屏幕半径跟随新点一起扩容，供随后的删除过程保持索引一致。
        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        # 注意这里把所有点的梯度累积和计数清零，并非只初始化新增点。
        # 每次调用都重新开始统计窗口，即使筛选出的新增点数为 0，也会清零。
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # 最大投影半径同样全部清零；阅读 densify_and_prune 后面的屏幕剪枝时需注意调用顺序。
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # 分裂针对“梯度大且体积尺度大”的高斯：每个父点采样 N 个更小子点，再删除父点。
    # 默认 N=2；若选中 M 个点，最终总点数净增加 (N-1)M。
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        # 此时可能刚执行过克隆，当前点数已大于计算 grads 时的点数。
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        # 把旧梯度补到当前点数；新克隆点补 0，正常正阈值下不会立刻再次分裂。
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        # 先筛选平均屏幕位置梯度足够大的点，意味着目前覆盖对图像误差仍敏感。
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # 再要求最大主轴标准差大于 percent_dense×场景半径，选择需要细化的大椭球。
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # 选中 M 点的尺度 [M,3] 重复 N 份，得到 [NM,3]；所有属性采用同样重复顺序。
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        # 在各父高斯局部主轴坐标系里，以零均值、对应主轴标准差生成偏移。
        means =torch.zeros((stds.size(0), 3),device="cuda")
        # 三个局部分量独立采样；此时尚未乘旋转，也尚未加世界中心。
        samples = torch.normal(mean=means, std=stds)
        # 父四元数转 [M,3,3] 旋转矩阵，再扩展到 [NM,3,3] 与采样一一对应。
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        # [NM,3,3]×[NM,3,1] 得到旋转后的偏移，加父中心后得到世界坐标子中心。
        # 这些随机偏移不沿梯度方向平移；梯度只负责决定哪些父点需要分裂。
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # 子标准差设为父标准差/(0.8N)，默认除以 1.6；再 log 写回原始尺度空间。
        # 这是上游的经验缩放规则，不是保证体积或总不透明度严格守恒的公式。
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        # 旋转、全部球谐系数和不透明度 logit 直接继承父点；没有按 N 平分不透明度。
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        # 临时半径也按同样索引复制，之后可随参数一起裁剪。
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        # 先追加全部 NM 子点；这个函数同时会重置全部点的稠密化统计。
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        # 构造“删除”掩码：[旧点中的被选父点=True，所有新子点=False]。
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # 删除父高斯并裁剪优化器动量，最终留下未选旧点与新子点。
        self.prune_points(prune_filter)

    # 克隆针对“梯度大但尺度小”的高斯：原点保留，另追加一份相同参数。
    # 源码没有按梯度方向移动克隆中心；刚克隆时两点重合，后续优化再调整。
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # 调用方传入 [P,1] 平均梯度范数，沿最后一维取 norm 后得到 [P] 判据。
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # 与分裂的大尺度条件互补：≤阈值的点走克隆，>阈值的点走分裂。
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # 用同一布尔索引取出六种原始属性，保持各新增点的位置/颜色/形状一一配对。
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # 克隆临时半径以维持逐点辅助数组长度和主参数一致。
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        # 只追加、不删除父点；新增参数独立存储，新增 Adam 动量为零。
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    # 一次自适应密度控制：平均梯度 → 小点克隆 → 大点分裂 → 清除无效/过大点。
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        # 按可见次数平均，避免仅因某点被更多训练视角看见就得到更大统计值。
        grads = self.xyz_gradient_accum / self.denom
        # 从未可见的点是 0/0→NaN，置 0 防止它们被梯度阈值误选。
        grads[grads.isnan()] = 0.0

        # 暂存当前视角光栅器返回的 [P] 像素半径，增删点时同步扩容/裁剪。
        self.tmp_radii = radii
        # 克隆先执行，会改变参数数量；grads 仍是调用前的长度，split 会自行补零。
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # α 小于阈值的点通常贡献很小，用实际 sigmoid 不透明度判定删除。
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        # max_screen_size 为真时额外启用尺寸筛选；None/0 时只按不透明度删除。
        if max_screen_size:
            # vs 指屏幕空间，判据单位是像素半径。
            # 注意本版本上面的 postfix 已把 max_radii2D 清零，所以不能宣称此处仍在使用
            # 本轮稠密化前累计的最大半径；这是阅读当前上游实现时必须追踪的状态细节。
            big_points_vs = self.max_radii2D > max_screen_size
            # ws 指世界空间，最大主轴标准差超过场景半径 10% 的高斯也进入删除集合。
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            # 逻辑或合并三个判据：满足任意一个就删，而不是必须同时满足。
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # 统一裁剪参数、动量与统计，避免不同数组的高斯编号失配。
        self.prune_points(prune_mask)
        # 此上游版本把临时半径赋给局部变量，但后续没有使用或返回该变量。
        tmp_radii = self.tmp_radii
        # 结束本轮后清除成员引用；下一次稠密化会重新从当前渲染结果赋值。
        self.tmp_radii = None

        # 释放 CUDA 缓存分配器中可回收的空闲块，不会删除仍被活跃张量引用的数据。
        torch.cuda.empty_cache()

    # 在 loss.backward() 后收集稠密化信号；渲染器的屏幕点张量必须保留梯度。
    # 当前渲染器传入 [K,1] 整数索引；本方法也能接收 [N] 布尔掩码。被选点仍可能因遮挡得到零梯度。
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # 取屏幕均值梯度的前两维 (x,y) 并求范数。当前 [K,1] 索引得到 [K,1,2]，范数为 [K,1,1]，
        # 与左侧高级索引结果匹配；若改用一维布尔掩码，则对应形状为 [K,2] 与 [K,1]。
        # 先逐视角取范数再累加，可避免不同视角梯度方向互相抵消；不是累加三维位置梯度。
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # 每个本轮有效点计数加 1，之后与梯度累积相除得到跨视角平均强度。
        self.denom[update_filter] += 1
