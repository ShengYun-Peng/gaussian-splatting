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

# 中文导读：相机同时提供监督数据（RGB/逆深度）与世界点投影到图像所需的矩阵。
# 注意三种容易混淆的顺序：图像张量为 [通道,高,宽]，resolution 为 (宽,高)，
# 位姿辅助函数先产生列向量矩阵，存进 Camera 时又转置为渲染器使用的布局。
import torch
from torch import nn
import numpy as np
# getWorld2View2 建立世界→相机变换；getProjectionMatrix 建立相机→裁剪空间投影。
# PILtoTorch 负责缩放图像并转为 [C,H,W] 浮点张量；cv2 用于逆深度缩放。
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

# 完整训练相机：继承 nn.Module，但这里保存的位姿和图像通常是固定数据，
# 不是 nn.Parameter；训练优化的是高斯及曝光，不会自动优化这些相机外参。
class Camera(nn.Module):
    # R 是“世界到相机旋转矩阵的转置”，T 是世界到相机的平移，T 不是相机中心。
    # FoVx/FoVy 单位为弧度；trans/scale 用于调整世界坐标中的相机中心。
    # is_test_view 描述该照片是否属于测试视角；is_test_dataset 描述当前对象来自哪套列表。
    # train_test_exp 时，同一测试照片可分别出现在训练和测试列表，使用不同的半幅掩码。
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        # 先完成 nn.Module 的初始化，再给相机写入成员。
        super(Camera, self).__init__()

        # uid 是当前相机列表的连续编号；colmap_id 保留读取器传来的原始元数据编号。
        self.uid = uid
        self.colmap_id = colmap_id
        # 保存原始 R/T 以供元数据导出；实际渲染使用下面构造的 4×4 矩阵。
        self.R = R
        self.T = T
        # 水平与垂直视场角决定投影缩放；图像缩放会改变像素焦距，但这里 FoV 保持不变。
        self.FoVx = FoVx
        self.FoVy = FoVy
        # 图像名同时是输出文件标识，以及 GaussianModel 查找曝光矩阵的键。
        self.image_name = image_name

        # data_device 控制监督图像、透明度掩码、深度数据的存放位置。
        try:
            # torch.device 解析设备字符串；成功解析并不证明 CUDA 实际可用或显存足够。
            self.data_device = torch.device(data_device)
        except Exception as e:
            # 解析异常时打印原因并使用 cuda；后面的实际数据迁移仍可能因设备不可用而失败。
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # 按 (宽,高) 缩放 PIL 图像，除以 255 后转成 [C,H,W]；RGB 通常为 C=3，RGBA 为 C=4。
        resized_image_rgb = PILtoTorch(image, resolution)
        # 只取前三通道作为监督颜色，alpha 独立处理，不混入颜色通道。
        gt_image = resized_image_rgb[:3, ...]
        # 成员先设 None，随后两个分支都会建立 [1,H,W] 掩码。
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            # 3:4 而非单索引 3：保留通道维，以便和 [3,H,W] 渲染结果广播相乘。
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            # 没有 alpha 通道时默认全图有效；ones_like 同时继承形状、类型和目标设备。
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        # 曝光评估协议下，仅测试视角启用左右半幅分离；普通训练照片不受此分支影响。
        if train_test_exp and is_test_view:
            if is_test_dataset:
                # 测试集合实例把左半幅设 0，保留右半幅用于评估；分界取宽度整除 2。
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                # 训练集合实例把右半幅设 0，保留左半幅；同一照片的两种实例使用互补区域。
                # 这里仅设置 alpha_mask，RGB 本身没有裁剪；掩码具体如何参与损失需沿 train.py 继续阅读。
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        # 监督颜色裁到 [0,1] 并移到 data_device；名称 original_image 指缩放后的训练参考图。
        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        # [C,H,W] 的第 2 维是宽，第 1 维是高，不可沿用 PIL 的下标理解。
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # 没有深度图时保持 None/False，训练循环就不会为此视图添加逆深度监督。
        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            # 深度掩码用相同尺寸初始化为全 1；ones_like 不复制 alpha_mask 的 0/1 数值。
            # 所以不能假设这个 depth_mask 自动继承 RGBA 或半幅图像掩码。
            self.depth_mask = torch.ones_like(self.alpha_mask)
            # cv2.resize 同样接收 (宽,高)；把逆深度监督对齐到 RGB 分辨率。
            # 值表示逆深度而非米制深度，因此这里不是在缩放相机世界坐标距离。
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            # 将输入中的负逆深度置 0；后面的仿射校正可能再次改变数值范围。
            self.invdepthmap[self.invdepthmap < 0] = 0
            # 有输入时先假定可靠，再依据深度尺度校准结果进行过滤。
            self.depth_reliable = True

            # depth_params 提供单目逆深度与 SfM 逆深度对齐的 scale、offset 和全场景 med_scale。
            if depth_params is not None:
                # 若该图的拟合尺度离全场景中位数太远（小于 0.2 倍或大于 5 倍），判为不可靠。
                # 这是一张图级别的可信度筛选，不是逐像素误差阈值。
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    # 训练循环可据此跳过该相机的深度项，并把深度掩码清零。
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                # 仅在尺度为正时应用 d_inv_aligned=scale·d_inv+offset。
                # 此处即使已判不可靠也可能继续计算校正，是否参与损失由可靠标记/掩码决定。
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            # 如果读入的逆深度带额外通道，只取首通道，统一成二维 [H,W]。
            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            # [None] 加通道维得到 [1,H,W]；from_numpy 先共享 CPU 数组，再迁移到目标设备。
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        # 下面的近远平面用于构造投影矩阵，单位与相机坐标一致。
        # 光栅器内部还可能有独立的可见性/近面判断，不能仅凭这两个值推断全部裁剪行为。
        self.zfar = 100.0
        self.znear = 0.01

        # 记录额外相机中心变换：辅助函数会先求 C2W，再执行 C←(C+trans)·scale。
        self.trans = trans
        self.scale = scale

        # 齐次外参 W 的前三行为 [Rᵀ|T]，末行为 [0,0,0,1]，按列向量计算 x_cam=W·x_world。
        # 辅助函数先按 trans/scale 校正相机中心，再返回相应 W。
        # 这里存 Wᵀ，所以以行向量理解时是 x_world_row @ Wᵀ。
        # 特别注意：矩阵末尾直接 .cuda()，即使 data_device='cpu'，渲染矩阵仍在 CUDA 上。
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # 投影矩阵同样转置为 Pᵀ；构造函数采用相机前方 z>0、齐次 w=z 的约定。
        # 透视除法后 x/y 才成为标准化设备坐标；这里没有执行该除法或像素坐标换算。
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # 行向量依次经过 Wᵀ、Pᵀ，因此合成矩阵为 WᵀPᵀ=(PW)ᵀ。
        # unsqueeze(0) 只增加大小为 1 的批量维以使用 bmm，结果再 squeeze 回 [4,4]。
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # (Wᵀ)⁻¹=(W⁻¹)ᵀ，因此世界坐标相机中心在逆矩阵下标为 3 的第四行前三列。
        # 若直接读取 T 会得到世界到相机的平移，通常不是相机在世界中的位置。
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
# 轻量相机用于交互查看等场景：只含渲染需要的尺寸/视场/矩阵，不携带 RGB 或深度监督。
class MiniCam:
    # 调用方已经准备好与完整 Camera 相同转置约定、设备和坐标系的矩阵；本类不再转换。
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        # 图像尺寸供光栅器分配输出；宽高与 FoV 共同决定投影到像素后的尺度。
        self.image_width = width
        self.image_height = height    
        # 按传入的垂直/水平顺序保存，注意参数列表 fovy 在前、fovx 在后。
        self.FoVy = fovy
        self.FoVx = fovx
        # 记录视锥近远平面元数据；已经给定的 full_proj_transform 不会因此重新计算。
        self.znear = znear
        self.zfar = zfar
        # 保存世界→相机矩阵以及世界→裁剪空间合成矩阵，布局与 Camera 一致。
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        # 转置约定下，逆世界到相机矩阵的最后一行给出世界空间相机中心。
        view_inv = torch.inverse(self.world_view_transform)
        # [3][:3] 与完整 Camera 中 [3,:3] 取相同三个分量，供视角相关球谐颜色计算。
        self.camera_center = view_inv[3][:3]

