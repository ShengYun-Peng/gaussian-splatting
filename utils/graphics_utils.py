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

# 中文导读：本文件集中放置点云数据结构、齐次坐标变换、相机外参和透视投影。
# 注意区分数学上的列向量矩阵与渲染调用中转置后使用的行向量矩阵。
import torch
import math
import numpy as np
from typing import NamedTuple

# 轻量只读字段容器：三个数组通常都是 [N, 3]。points 是世界坐标，colors 是 [0,1] RGB。
# normals 为法线；高斯初始化主要使用位置和颜色，不要求这里的法线已估计。
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

# 把 [P,3] 点集变换到另一坐标系。这里使用行向量右乘矩阵；
# 若手中是常见列向量矩阵 M，应按调用方约定传入 Mᵀ。该函数也允许带透视项的齐次变换。
def geom_transform_points(points, transf_matrix):
    # P 是点的个数；沿用输入 dtype 和 device，避免拼接时出现精度或 CPU/GPU 不匹配。
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    # 补齐齐次分量 1，得到 [P,4]，使旋转和平移可以通过同一次矩阵乘法完成。
    points_hom = torch.cat([points, ones], dim=1)
    # 矩阵增加一个批次维后，广播乘法输出 [1,P,4]；前三项尚未除以齐次坐标 w。
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    # 透视除法：xyz 除以 w。很小的常数减少 w=0 时的直接除零；它不是近裁剪或可见性测试。
    # 仿射矩阵通常给出 w=1；最后 squeeze 去掉仅由广播引入的批次维。
    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

# 构造列向量约定的世界到相机外参：x_camera = Rᵀ x_world + t。
# 输入 R 已按本项目约定转置存储；输入 t 不是相机在世界中的位置。
def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    # 最后一行补成 [0,0,0,1]，将三维刚体变换写成 4×4 齐次矩阵。
    Rt[3, 3] = 1.0
    return np.float32(Rt)

# 在外参构造基础上，可对世界中的相机中心做平移与缩放，再返回新的世界到相机矩阵。
# 默认 translate=0、scale=1 与普通外参一致；这不是自动把所有场景点一起归一化。
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    # 最后一行补成 [0,0,0,1]，将三维刚体变换写成 4×4 齐次矩阵。
    Rt[3, 3] = 1.0

    # 先求逆到相机到世界形式，才能直接读取/修改世界坐标中的相机中心。
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    # 操作顺序是“先平移，再整体缩放”；旋转部分保持原值。
    # 若把这作为整个场景的坐标重参数化，场景几何也需由调用方采用相同约定。
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    # 修改后的 C2W 再求逆回渲染需要的 W2C；输出固定为 float32。
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

# 构造对称针孔相机的透视投影矩阵；FoV 用弧度，近平面/远平面沿相机正 z 方向。
# 返回矩阵先按列向量书写；Camera 随后会转置它以匹配渲染端存储和乘法约定。
def getProjectionMatrix(znear, zfar, fovX, fovY):
    # 相似三角形：近裁剪面半高/近平面距离 = tan(垂直视场角/2)，半宽同理。
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    # 假设主点位于图像中心，视锥在 x/y 两个方向对称，所以 bottom=-top、left=-right。
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # 此处先创建 CPU 上的矩阵；其余元素保持为零，调用方负责迁移到渲染设备。
    P = torch.zeros(4, 4)

    # 正 z 朝前：裁剪坐标 w=z。不能直接套用 OpenGL 中相机看向负 z 的投影符号。
    z_sign = 1.0

    # 对称情况下，两个对角元素分别化简为 1/tan(fovX/2)、1/tan(fovY/2)。
    # 后续除以 w=z 后，把视锥左右/上下边缘映射到标准化坐标 ±1。
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    # 主点偏移项在当前对称视锥中为零；保留一般写法便于看清投影矩阵结构。
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    # w_clip=z_camera；z 项在除以 w 后将 znear 映到 0、zfar 映到 1。
    # 深度映射包含 1/z，因此标准化深度并不与实际距离线性对应。
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

# 视场角转像素焦距：f = 图像边长 / [2 tan(FoV/2)]。
# 例如处理水平方向就传宽度，处理垂直方向就传高度；这里不是毫米焦距。
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

# 上式的逆运算，返回弧度。相同像素焦距下，图像边长越大，对应视场角越大。
def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))