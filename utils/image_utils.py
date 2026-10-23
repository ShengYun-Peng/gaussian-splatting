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

# 中文导读：评估中的均方误差和峰值信噪比（PSNR）。
# 本文件把输入第 0 维视为要独立保留的维度，其含义取决于调用方传入的张量布局。
import torch

# 先平方误差，再把第 0 维之外的所有维度展平后求均值；输出 [img1.shape[0],1]。
# 输入 [N,C,H,W] 时得到每张图的 MSE；输入 [C,H,W] 时得到每个通道的 MSE。
# view 要求张量布局兼容；函数不检查形状匹配、不截断输入，也不添加掩码。
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

# PSNR=10 log₁₀(MAX²/MSE)=20 log₁₀(MAX/√MSE)，数值越大表示像素误差越小。
# 这里 MAX 固定为 1，因此默认输入已按 [0,1] 图像范围表示；输出单位为 dB。
def psnr(img1, img2):
    # 按第 0 维分别计算 MSE，保留 [N,1] 或 [C,1] 的输出形状。
    # 若调用方传 [C,H,W] 再对 PSNR 求均值，得到的是各通道 dB 的平均，
    # 通常不同于先把所有通道误差平均再计算一次 PSNR，因为对数不是线性运算。
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    # 完全一致的图像令 MSE=0，原公式会返回 +∞；这里没有通过 epsilon 人为设定分数上限。
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
