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

# 中文导读：训练用像素 L1 和结构相似度 SSIM 联合约束渲染。
# L1 更关注逐像素偏差，SSIM 比较局部亮度、对比度与结构；此文件返回的是 SSIM 相似度本身。
# train.py 中用 (1-λ)×L1 + λ×(1-SSIM) 转成需要最小化的损失。
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
# 可选导入光栅化扩展提供的融合 SSIM 前向/反向；普通 PyTorch 实现不依赖这些符号。
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
# 导入失败时仅忽略异常，未定义的 fusedssim 不会自动获得替代实现。
# 因此只有扩展确实可用时才可调用下方 fast_ssim；普通 ssim 仍可使用。
except:
    pass

# 输入动态范围按 1 处理（典型 RGB 为 [0,1]），C1=(0.01×1)²、C2=(0.03×1)²。
# 小常数稳定均值/方差接近零时的分母；它们不是可学习参数。
C1 = 0.01 ** 2
C2 = 0.03 ** 2

# 自定义自动微分桥梁：前向由扩展返回逐位置 SSIM 图，反向由配套内核传播渲染图梯度。
class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    # ctx 用来保存反向需要的常数和张量；img1 通常是可学习渲染结果，img2 是固定真值图。
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        # 保存反向计算所需数值；detach 不取消自定义 backward 对 img1 的返回梯度。
        # 它只避免保存的 img1 携带原计算历史，不能据此推断该实现支持高阶导数。
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    # opt_grad 是上游损失对 SSIM 图的梯度，与前向输出形状对应；反向内核再传回 img1。
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        # 返回位置必须对应 forward 的四个输入：C1、C2、img1、img2。
        # 只有 img1 获得梯度；常数和真值图返回 None，这不是对两张图都求导的通用接口。
        return None, None, grad, None

# 逐元素绝对误差再对全部维度求均值，通常包括颜色、高、宽（以及存在时的批次维）。
# 没有额外按有效像素数归一化；若输入已乘掩码，零值区域依然参与 mean 的分母。
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

# 返回均方误差 MSE，而非误差向量的欧氏长度；平方后对全部元素平均。
def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

# 生成一维离散高斯权重，中心索引为 window_size//2，默认 SSIM 会使用奇数窗口。
def gaussian(window_size, sigma):
    # 对每个窗口位置计算 exp(-距离²/(2σ²))；无需额外连续高斯常数，因为下一行会重新归一化。
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    # 使有限窗口内权重总和为 1，从而卷积表示局部加权平均，而不是随窗口大小改变亮度。
    return gauss / gauss.sum()

# 为每个颜色通道构造独立且相同的二维高斯卷积核，不在不同通道之间混合颜色。
def create_window(window_size, channel):
    # 固定 σ=1.5，生成 [K,1] 列向量；默认 K=11。
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    # 列向量与行向量外积得到可分离的 [K,K] 二维核，再补成 conv2d 所需 [1,1,K,K]。
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    # 扩展成 [C,1,K,K]；配合 groups=C，每组只卷积一个输入通道。
    # contiguous 实体化连续存储，Variable 是旧版 PyTorch 风格封装，默认不要求核梯度。
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

# 普通 PyTorch SSIM 入口，常见输入为 [N,C,H,W]；现代 conv2d 也接受无批次 [C,H,W]。
# 此处依赖输入形状相容且处于相同设备；函数不会自动把真值搬到渲染设备。
def ssim(img1, img2, window_size=11, size_average=True):
    # 从倒数第三维取通道数，兼容带/不带批次的图像；不是简单把第 0 维一律当作通道。
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    # 把新建的 CPU 高斯核移动到 img1 的 CUDA 卡，再把 dtype 转为 img1 的类型。
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # 窗口建好后进入实际公式；这个实现每次调用都会重建窗口，没有缓存。
    return _ssim(img1, img2, window, window_size, channel, size_average)

# 以同一高斯窗口估计每个像素附近的均值、方差和协方差，再得到局部 SSIM 图。
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # μ=G*I。分组卷积在各通道独立处理；奇数核配合 K//2 零填充可保持空间尺寸。
    # 边缘采用零填充，边界统计会受图像外侧的补零影响，并未重新归一化截断窗口。
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    # 缓存 μ₁²、μ₂²、μ₁μ₂，后面用于二阶中心矩及亮度相似度。
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 方差 σ²=E[I²]-E[I]²，协方差 σ₁₂=E[I₁I₂]-E[I₁]E[I₂]；E 都是高斯加权平均。
    # 这不是除以样本数减一的无偏统计估计，浮点相消也可能产生很小的负方差。
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # 局部函数使用同样的动态范围 1 稳定常数；如果改用 0～255 图像，需要相应调整公式。
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 第一比值比较局部均值（亮度），第二比值合并对比度和结构相似度。
    # 相同图像通常得到 1；SSIM 可能为负，因此不要把输出理解成严格 [0,1] 的概率。
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 默认把批次、通道和空间位置全部平均成单个标量，训练时再以 1-SSIM 作为惩罚。
    if size_average:
        return ssim_map.mean()
    else:
        # 此分支假定 SSIM 图是 [N,C,H,W]：依次消去 C、H、W，返回每个样本的 [N] 分数。
        # 若使用 [C,H,W] 无批次输入，三次 mean(1) 并不适用；调用方需按接口形状传入。
        return ssim_map.mean(1).mean(1).mean(1)


# 融合内核版本先返回 SSIM 图再全局平均；可用性与导入的扩展实现有关，不能仅凭函数名保证可调用。
def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()
