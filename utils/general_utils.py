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

# 中文导读：通用数值工具把优化参数转成有效几何量，并提供图像转换、学习率和随机状态。
# 其中若干函数显式在 CUDA 上分配张量，单独在 CPU 调用前必须理解这个原始设备约束。
import torch
import sys
from datetime import datetime
import numpy as np
import random

# Sigmoid 的反函数 logit：把 (0,1) 内的有效不透明度映射回可无约束优化的实数参数。
# 公式不做截断；输入 0/1 会产生无穷，越界会产生无效值，调用方应保证输入合法。
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

# 把 PIL 图像转换成浮点 [C,H,W] 张量；resolution 按 PIL 约定传入（宽，高）。
def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    # PIL 转 NumPy 后通常为 uint8 的 [H,W,C]；除以 255 将 8 位颜色映射到 [0,1]。
    # 这里沿用 PIL 默认 resize 行为，且仍是 CPU 张量，设备迁移交给调用方。
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    # 彩色/含透明通道图像把通道维从末尾挪到最前；不丢弃透明通道。
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        # 灰度 [H,W] 先补单通道得到 [H,W,1]，再变成 [1,H,W]，统一后续处理接口。
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

# 返回一个“输入训练步数、输出学习率”的闭包，不在这里修改优化器。
# 核心是对数空间线性插值：lr(t)=lr_init^(1-t) × lr_final^t。
# 可选延迟因子从 lr_delay_mult 平滑上升到 1；它与主指数衰减相乘。
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    # 内部函数捕获外层的起止学习率和调度参数，调用方每轮训练传入当前 step。
    def helper(step):
        # 负步数或起止都为零时直接停用更新；普通指数插值要求起止值为正。
        # 只把一个端点设成零并不会进入本分支，后面的 log(0) 可能造成无穷或 NaN。
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        # 只有显式设置正的 lr_delay_steps 才启用预热/延迟；仅设置 lr_delay_mult 不会生效。
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            # step=0 时因子为 lr_delay_mult；达到延迟步数时为 1；clip 让之后一直保持 1。
            # 正弦的四分之一周期提供平滑上升，避免学习率因子在预热结束时突变。
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        # 将进度截到 [0,1]，所以超过 max_steps 后主学习率保持 lr_final。max_steps 应大于零。
        t = np.clip(step / max_steps, 0, 1)
        # 几何插值使学习率按固定比例衰减，而不是每步减去固定常数。
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        # 最终学习率 = 延迟因子 × 指数调度值；返回标量供 optimizer.param_groups 使用。
        return delay_rate * log_lerp

    # 注意这里返回函数本身而非 helper(0) 的结果，因此训练阶段可以反复查询不同步数。
    return helper

# 把一批对称 3×3 协方差压成 [N,6]，减少传给光栅化器的数据量。
# 原函数名虽含 lowerdiag，实际读取的是上三角，顺序为 xx、xy、xz、yy、yz、zz。
def strip_lowerdiag(L):
    # 输出固定为当前 CUDA 设备的 float32；这里没有自动继承 L 的 device/dtype。
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    # 对称矩阵另外三个非对角元素由这些值唯一确定，不需要重复保存；本函数不检查对称性。
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

# 语义更清晰的别名，明确调用处要打包的是对称矩阵，内部仍使用同一存储顺序。
def strip_symmetric(sym):
    return strip_lowerdiag(sym)

# 把 [N,4] 四元数转换成 [N,3,3] 旋转矩阵；分量顺序为 (w,x,y,z)，不是 (x,y,z,w)。
# 高斯的四元数参数直接优化，因此每次使用前重新单位化。
def build_rotation(r):
    # 按每个高斯独立计算四元数的欧氏范数，输出 [N]。原代码不为零范数增加 epsilon。
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    # 用 [N,1] 广播归一化每行；单位四元数对应正交旋转，避免参数长度污染尺度。
    q = r / norm[:, None]

    # 批量构造矩阵；该分配固定在 CUDA，通常配合高斯模型中的 float32 参数使用。
    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    # 这里局部变量 r 被重用为实部 w；后续公式里的 r*z 指 w*z，不再表示整个输入四元数。
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    # 标准单位四元数到旋转矩阵公式：三行分别描述旋转后的坐标分量。
    # 例如单位四元数 (1,0,0,0) 给出单位矩阵，q 和 -q 给出相同旋转。
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    # 非对角项中的 ±w 分量决定旋转方向；交换四元数分量顺序会改变整个几何结果。
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    # 返回的 R 供列向量形式的 R @ v 使用；后续将局部椭球主轴旋转到世界坐标。
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

# 由三轴正尺度 s:[N,3] 和四元数 r:[N,4] 构造 L=R diag(s)。
# 调用方再计算 Σ=L Lᵀ=R diag(s²) Rᵀ，得到半正定协方差。
# s 表示主轴标准差（长度），s² 才是主轴方差；此函数本身不执行 exp 激活或检查正值。
def build_scaling_rotation(s, r):
    # 先分配批量对角尺度矩阵；非对角元素为零，三个尺度允许各向异性椭球。
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    # 先把可优化四元数归一化再变成旋转矩阵，使旋转与三个尺度参数各司其职。
    R = build_rotation(r)

    # 填入局部 x/y/z 三个主轴的尺度；这些轴随后由 R 的三列旋转到世界空间。
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    # 矩阵次序很重要：列向量先由 diag(s) 缩放，再由 R 旋转；交换顺序会改变协方差方向。
    L = R @ L
    return L

# 设置训练日志和基础随机种子，方便复现实验；固定种子不等于保证全部 CUDA 算子逐位确定。
def safe_state(silent):
    # 保存原输出流，包装器继续把文本写到它；避免在新 stdout 内递归调用自己。
    old_f = sys.stdout
    # 这个轻量输出代理实现 write/flush；silent=True 时丢弃 write 内容。
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                # 若本次写入以换行结尾，就给其中的换行附上本机时间；进度条式无换行文本原样转发。
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        # 让外层进度条/print(flush=True) 仍可及时刷新原输出流，避免日志长时间滞留缓冲区。
        def flush(self):
            old_f.flush()

    # 替换进程级 stdout，会影响之后整个进程的普通 print，而不只影响当前函数。
    sys.stdout = F(silent)

    # 分别固定 Python、NumPy、PyTorch 随机源；不同模块可能使用不同来源。
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # 显式选择当前进程可见的第 0 张 CUDA 卡；它受 CUDA_VISIBLE_DEVICES 映射影响。
    # 这一步要求可用的 CUDA 环境，因此 safe_state 不是通用的纯 CPU 初始化函数。
    torch.cuda.set_device(torch.device("cuda:0"))
