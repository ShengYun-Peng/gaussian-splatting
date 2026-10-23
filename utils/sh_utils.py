#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

# 中文导读：球谐（SH）用一组球面上的固定基函数表示随观察方向变化的颜色。
# 每个高斯、每个颜色通道保存自己的系数，渲染时计算“系数 × 对应方向的基函数”之和。
# 本文件采用实球谐的特定符号/排列约定，必须与 CUDA 光栅化器保持一致。
import torch

# 零阶常数 Y₀⁰=1/√(4π)。它与方向无关，因此零阶系数控制高斯的基础颜色。
C0 = 0.28209479177387814
# 一阶归一化因子 √(3/(4π))。下方基函数按 [-C1*y, C1*z, -C1*x] 排列。
# 这些负号是约定的一部分，不应为了“看起来对称”而改变。
C1 = 0.4886025119029199
# 二阶有 5 个基函数，对应系数索引 4～8；常数已吸收归一化因子和本实现的符号。
# 与标准球面角形式等价，这里直接使用单位方向 (x,y,z) 的二次多项式。
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
# 三阶有 7 个基函数，对应索引 9～15。累计到 l 阶共有 Σ(2k+1)=(l+1)² 个系数。
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
# 四阶有 9 个基函数，对应索引 16～24；当前 Python 实现实际支持到四阶。
# 默认训练上限通常为三阶，是否启用更高阶还应核对所用 CUDA 光栅化器的支持情况。
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


# 输入 sh:[...,C,K]，通常 C=3、K=(最大阶数+1)²；dirs:[...,3] 为单位观察方向。
# 函数只使用前 (deg+1)² 个系数，输出 [...,C]，可以保留多维批次前缀。
# 本函数不归一化 dirs；调用方必须先除以方向长度，否则多项式不再表示正确的球面基函数。
# 原英文说明中的“0-3”是旧描述，以后面的断言和四阶分支为准。
def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    # 限制可求值的阶数，并检查系数末维足够长；这里没有额外检查方向长度是否为 1。
    assert deg <= 4 and deg >= 0
    # 例如 deg=0/1/2/3/4 分别需要 1/4/9/16/25 个系数；“阶数”不是系数个数。
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    # 先求零阶项，对所有方向相同。sh[...,0] 取系数维第零项，保留颜色通道维 C。
    result = C0 * sh[..., 0]
    # 只在当前激活阶数允许时累加更高阶；训练可以逐渐提高 deg 而不改变系数存储大小。
    if deg > 0:
        # 切片 0:1 保留末尾单维 [...,1]，让同一方向基函数广播到 C 个颜色通道。
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        # 一阶三个方向项按索引 1/2/3 对应 -y、z、-x；它们允许近似随方向线性变化的颜色。
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        # 二阶从索引 4 开始。先复用平方/交叉乘积，避免为每个颜色通道反复计算相同方向量。
        if deg > 1:
            # 方向是单位向量时 xx+yy+zz=1，可把标准球谐表达式改写为这些笛卡尔多项式。
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            # 二阶依次累加 xy、yz、(2z²-x²-y²)、xz、(x²-y²)，共 5 个独立角向变化模式。
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            # 三阶索引 9～15，使用三次多项式刻画更细的角向变化；仍然对各颜色通道独立线性组合。
            if deg > 2:
                # 这 7 项依次对应本实现的三阶实球谐基顺序，加载/保存系数时必须保持相同顺序。
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                # 四阶索引 16～24。包含常数 1、3 的表达式已利用单位球约束，不适用于任意长度方向。
                if deg > 3:
                    # 逐项累加最后 9 个四阶基；保持表达式原样便于与同约定的 SH 实现逐项核对。
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    # 返回原始 SH 重建值，不加 0.5、不裁剪到 [0,1]。实际颜色偏置与限幅由渲染调用方处理。
    return result

# 把 RGB 初始颜色转换成零阶系数，令 C0×系数+0.5 恢复原 RGB。
# 这里只生成 DC（零阶）项；不能把该公式当成完整的高阶球谐拟合。
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

# RGB2SH 的逆变换，同样只针对零阶颜色映射；不含 gamma 转换，也不自动限幅。
# 输入系数为零时输出 0.5，因此零系数对应中灰而非黑色。
def SH2RGB(sh):
    return sh * C0 + 0.5