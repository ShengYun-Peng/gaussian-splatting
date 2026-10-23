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

// 中文精读：此文件把图像损失对颜色/逆深度的梯度，逆向传回每个高斯的可学习属性。
// 主要路径：像素损失 → α/颜色/二维中心/逆协方差 → 三维中心/协方差 → 尺度/旋转/球谐。
// 反向复用前向确定的覆盖、排序和阈值路径，不对这些离散决策自身求导。
// 阅读时 dL_dx 表示标量总损失 L 对 x 的导数，而不是 x 的前向数值。
#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ float sq(float x) { return x * x; }


// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
// 球谐反向包含两条链：颜色对系数的导数，以及颜色对观察方向再对三维中心的导数。
// 每个基函数 Y_k(d) 对其系数的导数就是 Y_k(d)，因此系数梯度很直接。
// 方向梯度需要逐项对 x/y/z 的多项式求导，再通过单位向量归一化反传。
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	// 保留归一化前的方向与长度；只缓存单位方向不足以求回对原始位置的正确梯度。
	// 相机位置在这个接口中视为固定量，所以这里只向高斯中心累积导数。
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	// 前向颜色 max(c,0) 的被截断通道在这里清零梯度。
	// 掩码来自同一次前向缓存；未截断通道继续传播，结果恰好为零时采用源码指定的未截断分支。
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	// 三个 vec3 分别存 RGB 三通道对方向分量 x/y/z 的导数。
	// 它们不是三个颜色值，而是颜色函数的三个偏导向量；最终与上游 RGB 梯度点乘。
	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	// 零阶颜色与方向无关，只影响系数梯度，不贡献位置梯度。
	// 同一个标量基值分别乘 R/G/B 的上游梯度，得到这个系数的三个可学习分量梯度。
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		// 一阶基函数顺序继续与前向一致：(-y,z,-x)。
		// 系数梯度使用基值本身；方向导数则取出对应系数，并乘基函数对该方向分量的导数。
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			// 二阶五个系数的导数仍只是各自基值，和当前系数数值无关。
			// 例如 Y_4=C·xy，则 ∂Y_4/∂x=C·y、∂Y_4/∂y=C·x，下面 dRGBdx/dy 正在累加这类项。
			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			// 多个基函数都依赖同一方向分量，因此使用 += 累加所有路径。
			// 注意 2z² 项对 z 的导数是 4z；系数的符号和乘二关系必须逐项与前向核对。
			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				// 三阶同理，先求七个基值以写系数梯度，再对七个三次多项式分别求方向偏导。
				// 这段长公式没有额外数值算法，主要是乘积法则；可用有限差分抽查任一方向分量。
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				// 例如 y(3x²-y²) 对 x 的导数为 6xy；x(4z²-x²-y²) 对 x 为 4z²-3x²-y²。
				// 后面的 dRGBdy/dz 是同一组基函数对其他方向的偏导，每项都乘其 RGB 系数。
				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	// 链式法则：dL/dx = Σ通道 (dL/dRGB通道)*(dRGB通道/dx)，y/z 同理。
	// 点乘把三通道影响汇总成一个方向分量梯度，再交给单位化的雅可比。
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	// 单位化的导数为 (I/||v||-vvᵀ/||v||³)；它去掉径向分量并按距离缩放。
	// 因此改变高斯中心不仅会移动屏幕足迹，也会改变视角相关颜色，两条路径都应贡献位置梯度。
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	// 这里必须累加：同一三维中心已有协方差、投影位置、可能的逆深度梯度。
	// 此核按高斯并行，一个线程处理一个高斯，因此这一步不需要 atomicAdd。
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
// 第一阶段预处理反向：把逆二维协方差的梯度变为原二维协方差梯度，
// 再传到三维协方差与中心；中心影响投影雅可比，所以即使只改变椭圆形状也有位置梯度。
// 这段单独发一个核，后面的 preprocessCUDA 再补齐屏幕中心和球谐等其他路径。
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* opacities,
	const float* dL_dconics,
	float* dL_dopacity,
	const float* dL_dinvdepth,
	float3* dL_dmeans,
	float* dL_dcov,
	bool antialiasing)
{
	auto idx = cg::this_grid().thread_rank();
	// 只处理前向实际保留的高斯；半径为零的项没有参与像素贡献，保留初始零梯度。
	// 半径本身是离散的支持范围标志，此处不计算「半径如何随参数变化」的导数。
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	// 反向渲染把 2×2 对称逆协方差梯度存到四槽中的 x/y/w，即索引 0/1/3。
	// y 保存按单个对称矩阵位置计算的非对角梯度，后面的公式会显式处理两份对称贡献。
	// 不能只看 float3 就直接与前向 conic 的三参数梯度逐元素等同。
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	// 与前向的视角范围截断相配，越界时对应 x 或 y 的梯度乘零。
	// 这是源码采用的分段导数；在精确边界上不可微，代码选择内部侧的值。
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	// 重新计算 W、J、Σ 及其投影，减少前向必须保留的中间张量数量。
	// 重新计算的矩阵约定要与 forward.cu 完全一致，尤其 GLM 构造器是按列填充。
	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float c_xx = cov2D[0][0];
	float c_xy = cov2D[0][1];
	float c_yy = cov2D[1][1];
	
	// 前向不论是否开启 antialiasing 都给对角线加 0.3，这里必须重复同样处理。
	// 若开启补偿，还需把有效不透明度 o_eff=o*sqrt(detΣ/det(Σ+hI)) 的梯度拆成两条：
	// 一条传给原 opacity，另一条通过行列式比值传给协方差。
	constexpr float h_var = 0.3f;
	float d_inside_root = 0.f;
	if(antialiasing)
	{
		const float det_cov = c_xx * c_yy - c_xy * c_xy;
		c_xx += h_var;
		c_yy += h_var;
		const float det_cov_plus_h_cov = c_xx * c_yy - c_xy * c_xy;
		const float h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability
		// dL_dopacity 此时其实是像素核得到的 dL/do_eff；先保存它才能计算两条支路。
		// 对原 opacity 的导数乘补偿系数；对补偿系数的导数乘原 opacity。
		// 当根号内比值被最小值截断时，该比值方向的导数置零。
		const float dL_dopacity_v = dL_dopacity[idx];
		const float d_h_convolution_scaling = dL_dopacity_v * opacities[idx];
		dL_dopacity[idx] = dL_dopacity_v * h_convolution_scaling;
		d_inside_root = (det_cov / det_cov_plus_h_cov) <= 0.000025f ? 0.f : d_h_convolution_scaling / (2 * h_convolution_scaling);
	} 
	else
	{
		c_xx += h_var;
		c_yy += h_var;
	}
	
	float dL_dc_xx = 0;
	float dL_dc_xy = 0;
	float dL_dc_yy = 0;
	if(antialiasing)
	{
		// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdx
		// https://www.wolframalpha.com/input?i=d+%28%28x*y+-+z%5E2%29%2F%28%28x%2Bw%29*%28y%2Bw%29+-+z%5E2%29%29+%2Fdz
		// 版本细节：此处 c_xx/c_yy 在上面的分支中已经加过 h_var，下面又代入带 w 的比值导数式。
		// 直接对前向 detΣ/det(Σ+hI) 求导时，通常应明确区分未滤波与已滤波的对角元素。
		// 因此这里应按固定源码理解，不能把它无条件宣称为该前向公式的精确解析导数；
		// 若研究抗锯齿梯度，应单独用有限差分核查这一处，本学习副本不修改原有计算。
		const float x = c_xx;
		const float y = c_yy;
		const float z = c_xy;
		const float w = h_var;
		const float denom_f = d_inside_root / sq(w * w + w * (x + y) + x * y - z * z);
		const float dL_dx = w * (w * y + y * y + z * z) * denom_f;
		const float dL_dy = w * (w * x + x * x + z * z) * denom_f;
		const float dL_dz = -2.f * w * z * (w + x + y) * denom_f;
		dL_dc_xx = dL_dx;
		dL_dc_yy = dL_dy;
		dL_dc_xy = dL_dz;
	}
	
	float denom = c_xx * c_yy - c_xy * c_xy;

	// 逆矩阵的导数含行列式平方的倒数；分母额外加 1e-7 是数值稳定近似。
	// 它避免过小分母，但也意味着这里与理想代数公式并不在所有输入上严格相等。
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		
		// 令 Σ=[[a,b],[b,c]]，逆矩阵为 [[c,-b],[-b,a]]/(ac-b²)。
		// 下面逐项展开 ∂L/∂a、∂L/∂b、∂L/∂c，并累加前面抗锯齿补偿产生的梯度。
		// 非对角的两份对称位置通过系数 2 计入，所以要与像素核中 0.5 的储存约定一起读。
		dL_dc_xx += denom2inv * (-c_yy * c_yy * dL_dconic.x + 2 * c_xy * c_yy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.z);
		dL_dc_yy += denom2inv * (-c_xx * c_xx * dL_dconic.z + 2 * c_xx * c_xy * dL_dconic.y + (denom - c_xx * c_yy) * dL_dconic.x);
		dL_dc_xy += denom2inv * 2 * (c_xy * c_yy * dL_dconic.x - (denom + 2 * c_xy * c_xy) * dL_dconic.y + c_xx * c_xy * dL_dconic.z);
		
		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		// 把 Σ屏幕=TᵀΣ世界T 的梯度传回三维协方差的六个压缩分量。
		// 先写 xx/yy/zz 三个对角项，每项由二维 a/b/c 的上游梯度线性组合而成。
		// T 的两列对应屏幕横纵投影方向，因此可以把长公式看成若干外积的逐元素展开。
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_dc_xx + T[0][0] * T[1][0] * dL_dc_xy + T[1][0] * T[1][0] * dL_dc_yy);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_dc_xx + T[0][1] * T[1][1] * dL_dc_xy + T[1][1] * T[1][1] * dL_dc_yy);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_dc_xx + T[0][2] * T[1][2] * dL_dc_xy + T[1][2] * T[1][2] * dL_dc_yy);
		
		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		// 再写 xy/xz/yz 三个非对角独立参数；每个参数同时影响矩阵中的两个对称位置。
		// 因此这里会出现乘二或两个交叉乘积相加，不能简单复制对角公式。
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_dc_xx + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][1] * dL_dc_yy;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_dc_xx + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_dc_xy + 2 * T[1][0] * T[1][2] * dL_dc_yy;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_dc_xx + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_dc_xy + 2 * T[1][1] * T[1][2] * dL_dc_yy;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	// 协方差投影对 T 也有梯度，因为 T=WJ 中 J 随中心深度和位置变化。
	// 六个 dL_dT 分量对应真正参与二维投影的两列三行；未使用方向无需展开。
	// 这一步确保移动中心时椭圆大小与形状的变化也能推动训练。
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xx +
	(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_xy;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xx +
	(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_xy;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xx +
	(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_xy;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc_yy +
	(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_dc_xy;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc_yy +
	(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_dc_xy;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc_yy +
	(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_dc_xy;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	// 由 T=WJ 反传得到 dL/dJ=Wᵀ(dL/dT)，相机旋转 W 本身在此固定。
	// 只有 fx/z、fy/z、-fx*x/z²、-fy*y/z² 四个非零雅可比元素需要继续求中心导数。
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	// 缓存 1/z、1/z²、1/z³，避免长公式重复除法。
	// 例如 ∂(fx/z)/∂z=-fx/z²，∂(-fx*x/z²)/∂z=2fx*x/z³。
	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	// 这是中心影响投影协方差这一支的相机坐标梯度。
	// x/y 路径带上前面的截断掩码；z 方向同时影响两个焦距缩放和两个透视交叉项。
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;
	// Account for inverse depth gradients
	// 逆深度 q=1/z，所以 dq/dz=-1/z²；像素核累积的 dL/dq 由此转换成深度梯度。
	// 它与协方差引起的位置梯度在相机空间相加，随后一次旋转转置映射回世界空间。
	if (dL_dinvdepth)
	dL_dtz -= dL_dinvdepth[idx] / (t.z * t.z);


	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	// 这是位置梯度的第一段，因此用赋值写入；后续 preprocessCUDA 与球谐反向再使用 +=。
	// 这个执行顺序解释了为何两个预处理核不能在没有依赖保证时随意并行。
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
// 三维协方差反向：已知 dL/dΣ，沿 Σ=MᵀM、M=S*R 回传到尺度与四元数。
// 正常训练的尺度来自 Python exp，旋转来自 Python normalize；这些上层激活的梯度由 PyTorch 接续。
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	// 与前向相同，当前 CUDA 不做四元数归一化；注释掉的除法不参与实际运行。
	// 因此最后输出的是对传入四元数的梯度；若它是 Python 归一化的结果，会继续经过该归一化节点。
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	// 压缩协方差中一个非对角参数代表两个对称矩阵项；展开梯度矩阵时各分一半。
	// 这样与矩阵微分中的 Frobenius 内积匹配，避免在 dL/dM=2M(dL/dΣ) 中重复计算。
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	// Σ=MᵀM 的微分包含 dMᵀM 与 MᵀdM 两项；当上游矩阵梯度对称时合并为 2M·梯度。
	// 这一步是尺度与旋转梯度的共同来源，下面再按 M=S*R 分流。
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	// 点积取出对应尺度方向的梯度；这里源码没有在最终写出的尺度梯度上再显式乘 mod。
	// 若 scale_modifier≠1，严格链式法则还应核对 s=mod*scale 这一层的因子；
	// 训练默认 mod=1 时这个区别消失，学习或数值梯度检查时不要忽略该前提。
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	// 旋转梯度需要乘相应尺度，因为 M=S*R 中每一行受到该方向尺度的调制。
	// 前面的转置让代码可用 GLM 的列访问表达这些行运算。
	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	// 把旋转矩阵九项对四元数四分量的导数逐项展开，累积出 dL/dr、dL/dx、dL/dy、dL/dz。
	// 公式里的 2 和 4 来自四元数旋转矩阵的二次项；分量顺序仍为实部在前。
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	// 末尾 dnormvdv 也被注释掉了，实际直接返回 dL_dq。
	// 保留原样是为了忠实解释本版本；不能仅凭上方英文「unnormalized quaternion」理解实际行为。
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
// 第二阶段预处理反向：补齐投影中心梯度、球谐颜色梯度，以及尺度/旋转梯度。
// 它接在 computeCov2DCUDA 之后执行，已有的三维中心梯度必须保留并相加。
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dopacity)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	// 对齐次除法 x_ndc=X/W 使用商法则：d(X/W)=dX/W-X*dW/W²，y 同理。
	// mul1/mul2 预先缓存 X/W²、Y/W²；下面三行分别对世界 x/y/z 求导。
	// 这里输入的 dL_dmean2D 已在像素核乘过 W图像/2、H图像/2，因此按归一化坐标继续链式法则。
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	// 只有使用球谐路径时才把 RGB 梯度传回球谐系数和视角方向。
	// 若调用者提供预计算颜色，该颜色梯度在 Python 侧沿其计算图继续传播。
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	// 只有使用尺度+旋转路径时才在 CUDA 展开协方差梯度。
	// 若上层已经计算 cov3D，六个协方差梯度直接交回上层自动求导，避免重复展开。
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
// 反向渲染仍按「块负责屏幕块、线程负责像素」组织，但改为从远到近遍历。
// 像素线程将贡献累加到共享的高斯参数梯度，因此必须使用原子加法避免线程间写入丢失。
// 逆序遍历可从最终 T 恢复每一步透射率，不必存储像素×高斯数量级的中间状态。
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_invdepths,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dinvdepths
)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];


	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	// 前向最终透射率 T_final=Π(1-α)；反向接受一项后用 T/=(1-α) 撤销该项。
	// 恢复出的 T 就是该高斯之前的透射率 T_i，颜色对其自身颜色的导数是 α_i*T_i。
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	// 列表遍历从末尾开始，contributor 每检查一项先减一。
	// last_contributor 是前向最后合成项的一基位置，所以 contributor>=last_contributor 的项都应跳过。
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	// accum_rec 保存当前高斯后方已经重建的合成颜色，暂不包括背景项。
	// last_alpha/last_color 保存上一次反向访问的那一层，下一步将它纳入后方颜色。
	// 逆深度使用同样的递推结构，因为它也按 α*T 加权。
	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	float dL_invdepth;
	float accum_invdepth_rec = 0;
	if (inside)
	{
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		if(dL_invdepths)
		dL_invdepth = dL_invdepths[pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_invdepth = 0;


	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	// 像素位置 u=(ndc+1)*W/2-0.5，所以 ∂u/∂ndc=W/2，纵向同理。
	// 因此返回的 means2D.x/y 梯度按归一化屏幕坐标缩放；不能直接当作纯像素坐标导数比较。
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			// 共享内存批次也按逆序装载，保证每个像素看到的顺序与前向完全相反。
			// 一个批次内颜色按 [通道,批内编号] 布局，便于线程协作搬运后供每个像素复用。
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];

			if(dL_invdepths)
			collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			// 重算与前向相同的 power、G、alpha 和阈值；仅对前向接受的项继续反传。
			// 版本细节：虽然前向对 alpha 截顶，下面并没有在饱和区显式乘 min 的零导数。
			// 因此饱和部分采用了近似反向，不能期待对所有输入都通过严格有限差分检查。
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			// 撤销当前层透射因子，恢复 T_i；alpha≤0.99 保证分母至少 0.01。
			// 恢复过程会有浮点误差，但比存储每个像素所有中间 T 节省很多显存。
			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				// 递推 C后方 = α上一层*c上一层 + (1-α上一层)*C更后方。
				// 当前 α 增大一方面增加当前颜色，另一方面遮住后方颜色，因此导数含 c当前-C后方。
				// 等各通道汇总后再乘 T_i，同时还需单独补上背景的遮挡导数。
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				// 一个高斯通常覆盖很多像素，多个线程同时写同一个颜色梯度，atomicAdd 保证加法不丢项。
				// 浮点原子加法的实际执行顺序可变化，因此运行结果可能存在很小的非确定性舍入差异。
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			// Propagate gradients from inverse depth to alphaas and
			// per Gaussian inverse depths
			if (dL_dinvdepths)
			{
			// 逆深度与颜色共享合成权重，所以同时贡献 dL/dα 和 dL/d(1/z)。
			// 这里累计的是每个高斯的逆深度梯度；稍后 computeCov2DCUDA 再乘 -1/z² 转成中心深度梯度。
			const float invd = 1.f / collected_depths[j];
			accum_invdepth_rec = last_alpha * last_invdepth + (1.f - last_alpha) * accum_invdepth_rec;
			last_invdepth = invd;
			dL_dalpha += (invd - accum_invdepth_rec) * dL_invdepth;
			atomicAdd(&(dL_dinvdepths[global_id]), dchannel_dcolor * dL_invdepth);
			}

			// 至此 c当前-C后方 已按各输出通道上游梯度求和，乘 T_i 完成当前 α 的前景导数。
			// last_alpha 更新后供下一次更靠前的高斯重建其后方颜色使用。
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			// 背景项为 T_final*背景色；∂T_final/∂α_i=-T_final/(1-α_i)。
			// 因此即使当前颜色与后方前景相同，改变 α 仍可能通过遮挡背景影响损失。
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			// 除去饱和近似后，α=o_eff*G，所以 dL/dG=o_eff*dL/dα、dL/do_eff=G*dL/dα。
			// G=exp(-0.5*dᵀQd)，对位移的导数为 -G*Qd，下面 gdx/gdy 是公共乘积。
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			// 位置梯度先由 G 对 d 求导，再乘从归一化坐标到像素的 W/2、H/2。
			// 二维占位张量有三个分量，但这里只更新 x/y；z 维不代表相机深度梯度。
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			// 这些是逆协方差 Q 的梯度，尽管原英文写作 covariance，实际不要混淆 Σ 与 Σ⁻¹。
			// Qxx/Qyy 的导数分别带 -0.5*dx²、-0.5*dy²；对称非对角项各存半份 -0.5*dx*dy。
			// 在 computeCov2DCUDA 中会合并两个对称位置，因此这里 y 槽的 0.5 不是漏掉交叉项。
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			// 这里先得到对有效不透明度 o_eff 的梯度；若抗锯齿开启，后续还要拆出原 opacity 与滤波因子。
			// 多个像素的贡献仍用原子加法求和，输出缓冲区必须在启动核前清零。
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

// 宿主侧按依赖顺序启动两个预处理核：先写入协方差/中心基础梯度，再补其余路径。
// 默认 CUDA 流中的顺序执行保证第二个核能读取并累加第一个核的结果。
void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const float* opacities,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	const float* dL_dinvdepth,
	float* dL_dopacity,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	bool antialiasing)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		opacities,
		dL_dconic,
		dL_dopacity,
		dL_dinvdepth,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		antialiasing);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot,
		dL_dopacity);
}

// 此包装函数只把参数交给编译期固定通道数的反向核。
// 返回梯度张量的内存分配与初始化在 rasterize_points.cu，自动求导输入顺序映射在 Python 桥接层。
void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_invdepths,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_dinvdepths)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_invdepths,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_dinvdepths
		);
}
