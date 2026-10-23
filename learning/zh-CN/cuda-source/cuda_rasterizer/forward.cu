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

// 中文精读：本文件负责「一个三维高斯 → 屏幕椭圆 → 像素颜色」的前向过程。
// 阅读顺序：computeCov3D → computeCov2D → preprocessCUDA → renderCUDA。
// P 表示高斯数，D 表示当前球谐阶数，M 表示每个高斯分配的球谐系数总数；
// 例如 D=3 需要 (3+1)^2=16 个系数，每个系数又有 R/G/B 三个分量。
// 这里没有优化器：输入已经由上层激活成物理量，输出和中间缓存供反向传播使用。
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
// 球谐颜色：c(d)=0.5+Σ_k SH_k(d)·系数_k；同一高斯从不同方向看可呈现不同颜色。
// 每个 CUDA 线程调用此函数处理一个高斯，glm::vec3 同时承载三个颜色通道。
// 这里的方向是「相机指向高斯」，请勿改成反方向：奇数阶球谐会因此改变符号。
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	// 先做世界坐标下的向量差，再单位化；球谐只依赖方向，不依赖距离。
	// 归一化也意味着颜色梯度会沿方向这条路径传回高斯中心；backward.cu 有对应链式法则。
	// 源码没有为零长度方向加保护，正常渲染依赖有效的相机与可见高斯位置。
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	// 连续存储按 [高斯编号, 系数编号, RGB] 排列，转换指针后每次跨过三个 float。
	// idx*max_coeffs 跳到当前高斯的全部球谐系数；即使当前阶数较低，分配容量也可能更大。
	// 0 阶常数项控制基础颜色，后面的分支逐步增加 1/2/3 阶方向细节。
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		// 一阶基函数在本实现中的顺序为 (-y, z, -x)，乘各自 RGB 系数后加总。
		// 不要用另一份球谐库的顺序直接替换；Python utils/sh_utils.py 与这里的约定必须一致。
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			// 二阶的五个基函数是 xy、yz、2z²-x²-y²、xz、x²-y²，前面的常数负责归一化。
			// 预先缓存平方与交叉项，避免后续多次重复乘法；三阶还会复用这些量。
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			// 三阶增加七个三次多项式，因此累计系数数目为 1+3+5+7=16。
			// 分支只读取当前激活阶数对应的系数，未激活的高阶系数此时不参与图像生成。
			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	// 加 0.5 与上层 RGB2SH 的中心化约定配套；它不是 sigmoid，也不把颜色限制到 [0,1]。
	// 下面只截断负颜色，因此大于 1 的颜色仍可保留，后续图像保存可能另外做截断。
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	// 记录每通道是否在截断前小于零，反向时这些通道的颜色梯度会乘零。
	// 该掩码必须对应本次前向的输入，不能在参数更新之后重新猜测。
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
// 三维协方差投影：数学上是 Σ屏幕 = J投影 · W相机 · Σ世界 · W相机ᵀ · J投影ᵀ。
// 透视投影是非线性的，这里用高斯中心处的一阶雅可比，把局部三维椭球近似成二维椭圆。
// 返回只存对称矩阵的三个独立量 (a,b,c)，对应 [[a,b],[b,c]]，单位是像素²。
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	// t 是相机坐标，x/y 是横纵方向，z 是沿相机前方的深度。
	// 随后计算的焦距 focal_x/focal_y 用像素为单位，因此得到的协方差自然处于像素坐标系。
	float3 t = transformPoint4x3(mean, viewmatrix);

	// 限制 x/z、y/z 的范围，避免画面边缘或画外点产生过大的透视雅可比。
	// 1.3 是扩大视野边界的经验系数；这里限制的是协方差计算中的局部投影，
	// 不会把真正的高斯中心坐标移到边界，中心投影仍由 preprocessCUDA 的 p_proj 决定。
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	// 透视投影 u=fx*x/z、v=fy*y/z，所以非零导数是 fx/z、fy/z、-fx*x/z²、-fy*y/z²。
	// glm::mat3 构造参数按列填充，源码里的 J 存的是常见纸面雅可比的转置形式。
	// 理解后面的 W*J 和 transpose(T) 时必须连同这一存储约定一起看，不能只按打印顺序读矩阵。
	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	// 这里只取世界到相机变换的旋转部分；平移影响中心 t，不直接改变协方差。
	// viewmatrix 使用辅助函数中的索引约定，W 在 GLM 中同样按相应转置形式组装。
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	// 三维对称协方差压缩为 [xx,xy,xz,yy,yz,zz] 六个 float；这里展开回 3×3。
	// Vrk 理论上对称，因此转置不改变数学值；代码保留 transpose 是为了统一约定。
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	// T 组合相机旋转与投影雅可比，再用 TᵀΣT 得到局部屏幕协方差。
	// 第三个投影方向设置成零，只需要左上角 2×2；返回值没有包含后面添加的低通滤波方差。
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
// 用三个正尺度与单位四元数构造半正定协方差，比直接学习六个矩阵项更容易保持合法。
// 数学上可写作 R diag(s²) Rᵀ；源码由于 GLM 构造约定写成 M=S*R、Σ=MᵀM。
// scale_modifier 是额外的整体尺寸系数，正常训练通常为 1，查看器可用它改变视觉尺寸。
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	// 注意原英文注释与实际执行不完全一致：/glm::length(rot) 已被注释，CUDA 此处没有归一化！
	// 正常调用由 Python 的 get_rotation 提供已归一化四元数；直接调用扩展时应自己满足这个前提。
	// 分量排列为 (实部 r, 虚部 x, y, z)，虽然 GLM 字段名依次叫 q.x/q.y/q.z/q.w。
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	// 任意 M 的 MᵀM 都是半正定矩阵；尺度为正且非零时通常正定。
	// 把尺度先平方再旋转与这种乘法形式等价，但必须使用匹配的矩阵方向和四元数约定。
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	// 只保存六个独立元素，节省每个高斯的缓存与带宽。
	// 之后的前向投影、反向梯度以及 Python strip_symmetric 都要遵守相同排列。
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
// 预处理核：一条线程负责一个高斯，生成供分块排序与像素渲染使用的紧凑属性。
// 它不遍历像素；高斯可能影响哪些像素块只由二维包围盒估计，真正的指数权重稍后逐像素算。
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	// 全局线程编号对应高斯编号；最后一个线程块可能超出 P，立即返回防止越界。
	// cg::this_grid().thread_rank() 在这里用于编号，并未进行跨线程块的全局同步。
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	// 先标记为「不参与」：任何后续早退都会保持半径和覆盖块数为零。
	// 这让后续扫描、复制和反向核无需为不同剔除原因分别维护标志。
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	// 这一版本的 in_frustum 实际主要检查相机深度 z>0.2；横纵裁剪条件在源码中被注释了。
	// 是否落到画面范围最终还需看 getRect 裁剪后的块数，不能把该函数当完整视锥测试。
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	// projmatrix 在上层通常已包含世界到相机与投影的复合变换。
	// 先得到齐次坐标 (x,y,z,w)，再除以 w 得到归一化设备坐标；1e-7 用来缓和除零问题。
	// 后面 ndc2Pix 才把横纵坐标从大致 [-1,1] 换算到像素中心约定。
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	// 协方差有两条互斥路径：上层已计算就直接读取；否则在 CUDA 用尺度和旋转构造。
	// 选哪条路径会改变反向梯度在哪里展开，但最终应描述同一个三维高斯。
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// 给二维协方差两个对角线都加 0.3 像素²，相当于与一个小的各向同性高斯滤波核卷积。
	// 它避免投影极小的高斯退化成比像素还尖锐的信号，并改善求逆稳定性。
	// 注意：无论 antialiasing 是否启用，这个低通扩宽都会发生；开关控制的是下面的透明度补偿。
	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	// 扩宽后椭圆面积随 sqrt(detΣ) 增大，因此乘 sqrt(detΣ原/detΣ扩宽) 降低峰值，
	// 用于近似保持二维高斯的积分质量；没有该补偿时只扩宽而不降低峰值。
	// 比值最小截到 0.000025，再开平方；这也是反向需要匹配的分段函数。
	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov;

	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	// 2×2 对称矩阵的逆为 1/(ac-b²)·[[c,-b],[-b,a]]。
	// conic=(A,B,C) 储存逆协方差三个独立量，稍后指数为 -0.5*(A*dx²+2B*dx*dy+C*dy²)。
	// 这里检查 det==0 后早退；并非一般性的数值合法性验证，调用者仍应提供有限且合理的输入。
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	// 二维特征值为 (a+c)/2 ± sqrt(((a-c)/2)²+b²)。最大特征值的平方根是最长主轴标准差。
	// 代码用 ceil(3*sqrt(λmax)) 构造保守圆半径，再覆盖椭圆的矩形块范围。
	// 根号内至少取 0.1 是实现中的稳定性/保守估计；它会使半径不同于精确特征值公式。
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	// getRect 给出半开区间 [rect_min,rect_max)，每个块含 BLOCK_X×BLOCK_Y 个像素。
	// 包围盒会包含椭圆外的像素或块，所以「覆盖块」只是候选集合；后续权重阈值还会跳过贡献。
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	// 排序使用相机空间中心深度，而不是欧氏距离，也不是每个像素光线与椭球表面的交点深度。
	// 同一个高斯在一个视图内共用此深度；这是速度与表达能力之间的重要实现取舍。
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = opacities[idx];


	// 把 A、B、C 与有效不透明度打包成 float4，像素核一次读取四个相关标量。
	// 这里的 opacity 已经是激活后的数值，通常来自 Python sigmoid；此处不会再次 sigmoid。
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };


	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
// 渲染核：一个 CUDA block 负责一个屏幕块，一个线程负责其中一个像素。
// 块内线程共同搬运同一批高斯到共享内存，再让每个像素复用这些数据，减少显存读取。
// point_list 已按「块编号优先、中心深度从近到远」排序，不能随意改变遍历顺序。
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	// pix_id 是图像平面内的线性位置；最终颜色写成 [通道,高度,宽度] 的通道优先布局。
	// 网格为覆盖整幅图而向上取整，最右/最下方线程可能越界，inside 决定是否计算和写回。
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	// 越界线程也不能提前退出整个函数，因为后面有块内同步和集体搬运。
	// 它们只跳过像素计算，仍参与共享内存加载与屏障，避免部分线程退出造成同步问题。
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	// range 是当前块在排序后高斯列表里的 [起始,结束) 下标。
	// 一轮最多搬运 BLOCK_SIZE 个高斯；rounds 向上取整，最后一轮用 toDo 限制有效数量。
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	// T 表示当前像素到达下一层前剩余的透射率，初始无遮挡时为 1。
	// C 累计已经接受的颜色；权重为 w_i=α_i·T_i，T_i=Π_{j<i}(1-α_j)。
	// contributor 是访问列表的位置计数，last_contributor 记录最后一个真正被合成的列表位置。
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		// 屏障后统计整个块已有多少线程停止；全部停止才可以结束这个块的加载循环。
		// 有些像素很早就不透明，有些仍需要更后面的高斯，因此不能仅凭单线程状态让整个块返回。
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		// 每条线程负责加载该批中的一个高斯，形成连续的全局内存访问。
		// block.sync() 保证整个批次写入共享内存之后，任一像素线程才开始读取。
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			// d 是「高斯投影中心减像素位置」，以像素为单位；椭圆没有进行概率密度归一化。
			// 因为高斯对 d 的二次型在变号后不变，前向颜色不受这一定义方向影响，但反向位置梯度会受影响。
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			// power=-0.5*dᵀΣ⁻¹d，应当非正；交叉项有两份，所以代码外写 -B*dx*dy。
			// 如果得到正值，源码直接跳过这一对像素/高斯，用来避开不合理的指数增长。
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// 有效 α=不透明度×exp(power)，再限到 0.99；它表示该高斯对此像素的覆盖强度。
			// 0.99 避免后续反向 T/(1-α) 发生除零；α<1/255 的贡献直接略去以节省计算。
			// 此阈值是工程近似，跳过的分支不会产生该像素对该高斯的梯度。
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			// 试算加入当前高斯后的透射率；若低于 1e-4，像素被标记结束。
			// 注意顺序：源码在颜色累加之前 continue，因此触发终止的这个高斯本身也没有被合成。
			// 保存的 T 仍是接受最后一个贡献者后的值，反向必须使用同样的截断边界。
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			// 按从近到远的顺序累积 C += c_i·α_i·T_i，这是带遮挡的透明度合成。
			// 前面的高斯减小后面所有高斯的权重；因此对 α 求导不能只考虑当前项。
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			// 逆深度图使用相同权重累计 Σ_i (1/z_i)·α_i·T_i。
			// 它没有除以 Σ_i α_i·T_i，也不是 1/(Σ_i z_i·α_i·T_i)；背景逆深度在这里视作零。
			// 变量名 expected_invdepth 不意味着已经条件归一化，使用深度损失时应认清这个定义。
			if(invdepth)
			expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			// 保存的是当前块列表的访问序号，包含前面被跳过的项，因此它不是「实际有效高斯数量」。
			// 反向据此忽略位于最后一次合成之后的高斯，再重算各项阈值以恢复对应关系。
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		// 只保存最终透射率和最后贡献者位置，避免为每个像素保存完整的逐高斯合成历史。
		// 反向可以从后往前用除以 (1-α) 重建中间透射率，这是节省显存的关键。
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			// 背景只乘剩余透射率：像素完全无遮挡时输出背景，前景不透明时背景影响很小。
			// 输出并未自动截断到 [0,1]；它仍是可供损失函数求导的浮点颜色。
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}

// 宿主侧包装函数负责发起 CUDA 核；模板通道数在编译时确定，可让通道循环展开。
// 这里只启动异步工作，调试同步与错误报告由上层 CHECK_CUDA 控制。
void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths, 
		depth);
}

// 预处理核使用一维 256 线程块，与渲染时二维 16×16 像素块是不同的工作划分。
// 形参很长时可按组阅读：输入几何/外观 → 相机 → 输出缓存 → 网格与控制开关。
void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing
		);
}
