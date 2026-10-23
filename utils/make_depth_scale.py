# 中文导读：为每张图估计单目逆深度到 COLMAP 逆深度的仿射对齐参数。
# 目标关系为 d_inv_colmap ≈ scale × d_inv_mono + offset；输出供 Camera 中的深度监督使用。
# 这里不训练单目深度模型，也不做稠密三维重建，而是借助 SfM 已知的稀疏对应点校准尺度。
# 注意：COLMAP 本身通常只有相似变换意义下的场景尺度，此步骤不自动恢复真实米制单位。
import numpy as np
import argparse
import cv2
# joblib 用线程并行处理多张图；每张图拟合一对 scale/offset，互不共享拟合结果。
from joblib import delayed, Parallel
import json
# 复用 COLMAP 格式读取器与四元数转换函数；通配符导入也带入该模块中公开的 os 名称。
# 所以后面的 os.path.join 虽未在本文件显式 import os，仍依赖这个上游导入行为。
from read_write_model import *

# 输入 key 是 COLMAP 图像记录编号；cameras 按相机编号索引，images 按图像编号索引。
# points3d_ordered 是按三维点 ID 直接寻址的 [最大点 ID+1,3] 数组，并非连续点列表。
# 返回图像名及两个对齐参数；深度图无法读取时返回 None，调用者会从输出 JSON 中省略该图。
def get_scales(key, cameras, images, points3d_ordered, args):
    # 先取当前照片的外参/二维观测，再通过 camera_id 获取其原始宽高和内参记录。
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    # point3D_ids 与 image_meta.xys 一一对应；-1 表示该二维特征没有关联重建出的三维点。
    # 源码此行读取全局 images_metas，而不是形参 images；命令行主流程传入同一对象，
    # 但单独调用 get_scales 时必须注意这个全局依赖，本注释保持原行为。
    pts_idx = images_metas[key].point3D_ids

    # 排除没有三维关联的负 ID，再排除超过索引数组范围的 ID；布尔乘法在此起逻辑与作用。
    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)

    # 同一个掩码同时筛选点 ID 和二维观测，维持“这个三维点出现在这个像素”的配对。
    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    # 存在对应点时按 ID 读取 [M,3] 世界坐标；此处假设所有有效 ID 在模型中确实存在。
    # 仅通过范围检查不会识别索引数组里的 ID 空洞；正常 COLMAP 关联应满足这项一致性。
    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx]
    else:
        # 没有任何关联时放一个三维零向量占位；二维观测仍为空，后面有效点数检查会失败。
        # 占位并不意味着把世界原点作为一个真实深度监督点；后续变换/除法仍会执行。
        pts = np.array([0, 0, 0])

    # COLMAP 的 qvec 顺序为 [w,x,y,z]，这里得到世界→相机旋转 R。
    # 这与 Camera 元数据中常见的“先转置再存 R”不同，本脚本直接使用读取器的旋转。
    R = qvec2rotmat(image_meta.qvec)
    # 按行保存点，所以世界→相机变换写为 pts @ R.T + t。
    # 所得第三分量 z 是沿相机光轴的深度，不是相机到点的欧氏距离。
    pts = np.dot(pts, R.T) + image_meta.tvec

    # 取每个相机坐标 z 的倒数，使其与单目模型输出的逆深度进行比较。
    # 当前源码未在除法前检查 z=0 或非有限值，极端/损坏数据可能先产生 inf/NaN。
    invcolmapdepth = 1. / pts[..., 2] 
    # 移除最后一个点号及扩展名，例如目录/a.b.jpg → 目录/a.b，保留可能存在的相对子目录。
    # 这里假设原图名称有扩展名，使用的命名规则与 scene/dataset_readers.py 保持一致。
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    # 读取同名 PNG 逆深度图；IMREAD_UNCHANGED 保留文件原位深，不先压成 8 位图。
    invmonodepthmap = cv2.imread(f"{args.depths_dir}/{image_meta.name[:-n_remove]}.png", cv2.IMREAD_UNCHANGED)
    
    # 文件缺失或 OpenCV 无法解码时返回 None；这与后面的“拟合不可靠，写零参数”不同。
    if invmonodepthmap is None:
        return None
    
    # 若 PNG 带多个通道，只使用第一个通道，统一为 [高,宽] 数组。
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]

    # 按该预处理约定将整数逆深度除以 65536 转为 float32，而不是除以 65535。
    # 这里假设输入就是对应的 16 位逆深度编码；普通米制深度/8 位可视化图不能直接混用。
    invmonodepthmap = invmonodepthmap.astype(np.float32) / (2**16)
    # 用高度比估计深度图相对 COLMAP 原图的缩放倍率 s，并对 x/y 使用同一个倍率。
    # 这隐含宽高等比例缩放的假设；不是根据实际深度图宽度再单独计算水平倍率。
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    # 用 COLMAP 已记录的二维特征坐标定位采样点，无需在此重新用焦距投影三维坐标。
    # 缩放到逆深度图坐标后转 float32，以便 cv2.remap 在非整数位置插值。
    maps = (valid_xys * s).astype(np.float32)
    # 只保留图像范围内且 COLMAP 逆深度为正的配对；正逆深度对应相机前方。
    # 水平上界仍使用原宽×高度比例 s，依赖前面提到的等比例缩放假设。
    # 这里没有额外检查 np.isfinite，因此正无穷不一定会被排除。
    valid = (
        (maps[..., 0] >= 0) * 
        (maps[..., 1] >= 0) * 
        (maps[..., 0] < cam_intrinsic.width * s) * 
        (maps[..., 1] < cam_intrinsic.height * s) * (invcolmapdepth > 0))
    
    # 至少需要 11 个有效对应点，并要求 COLMAP 逆深度有超过 1e-3 的变化范围。
    # 注意范围判断发生在下面 valid 过滤之前，使用的是当前全部关联点，不只最终保留的点。
    # 这些是上游的启发式门槛，不能据此保证单目逆深度本身有非零变化或拟合一定有限。
    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        # 同步筛选采样坐标与 COLMAP 逆深度，留下参与分布对齐的同一组观测。
        maps = maps[valid, :]
        invcolmapdepth = invcolmapdepth[valid]
        # 在这些稀疏二维位置双线性采样单目逆深度；BORDER_REPLICATE 对边界外插值邻居复制边值。
        # 一维坐标映射在 OpenCV 中形成单列采样结果，[...,0] 去掉该列维，得到 [有效点数]。
        # 所得两组数一一对应，但下方估计器匹配整体中位数/离散程度，并非逐点最小二乘回归。
        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]
        
        ## Median / dev
        # COLMAP 逆深度的中位数作为中心位置，比直接用均值更不容易被少数极端值拖动。
        t_colmap = np.median(invcolmapdepth)
        # 离散程度取“相对中位数的平均绝对偏差”：mean(|d−median(d)|)。
        # 它不是标准差，也不是常见稳健统计中的 median(|d−median(d)|)；极端离群点仍会影响这个均值。
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        # 对单目逆深度计算相同的中心与平均绝对偏差，使两端使用同一统计定义。
        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        # 尺度等于两种离散程度的比值，让仿射变换后的单目变化幅度与 COLMAP 对齐。
        # 源码没有对 s_mono=0 加 epsilon/回退分支；常量或异常单目图可能导致 inf/NaN。
        scale = s_colmap / s_mono
        # 偏移保证变换后的单目中位数落在 COLMAP 中位数上：scale·t_mono+offset=t_colmap。
        # 因此这是中位数与离散程度匹配，不是 RANSAC，也没有显式最小化所有配对点的残差。
        offset = t_colmap - t_mono * scale
    # 样本数量或 COLMAP 深度变化不足时，不强行给出尺度，约定输出 scale=offset=0。
    else:
        scale = 0
        offset = 0
    # 返回去掉最后扩展名的相对图像名；下游用这个键查找该照片的参数。
    # 零尺度用于表达未成功估计；最终可靠性还由 Camera 对照全场景尺度统计来判断，
    # 不能把所有零尺度/非有限值都视为已在本脚本中彻底过滤。
    return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}

# 命令行预处理入口；应先准备好 COLMAP sparse/0 模型及对应的逆深度 PNG。
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # base_dir 指数据集根目录；默认路径是上游示例，实际运行应传入自己的场景路径。
    parser.add_argument('--base_dir', default="../data/big_gaussians/standalone_chunks/campus")
    # depths_dir 可以独立指定，文件名需与图像名去掉扩展名后的路径对应。
    parser.add_argument('--depths_dir', default="../data/big_gaussians/standalone_chunks/campus/depths_any")
    # model_type 决定读取 .bin 或 .txt 格式；此脚本显式指定模型位于 sparse/0。
    parser.add_argument('--model_type', default="bin")
    args = parser.parse_args()


    # 一次性读取相机内参、图像外参与二维观测、三维点三个字典，供所有工作线程共享读取。
    cam_intrinsics, images_metas, points3d = read_model(os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")

    # COLMAP 点 ID 不必从零连续编号，先收集实际 ID 及其世界坐标。
    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    # 分配直接按 ID 索引的稠密数组，空洞保留为零；占用内存取决于最大 ID 而非实际点数。
    # 当前代码假设三维点非空；空模型在 pts_indices.max() 处就会失败。
    points3d_ordered = np.zeros([pts_indices.max()+1, 3])
    # 把实际点坐标散布到对应 ID 行，后面图像关联点可直接用整数数组批量索引。
    points3d_ordered[pts_indices] = pts_xyzs

    # depth_param_list = [get_scales(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas]
    # 对所有图像并行拟合；n_jobs=-1 使用 joblib 认为可用的全部处理器线程额度。
    # threading 后端共享上述大型数组，避免为每个进程复制点云；任务本身不使用 CUDA。
    depth_param_list = Parallel(n_jobs=-1, backend="threading")(
        # delayed 将每张照片的调用包装成任务；Parallel 返回与输入任务顺序对应的结果列表。
        delayed(get_scales)(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas
    )

    # 将逐图结果列表改为“图像名 → {scale,offset}”字典，便于训练按名称随机取相机。
    depth_params = {
        depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
        # 这里只移除 None，也就是图像未读取的情况；scale=0 或非有限估计值并未在此删除。
        for depth_param in depth_param_list if depth_param != None
    }

    # JSON 固定写入数据集 sparse/0/depth_params.json，会覆盖同名旧文件；此处不创建缺失目录。
    # 文件只存 scale/offset；scene/dataset_readers.py 加载后才补充全场景 med_scale，
    # Camera 随后检查相对尺度范围，并在 scale>0 时应用逆深度仿射变换。
    with open(f"{args.base_dir}/sparse/0/depth_params.json", "w") as f:
        # indent=2 便于人工检查。若上面产生 NaN/Infinity，默认 json.dump 可能写出这类非标准数值；
        # 本脚本没有在落盘前执行有限性检查，遇到异常图应先检查深度图与有效对应点。
        json.dump(depth_params, f, indent=2)

    # 上游保留的结束标记；打印 0 本身不报告有效拟合张数，也不证明每张图的参数都可靠。
    print(0)
