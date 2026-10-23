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

# 中文导读：把 COLMAP 实拍数据或 NeRF/Blender 合成数据统一成 SceneInfo。
# 这里解析相机元数据与初始化点云；真正的 GPU 相机张量由 utils/camera_utils.py 构建。
# 阅读顺序建议：CameraInfo → getNerfppNorm → 对应数据集的 SceneInfo 读取函数。
import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

# 单视图元数据容器。R:[3,3] 存世界到相机旋转的转置，T:[3] 存该外参的平移。
# 因此世界点的相机坐标是 Rᵀ @ X + T，相机在世界中的位置为 -R @ T。
class CameraInfo(NamedTuple):
    # COLMAP 分支使用内参 camera_id，同一内参可被多张照片共享；合成分支使用帧序号。
    uid: int
    R: np.array
    T: np.array
    # 垂直/水平视场角用弧度表示；尽管原注解写成 np.array，读取函数实际通常传标量。
    FovY: np.array
    FovX: np.array
    # 可选逆深度对齐参数：每张图的 scale/offset，以及全场景正 scale 的中位数 med_scale。
    depth_params: dict
    # image_path 用来重新打开原图；image_name 用于相机排序、曝光参数索引和结果文件命名。
    image_path: str
    image_name: str
    # 空字符串表示无逆深度文件；width/height 记录数据源原始分辨率，尚未下采样。
    depth_path: str
    width: int
    height: int
    # 标记该视图是否被划为测试视图；启用 train_test_exp 后，测试视图也可能进入训练相机列表。
    is_test: bool

# 整个场景的加载结果：初始化点云、训练/测试视图、场景尺度和原始 PLY 缓存路径。
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    # 字典包含 translate 与 radius。radius 后续用于空间学习率缩放及稠密化大小阈值。
    # 计算出归一化参数并不意味着此处已经把点云或相机坐标整体改写。
    nerf_normalization: dict
    ply_path: str
    # 记录数据类型，供后续深度图解码等逻辑选择约定。
    is_nerf_synthetic: bool

# 根据相机中心估计场景尺度，与 NeRF++ 的相机归一化思路相近。
# 输入应包含至少一个有效训练视图；半径来自相机分布，不是点云包围盒或物体实际半径。
def getNerfppNorm(cam_info):
    # 内部辅助函数接收多个 [3,1] 相机中心，返回平均中心 [3] 与最大离心距离。
    def get_center_and_diag(cam_centers):
        # 把每台相机的列向量沿列堆成 [3,N]，每一列对应一个世界坐标中的相机中心。
        cam_centers = np.hstack(cam_centers)
        # 沿相机维求平均，得到 [3,1]；保留单列维方便后续对全部相机广播相减。
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        # 每列计算到平均中心的欧氏距离，得到 [1,N]；变量 diagonal 实际取最大半径距离。
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        # flatten 把中心转换成普通长度为 3 的数组，方便后续相机坐标操作。
        return center.flatten(), diagonal

    cam_centers = []

    # 先还原 W2C，再求逆为 C2W；外参平移 T 并不等于世界中的相机中心。
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        # 使用 3:4 切片保留 [3,1] 列向量，而不是取成 [3]，以匹配 hstack 的堆叠约定。
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    # 最大相机中心距离额外放大 10%，作为场景尺度的余量；不是由优化得到的可学习参数。
    radius = diagonal * 1.1

    # 若调用方使用该平移，平均相机中心会移动到原点；此处仅记录建议平移。
    translate = -center

    return {"translate": translate, "radius": radius}

# 把 COLMAP 的图像外参记录与相机内参记录按 camera_id 关联，生成统一的视图列表。
# 图像记录是一张照片一个，而内参记录可能由多张照片共享。
def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        # 回车符让读取进度覆盖同一行；flush 使缓冲内容立即显示。
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        # extr 包含当前照片的姿态、文件名及内参索引；intr 包含模型、尺寸和焦距参数。
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        # COLMAP 四元数是世界到相机旋转，分量顺序为 (w,x,y,z)。
        # 先转成旋转矩阵，再转置存入本仓库的 R；之后 getWorld2View2 会再次转置还原。
        R = np.transpose(qvec2rotmat(extr.qvec))
        # 平移仍直接沿用 COLMAP 的世界到相机外参 t，不做旋转或取反。
        # COLMAP 相机坐标约定通常为 x 向右、y 向下、z 向前。
        T = np.array(extr.tvec)

        # 简单针孔模型参数是 [f,cx,cy]，横纵两个轴共用焦距 f；由尺寸分别求 FoV。
        # 当前实现只使用焦距，主点参数没有进入该数据结构和投影矩阵。
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        # 一般针孔模型参数是 [fx,fy,cx,cy]，横纵焦距可以不同；不能误把 fx 同时用于两个方向。
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            # 只接受已经去畸变的针孔数据；带径向/鱼眼畸变的模型应先经过 convert.py/COLMAP 去畸变流程。
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # 计算最后一个扩展名连同点号的长度，用于构造不含扩展名的深度键和 PNG 文件名。
        n_remove = len(extr.name.split('.')[-1]) + 1
        # 某视图没有匹配到对齐参数时保留 None；该分支仍允许构建视图元数据。
        depth_params = None
        if depths_params is not None:
            try:
                # 深度参数 JSON 按图像去掉扩展名的名字索引；命名必须与 COLMAP 图像记录一致。
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        # 保留 COLMAP 原始图像名及可能的子目录；深度文件使用同名、固定 .png 扩展名。
        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        # 把上述几何、路径和尺寸打包；is_test 由完整图像名是否出现在测试名单中决定。
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

# 读取初始化点云 PLY：每个顶点对应一个待初始化的三维高斯中心。
def fetchPly(path):
    plydata = PlyData.read(path)
    # 取 vertex 元素表；代码期望存在位置、RGB 和 nx/ny/nz 字段，并非读取任意 PLY 格式。
    vertices = plydata['vertex']
    # 先把三个长度为 N 的字段堆成 [3,N]，再转置成高斯模型使用的 [N,3]。
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    # 文件 RGB 为 8 位整数，除以 255 后进入 [0,1]；这一步不涉及 sRGB/线性色彩空间转换。
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # 法线照常读取；本项目生成的初始点云可以将法线留为全零。
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

# 把 [N,3] 位置和 RGB 写成带顶点属性的 PLY，供第一次加载后的后续运行复用。
# rgb 在此入口应已是 0～255 范围；下方结构化类型会转换成 uint8，不会替调用方归一化。
def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    # 结构化数组为每个顶点定义具名字段：f4 表示 float32，u1 表示 uint8。
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    # 当前初始化不估计法线，用零填充以满足统一 PLY 字段格式。
    normals = np.zeros_like(xyz)

    # 分配 N 条结构化记录，区别于常见的二维纯浮点矩阵。
    elements = np.empty(xyz.shape[0], dtype=dtype)
    # 临时拼成 [N,9]，再逐行转成 tuple，按 dtype 的字段顺序写入结构化数组。
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    # 声明这些记录属于 vertex 元素并写盘；没有面索引，因此输出是点云而不是三角网格。
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# COLMAP 场景总入口：读取稀疏重建、确定训练/测试视图、加载深度对齐信息和初始化点云。
def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        # 默认读取 sparse/0 下的二进制重建；images.bin 保存每张图像外参，cameras.bin 保存内参。
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        # 若二进制读取发生异常，则回退到文本格式；原代码捕获范围较宽，并不只处理文件缺失。
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # 逆深度对齐参数由预处理生成。仅在启用 depths 路径时才强制要求该 JSON 存在。
    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            # 聚集各图的深度尺度，仅使用正尺度估计典型量级，避免无效/反向尺度影响可靠性判定。
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            # 至少有一个正值就计算其中位数；没有时采用 0 作为回退值。
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            # 把相同 med_scale 附到每张图的参数中，Camera 随后用相对尺度阈值判断该图深度是否可靠。
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        # 已显式请求深度监督却找不到对齐参数时直接退出，避免静默使用缺少尺度对齐的深度。
        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    # eval 表示保留测试视图用于评估，不是 PyTorch 模块的 eval() 模式。
    if eval:
        # 原实现用路径中的字符串“360”触发 Mip-NeRF 360 风格的每八张留一张策略。
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            # 先按图像名字排序，确保测试集选择稳定，不依赖 COLMAP 字典的迭代顺序。
            cam_names = sorted(cam_names)
            # 从索引 0 开始每 llffhold 张留一张，因此默认第 1、9、17……张作为测试视图。
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            # 调用方把 llffhold 设为假值时，改为读取显式测试名单，每行一个完整图像名。
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        # 未启用评估时没有保留测试视图，常规训练将使用全部图像。
        test_cam_names_list = []

    # 允许从 images_2/images_4 等预处理目录读取图像；这里的目录选择与 loadCam 再缩放不同。
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), 
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    # 按名字稳定排序视图；后续训练仍会随机抽取相机，但缓存/导出顺序保持一致。
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # 普通模式排除测试视图；train_test_exp=True 时测试相机也进入训练列表。
    # 相应的 Camera 掩码负责把测试视图按半幅划分，不能把此处简单解释为完整测试图像参与监督。
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    # 测试相机列表依然保留被标记的视图；开启曝光评估协议时两个列表可能包含相同相机。
    test_cam_infos = [c for c in cam_infos if c.is_test]

    # 用实际训练相机分布估计场景半径，后续空间学习率与克隆/分裂阈值都与该尺度相关。
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 初始化使用 COLMAP 三角化出来的稀疏点；先缓存为统一 PLY 格式，避免每次重新转换。
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    # 只在 PLY 不存在时生成缓存；已有文件会直接复用，包括用户可能自行提供的点云。
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            # 读取三维坐标与颜色；第三个返回值（重投影误差）在高斯初始化阶段未使用。
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        # 原实现遇到 PLY 读取异常时返回 None；这只记录加载失败，不代表后续模型必能继续初始化。
        pcd = None

    # 把统一结果交给 scene.Scene；点云后续转成可训练高斯参数，相机后续转成渲染对象。
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

# 读取 NeRF 合成数据的 transforms_*.json。与 COLMAP 不同，JSON 直接提供相机到世界矩阵。
def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        # 该格式通常给出整个序列共享的水平视场角（弧度）；垂直视场角根据图像宽高推算。
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # frame.file_path 通常是不带扩展名的相对图像路径；默认补 .png。
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            # 4×4 列向量矩阵，前三列是相机坐标轴在世界中的方向，第四列是相机中心。
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            # 翻转相机的 y/z 轴，将 Blender/OpenGL 的“y 向上、z 向后”转成 COLMAP 的“y 向下、z 向前”。
            # 修改的是旋转部分的两列，相当于右乘 diag(1,-1,-1,1)；世界坐标和相机中心保持不变。
            # 同时翻转两个轴仍保持旋转行列式为 +1，属于基底约定转换。
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            # 变换方向从相机→世界变为世界→相机；之后存入与 COLMAP 分支一致的 R/T 约定。
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            # T 是逆变换中的平移列；不能把 c2w 的位置列直接当作 T。
            T = w2c[:3, 3]

            # cam_name 若为绝对路径，os.path.join 会直接使用它；正常 ModelParams 会将 source_path 绝对化。
            image_path = os.path.join(path, cam_name)
            # 合成数据以不带扩展名的 stem 作为图像标识；这一点与 COLMAP 分支保留完整名字不同。
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            # 统一为 [H,W,4]，最后一通道是透明度；原本没有 alpha 的图像会补成不透明。
            im_data = np.array(image.convert("RGBA"))

            # 这里选择白/黑背景用于局部 alpha 合成公式：前景 RGB×alpha + 背景×(1-alpha)。
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            # 颜色和 alpha 同时缩放到 [0,1]；alpha 切成 [H,W,1]，可对三个颜色通道广播。
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # 此版本只把 image 的尺寸用于 CameraInfo，没有把这里合成后的像素传给后续 Camera。
            # loadCam 会按 image_path 重新打开原图，真实监督像素与 alpha 处理应继续阅读 scene/cameras.py。
            # 这里保留上游原有 dtype=np.byte 写法；它是有符号字节，不应误读为通用图像编码建议。
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            # 先用宽度和水平 FoV 求像素焦距，再按同一焦距与高度求垂直 FoV，假定方形像素。
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            # 合成视图无 COLMAP 深度对齐 JSON，因此 depth_params=None；is_test 由所读 transforms 文件决定。
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

# NeRF 合成场景入口：读取训练/测试姿态，并用随机点初始化缺少 COLMAP 稀疏点云的场景。
def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    # 可选深度目录与数据集根路径拼接；空路径一路传递为“无深度监督”。
    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    # 不做留出评估时，把 transforms_test.json 中视图也并入训练列表，以使用全部观测。
    if not eval:
        # extend 原地追加相机记录；清空测试列表不会删除已经追加到训练列表中的记录。
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    # 用实际训练相机分布估计场景半径，后续空间学习率与克隆/分裂阈值都与该尺度相关。
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # 合成数据的缓存位于场景根目录，文件名小写 points3d，与 COLMAP 的 points3D 路径不同。
    ply_path = os.path.join(path, "points3d.ply")
    # 只在 PLY 不存在时生成缓存；已有文件会直接复用，包括用户可能自行提供的点云。
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        # 初始创建十万个随机高斯中心；这只是起始点数，训练中的稠密化/剪枝会改变点数。
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        # 每个坐标独立均匀采样到 [-1.3,1.3)，适配标准合成场景的物体范围；不是在球面上采样。
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        # 这里采样的是很小的零阶 SH 系数；经过 SH2RGB 后颜色接近 0.5 的中灰。
        # 不要把 shs 直接当作 [0,1] 的随机 RGB，二者有固定 C0 与 0.5 偏置的换算。
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        # 写盘前把颜色恢复到 0～255 范围；随后统一从 PLY 读取，初始颜色因此经过 uint8 量化。
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        # 原实现遇到 PLY 读取异常时返回 None；这只记录加载失败，不代表后续模型必能继续初始化。
        pcd = None

    # 把统一结果交给 scene.Scene；点云后续转成可训练高斯参数，相机后续转成渲染对象。
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

# 数据集类型到读取函数的分发表；scene.Scene 检测目录结构后选择 Colmap 或 Blender 入口。
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}