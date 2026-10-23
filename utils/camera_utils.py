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

# 中文导读：本文件把磁盘上的 CameraInfo 元数据转换成训练/渲染使用的 Camera 对象。
# 读取数据、选择分辨率、构造相机三个步骤分开；真正的张量转换和矩阵缓存见 scene/cameras.py。
from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2

# 模块级开关只控制“自动缩小大图”的提示是否已经输出，不影响后续图像的缩放计算。
WARNED = False

# 加载一张图像及可选逆深度图。id 是当前列表中的顺序编号，cam_info.uid 是数据集相机编号。
# resolution_scale 是场景调用方指定的额外缩放倍率；数值越大，最终宽高越小。
def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    # PIL 保留原始颜色/透明通道；此处只打开图像，后续由 Camera 按 resolution 缩放并转成张量。
    image = Image.open(cam_info.image_path)

    # 空路径表示没有深度监督。这里读取的量按项目约定是逆深度，不是直接以米计的距离。
    if cam_info.depth_path != "":
        try:
            # 两类数据使用不同的文件编码比例。IMREAD_UNCHANGED（-1）保留原始位深，再转为 float32。
            # 除以 512 或 65536 是解码约定，不是对任意深度文件都适用的通用归一化公式。
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        # 异常分支补充文件路径后重新抛出；OpenCV 对无法读取的文件也可能返回 None，
        # 此时后面的 astype 会触发其他异常，并由下方通用异常分支记录。
        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        # 用 None 明确关闭这台相机的深度分支，避免把缺失深度误当成有效的全零监督。
        invdepthmap = None
        
    # PIL 的尺寸顺序是（宽，高），而图像数组通常是（高，宽，通道）；不要互换。
    orig_w, orig_h = image.size
    # 常见整数 1/2/4/8 被解释成下采样倍数：例如 2 代表原图宽高各减半。
    # 两个缩放因子相乘后统一作用于宽高，保持长宽比；此分支使用 round 取整。
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        # 默认值 -1 启用自动策略：原始宽度超过 1600 时，先把目标宽度限制到 1600。
        # 这不是最终分辨率上限，因为下方还会应用 resolution_scale。
        if args.resolution == -1:
            if orig_w > 1600:
                # 声明修改模块级 WARNED，使遍历许多相机时只打印一次同样的提示。
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                # 求“原宽/目标宽”的缩小倍率，后面对高使用同一倍率以保持相机视野比例。
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            # 除 1/2/4/8/-1 以外的数值按目标宽度（像素）解释，而不是下采样倍数。
            global_down = orig_w / args.resolution
    

        # 把用户分辨率选择与调用方额外缩放合并；此分支使用 int 向下截断，和上方 round 略有不同。
        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # 交给 Camera 创建最终图像张量、透明度/深度掩码和投影矩阵。
    # 缩放图像不会改变以弧度计的 FoV；像素焦距随图像宽高一起缩放。
    # train_test_exp 和两个测试标记会参与测试视图的半幅掩码处理，不能简单视为重复字段。
    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test)

# 按输入顺序批量构造 Camera；训练时随机挑选视角发生在训练循环，不在这里。
def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    # enumerate 产生连续 uid；它与原始 COLMAP camera_id 不同，后者可能被多张图像共享。
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

# 导出供查看/记录用的相机描述。调用方传入的是拥有 width/FovX 等字段的相机元数据。
# 这里只输出位姿和焦距等 JSON 原生类型，不序列化图像或 GPU 张量。
def camera_to_JSON(id, camera : Camera):
    # 先还原列向量约定的世界到相机矩阵：[Rᵀ | T]。
    # 仓库中的 camera.R 存的是世界到相机旋转的转置，T 才是该外参平移。
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    # 注意原变量名 W2C 容易误导：Rt 已是世界到相机，因此求逆后实际是相机到世界 C2W。
    # 逆矩阵的平移列就是相机在世界坐标中的位置，不能直接用原来的 T 代替。
    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    # NumPy 数组转换成普通列表后可交给 json.dump；旋转矩阵按行导出。
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        # 根据垂直视场角与图像高度恢复像素焦距 fy；fx 同理，视场角单位都是弧度。
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry