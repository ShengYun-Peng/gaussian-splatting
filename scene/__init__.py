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

# 中文导读：Scene 是数据与模型的装配层，连接“磁盘数据集 → 相机 → GaussianModel”。
# 它不执行训练迭代，也不自己渲染；train.py/render.py 通过它取得相机和保存的模型。
# 主要流程：识别数据格式 → 读取场景元数据 → 构造各分辨率相机 → 初始化/加载高斯。
import os
import random
import json
# 辅助函数分别负责找最新保存轮次、按数据集类型选择读取器，以及相机张量化/JSON 导出。
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

# 一个 Scene 包含训练/测试相机集合、场景尺度与一个外部传入的高斯模型。
class Scene:

    # 类型注解只说明 gaussians 的预期类型，不会自动创建 GaussianModel 实例。
    gaussians : GaussianModel

    # load_iteration=None 表示从点云初始化，-1 表示选最新保存轮次，正整数选指定轮次。
    # resolution_scales 用来同时建立多个图像分辨率；这些相机共享同一场景几何。
    # 默认列表在本方法中只遍历、不修改；scale 在此处是图像分辨率因子而非三维场景尺度。
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        # model_path 是训练结果目录；source_path 则是原始图像/SfM 数据目录，两者用途不同。
        self.model_path = args.model_path
        # None 记录“尚未选择已保存模型”，后面的分支用它决定是否创建初始高斯。
        self.loaded_iter = None
        # 保存传入对象的引用，因此 Scene 对高斯的初始化/加载会直接影响训练持有的同一对象。
        self.gaussians = gaussians

        # 这里用真值判断：0 与 None 都不触发加载；通常实际保存轮次从正整数开始。
        if load_iteration:
            if load_iteration == -1:
                # 从 point_cloud/iteration_* 目录名中选最大编号，不通过文件修改时间判断最新。
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                # 用户指定具体编号时原样使用，实际 PLY 文件将在后面读取。
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # 字典键是 resolution_scale，值为该分辨率的 Camera 列表；训练和测试分别维护。
        self.train_cameras = {}
        self.test_cameras = {}

        # 优先识别 COLMAP：source_path/sparse 存在即走 SfM 读取器。
        # 若两种数据格式标记同时存在，因 if/elif 顺序，COLMAP 分支优先。
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # 读取器处理图像路径、深度路径、训练/测试划分及曝光实验模式。
            # 返回的 scene_info 仍以元数据、CPU 点云为主；此时尚未构造用于光栅器的 Camera 张量。
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        # NeRF 合成数据以 transforms_train.json 存储相机位姿，代码称其为 Blender 格式。
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            # 白背景选项用于合成 RGBA 图像的颜色合成；eval 决定是否保留独立测试划分。
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            # 缺少支持的数据格式标记就立即失败，避免误把空目录当成有效场景。
            assert False, "Could not recognize scene type!"

        # 只有新建场景时才保存初始输入快照；读取已有模型时无需重写这些文件。
        if not self.loaded_iter:
            # 把实际使用的初始点云复制为 input.ply，便于追溯初始化来源。
            # 二进制读取/写入保持 PLY 内容不变；model_path 的创建由调用方准备。
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            # 相机 JSON 只保存可序列化描述，不保存图像张量、梯度或 CUDA 对象。
            json_cams = []
            camlist = []
            # 先测试后训练合并相机信息；曝光实验模式可能让测试视图也出现在训练列表中。
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            # 为导出列表生成连续 id；camera_to_JSON 将位姿和焦距转成普通列表/数值。
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            # 在打乱相机顺序之前保存 cameras.json，让导出顺序不受下面随机排列影响。
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # 只在元数据列表上打乱一次，然后每个分辨率都按这个顺序构造相机。
        if shuffle:
            # 这样不同 resolution_scale 的同一列表下标仍对应同一个视角。
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # 测试列表也可打乱；渲染/度量仍依据相机内容进行，而不是依赖固定列表顺序。
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # 读取器通常以相机中心到平均中心的最大距离×1.1 定义 radius。
        # 这里把它作为位置学习率缩放和克隆/分裂阈值的参考长度。
        # 此赋值本身没有把所有点云/相机平移缩放到单位球，勿把“normalization”字样理解成已变换数据。
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 遍历请求的分辨率因子，为同一批视角生成图像尺寸不同的相机对象。
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            # 最后一个参数 False 表示这些对象来自训练集合；相机自身仍另有 is_test_view 标记。
            # 两种测试标记在 train_test_exp 的半幅图像掩码中有不同意义。
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            # 测试集合对应 True；没有测试视角时函数返回空列表，字典键仍会建立。
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        # 数据/相机准备完后，选择模型参数来源。
        if self.loaded_iter:
            # 加载选定轮次 PLY，并在启用曝光实验时尝试读取模型目录下的 exposure.json。
            # 这条路径恢复可渲染参数，不等于训练 checkpoint 恢复；Adam 动量不存于 PLY。
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            # 从初始点云创建中心/尺度/旋转/球谐/不透明度，同时为训练照片创建曝光矩阵。
            # cameras_extent 同时传入模型，成为位置学习率的场景尺度因子。
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    # 保存当前渲染模型；训练恢复所需的 checkpoint 由 train.py 另行 torch.save。
    def save(self, iteration):
        # 每个保存轮次有独立 point_cloud/iteration_N 子目录，便于对比优化过程。
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        # GaussianModel.save_ply 会创建该目录，写入六类逐点原始参数。
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        # 曝光属于图像而不是高斯点，单独按图像名保存，便于加载时稳定查找。
        exposure_dict = {
            # detach 去掉计算图，cpu/NumPy/list 把 GPU 张量转换成 JSON 能编码的嵌套列表。
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            # 遍历初始化时建立的图像名映射；这个方法假设曝光相关成员已准备好。
            for image_name in self.gaussians.exposure_mapping
        }

        # 注意 exposure.json 位于模型根目录，每次保存都会覆盖，并非每个迭代各保存一份。
        # 因此加载较早 PLY 时，默认能找到的曝光文件可能来自较晚保存轮次。
        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    # 按分辨率因子取训练视角；scale 必须是构造时已有的键，不会按需生成。
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    # 按同样规则取测试视角；返回原列表引用，调用方若要修改顺序可先 copy。
    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
