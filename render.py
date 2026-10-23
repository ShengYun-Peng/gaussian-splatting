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

# 中文精读：离线渲染入口，读取已经保存的 PLY 模型，逐个训练/测试相机输出预测图和参考图。
# 它不执行反向传播，也不会继续训练；先阅读 train.py，再用本文件理解训练后如何复现输出。
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


# 渲染一组相机。name 区分 train/test；iteration 用于构造输出目录，避免不同检查点互相覆盖。
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    # 预测图放在 renders，真值放在 gt；metrics.py 会读取这两个对应目录进行图像质量评估。
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    # 创建输出目录；exist_ok=True 允许重复运行，但同名编号图片会按原逻辑被覆盖。
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # 每次只渲染一台相机，返回 [3,H,W] 的浮点图；序号用于对齐预测图和参考图文件名。
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        # 只选参考图前三个颜色通道；Camera 将透明度另存为 alpha_mask，不能假定所有 RGBA 输入已合成背景。
        gt = view.original_image[0:3, :, :]

        # 当前实现这里读取模块级 args，而不是函数形参 train_test_exp；从命令行运行时二者通常一致。
        # 该选项只保存右半幅评估区域，配合训练/测试曝光分离策略；不要误以为输出尺寸总等于相机尺寸。
        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        # 两张图使用完全相同的五位编号，便于逐像素对应；PNG 是可视化输出，不保存原始浮点精度。
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

# 加载一次高斯场景，再按 skip_train/skip_test 控制是否导出相应相机集合。
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    # 离线评估无需梯度，关闭建图减少内存；render 中的梯度保留操作有异常保护以适配此路径。
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        # load_iteration=-1 会查找最新保存轮次；shuffle=False 保持读取相机顺序，便于比较导出图片。
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # 背景设置应与训练一致，否则透明区域的输出和评价会不同。
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    # sentinel=True 把模型参数默认值设为 None，便于区分“用户没传”与“用户明确覆盖保存配置”。
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    # 合并模型目录 cfg_args 与命令行参数，显式命令行值优先；只应读取自己可信的配置文件。
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    # 设置可复现实验的随机状态与 GPU 设备；该入口依然要求训练扩展和 CUDA 环境。
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)