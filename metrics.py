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

# 中文导读：对 render.py 保存的测试渲染与真值图像逐张计算 SSIM、PSNR、LPIPS。
# 目录约定为 模型目录/test/方法名/{renders,gt}/同名图片，最后保存均值和每视图结果 JSON。
from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

# 按渲染目录中的文件名配对真值；用列表保留每张图，允许不同视图拥有不同分辨率。
def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    # 目录遍历顺序没有排序，但预测/真值/文件名在同一循环同步追加，因此每个指标仍对应正确名字。
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        # 要求 gt 中存在完全相同的文件名；这里不做近似名称匹配或图像尺寸对齐。
        gt = Image.open(gt_dir / fname)
        # 常见 8 位 PIL 图像经 to_tensor 变为 [C,H,W]、浮点 [0,1]；补批次维后为 [1,C,H,W]。
        # 切片只保留前三个颜色通道并丢弃 alpha，然后放到当前 CUDA 设备。
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        # 真值采用同样转换。全部视图都会预先保存在 GPU 列表中，显存占用随图像总像素数增加。
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

# 逐模型目录评估，目录内可同时包含 ours_7000、ours_30000 等多个保存迭代/方法的渲染结果。
def evaluate(model_paths):

    # full_dict 保存每个场景/方法的图像平均指标，per_view_dict 保存按文件名索引的逐图指标。
    full_dict = {}
    per_view_dict = {}
    # 带 polytopeonly 后缀的两个字典在当前版本仅初始化，没有实际计算或输出，不代表额外指标分支。
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    # 每个场景都放在一个 try 中；某场景出错时原脚本会跳过它，继续处理后续模型。
    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            # 只读取测试渲染目录，训练图像评估没有在这个入口中单独执行。
            test_dir = Path(scene_dir) / "test"

            # 方法名直接取 test 下的子目录名，原实现没有验证每个项目是否都是合法方法目录。
            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                # 一次载入这个场景/方法的全部预测与真值，再逐对计算指标。
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                # 三个列表分别积累各图分数；先算每图指标再平均，每张图权重相同而不按像素数加权。
                ssims = []
                psnrs = []
                lpipss = []

                # 只展示评估进度；这里不训练、不反向传播，也不会更新高斯模型。
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    # SSIM 对局部统计和通道/空间求平均，越高越相似；输入保留大小为 1 的批次维。
                    ssims.append(ssim(renders[idx], gts[idx]))
                    # PSNR 的第 0 维此处是批次 N=1，因此对 RGB/全部像素一起计算单张图的 MSE。
                    # 这与把 [C,H,W] 直接传给 image_utils.psnr 得到三个通道分别的 PSNR 不同。
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    # LPIPS 使用本仓库封装的 VGG 感知特征距离，越低越相似；不是像素 L1/L2 的别名。
                    # 实际输入预处理由该封装决定，比较外部报告时应核对相同实现/预处理和网络版本。
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                # 把逐图单值组织成张量并求算术平均以打印。PSNR 是先对各图取 dB，再平均 dB。
                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                # 场景级指标通过 item() 转成 Python 数值，供 json.dump 序列化。
                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                # 按同步保存的 image_names 将各图分数写成字典，便于定位效果差的具体视角。
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            # 写场景的汇总结果；w 模式会覆盖该模型目录先前同名评估文件。
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            # 另存每张图的指标明细，不把不同方法/迭代的数值混在同一个均值中。
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        # 原实现捕获范围较宽且只输出模型名，具体异常栈没有在此打印；该提示不说明是哪一种指标失败。
        except:
            print("Unable to compute metrics for model", scene_dir)

# 命令行入口：明确选择当前进程可见的 cuda:0，评估脚本默认需要可用 CUDA。
if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    # nargs="+" 允许 -m 后传一个或多个模型目录，逐个执行相同评估。
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
