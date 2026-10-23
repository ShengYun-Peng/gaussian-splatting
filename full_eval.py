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

# 中文导读：批量基准评估编排脚本，顺序调用 train.py、render.py、metrics.py。
# 它本身不实现训练/渲染算法；理解单场景流程后再阅读这里的多数据集循环。
# 脚本在顶层执行，直接运行可能训练多个完整场景并覆盖相应评估输出。
import os
from argparse import ArgumentParser
import time

# Mip-NeRF 360 分室外/室内两组，是为了分别选择 images_4 和 images_2 输入目录。
mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
# 另外两组是 Tanks and Temples 与 Deep Blending 的固定场景名单，目录名需与列表一致。
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

# 先定义阶段开关和输出选项，再根据是否训练/渲染决定是否强制要求原始数据根目录。
parser = ArgumentParser(description="Full evaluation script parameters")
# 意图是跳过训练并复用已有模型；但下方 timing.txt 写入位于训练条件外，存在未定义计时变量问题。
# 这里仅解释上游行为，未修改脚本；跳过阶段运行时应注意该限制。
parser.add_argument("--skip_training", action="store_true")
# 跳过渲染时，后续指标阶段要求已有 test/*/renders 和 gt 图像结果。
parser.add_argument("--skip_rendering", action="store_true")
# 只跳过最终指标计算，前面的训练/渲染是否执行仍由各自开关决定。
parser.add_argument("--skip_metrics", action="store_true")
# 所有场景模型保存到 output_path/场景名，整体训练计时保存在 output_path/timing.txt。
parser.add_argument("--output_path", default="./eval")
# 训练时统一追加 -d depths2/，要求各场景已准备相应逆深度图与对齐参数。
parser.add_argument("--use_depth", action="store_true")
# 启用曝光补偿/半幅测试协议，并指定一套单独的曝光学习率与预热设置。
parser.add_argument("--use_expcomp", action="store_true")
# 训练时改用 sparse_adam，需安装支持该优化器的光栅化扩展；不是减少总训练步数。
parser.add_argument("--fast", action="store_true")
# 训练和渲染都追加 --antialiasing，保持高斯屏幕空间滤波配置一致。
parser.add_argument("--aa", action="store_true")




# 先只解析已知阶段开关，暂时允许数据集路径参数出现在未知列表中；稍后会补齐参数定义。
args, _ = parser.parse_known_args()

# 按固定顺序拼出全部场景名，后面构造 all_sources 时必须使用相同分组/顺序。
all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

# 只要训练或渲染至少执行一个阶段，就需要三个数据集根路径；两者都跳过时理论上只需模型输出。
if not args.skip_training or not args.skip_rendering:
    # 每个参数是数据集总根目录，实际场景路径会在循环中再拼接 bicycle、room 等子目录名。
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
    parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    parser.add_argument("--deepblending", "-db", required=True, type=str)
    # 加入条件参数后重新完整解析，检查必须路径和未知选项；不是把第一次解析结果直接复用到底。
    args = parser.parse_args()
# 训练阶段按场景顺序串行启动子进程，当前脚本等待每个 train.py 退出后才继续下一个。
if not args.skip_training:
    # 统一关闭交互查看器、减少日志，并启用测试留出划分。
    # 把训练期测试迭代设为 -1，使正常正数迭代中不做该测试；保存迭代仍由 train.py 默认配置决定。
    common_args = " --disable_viewer --quiet --eval --test_iterations -1 "
    
    # 把同一抗锯齿标志传入当前阶段的子脚本；训练和离线渲染各有一处同样分支。
    if args.aa:
        common_args += " --antialiasing "
    # 该开关只向 train.py 追加 -d；此处没有给渲染命令追加深度路径。
    # render.py 仍可能从已保存的 cfg_args 恢复 depths 配置，因此不意味着渲染阶段一定不读深度。
    if args.use_depth:
        common_args += " -d depths2/ "

    # 曝光协议必须在训练和渲染两侧一致；训练还设置曝光学习率，渲染侧只需启用已训练的变换。
    if args.use_expcomp:
        # 曝光学习率从 0.001 衰减到 0.0001，同时用 5000 步延迟将初始倍率 0.001 平滑升回 1。
        common_args += " --exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp "

    # 仅切换优化器类型；是否成功由 train.py/扩展的可用性决定。
    if args.fast:
        common_args += " --optimizer_type sparse_adam "

    # 记录当前数据集组的墙钟起点；每组结束后除以 60，得到整组训练耗时（分钟）。
    start_time = time.time()
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        # Mip-NeRF 360 室外选用预处理好的四倍下采样目录，模型输出按场景分开。
        # 这里使用 shell 字符串拼接且不检查返回状态；某个子进程失败不会自动终止整个编排。
        os.system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + common_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        # 室内选择两倍下采样图像；输入目录应与先前 convert.py 或数据集预处理产物一致。
        os.system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + common_args)
    # Mip-NeRF 360 的计时包含室外与室内两组全部场景，而不是每个场景的平均时间。
    m360_timing = (time.time() - start_time)/60.0

    # 记录当前数据集组的墙钟起点；每组结束后除以 60，得到整组训练耗时（分钟）。
    start_time = time.time()
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        # Tanks/Deep Blending 使用默认 images 目录；loadCam 的分辨率策略仍可能在加载时进一步缩图。
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    tandt_timing = (time.time() - start_time)/60.0

    # 记录当前数据集组的墙钟起点；每组结束后除以 60，得到整组训练耗时（分钟）。
    start_time = time.time()
    for scene in deep_blending_scenes:
        source = args.deepblending + "/" + scene
        # Tanks/Deep Blending 使用默认 images 目录；loadCam 的分辨率策略仍可能在加载时进一步缩图。
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    db_timing = (time.time() - start_time)/60.0

# 现有代码无条件写入计时：若使用 --skip_training，m360_timing/tandt_timing/db_timing 未赋值。
# 此外父输出目录必须已存在。此处说明原脚本限制，不把“可跳过训练”误写成已验证保证。
with open(os.path.join(args.output_path,"timing.txt"), 'w') as file:
    file.write(f"m360: {m360_timing} minutes \n tandt: {tandt_timing} minutes \n db: {db_timing} minutes\n")

# 离线渲染阶段只生成测试视图，分别评估 7000 与 30000 次训练保存的高斯模型。
if not args.skip_rendering:
    # 按 all_scenes 的相同顺序拼装每个场景的数据路径，之后 zip 按位置一一配对。
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)
    
    # 重建渲染阶段专用参数：quiet 减少日志、eval 使用测试划分、skip_train 不输出训练视图。
    common_args = " --quiet --eval --skip_train"
    
    # 把同一抗锯齿标志传入当前阶段的子脚本；训练和离线渲染各有一处同样分支。
    if args.aa:
        common_args += " --antialiasing "
    # 曝光协议必须在训练和渲染两侧一致；训练还设置曝光学习率，渲染侧只需启用已训练的变换。
    if args.use_expcomp:
        common_args += " --train_test_exp "

    # 两个列表按顺序配对；zip 会在较短列表结束时停止，因此维护场景列表时需同时维护路径列表。
    for scene, source in zip(all_scenes, all_sources):
        # 载入明确的 7000 次迭代模型，而非“最新”模型；对应 point_cloud/iteration_7000 必须已保存。
        os.system("python render.py --iteration 7000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        # 再渲染 30000 次迭代结果；render.py 按迭代命名子目录，便于 metrics.py 分别统计。
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

# 指标阶段一次把所有模型目录传给 metrics.py；每个目录下可评估多个迭代结果。
if not args.skip_metrics:
    # 构造 -m 接受的多个路径参数；下方对各模型路径加双引号，以保留路径里的普通空格。
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    # 调用最终 SSIM/PSNR/LPIPS 评估并写 results.json/per_view.json；这里同样没有检查子进程退出状态。
    os.system("python metrics.py -m " + scenes_string)
