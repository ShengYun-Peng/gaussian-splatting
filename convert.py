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

# 中文导读：把原始照片转换成高斯训练能读取的 COLMAP 去畸变场景。
# 流水线为 SIFT 特征提取 → 特征匹配 → 稀疏重建/束调整 → 图像去畸变 → 可选多尺度图像。
# 这是实际运行外部程序并写入数据目录的预处理脚本，代码在模块顶层执行。
import os
import logging
from argparse import ArgumentParser
import shutil

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
# 命令行选项控制数据位置、相机模型以及 COLMAP/ImageMagick 可执行文件路径。
parser = ArgumentParser("Colmap converter")
# 该开关在下方转成 SiftExtraction/SiftMatching.use_gpu=0；它只显式配置这两项 GPU 选项。
parser.add_argument("--no_gpu", action='store_true')
# 跳过的不只是匹配，而是整个特征提取、匹配、mapper 区块；要求已有 distorted/sparse/0。
parser.add_argument("--skip_matching", action='store_true')
# 源目录内原始照片应放在 input/。输出的 images/、sparse/0 和 distorted/ 也位于该目录下。
parser.add_argument("--source_path", "-s", required=True, type=str)
# 原始照片默认用 OPENCV 畸变模型估计内参；随后去畸变阶段再生成训练可用的理想针孔相机。
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
# 除原始去畸变图像外，再创建 images_2、images_4、images_8，供训练选择不同输入分辨率。
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
args = parser.parse_args()
# 显式路径会给可执行文件加引号，否则从 PATH 查找 colmap/magick。
# 下方仍直接拼接 source_path 等参数，原脚本没有统一引用这些路径，因此带空格路径可能解析失败。
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
# COLMAP 命令行以整数 1/0 接受这两个 GPU 开关，和 argparse 的布尔值作一次转换。
use_gpu = 1 if not args.no_gpu else 0

# 完整 SfM 流程的中间结果单独保存在 distorted，保留原始畸变相机模型。
if not args.skip_matching:
    # exist_ok=True 允许目录已存在；不会自动清空之前的数据库或重建产物。
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    # 阶段一：把 input 图像的局部 SIFT 特征存入 SQLite 数据库 distorted/database.db。
    # ImageReader.single_camera=1 假定所有图像共享同一套相机内参；camera_model 指定所选畸变模型。
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    # 通过 shell 同步运行命令，程序返回后才继续下一阶段；os.system 的返回状态非零就视为失败。
    exit_code = os.system(feat_extracton_cmd)
    # 原脚本对本阶段外部命令的非零返回状态立即记录错误并退出，避免继续使用不完整产物。
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    # 阶段二：穷举图像对匹配特征，并将对应关系/几何验证结果存回同一数据库。
    # 穷举匹配候选数随照片数近似平方增长，适合先从少量图像理解整个重建流程。
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    # 原脚本对本阶段外部命令的非零返回状态立即记录错误并退出，避免继续使用不完整产物。
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    # 阶段三：根据匹配初始化并增量注册相机、三角化三维点，再做束调整联合优化位姿与几何。
    # 这里把全局束调整的函数收敛容差设置为 1e-6；实际收敛速度取决于数据和 COLMAP 版本。
    # 输出可能包含多个重建子模型，后续脚本固定选择 distorted/sparse/0。
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    # 原脚本对本阶段外部命令的非零返回状态立即记录错误并退出，避免继续使用不完整产物。
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
# 阶段四：将原始带畸变图像与 sparse/0 位姿转换成理想针孔相机对应的去畸变图像/内参。
# output_type=COLMAP 使结果仍采用 COLMAP 格式；这个阶段在 skip_matching=True 时仍会执行。
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

# 去畸变输出的相机/图像/点记录可能位于 sparse 根目录，下面统一整理成训练期要求的 sparse/0。
files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
# 遍历目录内容逐项整理；原代码没有按扩展名筛选，只对名字为 0 的目录作特殊处理。
for file in files:
    # 跳过刚创建/已有的目标目录，避免把 sparse/0 移进自身。
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    # 实际是移动，而不是上方旧英文注释所写的复制；处理后记录位于 sparse/0 下。
    shutil.move(source_file, destination_file)

# 可选多尺度准备：每种尺寸都从原始去畸变图像拷贝开始，避免反复缩放累积插值误差。
if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    # 目录后缀表示宽高各缩小的倍数；面积/像素数分别约为原图的 1/4、1/16、1/64。
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    # 读取 image_undistorter 生成的图像目录；本循环假定这些项目都是可处理的图像文件。
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        # 复制文件及元数据，再用 ImageMagick mogrify 原地将副本宽高缩到 50%。原始 images 不变。
        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        # 另一份独立副本缩到 25%，对应训练时 -i images_4。
        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        # 最后一份独立副本缩到 12.5%，宽高分别为原图八分之一。
        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

# 只有所有启用阶段的命令成功返回，脚本才会到达此完成提示。
print("Done.")
