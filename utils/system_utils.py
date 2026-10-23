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

# 中文导读：文件系统小工具，分别处理可重复创建目录与从保存目录名推断最新训练迭代。
from errno import EEXIST
from os import makedirs, path
import os

# 实现类似命令行 mkdir -p 的语义：递归创建缺失父目录，已有目录也视为成功。
def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        # 先直接尝试创建，避免“先检查存在、再创建”在并发场景中发生状态变化。
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        # 只有“已存在”且确实是目录才忽略；同名普通文件、权限不足等其他错误应继续抛出。
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

# 在 point_cloud 等目录中查找最大的保存迭代号，通常文件夹名为 iteration_7000、iteration_30000。
def searchForMaxIteration(folder):
    # 按下划线拆分后把最后一段转为整数，按数值而非字符串比较。
    # 原实现不筛选目录/前缀，因此要求目录内每个条目都满足“末段为整数”的命名约定。
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    # 取最新迭代。空目录会使 max 抛异常；该函数没有“未保存任何迭代”的默认返回值。
    return max(saved_iters)
