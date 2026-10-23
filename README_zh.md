# 三维高斯泼溅：中文源码学习版

这个 fork 在原始实现中加入详细中文注释，并提供一份可从头运行的中文学习 notebook。重点解释每个核心步骤的输入输出、张量形状、数学关系、坐标约定、梯度路径和执行顺序。

主仓库基准版本为 `54c035f7834b564019656c3e3fcc3646292f727d`，光栅器基准版本为 `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0`。它包含后来加入的深度正则、曝光补偿、抗锯齿及可选加速接口。讲解以这些版本的实际代码为准。

## 从这里开始

- **[打开中文学习 notebook](learning/zh-CN/gaussian_splatting_zh.ipynb)**：20 节讲解、17 个代码实验，包含已经执行的结果。
- **[阅读验证记录](learning/zh-CN/VALIDATION_zh.md)**：本地与 hsg 的实际检查，以及未验证的 GPU 项目。
- **[阅读中文 CUDA 源码](learning/zh-CN/cuda-source/README.md)**：对应版本的自动微分桥接、前向、反向和光栅调度。
- **[查看原始安装与数据指南](README.md)**：软件依赖、数据准备和完整训练命令仍以原文为准。

notebook 的全部代码单元只使用 Python 标准库，不需要显卡、PyTorch 或数据集。可以在 Jupyter 中顺序运行，也可以在仓库根目录运行：

```bash
# 执行全部数学实验与断言，不修改笔记文件。
python learning/zh-CN/run_notebook.py

# 可选：将本次执行输出更新到笔记文件。
python learning/zh-CN/run_notebook.py --output learning/zh-CN/gaussian_splatting_zh.ipynb
```

这些实验帮助理解算法；原始训练代码仍需 CUDA 及编译好的扩展。

## 推荐精读顺序

| 顺序 | 源码 | 重点 |
|---|---|---|
| 1 | [arguments/__init__.py](arguments/__init__.py) | 参数分组、默认值、配置合并 |
| 2 | [scene/dataset_readers.py](scene/dataset_readers.py) | COLMAP/合成数据、点云、相机、逆深度 |
| 3 | [scene/cameras.py](scene/cameras.py)、[utils/camera_utils.py](utils/camera_utils.py) | 图像尺寸、外参、掩码、矩阵布局 |
| 4 | [scene/gaussian_model.py](scene/gaussian_model.py) | 可训练参数、激活、初始化、Adam 状态、增密裁剪 |
| 5 | [gaussian_renderer/__init__.py](gaussian_renderer/__init__.py) | 颜色/协方差路径、屏幕梯度、曝光、返回张量 |
| 6 | [train.py](train.py) | 抽相机、渲染、损失、反传、保存、结构调整、更新 |
| 7 | [utils/general_utils.py](utils/general_utils.py)、[utils/graphics_utils.py](utils/graphics_utils.py)、[utils/sh_utils.py](utils/sh_utils.py) | 协方差、四元数、投影、球谐与学习率 |
| 8 | [utils/loss_utils.py](utils/loss_utils.py)、[utils/image_utils.py](utils/image_utils.py) | L1、局部结构相似度、误差与峰值信噪比 |
| 9 | [CUDA 学习源码](learning/zh-CN/cuda-source/README.md) | 分块、前缀和、排序、透明度融合、逆序反传 |
| 10 | [render.py](render.py)、[metrics.py](metrics.py)、[full_eval.py](full_eval.py) | 模型加载、图像导出、评价与批量运行 |

辅助阅读：[scene/__init__.py](scene/__init__.py) 负责装配场景；[scene/colmap_loader.py](scene/colmap_loader.py) 解释二进制记录与位姿；[convert.py](convert.py) 串起数据预处理；[gaussian_renderer/network_gui.py](gaussian_renderer/network_gui.py) 解释查看器协议。

建议每次追一个函数：先写出输入形状，再逐行追踪每个中间变量，最后找到返回值的使用者。重点代码块的中文注释解释“为什么”，原始表达式给出精确边界条件。第三方查看器、GLM、LPIPS 等依赖没有整库翻译。

## 子模块中的中文注释如何使用

CUDA 光栅器是独立 Git 子模块，主仓库只记录它的提交号。只修改本地子模块而不发布其提交，其他人克隆主仓库时就看不到修改。因此本 fork 在主仓库内保存了带注释的源码副本和补丁，同时保持原子模块引用不变。

默认直接阅读 `learning/zh-CN/cuda-source/`。该目录是学习副本，构建脚本仍使用 `submodules/diff-gaussian-rasterization/`。若希望在实际编译的子模块文件中查看相同注释，可从仓库根目录执行：

```bash
# 先验证补丁与当前子模块匹配；需要子模块处于基准版本且没有冲突修改。
git -C submodules/diff-gaussian-rasterization apply --check ../../learning/zh-CN/rasterizer-zh.patch

# 将中文注释应用到本地子模块；这会让子模块显示为有本地修改。
git -C submodules/diff-gaussian-rasterization apply ../../learning/zh-CN/rasterizer-zh.patch

# 如需撤销这个注释补丁，先检查逆向应用，再执行撤销。
git -C submodules/diff-gaussian-rasterization apply --reverse --check ../../learning/zh-CN/rasterizer-zh.patch
git -C submodules/diff-gaussian-rasterization apply --reverse ../../learning/zh-CN/rasterizer-zh.patch
```

补丁只适用于锁定版本，不会自动修改上游仓库。原始许可与版权声明保留在每个文件中；学习副本及注释遵循相应原文件的许可条件。

## 如何确认没有改动算法

```bash
# 请在尚未应用子模块注释补丁的干净子模块状态下运行。
python learning/zh-CN/verify_annotations.py
```

验证脚本逐文件对比基准版本：Python 语法树与非注释符号序列完全相同，原始每一行都保留，只插入整行注释；所有原有 Python 文件可编译；子模块提交号保持不变；CUDA 学习副本保留原始行，补丁可干净应用。验证范围不包含运行时间、显存占用或完整 GPU 训练数值结果。

学习注释也会指出原代码的特殊行为，例如 `depth` 实际输出未归一化的逆深度、可见性结果是整数索引、最终迭代不执行优化器更新、某些检查点未包含曝光状态。这些是当前实现的阅读提示，没有借注释任务改变代码行为。
