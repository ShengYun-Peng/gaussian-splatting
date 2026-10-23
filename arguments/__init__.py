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

# 中文导读：用参数类的成员变量同时声明命令行选项和默认值，再按加载/渲染/优化职责分组。
# 调用链通常是：创建参数组 → parser.parse_args() → 各组 extract(args) → 交给训练或渲染。
from argparse import ArgumentParser, Namespace
import sys
import os

# 空属性容器，extract 用 setattr 把本组参数装进去；本身不计算、不验证参数。
class GroupParams:
    pass

# 自动注册参数的基类。子类要先设置 self.xxx 默认值，再调用这里的构造函数。
class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        # 命令行帮助中按 name 分组，便于查看；这些组仍共用同一个 ArgumentParser 和参数命名空间。
        group = parser.add_argument_group(name)
        # vars(self) 读取实例字典，因此遍历的是子类先前设置的参数和默认值。
        # 参数名称、默认值和 Python 类型在这里被统一转换成 argparse 规则。
        for key, value in vars(self).items():
            shorthand = False
            # 成员名开头的下划线表示额外注册单字母短选项；命令行长选项不保留下划线。
            # 例如 _source_path → --source_path 和 -s；_model_path → --model_path 和 -m。
            if key.startswith("_"):
                shorthand = True
                # 只修改局部变量 key，不会重命名实例原成员；extract 之后仍需兼容有/无下划线两种名称。
                key = key[1:]
            # 先记住原默认值类型，如 int/float/str/bool；即使稍后把默认值改成 None，解析类型仍正确。
            t = type(value)
            # fill_none 用 None 表示“用户没在命令行指定此项”，方便后续合并模型目录中的 cfg_args。
            # 否则普通默认值可能无意覆盖训练时保存的配置。
            value = value if not fill_none else None 
            # 带短选项和只带长选项两条分支的转换逻辑相同，区别仅在是否增加 -首字母。
            if shorthand:
                # 布尔项用 store_true：出现开关就设为 True，不要求在其后再写 True/False。
                # 这里没有生成 --no-xxx；如果已保存配置为 True，不能用这个开关显式覆盖成 False。
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    # 非布尔项按默认值类型转换命令行文本，如把 "30000" 转为 int；范围合法性由后续逻辑负责。
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                # 布尔项用 store_true：出现开关就设为 True，不要求在其后再写 True/False。
                # 这里没有生成 --no-xxx；如果已保存配置为 True，不能用这个开关显式覆盖成 False。
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    # 从全局 Namespace 中只抽取这个参数组声明过的字段，返回不含前导下划线的属性容器。
    def extract(self, args):
        group = GroupParams()
        # arg 是（参数名，值）二元组；遍历的是已经解析/合并后的最终命名空间。
        for arg in vars(args).items():
            # 既匹配 sh_degree 这类普通成员，也匹配 source_path 对应的内部 _source_path。
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                # 保留 parser 输出的公开参数名，后续调用方统一写 dataset.source_path，而不用知道下划线约定。
                setattr(group, arg[0], arg[1])
        return group

# 数据加载与模型表示相关选项；sentinel=True 时默认填 None，常用于加载已有训练配置。
class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        # 每个颜色通道最多使用三阶 SH，共 (3+1)²=16 个系数；实际激活阶数会在训练中逐步提高。
        self.sh_degree = 3
        # 数据集根目录，对应 -s；目录内应有 COLMAP sparse 或 NeRF transforms_*.json 等结构。
        self._source_path = ""
        # 模型输出/加载目录，对应 -m；用于高斯 PLY、cfg_args、相机描述和曝光配置等。
        self._model_path = ""
        # 相对数据集根目录的图像子目录，对应 -i；可改成预先缩放后的 images_2 等目录。
        self._images = "images"
        # 可选逆深度图目录，对应 -d；空字符串关闭深度文件读取及相应监督数据加载。
        self._depths = ""
        # 图像加载分辨率，对应 -r：1/2/4/8 表示缩小倍数，其他正值表示目标宽度。
        # 默认 -1 会在原图宽度超过 1600 时自动缩放；具体规则见 utils/camera_utils.py。
        self._resolution = -1
        # 白色渲染背景开关，对应 -w；默认黑色。合成数据有透明区域时尤其需要保持数据/背景一致。
        self._white_background = False
        # 启用包含测试视图半幅训练/半幅评估及曝光变换的协议；会影响相机划分、掩码和渲染。
        # 不是简单让完整测试图像与训练图像混在一起，细节见 dataset_readers.py、cameras.py 和 train.py。
        self.train_test_exp = False
        # 相机监督图像/逆深度张量的存放设备；设为 cpu 可以减少常驻显存，但训练时仍会搬到 CUDA。
        # 这个参数不表示整个高斯优化与 CUDA 光栅化器已支持 CPU 运行。
        self.data_device = "cuda"
        # 启用数据集的训练/测试留出划分；它与 torch.nn.Module.eval() 是不同概念。
        self.eval = False
        # 参数默认值全部声明后，统一交给基类注册；sentinel 控制未指定参数是否保留为 None。
        super().__init__(parser, "Loading Parameters", sentinel)

    # 从全局 Namespace 中只抽取这个参数组声明过的字段，返回不含前导下划线的属性容器。
    def extract(self, args):
        g = super().extract(args)
        # 把数据根目录绝对化，避免之后拼接图像路径时重复附加相对根目录。
        g.source_path = os.path.abspath(g.source_path)
        return g

# 渲染流水线开关：决定部分数学计算在 Python/PyTorch 端还是扩展内部执行。
class PipelineParams(ParamGroup):
    def __init__(self, parser):
        # True 时在 Python 里按视线方向求 SH 颜色，再传预计算 RGB；False 时由光栅化器求值。
        self.convert_SHs_python = False
        # True 时用尺度和四元数在 Python 端构造/打包三维协方差；False 时传尺度与旋转给扩展。
        self.compute_cov3D_python = False
        # 向光栅化扩展传调试标志，出错时便于保留诊断信息；具体行为由扩展实现。
        self.debug = False
        # 传给光栅化器的抗锯齿开关，用于其屏幕空间滤波/补偿路径，不是提高输入图像分辨率。
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

# 训练默认超参数。这里只声明数值，真正的优化器分组在 GaussianModel.training_setup。
# 优化器操作的是原始参数：位置、SH、logit 不透明度、对数尺度、四元数及曝光变换。
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # 训练总迭代数；每轮通常采样一个相机，不能等同于对完整数据集训练三万轮 epoch。
        self.iterations = 30_000
        # 位置学习率起点，创建优化器时还会乘场景空间尺度 spatial_lr_scale。
        self.position_lr_init = 0.00016
        # 位置学习率终点，默认比起点小 100 倍；两者在对数空间插值。
        self.position_lr_final = 0.0000016
        # 位置调度的延迟倍率参数。当前 training_setup 没有给位置调度传 lr_delay_steps，
        # 因此延迟步数仍为 0，这个倍率在当前位置调度调用中实际不启用预热。
        self.position_lr_delay_mult = 0.01
        # 位置指数衰减达到终点所需步数；超过该步数后保持最终学习率。
        self.position_lr_max_steps = 30_000
        # SH 零阶颜色学习率；当前实现对其余高阶系数使用 feature_lr/20，降低高频外观更新幅度。
        self.feature_lr = 0.0025
        # 不透明度原始 logit 参数的学习率；实际 alpha 通过 sigmoid 激活进入 (0,1)。
        self.opacity_lr = 0.025
        # 对数尺度参数的学习率；实际三轴标准差通过 exp 得到正值。
        self.scaling_lr = 0.005
        # 四元数参数学习率；使用时重新单位化，使它表示纯旋转。
        self.rotation_lr = 0.001
        # 每图曝光/颜色仿射变换的学习率起点，曝光参数由单独的 Adam 优化器管理。
        self.exposure_lr_init = 0.01
        # 曝光学习率指数衰减终点，调度跨度使用总 iterations。
        self.exposure_lr_final = 0.001
        # 曝光预热/延迟步数，默认 0 表示关闭；若改为正值，下一项才控制起始倍率。
        self.exposure_lr_delay_steps = 0
        # 启用曝光预热时从该倍率平滑回到 1；默认延迟步数为 0，所以不会把曝光学习率乘零。
        self.exposure_lr_delay_mult = 0.0
        # 相对场景半径的尺寸阈值：最大轴尺度小的候选高斯可克隆，大的候选高斯可分裂。
        # 它不是“每次增加 1% 点”或“保留 1% 密集区域”的比例。
        self.percent_dense = 0.01
        # 图像损失混合权重 λ：(1-λ)×L1 + λ×(1-SSIM)，默认两项权重为 0.8/0.2。
        self.lambda_dssim = 0.2
        # 处于允许稠密化的训练区间时，每隔 100 步执行一次候选筛选、克隆/分裂与剪枝。
        self.densification_interval = 100
        # 每隔这些步数把有效不透明度压回较小上限，再继续优化；避免长期高不透明度阻碍调整。
        self.opacity_reset_interval = 3000
        # 稠密化开始阈值；训练循环使用 iteration > 此值，先积累一段梯度统计再调整点数。
        self.densify_from_iter = 500
        # 稠密化结束阈值；训练循环使用 iteration < 此值，后半程通常固定拓扑继续优化参数。
        self.densify_until_iter = 15_000
        # 高斯投影位置的累计平均梯度阈值，用于找仍难以拟合图像的区域；不是直接阈值化颜色误差。
        self.densify_grad_threshold = 0.0002
        # 可用且可靠的逆深度监督 L1 权重起点；没有深度数据时不产生该监督项。
        self.depth_l1_weight_init = 1.0
        # 逆深度损失权重的指数衰减终点，逐渐减弱其在总训练损失中的影响。
        self.depth_l1_weight_final = 0.01
        # 训练时是否每步随机采样 RGB 背景；False 使用数据集指定的固定黑/白背景。
        self.random_background = False
        # 默认 Adam；sparse_adam 路径依赖相应版本的 CUDA 光栅化扩展，是否可用由训练代码检查。
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

# 加载已有模型时合并两类参数：训练保存的 cfg_args 与本次命令行显式覆盖值。
# 只有默认填 None 的参数才能可靠区分“没有输入”和“用户有意选择默认值”。
def get_combined_args(parser : ArgumentParser):
    # 跳过 argv[0]（脚本路径），剩下的才是待解析命令行参数。
    cmdlne_string = sys.argv[1:]
    # 先准备一个空 Namespace 表达式，若未得到配置内容则保留此回退值。
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        # 配置位于模型目录，保存格式是 Namespace(...) 的 Python 表达式，不是 JSON。
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    # 原代码仅捕获 TypeError（例如 model_path 为 None），没有捕获文件不存在的 FileNotFoundError。
    # 因此不能据打印信息推断所有“找不到 cfg_args”的情况都会安全回退。
    except TypeError:
        print("Config file not found at")
        pass
    # 按上游格式把 Namespace(...) 文本还原为对象；eval 会执行 Python 表达式。
    # 这要求 cfg_args 来源可信，不能把它理解成只读取数据的 JSON 解析器。
    args_cfgfile = eval(cfgfile_string)

    # 先以训练配置作为底稿，copy 避免直接修改从文件恢复的对象字典。
    merged_dict = vars(args_cfgfile).copy()
    # 逐项检查本次解析结果；非 None 值优先覆盖文件值，None 保留文件配置。
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    # 恢复统一的属性访问接口，后面仍然可用 args.model_path、args.sh_degree 等形式访问。
    return Namespace(**merged_dict)
