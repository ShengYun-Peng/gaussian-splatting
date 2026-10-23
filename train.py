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

# 中文精读：本文件把“读取场景 → 选相机 → 可微渲染 → 图像损失 → 反传 → 增密/裁剪 → 优化”串起来。
# 建议先顺序阅读 training，再进入 gaussian_renderer.render 和 GaussianModel。
# 这里是当前仓库版本，包含原始三维高斯算法之外的曝光补偿、逆深度监督和可选加速。
# 注释只解释原有行为；原始变量名、表达式、执行顺序与版权说明均保持原样。
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
# 可选依赖只影响对应功能：没有 TensorBoard 仍可训练，只是不写可视化日志。
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# 融合结构相似度扩展把多个操作合并到 CUDA 内核；缺少时回退到 PyTorch 的 ssim。
# 注意这里捕获所有异常，因此导入失败既可能是未安装，也可能是扩展二进制不兼容。
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

# 稀疏 Adam 属于另一个可选加速版本；当前锁定的标准光栅器不一定导出此类。
# 该导入结果还决定球谐直流分量是否与高阶系数分开传给光栅器，二者接口需匹配。
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

# 训练入口的三组配置各司其职：dataset 管数据/模型目录，opt 管学习率/迭代，pipe 管渲染实现。
# 三个 iterations 列表分别控制评估、导出场景、保存优化器检查点；它们不是三段训练。
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    # 显式要求稀疏优化器却没有对应扩展时立即停止，避免悄悄使用不同优化器。
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    # 记录恢复训练的起点；从头训练时下方会加一，所以实际迭代编号从 1 开始。
    first_iter = 0
    # 先建立输出目录并写入 cfg_args；后续渲染脚本用它恢复数据相关配置。
    tb_writer = prepare_output_and_logger(dataset)
    # 这里只创建参数容器与激活函数，还没有点；Scene 随后读取相机与初始点云并填充参数。
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    # Scene 把数据读取器、训练/测试相机与 GaussianModel 连接起来；场景尺度会影响位置学习率。
    scene = Scene(dataset, gaussians)
    # 为位置、球谐颜色、不透明度、尺度、旋转分别建立参数组，同时准备增密统计量。
    gaussians.training_setup(opt)
    # 恢复 checkpoint 中的高斯参数、优化器状态和迭代号。
    # 当前 capture 元组未保存曝光参数与曝光优化器，所以不要把它理解为所有状态的完整快照。
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 背景是透射率剩余部分的颜色：白底常用于合成数据，真实照片通常使用黑底。
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # CUDA 运算异步提交；使用设备事件测量 GPU 这段时间，不能用两次普通 CPU 时钟简单代替。
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 只有配置要求且扩展存在才执行稀疏更新；“扩展可用”并不等于“选用了稀疏优化器”。
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    # 逆深度损失权重随训练指数衰减。这里复用了学习率调度函数，但返回值用于损失加权。
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    # 拷贝相机列表，用 pop 做不放回随机抽样；原始 Scene 列表不会被清空。
    # 一轮列表耗尽才重新拷贝，因此每个训练相机在每轮抽样中恰好出现一次。
    viewpoint_stack = scene.getTrainCameras().copy()
    # 相机索引列表与相机栈同步弹出；本版本取得的 vind 后面没有使用，保留原逻辑。
    viewpoint_indices = list(range(len(viewpoint_stack)))
    # 日志平滑值只用于显示，不参与反向传播，也不会改变用于优化的 loss。
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # 进度条总步数排除已经完成的迭代；range 的终点不含，而下方训练循环使用 +1 包含最后一步。
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # 每次循环取一张图像监督所有当前高斯，并可能改变高斯总数。这里没有神经网络小批量数据加载器。
    for iteration in range(first_iter, opt.iterations + 1):
        # 可选网络查看器可以请求任意相机的预览，并通过 do_training 控制何时继续训练。
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                # 接收相机、继续标志、球谐/协方差计算路径及预览尺度；网络预览可临时改变 pipe 的开关。
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    # 渲染结果原为浮点通道优先张量；先限幅、转为 0～255 整数，再转为高宽通道排列并拷贝到 CPU 发送。
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                # 允许训练且没有要求在末步维持连接时退出预览循环；否则继续服务查看器。
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            # 查看器断开或通信失败只清空连接，下一轮可重连，不终止核心训练。
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # 更新位置与曝光的学习率；其他参数组沿用各自设置，不是对所有参数乘同一衰减系数。
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 逐步开放球谐阶数：先用方向无关的颜色稳定几何，再允许更复杂的视角相关外观。
        # oneupSHdegree 会限制在 max_sh_degree 内；它不改变已分配系数张量的大小。
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # 当前一轮相机采样结束，重新装填栈；随机打乱的效果来自每次随机选择一个下标并移除。
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        # randint 的上下界都包含；两个列表必须使用同一 rand_idx 才能保持对应关系。
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        # 命令行 debug_from 使用从零计数的语义，而训练 iteration 从一开始，所以这里减一比较。
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 随机背景每步抽取一个 RGB 向量，并不是为每个像素单独抽背景；可帮助透明区域训练。
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # 可微渲染返回 RGB 图、屏幕坐标梯度占位张量、可见索引、像素半径和逆深度图。
        # 后面不仅用 RGB 计算损失，还会用屏幕坐标梯度决定哪些高斯需要增加表示能力。
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        # image 为 [3,H,W]；半径为 [N]；viewspace_points 为 [N,3]，前两列梯度用于增密。
        # 本版本 visibility_filter 实际是 nonzero 返回的 [K,1] 整数索引，名字并不表示它是布尔掩码。
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 相机携带透明度/训练区域掩码时先乘到预测图。这里没有将损失除以有效像素数量。
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        # 监督目标是该相机保存的 RGB 图，移到 GPU 与预测对齐；形状也是 [3,H,W]。
        gt_image = viewpoint_cam.original_image.cuda()
        # 逐通道逐像素绝对误差取均值，量纲与图像值一致；不会平方后放大少数大误差。
        Ll1 = l1_loss(image, gt_image)
        # 融合实现要求批维，因此给 [3,H,W] 增加为 [1,3,H,W]；普通实现由 utils.loss_utils 封装。
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        # 目标为 (1−λ)·L1 + λ·(1−SSIM)：结构相似度越大越好，减一转换为越小越好的惩罚。
        # 这里代码直接用 1−SSIM，不额外乘常见 DSSIM 定义中的 1/2；以实际表达式为准。
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        # 深度监督是可选项；没有可靠深度时本轮只优化图像损失。
        Ll1depth_pure = 0.0
        # 必须同时满足调度权重大于零与相机深度可靠；不是每个数据集都天然有深度图。
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            # 虽然键名叫 depth，此版本 CUDA 输出的是按透明度权重累积的逆深度，不是普通米制深度。
            # 目标 mono_invdepth 也需经过数据读取器中的尺度/偏移对齐；两者物理意义必须一致。
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            # 先乘可靠区域掩码再对整张图取均值；无效像素贡献零，但仍计入分母。
            # 这与“只对有效像素求平均”不同，因此有效区域比例会影响该损失的大小。
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            # 把调度权重乘上深度误差后加到总损失；下面 .item() 仅把日志变量转为 Python 数值。
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # 从图像/逆深度损失启动反传，经自定义光栅器反向计算位置、颜色、不透明度、尺度和旋转梯度。
        # 屏幕坐标占位张量的 grad 也在此被填入，随后供增密统计使用。
        loss.backward()

        iter_end.record()

        # 此后是日志、结构调整与优化器更新，不需再为这些管理操作建立自动微分计算图。
        # no_grad 不会清除前面已经算出的参数梯度。
        with torch.no_grad():
            # Progress bar
            # 指数滑动平均保留 60% 历史和 40% 本轮值；显示曲线更平滑，优化仍使用原始损失。
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # 日志/评估发生在本轮优化器 step 之前；保存的场景也在下面的增密与更新之前导出。
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            # Scene.save 写 PLY 与曝光文件，适合渲染/查看；不同于含 Adam 状态的 checkpoint。
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            # 只有训练前期调整高斯数量；后期固定表示结构，使已有高斯有时间收敛。
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                # 对本轮有屏幕覆盖的高斯更新历史最大像素半径，用于识别屏幕上过大的高斯。
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 累计屏幕平面位置梯度的二范数与观测次数。
                # 梯度大表示该处投影位置的微小变化能显著影响损失，是增密的启发式信号，不是误差图本身。
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 经过预热且到达指定间隔才真正增密；每一步收集统计，不等于每一步都新增点。
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # 第一次不透明度重置之后才启用额外的像素半径裁剪门槛；此前传 None 关闭这部分筛选。
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # 小而梯度大的高斯克隆，大而梯度大的高斯分裂；随后去除过透明/过大的点。
                    # 该过程会替换 Parameter 对象并同步调整 Adam 状态，细节必须结合 gaussian_model.py 阅读。
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                # 周期性限制不透明度，避免部分高斯长期以高遮挡阻止其他点获得有效学习信号。
                # 白背景场景在增密起点还额外触发一次；这不是直接把所有高斯删除。
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            # 原代码最后一轮计算损失与保存，但不执行 optimizer.step；学习时注意这一边界条件。
            # 如果本轮增密替换了 Parameter，新参数的梯度状态需按 optimizer 管理函数理解，不能假定原对象仍在。
            if iteration < opt.iterations:
                # 先更新曝光仿射参数，再更新高斯。曝光是所有相机共享一个大 Parameter，启用时未选行通常为零梯度，
                # 仍可能因 Adam 的历史动量改变；整个曝光分支未参与计算时才通常没有该 Parameter 的梯度。
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    # 稀疏优化器接收一维布尔可见掩码，这里与 render 返回的 [K,1] 索引形式不同。
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    # 默认 Adam 根据各参数组已有梯度更新；zero_grad(set_to_none=True) 释放梯度引用，供下轮重新累计。
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # 检查点保存在本轮结构调整与优化之后，而上方 PLY 保存在之前；相同迭代编号不代表保存时刻完全相同。
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

# 输出准备：如果没有指定目录，使用调度作业号或随机标识构造目录，并保存运行配置。
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    # 保存 argparse.Namespace 的文本形式，供 render.py 等脚本合并参数；它不是模型权重文件。
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

# 评估只在 testing_iterations 指定的轮次触发，使用当前模型渲染测试相机和少量训练相机。
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        # 释放缓存分配器中未使用的显存，不能释放仍被活跃张量占用的内存；评估前这样做腾出余量。
        torch.cuda.empty_cache()
        # 训练集取固定的几个索引作为快速观察样本；取模适配少于 30 个相机的情况，可能重复抽到同一相机。
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # 启用训练/测试曝光实验时，评估右半幅；左半幅可用于拟合相机曝光，不能把两半混为同一评测区域。
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    # 每张图先算指标再累加，最后按相机数平均；这不等于把所有像素拼接后算统一 PSNR。
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

# 脚本入口：定义配置、设随机状态、可选启动查看器，最后调用唯一的训练入口。
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    # 无论用户指定哪些保存轮次，都追加最终轮次，确保最终场景得到导出。
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # 初始化 Python/NumPy/PyTorch 随机种子并设置 CUDA 设备；固定种子不代表所有 GPU 原子操作都严格确定。
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # 集群上无交互训练时可传 --disable_viewer，省去查看器监听；训练本身仍需要 CUDA。
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    # 异常追踪便于定位反传中的无效值，但会增加运行开销，正常学习实验可保持默认关闭。
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
