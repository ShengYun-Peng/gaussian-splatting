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

# 中文导读：训练进程与交互查看器之间的 TCP 通信，用查看器传入的相机渲染预览。
# 这个协议同时传训练暂停/继续、渲染计算路径和高斯显示尺度等控制值。
import torch
import traceback
import socket
import json
from scene.cameras import MiniCam

# 默认只绑定本机回环地址，端口 6009；训练入口可以通过 init 传入不同地址和端口。
host = "127.0.0.1"
port = 6009

# conn 是当前已接受的连接，addr 是对端地址；没有连接时训练循环可以继续正常训练。
conn = None
addr = None

# 创建 IPv4 TCP 监听 socket，尚未绑定/开始监听；创建模块时只分配该对象。
listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 初始化监听器；这与单次客户端连接 conn 是两个不同 socket。
def init(wish_host, wish_port):
    global host, port, listener
    host = wish_host
    port = wish_port
    # 绑定本地地址后 listen 进入监听状态，等待查看器主动连接。
    listener.bind((host, port))
    listener.listen()
    # 监听 socket 使用非阻塞模式，因此没有连接时 accept 会立刻抛异常，不让训练等待。
    listener.settimeout(0)

# 训练循环定期尝试接受连接；失败时安静返回，下轮还可重试。
def try_connect():
    global conn, addr, listener
    try:
        # 成功时得到专属于该客户端的新 socket；listener 仍负责后续连接请求。
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        # 新连接设为阻塞读取：连接后等待完整控制消息会暂停当前训练循环，符合交互查看器控制流程。
        conn.settimeout(None)
    # 这里不仅忽略“当前没有客户端”，也忽略其他 accept 异常；原实现没有更细的重试分类。
    except Exception as inst:
        pass
            
# 入站消息格式：4 字节小端无符号长度 N，随后 N 字节 UTF-8 编码的 JSON。
def read():
    global conn
    # TCP 是字节流，recv(4) 理论上可能只返回 1～3 字节；原实现没有循环补齐长度头。
    messageLength = conn.recv(4)
    # 按协议的小端序还原 JSON 字节数，不能用本机默认字节序或文本整数解析替代。
    messageLength = int.from_bytes(messageLength, 'little')
    # 消息体也只调用一次 recv；真实网络下可能出现短读，随后 JSON 解码会失败。
    # 这是当前简化协议读取器的限制，sendall 并不能保证对端一次 recv 就取到全部内容。
    message = conn.recv(messageLength)
    # 先解码字节为文本，再解析为字典；预览相机和控制开关在 receive 中按固定键取出。
    return json.loads(message.decode("utf-8"))

# 出站格式与入站不同：可选原始 RGB 图像字节，之后是 4 字节验证字符串长度与 ASCII 字符串。
# 图像没有单独长度头，查看器根据自己请求的宽高和每像素 3 字节来解析图像部分。
def send(message_bytes, verify):
    global conn
    # 训练调用方已把渲染转换为 [H,W,3] uint8 连续字节；这里直接发送，不做 PNG/JPEG 编码。
    if message_bytes != None:
        # sendall 会尝试发送全部字节，避免本地一次 send 只发出部分数据；失败仍会抛异常。
        conn.sendall(message_bytes)
    # verify 通常是源数据路径，按 ASCII 编码发送；ASCII 下字符数与字节数一致。
    # 原接口不接受非 ASCII 验证字符串，bytes(verify, "ascii") 会因此失败。
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))

# 把 JSON 控制消息解码成 MiniCam 与五个控制值，供 train.py 中的交互循环直接解包。
def receive():
    message = read()

    # 请求渲染分辨率来自查看器，不一定与训练相机图像分辨率一致。
    width = message["resolution_x"]
    height = message["resolution_y"]

    # 零宽或零高表示本次没有有效预览相机，函数会在末尾返回六个 None。
    if width != 0 and height != 0:
        try:
            # 是否继续训练由查看器控制；FoV 是以弧度表示的垂直/水平视场角，znear/zfar 是裁剪参数。
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            # 这两个布尔值分别选择 Python SH 求色和 Python 协方差构造路径，会被训练端写入 pipeline。
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            # 训练达到末步后是否继续保持查看器服务，由训练循环结合迭代计数决定。
            keep_alive = bool(message["keep_alive"])
            # 预览时额外缩放高斯尺度，用于观察点的覆盖范围；不在这里修改模型保存的尺度参数。
            scaling_modifier = message["scaling_modifier"]
            # 把查看器提供的扁平矩阵按协议重排成 [4,4] 并放到 CUDA。
            # 这里不再做 transpose，假定发送端序列与项目要求的矩阵存储顺序已匹配。
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            # 视图矩阵第 1、2 列（从 0 计）分别取反，适配查看器与训练端的相机 y/z 方向约定。
            # 该操作建立在当前转置矩阵存储方式上，不能直接照搬为任意列向量矩阵的同名操作。
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            # 另读发送端已经合成的视图投影矩阵；它不是仅含透视投影的 P。
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            # 合成矩阵按此协议只翻转第 1 列；不要误以为必须和上面的视图矩阵翻转完全相同列。
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            # 构造只含渲染几何信息的轻量相机，没有训练真值图像/逆深度；MiniCam 从逆视图矩阵取相机中心。
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        # 字段缺失、矩阵形状错误或 CUDA 转换失败时打印异常栈并抛回训练端，由其清理连接状态。
        except Exception as e:
            print("")
            traceback.print_exc()
            raise e
        # 返回顺序与 train.py 的六变量解包必须一致，尤其注意 train 和 keep_alive 的作用不同。
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return None, None, None, None, None, None