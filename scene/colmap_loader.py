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

# 中文导读：读取 COLMAP 稀疏重建的相机、图像位姿与三维点，兼容文本和二进制文件。
# 二进制布局由格式字符串精确描述；文件里的 ID 是标识符，不保证连续，不能直接当数组下标。
# 这里负责解码，dataset_readers.py 再负责选择支持的相机模型、转换约定与组装场景。
import numpy as np
import collections
import struct

# 模型定义表记录编号、名称和内参个数，用于确定 cameras.bin 后续应读取多少个 double。
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
# Camera 表示内参记录：同一相机内参 ID 可以被多张不同姿态的图像共享。
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
# Image 表示每张图的外参及二维观测。qvec=(w,x,y,z)，外参满足 X_camera=R(q)X_world+t。
# xys:[M,2] 保存二维点坐标，point3D_ids:[M] 保存对应三维点 ID，未关联点通常为 -1。
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
# 完整三维点记录还可保存观测轨迹；本文件实际读取点云的入口只返回坐标、颜色和误差数组。
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
# 常见模型内参排列：SIMPLE_PINHOLE 为 [f,cx,cy]，PINHOLE 为 [fx,fy,cx,cy]。
# 其他模型还带径向/切向等畸变参数；能解码这些记录不等于训练流水线支持带畸变图像。
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
# 建立编号/名称两种查找表，避免每读一台相机都遍历模型集合。
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


# 单位四元数转 3×3 旋转矩阵，采用 COLMAP 的实部在前顺序 (w,x,y,z)。
# 原函数不重新归一化输入；读取的四元数应已合法。此处返回未转置的世界到相机旋转。
# 相机在世界坐标中的中心为 -Rᵀt，而非外参中的 t。
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

# 旋转矩阵转四元数：通过对称特征值问题提取单位四元数，避免只依赖旋转迹的分段公式。
def rotmat2qvec(R):
    # 按 NumPy 展平顺序拆出九个元素；变量命名与后续 K 公式配套，不要单独交换变量顺序。
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    # 构造 4×4 矩阵的下三角信息，除以 3 只统一整体尺度，不改变特征向量。
    # 上三角此处填零；后面的 eigh 默认读取下三角并按对称矩阵解释它。
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    # 对实对称矩阵求特征值/特征向量；eigvecs 的每一列对应一个特征向量。
    eigvals, eigvecs = np.linalg.eigh(K)
    # 取最大特征值对应向量，并把内部分量顺序重排成公开接口的 (w,x,y,z)。
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    # q 与 -q 表示同一个旋转；统一令实部非负，减少保存/比较时的符号歧义。
    if qvec[0] < 0:
        qvec *= -1
    return qvec

# 在不可变字段容器上添加便捷方法，可直接由图像记录计算其旋转矩阵。
class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

# 二进制解码底层工具：默认小端 <，字段不使用本机对齐填充。
# 常见格式：Q=无符号 64 位整数，q=有符号 64 位整数，i=有符号 32 位整数，
# d=64 位浮点，B=无符号单字节，c=单字节字符；num_bytes 必须与格式总长度一致。
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    # 读取恰当长度并解包为 Python 元组；若文件被截断、长度不足，struct.unpack 会抛异常。
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

# 读取 points3D.txt，返回 xyzs:[N,3]、rgbs:[N,3]、errors:[N,1]。
# 每行结构为 ID X Y Z R G B ERROR 后接观测轨迹；初始化高斯只需要前面的几何/颜色。
def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    # 先占位并统计有效记录数，再一次性分配 NumPy 数组；因此文本点云会读取两遍。
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                # 忽略空行和 # 开头注释行，其余行各代表一个三维点。
                num_points += 1


    # 按有效点数预分配；未指定 dtype 时默认为浮点，RGB 读取后也会存进浮点数组。
    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                # 跳过 elems[0] 的点 ID，读取三个世界坐标。输出数组按文件顺序紧凑排列，不保留稀疏 ID。
                xyz = np.array(tuple(map(float, elems[1:4])))
                # 读取 0～255 的 RGB；此处不除以 255，归一化由后续 PLY/场景读取流程完成。
                rgb = np.array(tuple(map(int, elems[4:7])))
                # ERROR 为 COLMAP 点重投影误差；后续可用于诊断，但高斯初始化在上层忽略此返回值。
                error = np.array(float(elems[7]))
                # 逐行填入预分配数组，再增加计数；后面的 TRACK 项在此文本入口中没有解析。
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors

# 读取 points3D.bin。固定字段之后还有变长轨迹，即使不使用轨迹也必须消费这些字节才能对齐下一条。
def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        # 文件开头 8 字节是三维点总数 Q；[0] 从单元素元组中取标量。
        num_points = read_next_bytes(fid, 8, "Q")[0]

        # 按记录数分配坐标、颜色和误差；数组行号 p_id 与文件中的 POINT3D_ID 不一定相同。
        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            # 每条点记录固定部分共 43 字节：ID 8 + XYZ 3×8 + RGB 3×1 + ERROR 8。
            # 小端标准尺寸模式无结构体对齐字节，所以不是某些 C 结构体的 sizeof 大小。
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            # 跳过第 0 项点 ID，按固定字段顺序切出 XYZ、RGB、ERROR。
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            # 固定部分后先读轨迹长度 Q，再读该点被哪些图像中的哪些二维特征观测到。
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            # 每个轨迹元素由 image_id 与 point2D_idx 两个 32 位整数构成，共 8 字节。
            # 即使本函数不返回 track_elems，也必须读取它们，使文件指针位于下一点的起始处。
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

# 读取 cameras.txt，每行包含 CAMERA_ID MODEL WIDTH HEIGHT PARAMS...。
# 此文本入口有更严格的 PINHOLE 断言，与二进制解码器和上层 SIMPLE_PINHOLE 支持范围不同。
def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                # 相机 ID 作为字典键；它不是本文件的行号，同一 ID 可供多张图像复用。
                camera_id = int(elems[0])
                model = elems[1]
                # 保留原实现约束：文本相机仅接受 PINHOLE。不能因为模型表里列出其他模型就认为这里都可加载。
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                # PINHOLE 参数按 [fx,fy,cx,cy] 读成浮点数组，焦距/主点都采用像素单位。
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

# 读取 images.bin：一张注册图像一个外参记录，同时携带该图的全部二维特征/三维点关联。
def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        # 开头的无符号 64 位计数表示成功注册到重建中的图像数量，未必等于输入目录图片总数。
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            # 固定外参记录 64 字节：image_id 4 + 四元数 4×8 + 平移 3×8 + camera_id 4。
            # 格式 idddddddi 中的七个 d 对应四元数与平移，两个 i 分别是图像 ID 和内参 ID。
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            # 四元数顺序为 (w,x,y,z)，与 qvec2rotmat 一致；tvec 是世界到相机平移。
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            # 固定字段后是以 NUL（零字节）结尾的变长文件名，不是预先带长度的字符串。
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            # 逐字节读到 NUL 后停止；原实现逐字节做 UTF-8 解码，遇到多字节非 ASCII 文件名可能失败。
            # 结束符本身已经被消费，下一字节随即是二维点数量字段。
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            # 图像名之后的 Q 指明二维特征数 M；每个特征随后固定占 24 字节。
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            # 每条观测是 x(double)、y(double)、point3D_id(int64)，共 8+8+8 字节。
            # 三维点 ID 使用有符号 q，才能表示尚未三角化关联的 -1。
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            # 扁平元组按步长 3 拆出 x/y/ID；column_stack 将两列坐标组织为 [M,2]。
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            # 按图像 ID 存入字典；保持原始世界到相机姿态，上层读取器再将 R 转置成仓库约定。
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


# 读取 cameras.bin，根据模型编号决定每条相机记录附带多少内参 double。
def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        # 开头是内参记录总数，不是图像数；多张图片共享内参时 num_cameras 会小得多。
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            # 固定头为 24 字节：camera_id 4 + model_id 4 + width 8 + height 8，即 iiQQ。
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            # 通过模型表把二进制 model_id 解释为名称；未知/新增编号不在表中时会触发 KeyError。
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            # 不同模型内参数量不同，例如 PINHOLE 4 个 double，OPENCV 8 个 double。
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            # 按模型要求精确读取内参，之后文件指针即位于下一相机记录。
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        # 校验唯一相机 ID 的数量与文件声明一致；重复 ID 会覆盖字典项，从而使此断言失败。
        assert len(cameras) == num_cameras
    return cameras


# 读取 images.txt。每张图使用两行：第一行是位姿/相机 ID/文件名，第二行是二维特征三元组。
# 即使一张图没有二维点，也需要消费其第二行，不能把所有非空行一概当外参记录。
def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                # 第一行按 ID、qw qx qy qz、tx ty tz、camera_id、name 的顺序解析。
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                # 原解析器按空白分词，直接取第 10 个字段作文件名；这里不支持含空格的完整名字拼回。
                image_name = elems[9]
                # 立即读取该图的第二行；每三个字段依次为 x、y、point3D_id。
                elems = fid.readline().split()
                # 步长切片分别取 x 和 y 拼成 [M,2]，并单独构造长度 M 的三维点 ID 数组。
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


# 读取 COLMAP 稠密深度/法线等二进制数组，格式与上面的稀疏重建文件不同。
# 前缀是文本“宽&高&通道&”，之后紧接 float32 数组，既不能按纯文本读，也不是 NumPy .npy。
def read_colmap_bin_array(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py

    :param path: path to the colmap binary file.
    :return: nd array with the floating point values in the value
    """
    with open(path, "rb") as fid:
        # 先从文本头提取三个整数，确定后续数组尺寸；解析时仅选前三个以 & 分隔的字段。
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        # 文本读取器可能读取额外缓冲字节，因此重新从文件开头定位二进制数据边界。
        fid.seek(0)
        # 按字节数到第三个 &，之后的位置就是 float32 有效负载的开始。
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                # 消费第三个分隔符后停止扫描。原循环未单独处理损坏文件的提前 EOF，要求输入头格式完整。
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        # 从当前位置读取余下 float32。此调用使用 NumPy 本机字节序，常见运行环境为小端。
        array = np.fromfile(fid, np.float32)
    # COLMAP 数组采用与该 Fortran 顺序 reshape 配套的存储布局，先恢复 [W,H,C]。
    array = array.reshape((width, height, channels), order="F")
    # 交换宽高变为常见 [H,W,C]；squeeze 删除所有长度为 1 的维，单通道通常得到 [H,W]。
    return np.transpose(array, (1, 0, 2)).squeeze()
