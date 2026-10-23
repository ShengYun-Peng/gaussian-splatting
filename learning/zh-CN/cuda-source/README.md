# CUDA 光栅器中文逐段精读

这里是原始 CUDA 子模块的**带中文注释参考副本**，不是训练时直接编译或导入的代码。主仓库仍使用 `submodules/diff-gaussian-rasterization/`，其源码与 gitlink 未被修改。为了让你在自己的主仓库中直接查看 CUDA 中文讲解，同时保留可复现的上游依赖，学习副本与可选补丁分别保存。

来源仓库：[graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)。固定提交：`9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0`。本目录每个 `.py`、`.cu`、`.cpp`、`.h` 的相对路径，都与原子模块内同名源文件一一对应；仅新增中文行注释，保留原版权声明。原许可全文见 [LICENSE.md](LICENSE.md)，增加注释不改变原许可。

## 建议阅读顺序

1. `diff_gaussian_rasterization/__init__.py`：Python 输入约定、屏幕梯度占位、自定义自动求导与缓存。
2. `ext.cpp` 与 `rasterize_points.cu`：扩展注册、张量到指针、前向缓存和全零梯度分配。
3. `cuda_rasterizer/rasterizer_impl.cu`：前缀和、实例复制、排序键、基数排序、块区间和调度。
4. `cuda_rasterizer/forward.cu`：球谐颜色、三维协方差、投影雅可比、二维滤波、透明度合成与逆深度。
5. `cuda_rasterizer/backward.cu`：逆序重建、遮挡导数、原子累加、矩阵求逆与参数链式法则。
6. `cuda_rasterizer/auxiliary.h` 与各接口头文件：矩阵索引、归一化导数、像素中心、近裁剪和内存布局。

注释解释公式、张量/数组形状、变量含义、分支原因与前后向的对应关系。不会把源码中的近似或版本细节包装成严格数学结论：特别标注了四元数归一化实际在 Python 侧、只检查近裁剪的 `in_frustum`、触发提前终止的当前高斯不被合成、逆深度未归一化、alpha 截顶的近似反向、非单位 `scale_modifier` 梯度因子，以及抗锯齿反向中已滤波对角元素的使用。后两处建议在研究相应模式时做针对性有限差分；本次注释不修改这些原有行为。

## 可选：让本地子模块也显示中文注释

在主仓库根目录执行：

```bash
git -C submodules/diff-gaussian-rasterization apply --check ../../learning/zh-CN/rasterizer-zh.patch
git -C submodules/diff-gaussian-rasterization apply ../../learning/zh-CN/rasterizer-zh.patch
```

该补丁只增加注释，路径相对原子模块根目录，要求原子模块处于上述固定版本且没有冲突修改。执行第二条命令会使本地子模块工作区显示为已修改；它不会自动提交或推送子模块，也不影响上游远程仓库。训练构建仍使用子模块里的文件。

需要撤销本补丁时，在确认仍只有本补丁改动的情况下先检查再反向应用：

```bash
git -C submodules/diff-gaussian-rasterization apply --reverse --check ../../learning/zh-CN/rasterizer-zh.patch
git -C submodules/diff-gaussian-rasterization apply --reverse ../../learning/zh-CN/rasterizer-zh.patch
```

## 校验范围

生成时检查 Python 抽象语法树不变，并用 Clang 原始词法分析器比较 C++/CUDA 的非注释、非空白 token；字符串、字符字面量和预处理 token 都保留比较。补丁通过 `git apply --check`，在临时副本中应用后逐字节比对参考副本。该检查证明注释没有改变代码 token；它不等同于编译了 CUDA 扩展或验证了全部数值梯度。
