# GPU 计算: CUDA 编程入门

GPU 加速计算是指同时利用图形处理器 (GPU) 和 CPU，加快科学、分析、工程、消费和企业应用程序的运行速度。GPU 加速器于 2007 年由 NVIDIA 率先推出，现已在世界各地为政府实验室、高校、公司以及中小型企业的高能效数据中心提供支持。GPU 能够使从汽车、手机和平板电脑到无人机和机器人等平台的应用程序加速运行.
GPU 加速计算可以提供非凡的应用程序性能，能将应用程序计算密集部分的工作负载转移到 GPU，同时仍由 CPU 运行其余程序代码。从用户的角度来看，应用程序的运行速度明显加快. 理解 GPU 和 CPU 之间区别的一种简单方式是比较它们如何处理任务。CPU 由专为顺序串行处理而优化的几个核心组成，而 GPU 则拥有一个由数以千计的更小、更高效的核心（专为同时处理多重任务而设计）组成的大规模并行计算架构。


CUDA是建立在NVIDIA的CPUs上的一个通用并行计算平台和编程模型，基于CUDA编程可以利用GPUs的并行计算引擎来更加高效地解决比较复杂的计算难题。近年来，GPU最成功的一个应用就是深度学习领域，基于GPU的并行计算已经成为训练深度学习模型的标配。


# 高级课程
* [GPU 计算: CUDA 编程从入门到精通](https://www.udemy.com/course/cuda-yxp/?couponCode=064748E49AF03019A837)

# 部分课程
* [YouTube: CUDA 编程入门](https://www.youtube.com/playlist?list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi)

# Remark
The codes are only tested under Linux, where a little system programming is required. However, user may
try to compile and run under Windows. But Windows environment isn't tested. Thanks.

Some codes requires computer capability is 8.0 or higher.

# 课程列表

1-0-什么是 GPU 计算\
1-1-GPU 架构综述\
1-1-处理器空间\
1-1-内存空间\
1-2-计算能力\
1-3-如何编写 CUDA 程序\
1-4-如何编译 CUDA 程序\
1-5-CUDA 函数修饰符\
1-6-内存修饰符\
1-7-内建向量\
1-8-内建变量\
2-0-CUDA 编程模型\
2-1-硬件映射\
2-2-向量加法示例\
2-3-主机函数: __host__\
2-4-设备函数: __device__\
2-5-核函数: __global__\
2-6-网格\
2-7-线程块\
2-8-网格维度: gridDim\
2-9-线程块维度: blockDim\
2-10-线程块 ID: blockIdx\
2-11-线程 ID: threadIdx\
2-12-线程调度\
2-13-线程块与线程编号映射\
2-14-向量加法示例\
2-15-如何启动核函数\
2-16-线程执行顺序\
3-0-GPU 内存\
3-1-CPU 内存\
3-2-页锁定内存\
4-0-GPU 内存管理综述\
4-1-CPU 内存管理综述\
4-2-页锁定内存\
4-3-1-全局内存\
4-3-2-全局内存示例\
4-4-1-共享内存\
4-4-2-共享内存 bank 冲突\
4-4-3-共享内存规约算法\
4-5-1-内存拷贝\
4-5-2-示例\
4-6-向量操作\
4-7-稀疏矩阵向量乘法\
4-8-地址空间\
5-0-同步\
5-1-核函数同步\
5-2-线程块同步\
5-3-Warp 同步\
5-4-Warp 同步深入理解\
6-1-什么是规约算法- 如何并行\
6-2-并行规约算法-1: 二叉树算法\
6-3-并行规约算法-2: 改进 warp divergence\
6-4-并行规约算法-3: 改进共享内存访问\
6-5-并行规约算法-4: 改进全局内存访问\
6-6-并行规约算法-5: warp 内循环展开\
6-7-并行规约算法-6: 完全循环展开\
6-8-并行规约算法：成功优化的关键\
6-9-完整并行规约算法：三阶段算法与完整代码\
6-10-并行规约算法应用: 内积\
7-0-线程调度概述\
7-1-Warp 投票函数\
7-2-Warp 匹配函数\
7-3-Warp 规约函数\
7-4-Warp 内通信: 数据交换\
8-1-Cooperative Groups\
8-2-隐式组类型\
8-3-显式组类型\
8-4-Coalesced Groups\
8-5-组划分\
8-6-Group Labeled Partition\
8-7-Group Binary Partition\
8-8-组同步\
8-9-网格同步\
9-1-CUDA 程序概述\
9-2-CUDA 程序优化: 探索并行化\
9-3-CUDA 程序优化: GPU 内存优化策略\
9-4-CUDA 程序优化: 指令优化\
