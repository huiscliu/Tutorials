# GPU 计算: CUDA 编程入门

GPU 加速计算是指同时利用图形处理器 (GPU) 和 CPU，加快科学、分析、工程、消费和企业应用程序的运行速度。GPU 加速器于 2007 年由 NVIDIA 率先推出，现已在世界各地为政府实验室、高校、公司以及中小型企业的高能效数据中心提供支持。GPU 能够使从汽车、手机和平板电脑到无人机和机器人等平台的应用程序加速运行.
GPU 加速计算可以提供非凡的应用程序性能，能将应用程序计算密集部分的工作负载转移到 GPU，同时仍由 CPU 运行其余程序代码。从用户的角度来看，应用程序的运行速度明显加快. 理解 GPU 和 CPU 之间区别的一种简单方式是比较它们如何处理任务。CPU 由专为顺序串行处理而优化的几个核心组成，而 GPU 则拥有一个由数以千计的更小、更高效的核心（专为同时处理多重任务而设计）组成的大规模并行计算架构。


CUDA是建立在NVIDIA的CPUs上的一个通用并行计算平台和编程模型，基于CUDA编程可以利用GPUs的并行计算引擎来更加高效地解决比较复杂的计算难题。近年来，GPU最成功的一个应用就是深度学习领域，基于GPU的并行计算已经成为训练深度学习模型的标配。


# 课程列表

* [CUDA 编程入门: 什么是 GPU 计算](https://www.youtube.com/watch?v=QLfF5sT23f8&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi)
* [CUDA 编程入门 01 GPU 硬件架构综述](https://www.youtube.com/watch?v=KsSWzijqDuk&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=2)
* [CUDA 编程入门 02 CUDA 编程模型](https://www.youtube.com/watch?v=B9bnZZWDKT0&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=3)
* [CUDA 编程入门 02-1 向量加法 程序解析](https://www.youtube.com/watch?v=LbizO1xn-Ng&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=4)
* [CUDA 编程入门 02-2 Grid, block, warp, thread 详细介绍](https://www.youtube.com/watch?v=ofzl4Ao1_SI&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=5)
* [CUDA 编程入门 03 GPU 内存介绍](https://www.youtube.com/watch?v=Ujdh4HzS4ac&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=6)
* [CUDA 编程入门 04 GPU 内存如何管理](https://www.youtube.com/watch?v=z9IxfJdhYqY&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=7)
* [CUDA 编程入门 04-1 内存管理 代码示例](https://www.youtube.com/watch?v=BctnmrFfXLs&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=8)
* [CUDA 编程入门 05 CUDA 程序执行与硬件映射](https://www.youtube.com/watch?v=KV9tDR69nUM&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=9)
* [CUDA 编程入门 06-1 什么是规约算法: 如何并行](https://www.youtube.com/watch?v=nu10CcUESQI&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=10)
* [CUDA 编程入门 06-2 并行规约算法 -1- 二叉树算法](https://www.youtube.com/watch?v=RZU35_9RzvE&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=11)
* [CUDA 编程入门 06-3 并行规约算法 -2- 改进 warp divergence](https://www.youtube.com/watch?v=HsneVj2svRU&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=12)
* [CUDA 编程入门 06-4 并行规约算法 -3- 改进共享内存访问 消除冲突](https://www.youtube.com/watch?v=wCYHVDTY448&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=13)
* [CUDA 编程入门 06-5 并行规约算法 -4- 改进全局内存访问](https://www.youtube.com/watch?v=DQ45jRg_iBY&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=14)
* [CUDA 编程入门 06-6 并行规约算法 -5- warp 内循环展开](https://www.youtube.com/watch?v=eRkEDNEYNDQ&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=15)
* [CUDA 编程入门 06-7 并行规约算法 -6- 完全循环展开](https://www.youtube.com/watch?v=hB0J0JurwAE&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=16)
* [CUDA 编程入门 06-8 并行规约算法：成功优化的关键](https://www.youtube.com/watch?v=MBf6WsNfaro&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=17)
* [CUDA 编程入门 06-9 完整并行规约算法： 三阶段算法与完整代码](https://www.youtube.com/watch?v=aBImtsYJKQM&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=18)
* [CUDA 编程入门 06-10 并行规约算法应用: 内积](https://www.youtube.com/watch?v=_4CI9GF2NAM&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=19)
* [CUDA 编程入门 07-0 CUDA 程序优化技巧](https://www.youtube.com/watch?v=5K51Sm21Ujo&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=20)
* [CUDA 编程入门 07-1 CUDA 程序优化: 探索并行化](https://www.youtube.com/watch?v=OPxqpf-DBsE&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=21)
* [CUDA 编程入门 07-2 CUDA 程序优化: GPU 内存优化策略](https://www.youtube.com/watch?v=AJArARLVPSk&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=22)
* [CUDA 编程入门 07-3 CUDA 程序优化: 指令优化](https://www.youtube.com/watch?v=BIFm-QzdDlI&list=PLSVM68VUM1eWsEX0yPliaL3pTZoKqJWfi&index=23)


# Remark
The codes only work for Linux and Mac OSX, since a little system programming is required. However, user may
try to compile under Windows by changing
```
#define USE_UNIX 1
```
to 
```
#define USE_UNIX 0
```

Windows environment isn't tested. Thanks.
