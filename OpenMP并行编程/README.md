
# OpenMP 并行编程: 从入门到精通

目前主流的CPU均是多核的, 少则双核, 在高端CPU中, 可以看到64核心. 这些CPU的计算能力是很强大的. 如果我们的程序是串行的, 我们只是使用了一个核心. 如果我们能使用所有的核心, 那么程序的性能会大幅度提高的. OpenMP 是一种用于共享内存并行系统的多线程程序设计方案，它是目前最成熟最主流的多核并行编程标准. 基本上, 所有的编译器都支持, 例如 GCC, Intel C++, Visual Studio. OpenMP 是一个语言标准, 它支持的编程语言包括C、C++和Fortran. 本课程详细讲述 OpenMP 的基本概念和很多高级主题. 通过本课程的学习, 您会成为多核并行编程的专家.


课程将讲述: 1) 各种并行构造; 2) 常用函数调用, 如何设置与获取线程数, 线程 ID; 3) 环境变量, 如何控制程序运行行为; 4) 多核 CPU 架构以及各种细节; 5) 内存多层次架构, UMA, NUMA, ccNUMA; 6) 任务划分方式, 例如循环, sections, single, master; 7) 规约操作以及用户自定义规约算子; 8) 任务构造; 9) 各种常用数据属性, 例如default, private, shared, firstprivate, lastprivate, threadprivate, copyin, copyprivate; 10) 高级主题等.

* [YouTube](https://www.youtube.com/playlist?list=PLSVM68VUM1eWrdw3w8cCKHLYDi3-k3T9o)

课程目录如下: \
00: 内容简介\
01.1: CPU 架构\
01.2: 系统架构\
01.3: 内存组织\
01.4: 非一致内存访问 (NUMA)\
01.5: 并行计算简介\
01.6: 并行机/超级计算机\
01.7: 并行编程\
02.1: 什么是 OpenMP\
02.2: 术语\
02.3: OpenMP 内存模型\
03: 如何编写 OpenMP 程序\
04: 如何编译 OpenMP 程序\
05: 如何执行 OpenMP 程序\
06.1: parallel 构造\
06.2: 设置线程数\
06.3: 获取线程数\
06.4: 获取线程编号 (ID)\
06.5: 例子: 向量加法\
07.0: 工作分配\
07.1: loop\
07.2: sections\
07.3: single\
07.4: master\
07.5: workshare\
07.6: 组合构造\
08.1: 规约 (reduction)\
08.2: 自定义规约\
09.1: 什么是任务\
09.2: task 构造\
09.3: 任务调度\
09.4: 例子: 生成任务\
10: 同步\
11.0: 数据属性\
11.1: default\
11.2: shared\
11.3: private\
11.4: firstprivate\
11.5: lastprivate\
11.6: threadprivate\
11.7: copyin\
11.8: copyprivate\
12.1: 非一致内存访问 (NUMA)\
12.2: 一致性高速缓存非均匀存储访问模型 (ccNUMA)\
12.3: ccNUMA 高速缓存探测策略\
12.4: 内存优化\
12.5: C/C++ 内存管理\
12.6: NUMA-Aware 编程\
12.7: 并行计算通信建模\
12.8: 区域分解\
12.9: 网格划分\
13: 总结

