
# 代数多重网格入门

多重网格方法 (Multi-Grid) 就是由对偏微分方程里得出的代数方程组的求解的研究引发出来的一种计算方法，现在多重网格方法的研究依然是一个热点，特别是在非线性非对称问题的求解上的使用。
代数多重网格（AMG）是利用几何多重网格 (Geometric Multigrid, GMG) 的一些重要原则和理念发展起来的不依赖于实际几何网格的多重网格方法。它继承了几何多重网格的主要优点，并且可以被用于更多类型的线性方程组。

* [代数多重网格入门介绍](https://www.youtube.com/playlist?list=PLSVM68VUM1eXHQEKh0WyYen2Jii_0EqCX)
  * [01 什么是多重网格](https://www.youtube.com/watch?v=R4eivWGy_Pg&index=1&list=PLSVM68VUM1eXHQEKh0WyYen2Jii_0EqCX)
  * [02 为什么单层网格效果不好](https://www.youtube.com/watch?v=_t3Vp5UTqrQ&list=PLSVM68VUM1eXHQEKh0WyYen2Jii_0EqCX&index=2)
  * [03 粗网格矫正以及多层网格结构](https://www.youtube.com/watch?v=YAaK9fizCZA&list=PLSVM68VUM1eXHQEKh0WyYen2Jii_0EqCX&index=3)
  * [04 代数多重网格的组装与求解阶段](https://www.youtube.com/watch?v=YviwqHZ65WA&list=PLSVM68VUM1eXHQEKh0WyYen2Jii_0EqCX&index=4)
  * [05 如何定义依赖集与影响集](https://www.youtube.com/watch?v=OppYZjVnf-4&list=PLSVM68VUM1eXHQEKh0WyYen2Jii_0EqCX&index=5)
  * [06 选取粗网格的两个基本原则](https://www.youtube.com/watch?v=yyQWwJqx7nU&list=PLSVM68VUM1eXHQEKh0WyYen2Jii_0EqCX&index=6)
  * [07 粗网格选取经典算法](https://www.youtube.com/watch?v=f_tYUxQ7ZuI&list=PLSVM68VUM1eXHQEKh0WyYen2Jii_0EqCX&index=7)
  * [08 插值算子矩阵如何构造](https://www.youtube.com/watch?v=sRUv7e8LyOo&list=PLSVM68VUM1eXHQEKh0WyYen2Jii_0EqCX&index=8)
  * [09 代数多重网格的复杂度, 性能](https://www.youtube.com/watch?v=3rgO4ggB0NU&index=9&list=PLSVM68VUM1eXHQEKh0WyYen2Jii_0EqCX)
  * [10 并行代数多重网格](https://www.youtube.com/watch?v=gwrdu_6KoZ0&index=10&list=PLSVM68VUM1eXHQEKh0WyYen2Jii_0EqCX)
  * [11 GPU加速代数多重网格](https://www.youtube.com/watch?v=evDlAd1ddnA&index=11&list=PLSVM68VUM1eXHQEKh0WyYen2Jii_0EqCX)
  * [12 开源串行 AMG 软件包](https://www.youtube.com/watch?v=FYth7D4Zwkk&list=PLSVM68VUM1eXHQEKh0WyYen2Jii_0EqCX&index=12)
  * [13 开源并行 AMG 软件包 BoomerAMG](https://www.youtube.com/watch?v=IV7pTshKE3Y&list=PLSVM68VUM1eXHQEKh0WyYen2Jii_0EqCX&index=13)



# 开源代数多重网格软件包

## 串行
* [FASP](http://fasp.sourceforge.net/)
* [SXAMG](https://github.com/huiscliu/sxamg)
* [HSL MI20](http://www.hsl.rl.ac.uk/catalogue/hsl_mi20.html)

## 并行
* [Hypre](https://computation.llnl.gov/projects/hypre-scalable-linear-solvers-multigrid-methods)
