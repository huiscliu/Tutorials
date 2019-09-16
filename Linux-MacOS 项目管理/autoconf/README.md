# projmgmt
Linux /Mac OS Project Management

**projmgmt** defines a framework for project management, which supports C, C++ and Fortran. However, this framework
can be modified to meet other requirements.

The framework assumes 1) header files (.h) are placed in **include**; 2) implementation files (.c, .cxx, .c++, .cpp, .C, .f,
.for) are placed in **src**.

# How to Use

1) change projmgmt to your own project in **Makefile.inc.in** and **configure.in**, such as btree.

2) modify **configure.in** to add/remove external packages, functions and files;

3) run: **autoconf**

4) remove all headers in **include** and source files in **src**, and put your codes to **include** and
**src**;

5) go to **src**, run: **../utils/makedeps** to generate Makefile.dep

# How to Configure
```
./configure

./configure --help
```

# How to Compile
```
make
```

# How to Clean Objects
```
make clean
```

# How to Clean All
```
make distclean
```
