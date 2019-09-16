LIBNAME         = projmgmt
CPP             = /usr/lib64/ccache/gcc -E
CPPFLAGS        = -I/home/liuhui/codehub/projmgmt/autoconf/include  -I/usr/local/include -I/usr/local/superlu/include

CXX             = g++
CXXFLAGS        = -g -O2 -Wno-deprecated -Wall -std=c++11 -Wmissing-declarations -Wunused -Wuninitialized -Wvla -Wformat -Wformat-security -Wmain -Wsequence-point -Wshadow -Waddress -Wcomments -Wmain -Wsequence-point -Wempty-body -Wmissing-field-initializers -Wmaybe-uninitialized -Wno-return-local-addr -Wreturn-type -Wsequence-point -Wundef -Wunknown-pragmas -Wunused-label

CC              = /usr/lib64/ccache/gcc 
CFLAGS          = -g -O2 -Wall -Wextra -Wmissing-declarations -Wmissing-prototypes -Wimplicit-function-declaration -Wunused -Wuninitialized -Wvla -Wformat -Wformat-security -Wmain -Wsequence-point -Wshadow -Wbad-function-cast -Wcomments -Wdeclaration-after-statement -Waddress -Wpointer-sign -Wempty-body -Wnested-externs -Wold-style-declaration -Wold-style-definition -Wmissing-field-initializers -Wmaybe-uninitialized -Wno-return-local-addr -Wreturn-type -Wsequence-point -Wundef -Wunknown-pragmas -Wunused-label

FC              = /usr/bin/gfortran 
FCFLAGS         = -g -O2
FCLIBS          =  -L/usr/lib/gcc/x86_64-redhat-linux/9 -L/usr/lib/gcc/x86_64-redhat-linux/9/../../../../lib64 -L/lib/../lib64 -L/usr/lib/../lib64 -L/usr/lib/gcc/x86_64-redhat-linux/9/../../.. -lgfortran -lm -lquadmath

LINKER          = g++
LDFLAGS         = -L/home/liuhui/codehub/projmgmt/autoconf/src  -L/usr/local/lib -L/usr/local/superlu/lib
LIBS            = -lprojmgmt  -lsuperlu  -lblas -L/usr/local/lib  -lm  -L/usr/lib/gcc/x86_64-redhat-linux/9 -L/usr/lib/gcc/x86_64-redhat-linux/9/../../../../lib64 -L/lib/../lib64 -L/usr/lib/../lib64 -L/usr/lib/gcc/x86_64-redhat-linux/9/../../.. -lgfortran -lm -lquadmath -lm

AR              = ar
RANLIB          = ranlib

prefix		= /usr/local/projmgmt
exec_prefix	= ${prefix}
bindir		= ${exec_prefix}/bin
libdir		= ${exec_prefix}/lib
includedir	= ${prefix}/include
RPATH_FLAG	= -Wl,-rpath,

%.o : %.cxx
	${CXX} ${CXXFLAGS} ${CPPFLAGS} -c $*.cxx
	@echo

%.o : %.c
	${CC} ${CFLAGS} ${CPPFLAGS} -c $*.c
	@echo

%.o : %.f
	${FC} ${FCFLAGS} -c $*.f
	@echo

% : %.o
	${LINKER} ${LDFLAGS} -o $@ $*.o ${LIBS}
	@echo

