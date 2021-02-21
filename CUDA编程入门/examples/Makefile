
CUCC     = nvcc
CUFLAGS  = -O3 -arch=sm_80 -Xcompiler "-Wall -Wextra"

SRC_CU   = $(wildcard *.cu)
DIR_CU   = $(notdir $(SRC_CU))
BIN_CU  = $(patsubst %.cu,%,$(DIR_CU))

default: ${BIN_CU}

% : %.cu
	${CUCC} ${CUFLAGS} -o $@ $*.cu

clean:
	rm -fv ${BIN_CU}
