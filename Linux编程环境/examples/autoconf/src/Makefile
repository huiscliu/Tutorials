
.SUFFIXES: .cxx

default: lib
all: lib

include ../Makefile.inc
include Makefile.dep

SRC_CXX   = $(wildcard *.cxx)
DIR_CXX   = $(notdir $(SRC_CXX))
OBJS_CXX  = $(patsubst %.cxx,%.o,$(DIR_CXX))

SRC_C   = $(wildcard *.c)
DIR_C   = $(notdir $(SRC_C))
OBJS_C  = $(patsubst %.c,%.o,$(DIR_C))

SRC_FOR   = $(wildcard *.f)
DIR_FOR   = $(notdir $(SRC_FOR))
OBJS_FOR  = $(patsubst %.f,%.o,$(DIR_FOR))

OBJS     := $(OBJS_CXX) $(OBJS_C) $(OBJS_FOR)

lib: lib$(LIBNAME).a

lib$(LIBNAME).a: ${OBJS}
	${AR} -ru $@ $(OBJS)
	${RANLIB} $@

.PHONY: clean

clean:
	@rm -fv lib$(LIBNAME).a
	@rm -fv *.o 
