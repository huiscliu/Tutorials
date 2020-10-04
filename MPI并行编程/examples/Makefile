
.PHONY: defaullt all clean

default: all

CC      = mpicc
CFLAGS  = -Wall -Wextra -O2 -g
LDFLAGS = -lmpi -lm

SRC     = $(wildcard *.c)
DIR     = $(notdir $(SRC))
BINS    = $(patsubst %.c,%,$(DIR))


all: $(BINS)


% : %.c
	${CC} -o $@ $*.c $(CFLAGS) ${LDFLAGS} 

clean:
	rm -f $(BINS)
