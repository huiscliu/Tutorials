
.PHONY: defaullt all clean

default: all

CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -g -fopenmp
LDFLAGS = -lgomp -lm

SRC     = $(wildcard *.c)
DIR     = $(notdir $(SRC))
BINS    = $(patsubst %.c,%,$(DIR))


all: $(BINS)


% : %.c
	${CC} -o $@ $*.c $(CFLAGS) ${LDFLAGS} 

clean:
	rm -f $(BINS)
