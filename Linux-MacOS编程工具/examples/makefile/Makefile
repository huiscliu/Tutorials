
default: main

CC     = gcc

CFLAGS = -Wall -Wextra -O2 -g -Wmissing-declarations -Wmissing-prototypes -Wimplicit-function-declaration -Wunused -Wuninitialized -Wvla -Wformat -Wformat-security -Wmain -Wsequence-point -Wshadow -Wbad-function-cast -Wcomments -Wdeclaration-after-statement -Waddress -Wpointer-sign -Wempty-body -Wnested-externs -Wold-style-declaration -Wold-style-definition -Wmissing-field-initializers -Wmaybe-uninitialized -Wno-return-local-addr -Wreturn-type -Wsequence-point -Wundef -Wunknown-pragmas -Wunused-label

main: main.o utils.o add.o
	$(CC) -o main main.o utils.o add.o
	@echo ""

main.o: main.c utils.h add.h
	$(CC) $(CFLAGS) -c main.c
	@echo ""

utils.o: utils.c utils.h
	$(CC) $(CFLAGS) -c utils.c
	@echo ""

add.o: add.c add.h
	$(CC) $(CFLAGS) -c add.c
	@echo ""

clean:
	rm -fv main *.o
