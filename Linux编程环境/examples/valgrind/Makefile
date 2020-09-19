
default: main

CC     = gcc

CFLAGS = -Wall -Wextra -O2 -g -Wmissing-declarations -Wmissing-prototypes -Wimplicit-function-declaration -Wunused -Wuninitialized -Wvla -Wformat -Wformat-security -Wmain -Wsequence-point -Wshadow -Wbad-function-cast -Wcomments -Wdeclaration-after-statement -Waddress -Wpointer-sign -Wempty-body -Wnested-externs -Wold-style-declaration -Wold-style-definition -Wmissing-field-initializers -Wmaybe-uninitialized -Wno-return-local-addr -Wreturn-type -Wsequence-point -Wundef -Wunknown-pragmas -Wunused-label

main: main.o 
	$(CC) -o main main.o 
	@echo ""

main.o: main.c 
	$(CC) $(CFLAGS) -c main.c
	@echo ""

clean:
	rm -fv main *.o
