CC := gcc
CFLAGS := -g -Wall -Werror -Wno-unused-function -Wno-unused-variable

all: main 
clean:
	rm -rf file_handling *.dSYM
main: file_handling.c black.c black.h
	$(CC) $(CFLAGS) -o file_handling file_handling.c black.c 

