CC := gcc
CFLAGS := -g -Wall -Werror -Wno-unused-function -Wno-unused-variable


all: main
clean:
	rm -rf file_handling *.dSYM
main: 
	$(CC) $(CFLAGS) -o file_handling file_handling.c black.c  
