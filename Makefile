CC := nvcc
CFLAGS := -g 

all: main
clean:
	rm -rf main *.dSYM *.o

main: main.cu file_handling.h black.h time_util.h Makefile
	$(CC) $(CFLAGS) -o main main.cu
