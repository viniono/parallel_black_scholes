CC := nvcc
CFLAGS := -g 

all: main
clean:
	rm -rf main *.dSYM *.o

main: main.cu file_handling.cuh black.cuh time_util.cuh 
	$(CC) $(CFLAGS) -o main main.cu 
