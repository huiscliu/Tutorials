
default: vec_add 1B-reduction mem mem-host reduction_3

vec_add: vec_add.cu
	nvcc -O3 -Xcompiler "-Wall -Wextra" vec_add.cu -o vec_add

1B-reduction: 1B-reduction.cu
	nvcc -O3 -Xcompiler "-Wall -Wextra" 1B-reduction.cu -o 1B-reduction

mem: mem.cu
	nvcc -O3 -Xcompiler "-Wall -Wextra" mem.cu -o mem

mem-host: mem-host.cu
	nvcc -O3 -Xcompiler "-Wall -Wextra" mem-host.cu -o mem-host

reduction_3: reduction_3.cu
	nvcc -O3 -Xcompiler "-Wall -Wextra" reduction_3.cu -o reduction_3

clean:
	rm -f vec_add 1B-reduction reduction_3 mem mem-host
