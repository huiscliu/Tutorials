
default: vec_add 1B-reduction mem mem-host reduction_1 reduction_2  vec_dot

vec_add: vec_add.cu
	nvcc -O3 -Xcompiler "-Wall -Wextra" vec_add.cu -o vec_add

1B-reduction: 1B-reduction.cu
	nvcc -O3 -Xcompiler "-Wall -Wextra" 1B-reduction.cu -o 1B-reduction

mem: mem.cu
	nvcc -O3 -Xcompiler "-Wall -Wextra" mem.cu -o mem

mem-host: mem-host.cu
	nvcc -O3 -Xcompiler "-Wall -Wextra" mem-host.cu -o mem-host

reduction_1: reduction_1.cu
	nvcc -O3 -Xcompiler "-Wall -Wextra" reduction_1.cu -o reduction_1

reduction_2: reduction_2.cu
	nvcc -O3 -Xcompiler "-Wall -Wextra" reduction_2.cu -o reduction_2

vec_dot: vec_dot.cu
	nvcc -O3 -Xcompiler "-Wall -Wextra" vec_dot.cu -o vec_dot

clean:
	rm -f vec_add 1B-reduction reduction_1 reduction_2 mem mem-host vec_dot
