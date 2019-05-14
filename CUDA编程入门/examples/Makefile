
default: vec_add 1B-reduction asum mem mem-host

vec_add: vec_add.cu
	nvcc -O3 -Xcompiler "-Wall -Wextra" vec_add.cu -o vec_add

1B-reduction: 1B-reduction.cu
	nvcc -O3 -Xcompiler "-Wall -Wextra" 1B-reduction.cu -o 1B-reduction

asum: asum.cu
	nvcc -O3 -Xcompiler "-Wall -Wextra" asum.cu -o asum

mem: mem.cu
	nvcc -O3 -Xcompiler "-Wall -Wextra" mem.cu -o mem

mem-host: mem-host.cu
	nvcc -O3 -Xcompiler "-Wall -Wextra" mem-host.cu -o mem-host

clean:
	rm -f vec_add 1B-reduction asum mem mem-host
