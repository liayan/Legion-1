NVCC = nvcc
CFLAGS = -I/home/zhzh/bam/src/linux

all: decouple

test: test.cu common.cuh iostack.cuh ssdqp.cuh
	$(NVCC) $(CFLAGS) -o $@ $< -g

decouple: test_read.cu common.cuh iostack_decouple.cuh ssdqp.cuh
	$(NVCC) $(CFLAGS) -o $@ $< -g

clean:
	rm -f test decouple
