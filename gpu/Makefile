NVCC = nvcc
ARCH ?= -arch=sm_70
CXXSTD = -std=c++14
CFLAGS = -O3 $(CXXSTD)

TARGET = reduction
SRC = main.cu

all: $(TARGET)

$(TARGET): $(SRC) reduction.cuh sum_max_ops.cuh
	$(NVCC) $(CFLAGS) $(ARCH) -o $@ $(SRC)

clean:
	rm -f $(TARGET)
