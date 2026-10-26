DS    := /opt/nvidia/deepstream/deepstream
CUDA  := /usr/local/cuda-12.1/targets/x86_64-linux
INC   := -I$(DS)/sources/includes -I$(CUDA)/include -Isrc -I/usr/include/eigen3
LIBS  := -L$(CUDA)/lib -L$(DS)/lib -lcudart -lnvinfer -lnvbufsurface -lnvbufsurftransform
CXXF  := -std=c++14 -fPIC -O2 -Wall
SRC   := src/tracker_api.cpp src/tracker.cpp src/hungarian.cpp src/kalman.cpp \
         src/reid.cpp src/crops.cpp
TARGET := libmoj.so

all: $(TARGET)

$(TARGET): $(SRC)
	g++ $(CXXF) $(INC) -shared -o $@ $^ $(LIBS)

clean:
	rm -f $(TARGET)
