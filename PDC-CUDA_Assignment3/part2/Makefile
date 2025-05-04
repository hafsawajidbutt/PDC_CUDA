EXECUTABLE := part2

CU_FILES   := prefix_sum.cu find_repeats.cu

CC_FILES   := main.cpp

OBJDIR=objs
CXXFLAGS=-O3 -Wall
NVCC=nvcc
NVCCFLAGS=-O3 -m64

OBJS=$(OBJDIR)/main.o $(OBJDIR)/prefix_sum.o $(OBJDIR)/find_repeats.o

.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
    mkdir -p $(OBJDIR)/

clean:
    rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
    $(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(OBJDIR)/%.o: %.cpp
    $(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
    $(NVCC) $< $(NVCCFLAGS) -c -o $@
