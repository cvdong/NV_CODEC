# makefile
cc        := g++
name      := libffhdd.so
workdir   := workspace
srcdir    := src
objdir    := objs
stdcpp    := c++11
cuda_home := /home/env/trt8cuda115cudnn8
cuda_arch := 

cpp_srcs := $(shell find $(srcdir) -name "*.cpp")
cpp_objs := $(cpp_srcs:.cpp=.cpp.o)
cpp_objs := $(cpp_objs:$(srcdir)/%=$(objdir)/%)
cpp_mk   := $(cpp_objs:.cpp.o=.cpp.mk)

cu_srcs := $(shell find $(srcdir) -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cu.o)
cu_objs := $(cu_objs:$(srcdir)/%=$(objdir)/%)
cu_mk   := $(cu_objs:.cu.o=.cu.mk)

# lib
link_cuda      := cuda cudart cudnn nvcuvid nvidia-encode
link_ffmpeg    := avcodec avformat swresample swscale avutil
link_sys       := stdc++ dl python3.8
link_librarys  := $(link_cuda) $(link_sys) $(link_ffmpeg)

nvcc              := $(cuda_home)/bin/nvcc -ccbin=$(cc)
include_cuda      := $(cuda_home)/include/cuda
current_dir       := $(shell pwd)

lib_all        	  := $(cuda_home)/lib64  $(current_dir)ffmpeg/lib  /lib/x86_64-linux-gnu /home/anaconda3/envs/yolo/lib

# .h
include_paths := src    \
    $(include_cuda)     \
	/home/anaconda3/envs/yolo/include/python3.8 \
	ffmpeg/include      \
	cuvid-include        

library_paths := $(lib_all)

empty := 
library_path_export := $(subst $(empty) $(empty),:,$(library_paths))

run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

# compile
cpp_compile_flags := -std=$(stdcpp) -w -g -O3 -m64 -fPIC -fopenmp -pthread
cu_compile_flags  := -std=$(stdcpp) -w -g -O3 -m64 $(cuda_arch) -Xcompiler "$(cpp_compile_flags)"
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        += $(library_paths) $(link_librarys) $(run_paths)

ifneq ($(MAKECMDGOALS), clean)
-include $(cpp_mk) $(cu_mk)
endif

$(name)   : $(workdir)/$(name)

# make run link
run       : $(name)
	@cd $(workdir) && python test.py

$(workdir)/$(name) : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) -shared $^ -o $@ $(link_flags)

$(objdir)/%.cpp.o : $(srcdir)/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ $(cpp_compile_flags)

$(objdir)/%.cu.o : $(srcdir)/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)

$(objdir)/%.cpp.mk : $(srcdir)/%.cpp
	@echo Compile depends C++ $<
	@mkdir -p $(dir $@)
	@$(cc) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)

$(objdir)/%.cu.mk : $(srcdir)/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)

clean :
	@rm -rf $(objdir) $(workdir)/$(name) $(workdir)/libffhdd.so

.PHONY : clean run $(name)

# 链接库
export LD_LIBRARY_PATH:=$(library_path_export)