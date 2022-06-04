#!/bin/bash

NVCC=/usr/local/cuda/bin/nvcc

if [ ! -d bin ];then
	mkdir bin
else
	rm -f bin/*
fi

$NVCC src/main_gpu.cu -o bin/hello

$NVCC src/vec_add.cu -o bin/vec_add

$NVCC src/matrixMultiple.cu -o bin/mat_mul -lcublas

