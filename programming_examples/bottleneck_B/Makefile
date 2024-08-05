# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

include ${srcdir}/../makefile-common

VPATH := ${srcdir}/kernels

mlirFileName=aie2_bn10
testFileName=test_wts_L2

all: ${srcdir}/build/bn10_conv2dk1_fused_relu.o run

print:
	${powershell} python3 ${srcdir}/bn10.py -p

run: ${srcdir}/build/bn10_conv2dk1_fused_relu.o
	mkdir -p ${srcdir}/build
	cd ${srcdir}/build && ${powershell} python3 ${srcdir}/bn10.py

clean:
	rm -rf ${srcdir}/build ${srcdir}/__pycache__

${srcdir}/build/bn10_conv2dk1_fused_relu.o: ${VPATH}/bn_conv2dk1_relu.cc
	mkdir -p ${srcdir}/build
	cd ${srcdir}/build && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -d -DSCALAR -DBN10 -DINT8_ACT -c $< -o ${@F}