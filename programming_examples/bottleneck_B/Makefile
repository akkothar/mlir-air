# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

include ${srcdir}/../makefile-common

# TODO: this should be to kernels in mlir-aie, once they are merged
VPATH := ${srcdir}/kernels

mlirFileName=aie2_bn10
testFileName=test_wts_L2

#all: ${srcdir}/build/bn10_conv2dk1_fused_relu.o ${srcdir}/build/bn10_conv2dk3_dw.o \
	${srcdir}/build/bn10_conv2dk1_ui8.o ${srcdir}/build/bn11_conv2dk1_fused_relu.o \
	${srcdir}/build/bn11_conv2dk3_dw.o ${srcdir}/build/bn11_conv2dk1_skip.o \
	${srcdir}/build/bn12_conv2dk1_fused_relu.o ${srcdir}/build/bn12_conv2dk3_dw_stride2.o \
	${srcdir}/build/bn12_conv2dk1_ui8.o run
all: run

print:
	${powershell} python3 ${srcdir}/bn10.py -p

run: ${srcdir}/build/bn10_conv2dk1_fused_relu.o ${srcdir}/build/bn10_conv2dk3_dw.o ${srcdir}/build/bn10_conv2dk1_ui8.o
	mkdir -p ${srcdir}/build
	cd ${srcdir}/build && ${powershell} python3 ${srcdir}/bn10.py -v

clean:
	rm -rf ${srcdir}/build ${srcdir}/__pycache__

# bn10
${srcdir}/build/bn10_conv2dk1_fused_relu.o: ${VPATH}/bn_conv2dk1_relu.cc
	mkdir -p ${srcdir}/build
	cd ${srcdir}/build && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -d -DSCALAR -DBN10 -DINT8_ACT -c $< -o ${@F}

${srcdir}/build/bn10_conv2dk3_dw.o: ${VPATH}/bn_conv2dk3_dw.cc
	mkdir -p ${srcdir}/build
	cd ${srcdir}/build && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -d -DSCALAR -DBN10 -DSTRIDE1 -c $< -o ${@F}

${srcdir}/build/bn10_conv2dk1_ui8.o: ${VPATH}/bn_conv2dk1_i8.cc
	mkdir -p ${srcdir}/build
	cd ${srcdir}/build && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBN10 -DSCALAR -c $< -o ${@F}

# bn11
${srcdir}/build/bn11_conv2dk1_fused_relu.o: ${VPATH}/bn_conv2dk1_relu.cc
	mkdir -p ${srcdir}/build
	cd ${srcdir}/build && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -d -DSCALAR -DBN11 -DINT8_ACT -c $< -o ${@F}

${srcdir}/build/bn11_conv2dk3_dw.o: ${VPATH}/bn_conv2dk3_dw.cc
	mkdir -p ${srcdir}/build
	cd ${srcdir}/build && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -d -DSCALAR -DBN11 -DSTRIDE1 -c $< -o ${@F}

${srcdir}/build/bn11_conv2dk1_skip.o: ${VPATH}/bn_conv2dk1_skip.cc
	mkdir -p ${srcdir}/build
	cd ${srcdir}/build && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBN11 -DSCALAR -c $< -o ${@F}

# bn12
${srcdir}/build/bn12_conv2dk1_fused_relu.o:${VPATH}/ bn_conv2dk1_relu.cc
	mkdir -p ${srcdir}/build
	cd ${srcdir}/build && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -d -DSCALAR -DBN12 -DINT8_ACT -c $< -o ${@F}

${srcdir}/build/bn12_conv2dk3_dw_stride2.o: ${VPATH}/bn_conv2dk3_dw.cc
	mkdir -p ${srcdir}/build
	cd ${srcdir}/build && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -d -DSCALAR -DSTRIDE2 -DBN12 -c $< -o ${@F}

${srcdir}/build/bn12_conv2dk1_ui8.o: ${VPATH}/bn_conv2dk1_i8.cc
	mkdir -p ${srcdir}/build
	cd ${srcdir}/build && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -DBN12 -DSCALAR  -c $< -o ${@F}