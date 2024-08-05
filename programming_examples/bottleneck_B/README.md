<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// 
//===----------------------------------------------------------------------===//-->

# WIP: Bottleneck Block B

This example is based on bottleneck block B found [here](https://github.com/Xilinx/mlir-aie/tree/dataflow_mobilenet/programming_examples/ml/mobilenet/bottleneck_B).


First Core: Conv + ReLU (1x1)
Second Core: DConv + ReLU (3x3)
Third Core: Conv 1x1 C + Skip Add (two functions)