# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp
from air.backend.xrt_runner import XRTRunner, type_mapper

bneck_10_InW1 = 14
bneck_10_InH1 = 14
bneck_10_InC1 = 80
bneck_10_OutC1 = 480

bneck_10_InW2 = 14
bneck_10_InH2 = 14
bneck_10_OutC2 = bneck_10_OutC1

bneck_10_InW3 = 14
bneck_10_InH3 = 14
bneck_10_OutC3 = 112

OutC = bneck_10_OutC3
OutH = bneck_10_InH3
OutW = bneck_10_InW3

activationsInSize32b = (bneck_10_InW1 * bneck_10_InH1 * bneck_10_InC1) // 4
acitivationsOutSize32b = (OutW * OutH * OutC) // 4

bn10_totalWeightsSize32b = (
    bneck_10_InC1 * bneck_10_OutC1
    + 3 * 3 * bneck_10_OutC2 * 1
    + bneck_10_OutC2 * bneck_10_OutC3
) // 4

totalWeightsSize32b_complete = bn10_totalWeightsSize32b

scale_factors = {
    "BN10": {"conv1x1_1": 9, "conv3x3": 8, "conv1x1_2": 9, "skip_add": 0},
    "BN11": {"conv1x1_1": 9, "conv3x3": 8, "conv1x1_2": 12, "skip_add": 1},
}


@module_builder
def build_module(bn10_scaleFactor1=10, bn10_scaleFactor2=7, bn10_scaleFactor3=9):
    # define types
    uint8_ty = IntegerType.get_unsigned(8)
    int8_ty = IntegerType.get_signless(8)
    int32_ty = IntegerType.get_signless(32)
    # ************************ bneck10 ************************
    ty_bneck_10_layer1_in = MemRefType.get(
        (
            bneck_10_InW1,
            1,
            bneck_10_InC1,
        ),
        int8_ty,
    )
    ty_bneck_10_layer2_in = MemRefType.get(
        (
            bneck_10_InW2,
            1,
            bneck_10_OutC1,
        ),
        uint8_ty,
    )
    ty_bneck_10_layer3_in = MemRefType.get(
        (
            bneck_10_InW3,
            1,
            bneck_10_OutC2,
        ),
        uint8_ty,
    )

    # define wts
    ty_bneck_10_layer1_wts = MemRefType.get((bneck_10_InC1 * bneck_10_OutC1,), int8_ty)
    ty_bneck_10_layer2_wts = MemRefType.get((3 * 3 * bneck_10_OutC2 * 1,), int8_ty)
    ty_bneck_10_layer3_wts = MemRefType.get((bneck_10_OutC2 * bneck_10_OutC3,), int8_ty)
    ty_bneck_10_all_wts = MemRefType.get(
        (
            bneck_10_InC1 * bneck_10_OutC1
            + 3 * 3 * bneck_10_OutC2 * 1
            + bneck_10_OutC2 * bneck_10_OutC3,
        ),
        int8_ty,
    )

    # output
    ty_bneck_10_layer1_out = MemRefType.get(
        (
            bneck_10_InW2,
            1,
            bneck_10_OutC1,
        ),
        uint8_ty,
    )
    ty_bneck_10_layer2_out = MemRefType.get(
        (
            bneck_10_InW3,
            1,
            bneck_10_OutC2,
        ),
        uint8_ty,
    )
    ty_bneck_10_layer3_out = MemRefType.get(
        (
            bneck_10_InW3,
            1,
            bneck_10_OutC3,
        ),
        int8_ty,
    )
    # ************************ bneck11 ************************
    # input
    ty_bneck_11_layer1_in = MemRefType.get(
        (
            bneck_10_InW3,
            1,
            bneck_10_OutC3,
        ),
        int8_ty,
    )

    # AIE Core Function declarations
    # ************************ bneck10 ************************
    bn10_conv2dk1_fused_relu = external_func(
        "bn10_conv2dk1_relu_i8_ui8",
        inputs=[
            ty_bneck_10_layer1_in,
            ty_bneck_10_layer1_wts,
            ty_bneck_10_layer1_out,
            int32_ty,
            int32_ty,
            int32_ty,
            int32_ty,
        ],
    )
    bn10_conv2dk3_dw = external_func(
        "bn10_conv2dk3_dw_stride1_relu_ui8_ui8",
        inputs=[
            ty_bneck_10_layer2_in,
            ty_bneck_10_layer2_in,
            ty_bneck_10_layer2_in,
            ty_bneck_10_layer2_wts,
            ty_bneck_10_layer2_out,
            int32_ty,
            int32_ty,
            int32_ty,
            int32_ty,
            int32_ty,
            int32_ty,
            int32_ty,
            int32_ty,
        ],
    )
    bn10_conv2dk1_ui8 = external_func(
        "bn10_conv2dk1_ui8_i8",
        inputs=[
            ty_bneck_10_layer3_in,
            ty_bneck_10_layer3_wts,
            ty_bneck_10_layer3_out,
            int32_ty,
            int32_ty,
            int32_ty,
            int32_ty,
        ],
    )

    activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
    weightsInL3_ty = MemRefType.get((totalWeightsSize32b_complete,), int32_ty)
    activationsOutL3_ty = MemRefType.get((acitivationsOutSize32b,), int32_ty)

    @FuncOp.from_py_func(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
    def sequence(inputFromL3, weightsFromL3, outputToL3):

        @launch(operands=[inputFromL3, weightsFromL3, outputToL3])
        def launch_body(inputL3, weightsL3, outputL3):

            @segment(name="seg")
            def segment_body():
                # We want to store our data in L1 memory
                mem_space_l2 = IntegerAttr.get(T.i32(), MemorySpace.L2)

                # This is the type definition of the tile
                tile_type_l2 = MemRefType.get(
                    shape=(10, 10),
                    element_type=int32_ty,
                    memory_space=mem_space_l2,
                )

                # We must allocate a buffer of tile size for the input/output
                tile_in_l2 = AllocOp(tile_type_l2, [], [])

                # The herd sizes correspond to the dimensions of the contiguous block of cores we are hoping to get.
                # We just need one compute core, so we ask for a 1x1 herd
                @herd(name="copyherd", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):

                    # We want to store our data in L1 memory
                    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)

                    # This is the type definition of the tile
                    tile_type_l1 = MemRefType.get(
                        shape=(10, 10),
                        element_type=int32_ty,
                        memory_space=mem_space_l1,
                    )

                    # We must allocate a buffer of tile size for the input/output
                    tile_in_l1 = AllocOp(tile_type_l1, [], [])
                    tile_out_l1 = AllocOp(tile_type_l1, [], [])

                    # Deallocate our L1 buffers
                    DeallocOp(tile_in_l1)
                    DeallocOp(tile_out_l1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the segment_alloc example",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
    )
    args = parser.parse_args()

    mlir_module = build_module(
        bn10_scaleFactor1=scale_factors["BN10"]["conv1x1_1"],
        bn10_scaleFactor2=scale_factors["BN10"]["conv3x3"],
        bn10_scaleFactor3=scale_factors["BN10"]["conv1x1_2"],
    )

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    activationsIn = np.zeros(shape=(activationsInSize32b,), dtype=np.int32)
    weightsIn = np.zeros(shape=(totalWeightsSize32b_complete,), dtype=np.int32)
    activationsOut = np.zeros(shape=(acitivationsOutSize32b,), dtype=np.int32)

    runner = XRTRunner(verbose=args.verbose, experimental_passes=True)
    exit(
        runner.run_test(
            mlir_module,
            inputs=[activationsIn, weightsIn],
            expected_outputs=[activationsOut],
        )
    )
