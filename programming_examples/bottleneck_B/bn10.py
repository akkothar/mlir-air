# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner

range_ = for_

scale_factors = {
    "BN10": {"conv1x1_1": 9, "conv3x3": 8, "conv1x1_2": 9, "skip_add": 0},
    "BN11": {"conv1x1_1": 9, "conv3x3": 8, "conv1x1_2": 12, "skip_add": 1},
    "BN12": {"conv1x1_1": 8, "conv3x3": 7, "conv1x1_2": 10, "skip_add": 0},
}

bn10_InW1 = 8
bn10_InH1 = 8
bn10_InC1 = 16
bn10_OutC1 = 64

bn10_InW2 = 8
bn10_InH2 = 8
bn10_OutC2 = bn10_OutC1

bn10_InW3 = 8
bn10_InH3 = 8
bn10_OutC3 = 32

OutC = bn10_OutC3
OutH = bn10_InH3
OutW = bn10_InW3

bn10_layer1_wts_size = bn10_InC1 * bn10_OutC1
bn10_layer2_wts_size = 3 * 3 * bn10_OutC2 * 1
bn10_layer3_wts_size = bn10_OutC2 * bn10_OutC3
bn10_total_wts_size = bn10_layer1_wts_size + bn10_layer2_wts_size + bn10_layer3_wts_size

activationsInSize32b = (bn10_InW1 * bn10_InH1 * bn10_InC1) // 4
activationsOutSize32b = (OutW * OutH * OutC) // 4
bn10_totalWeightsSize32b = bn10_total_wts_size // 4


@module_builder
def build_module(bn10_scaleFactor1=10, bn10_scaleFactor2=7, bn10_scaleFactor3=9):
    # Define types
    uint8_ty = IntegerType.get_unsigned(8)
    int8_ty = IntegerType.get_signless(8)
    int32_ty = IntegerType.get_signless(32)
    uint32_ty = IntegerType.get_unsigned(32)

    activationsInL3_ty = MemRefType.get((activationsInSize32b,), int32_ty)
    # TODO: should include everything, not just bn10 in future
    weightsInL3_ty = MemRefType.get((bn10_totalWeightsSize32b,), int32_ty)
    activationsOutL3_ty = MemRefType.get((activationsOutSize32b,), int32_ty)

    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)

    # define inputs
    bn10_layer1_in_ty = MemRefType.get(
        (
            bn10_InW1,
            1,
            bn10_InC1,
        ),
        int8_ty,
        memory_space=mem_space_l1,
    )
    bn10_layer2_in_ty = MemRefType.get(
        (
            bn10_InW2,
            1,
            bn10_OutC1,
        ),
        uint8_ty,
        memory_space=mem_space_l1,
    )
    bn10_layer3_in_ty = MemRefType.get(
        (
            bn10_InW3,
            1,
            bn10_OutC2,
        ),
        uint8_ty,
        memory_space=mem_space_l1,
    )

    # define outputs
    bn10_layer1_out_ty = bn10_layer2_in_ty
    bn10_layer2_out_ty = bn10_layer3_in_ty
    bn10_layer3_out_ty = MemRefType.get(
        (
            bn10_InW3,
            1,
            bn10_OutC3,
        ),
        int8_ty,
        memory_space=mem_space_l1,
    )

    # define wts
    bn10_layer1_wts_ty = MemRefType.get(
        (bn10_layer1_wts_size,), int8_ty, memory_space=mem_space_l1
    )
    bn10_layer2_wts_ty = MemRefType.get(
        (bn10_layer2_wts_size,), int8_ty, memory_space=mem_space_l1
    )
    bn10_layer3_wts_ty = MemRefType.get(
        (bn10_layer3_wts_size,), int8_ty, memory_space=mem_space_l1
    )

    # AIE Core Function declarations
    # ************************ bneck10 ************************
    bn10_conv2dk1_fused_relu = external_func(
        "bn10_conv2dk1_relu_i8_ui8",
        inputs=[
            bn10_layer1_in_ty,
            bn10_layer1_wts_ty,
            bn10_layer1_out_ty,
            int32_ty,
            int32_ty,
            int32_ty,
            int32_ty,
        ],
    )
    bn10_conv2dk3_dw = external_func(
        "bn10_conv2dk3_dw_stride1_relu_ui8_ui8",
        inputs=[
            bn10_layer2_in_ty,
            bn10_layer2_in_ty,
            bn10_layer2_in_ty,
            bn10_layer2_wts_ty,
            bn10_layer2_out_ty,
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
            bn10_layer3_in_ty,
            bn10_layer3_wts_ty,
            bn10_layer3_out_ty,
            int32_ty,
            int32_ty,
            int32_ty,
            int32_ty,
        ],
    )

    # declare channels
    ChannelOp("bn10_act_memtile_layer1")
    ChannelOp("bn10_act_layer1_layer2")
    ChannelOp("bn10_act_layer2_layer3")
    ChannelOp("bn10_act_layer3_memtile")

    ChannelOp("bn10_wts_layer1")
    ChannelOp("bn10_wts_layer2")
    ChannelOp("bn10_wts_layer3")

    @FuncOp.from_py_func(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
    def sequence(activationsIn, weightsIn, activationsOut):

        @launch(operands=[activationsIn, weightsIn, activationsOut])
        def launch_body(activationsInL3, weightsInL3, activationsOutL3):

            ChannelPut("bn10_act_memtile_layer1", activationsInL3)

            ChannelPut(
                "bn10_wts_layer1",
                weightsInL3,
                sizes=(bn10_layer1_wts_size,),
                offsets=(0,),
                strides=(1,),
            )
            ChannelPut(
                "bn10_wts_layer2",
                weightsInL3,
                sizes=(bn10_layer2_wts_size,),
                offsets=(bn10_layer1_wts_size,),
                strides=(1,),
            )
            ChannelPut(
                "bn10_wts_layer3",
                weightsInL3,
                sizes=(bn10_layer3_wts_size,),
                offsets=(bn10_layer1_wts_size + bn10_layer2_wts_size,),
                strides=(1,),
            )

            ChannelGet("bn10_act_layer3_memtile", activationsOutL3)

            @segment(name="seg")
            def segment_body():

                @herd(
                    name="bn10_layer1",
                    sizes=[1, 1],
                    link_with="bn10_conv2dk1_fused_relu.o",
                )
                def herd_body(tx, ty, sx, sy):
                    weights_in = AllocOp(bn10_layer1_wts_ty, [], [])
                    ChannelGet("bn10_wts_layer1", weights_in)

                    for _ in range_(bn10_InH1):
                        activations_in = AllocOp(bn10_layer1_in_ty, [], [])
                        activations_out = AllocOp(bn10_layer1_out_ty, [], [])

                        ChannelGet("bn10_act_memtile_layer1", activations_in)

                        call(
                            bn10_conv2dk1_fused_relu,
                            inputs=[
                                activations_in,
                                weights_in,
                                activations_out,
                                bn10_InW1,
                                bn10_InC1,
                                bn10_OutC1,
                                bn10_scaleFactor1,
                            ],
                            input_types=[
                                bn10_layer1_in_ty,
                                bn10_layer1_wts_ty,
                                bn10_layer1_out_ty,
                                int32_ty,
                                int32_ty,
                                int32_ty,
                                int32_ty,
                            ],
                        )

                        ChannelPut("bn10_act_layer1_layer2", activations_out)

                        DeallocOp(activations_in)
                        DeallocOp(activations_out)
                        yield_([])

                    DeallocOp(weights_in)

                @herd(name="bn10_layer2", sizes=[1, 1], link_with="bn10_conv2dk3_dw.o")
                def herd_body(tx, ty, sx, sy):
                    weights_in = AllocOp(bn10_layer2_wts_ty, [], [])
                    ChannelGet("bn10_wts_layer2", weights_in)

                    # Preamble: top row
                    activations_in_0 = AllocOp(bn10_layer2_in_ty, [], [])
                    activations_in_1 = AllocOp(bn10_layer2_in_ty, [], [])

                    ChannelGet("bn10_act_layer1_layer2", activations_in_0)
                    ChannelGet("bn10_act_layer1_layer2", activations_in_1)

                    activations_out_0 = AllocOp(bn10_layer2_out_ty, [], [])

                    call(
                        bn10_conv2dk3_dw,
                        inputs=[
                            activations_in_0,
                            activations_in_0,
                            activations_in_1,
                            weights_in,
                            activations_out_0,
                            bn10_InW2,
                            1,
                            bn10_OutC2,
                            3,
                            3,
                            0,
                            bn10_scaleFactor2,
                            0,
                        ],
                        input_types=[
                            bn10_layer2_in_ty,
                            bn10_layer2_in_ty,
                            bn10_layer2_in_ty,
                            bn10_layer2_wts_ty,
                            bn10_layer2_out_ty,
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
                    ChannelPut("bn10_act_layer2_layer3", activations_out_0)

                    DeallocOp(activations_in_0)
                    DeallocOp(activations_in_1)
                    DeallocOp(activations_out_0)

                    # middle
                    for _ in range_(bn10_InH2 - 2):
                        activations_in_2 = AllocOp(bn10_layer2_in_ty, [], [])
                        activations_in_3 = AllocOp(bn10_layer2_in_ty, [], [])
                        activations_in_4 = AllocOp(bn10_layer2_in_ty, [], [])

                        ChannelGet("bn10_act_layer1_layer2", activations_in_2)
                        ChannelGet("bn10_act_layer1_layer2", activations_in_3)
                        ChannelGet("bn10_act_layer1_layer2", activations_in_4)

                        activations_out_1 = AllocOp(bn10_layer2_out_ty, [], [])

                        res = call(
                            bn10_conv2dk3_dw,
                            [
                                activations_in_2,
                                activations_in_3,
                                activations_in_4,
                                weights_in,
                                activations_out_1,
                                bn10_InW2,
                                1,
                                bn10_OutC2,
                                3,
                                3,
                                1,
                                bn10_scaleFactor2,
                                0,
                            ],
                        )
                        ChannelPut("bn10_act_layer2_layer3", activations_out_1)

                        DeallocOp(activations_in_2)
                        DeallocOp(activations_in_3)
                        DeallocOp(activations_in_4)
                        DeallocOp(activations_out_1)
                        yield_([])

                    # last part
                    activations_in_5 = AllocOp(bn10_layer2_in_ty, [], [])
                    activations_in_6 = AllocOp(bn10_layer2_in_ty, [], [])

                    ChannelGet("bn10_act_layer1_layer2", activations_in_5)
                    ChannelGet("bn10_act_layer1_layer2", activations_in_6)

                    activations_out_2 = AllocOp(bn10_layer2_out_ty, [], [])

                    call(
                        bn10_conv2dk3_dw,
                        [
                            activations_in_5,
                            activations_in_6,
                            activations_in_6,
                            weights_in,
                            activations_out_2,
                            bn10_InW2,
                            1,
                            bn10_OutC2,
                            3,
                            3,
                            2,
                            bn10_scaleFactor2,
                            0,
                        ],
                        input_types=[
                            bn10_layer2_in_ty,
                            bn10_layer2_in_ty,
                            bn10_layer2_in_ty,
                            bn10_layer2_wts_ty,
                            bn10_layer2_out_ty,
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
                    ChannelPut("bn10_act_layer2_layer3", activations_out_2)

                    DeallocOp(activations_in_5)
                    DeallocOp(activations_in_6)
                    DeallocOp(activations_out_2)

                    DeallocOp(weights_in)

                @herd(name="bn10_layer3", sizes=[1, 1], link_with="bn10_conv2dk1_ui8.o")
                def herd_body(tx, ty, sx, sy):
                    weights_in = AllocOp(bn10_layer3_wts_ty, [], [])
                    ChannelGet("bn10_wts_layer3", weights_in)

                    for _ in range_(bn10_InH3):
                        activations_in = AllocOp(bn10_layer3_in_ty, [], [])
                        activations_out = AllocOp(bn10_layer3_out_ty, [], [])

                        ChannelGet("bn10_act_layer2_layer3", activations_in)

                        call(
                            bn10_conv2dk1_ui8,
                            inputs=[
                                activations_in,
                                weights_in,
                                activations_out,
                                bn10_InW3,
                                bn10_OutC2,
                                bn10_OutC3,
                                bn10_scaleFactor3,
                            ],
                            input_types=[
                                bn10_layer3_in_ty,
                                bn10_layer3_wts_ty,
                                bn10_layer3_out_ty,
                                int32_ty,
                                int32_ty,
                                int32_ty,
                                int32_ty,
                            ],
                        )

                        ChannelPut("bn10_act_layer3_memtile", activations_out)

                        DeallocOp(activations_in)
                        DeallocOp(activations_out)
                        yield_([])

                    DeallocOp(weights_in)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the bottleneck block B example",
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

    activationsIn = np.full(shape=(ACTIVATIONS_IN_SIZE,), fill_value=1, dtype=np.int32)
    weightsIn = np.zeros(shape=(WEIGHTS_SIZE,), dtype=np.int32)
    for i in range(WEIGHTS_SIZE):
        if i < WEIGHTS_SIZE_LAYER1:
            weightsIn[i] = 1
        elif i < WEIGHTS_SIZE_LAYER1 + WEIGHTS_SIZE_LAYER2:
            weightsIn[i] = 3
        else:
            weightsIn[i] = 7
    activationsOut = np.zeros(shape=(ACTIVATIONS_OUT_SIZE,), dtype=np.int32)

    runner = XRTRunner(verbose=args.verbose, experimental_passes=False)
    exit(
        runner.run_test(
            mlir_module,
            inputs=[activationsIn, weightsIn],
            expected_outputs=[activationsOut],
        )
    )
