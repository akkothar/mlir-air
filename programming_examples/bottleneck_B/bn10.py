# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner

range_ = for_

bn10_InW1 = 8  # 14
bn10_InH1 = 8  # 14
bn10_InC1 = 16  # 80
bn10_OutC1 = 64  # 480

bn10_InW2 = 8  # 14
bn10_InH2 = 8  # 14
bn10_OutC2 = bn10_OutC1

bn10_InW3 = 8  # 14
bn10_InH3 = 8  # 14
bn10_OutC3 = 32  # 112

OutC = bn10_OutC3
OutH = bn10_InH3
OutW = bn10_InW3

bn10_layer1_wts_size = bn10_InC1 * bn10_OutC1
bn10_layer2_wts_size = 3 * 3 * bn10_OutC2 * 1
bn10_layer3_wts_size = bn10_OutC2 * bn10_OutC3
bn10_total_wts_size = bn10_layer1_wts_size + bn10_layer2_wts_size + bn10_layer3_wts_size
# This is so we can correctly convert to/from int8/int32
assert bn10_layer1_wts_size % 4 == 0
assert bn10_layer2_wts_size % 4 == 0
assert bn10_layer3_wts_size % 4 == 0

activationsInSize = bn10_InW1 * bn10_InH1 * bn10_InC1
activationsOutSize = OutW * OutH * OutC
bn10_totalWeightsSize = bn10_total_wts_size
assert bn10_totalWeightsSize % 4 == 0
bn10_totalWeightsSize32b = bn10_total_wts_size // 4


def set_memory(mem_region, mem_type, value, weight):
    val = arith.addi(value, weight)
    sizes = mem_type.shape
    assert len(sizes) == 3
    for i in range_(sizes[0]):
        for j in range_(sizes[1]):
            for k in range_(sizes[2]):
                store(val, mem_region, [i, j, k])
                yield_([])
            yield_([])
        yield_([])


@module_builder
def build_module():
    # Define types
    uint8_ty = IntegerType.get_unsigned(8)
    int8_ty = IntegerType.get_signless(8)
    int32_ty = IntegerType.get_signless(32)

    # Input/output types. Notice we treat weights as i32 for easier channel sizes/strides/offsets
    activationsInL3_ty = MemRefType.get((activationsInSize,), int8_ty)
    activationsOutL3_ty = MemRefType.get((activationsOutSize,), int8_ty)
    # Notice we treat weights as i32 for easier channel sizes/strides/offsets
    weightsInL3_ty = MemRefType.get((bn10_totalWeightsSize32b,), int32_ty)

    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)

    # Define inputs
    bn10_layer1_in_ty = MemRefType.get(
        shape=(
            bn10_InW1,
            1,
            bn10_InC1,
        ),
        element_type=int8_ty,
        memory_space=mem_space_l1,
    )
    bn10_layer2_in_ty = MemRefType.get(
        shape=(
            bn10_InW2,
            1,
            bn10_OutC1,
        ),
        element_type=int8_ty,  # TODO: uint8_ty,
        memory_space=mem_space_l1,
    )

    bn10_layer3_in_ty = MemRefType.get(
        (
            bn10_InW3,
            1,
            bn10_OutC2,
        ),
        int8_ty,  # TODO: uint8_ty,
        memory_space=mem_space_l1,
    )

    # Define outputs
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

    # Define weights
    bn10_layer1_wts_ty = MemRefType.get(
        (bn10_layer1_wts_size,), int8_ty, memory_space=mem_space_l1
    )
    bn10_layer2_wts_ty = MemRefType.get(
        (bn10_layer2_wts_size,), int8_ty, memory_space=mem_space_l1
    )
    bn10_layer3_wts_ty = MemRefType.get(
        (bn10_layer3_wts_size,), int8_ty, memory_space=mem_space_l1
    )

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
                sizes=(bn10_layer1_wts_size // 4,),
                offsets=(0,),
                strides=(1,),
            )
            ChannelPut(
                "bn10_wts_layer2",
                weightsInL3,
                sizes=(bn10_layer2_wts_size // 4,),
                offsets=(bn10_layer1_wts_size // 4,),
                strides=(1,),
            )
            ChannelPut(
                "bn10_wts_layer3",
                weightsInL3,
                sizes=(bn10_layer3_wts_size // 4,),
                offsets=((bn10_layer1_wts_size + bn10_layer2_wts_size) // 4,),
                strides=(1,),
            )

            ChannelGet("bn10_act_layer3_memtile", activationsOutL3)

            @segment(name="seg")
            def segment_body():
                @herd(name="bn10_layer1", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):
                    weights_in = AllocOp(bn10_layer1_wts_ty, [], [])
                    ChannelGet("bn10_wts_layer1", weights_in)

                    bytes_input = np.prod(bn10_layer1_in_ty.shape) * (bn10_InH1)
                    expected_bytes_input = np.prod(activationsInL3_ty.shape)
                    print(
                        f"bn10_layer1 TOTAL INPUT: {bytes_input}, EXPECTED INPUT: {expected_bytes_input}"
                    )
                    assert bytes_input == expected_bytes_input

                    for _ in range_(bn10_InH1):
                        activations_in = AllocOp(bn10_layer1_in_ty, [], [])
                        activations_out = AllocOp(bn10_layer1_out_ty, [], [])

                        ChannelGet("bn10_act_memtile_layer1", activations_in)

                        c0 = arith.ConstantOp.create_index(0)
                        set_memory(
                            activations_out,
                            bn10_layer1_out_ty,
                            load(weights_in, [c0]),
                            load(activations_in, [c0, c0, c0]),
                        )

                        ChannelPut("bn10_act_layer1_layer2", activations_out)

                        DeallocOp(activations_in)
                        DeallocOp(activations_out)
                        yield_([])
                    DeallocOp(weights_in)

                @herd(name="bn10_layer2", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):
                    weights_in = AllocOp(bn10_layer2_wts_ty, [], [])
                    ChannelGet("bn10_wts_layer2", weights_in)

                    actual_count = 2 + (bn10_InH2 - 2)
                    bytes_input = np.prod(bn10_layer2_in_ty.shape) * actual_count
                    expected_bytes_input = np.prod(bn10_layer1_out_ty.shape) * bn10_InH1
                    print(
                        f"bn10_layer2 TOTAL INPUT: {bytes_input}, EXPECTED INPUT: {expected_bytes_input}"
                    )
                    assert bytes_input == expected_bytes_input

                    # Preamble: top row
                    activations_in = []

                    activations_in.append(AllocOp(bn10_layer2_in_ty, [], []))
                    activations_in.append(AllocOp(bn10_layer2_in_ty, [], []))

                    ChannelGet("bn10_act_layer1_layer2", activations_in[0])
                    ChannelGet("bn10_act_layer1_layer2", activations_in[1])

                    activations_out_0 = AllocOp(bn10_layer2_out_ty, [], [])

                    c0 = arith.ConstantOp.create_index(0)
                    set_memory(
                        activations_out_0,
                        bn10_layer2_out_ty,
                        load(weights_in, [c0]),
                        load(activations_in[0], [c0, c0, c0]),
                    )
                    ChannelPut("bn10_act_layer2_layer3", activations_out_0)

                    # middle
                    for _ in range_(bn10_InH2 - 2):
                        activations_in.append(AllocOp(bn10_layer2_in_ty, [], []))
                        ChannelGet("bn10_act_layer1_layer2", activations_in[2])

                        activations_out_1 = AllocOp(bn10_layer2_out_ty, [], [])

                        c0 = arith.ConstantOp.create_index(0)
                        set_memory(
                            activations_out_1,
                            bn10_layer2_out_ty,
                            load(weights_in, [c0]),
                            load(activations_in[0], [c0, c0, c0]),
                        )

                        ChannelPut("bn10_act_layer2_layer3", activations_out_1)

                        to_dealloc = activations_in.pop()
                        DeallocOp(to_dealloc)

                        yield_([])

                    activations_out_2 = AllocOp(bn10_layer2_out_ty, [], [])

                    c0 = arith.ConstantOp.create_index(0)
                    set_memory(
                        activations_out_2,
                        bn10_layer2_out_ty,
                        load(weights_in, [c0]),
                        load(activations_in[0], [c0, c0, c0]),
                    )
                    ChannelPut("bn10_act_layer2_layer3", activations_out_2)

                    bytes_output = np.prod(bn10_layer3_in_ty.shape) * bn10_InH2
                    expected_bytes_output = np.prod(bn10_layer3_in_ty.shape) * bn10_InH3
                    print(
                        f"bn10_layer2 TOTAL OUTPUT: {bytes_output}, EXPECTED OUTPUT: {expected_bytes_output}"
                    )
                    assert bytes_output == expected_bytes_output
                    DeallocOp(activations_in[0])
                    DeallocOp(activations_in[1])

                @herd(name="bn10_layer3", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):
                    weights_in = AllocOp(bn10_layer3_wts_ty, [], [])

                    ChannelGet("bn10_wts_layer3", weights_in)

                    for input_tile in range_(bn10_InH3):
                        activations_in = AllocOp(bn10_layer3_in_ty, [], [])
                        activations_out = AllocOp(bn10_layer3_out_ty, [], [])

                        ChannelGet("bn10_act_layer2_layer3", activations_in)

                        c0 = arith.ConstantOp.create_index(0)
                        set_memory(
                            activations_out,
                            bn10_layer3_out_ty,
                            arith.index_cast(
                                bn10_layer3_out_ty.element_type, input_tile
                            ),  # load(weights_in, [c0]),
                            load(activations_in, [c0, c0, c0]),
                        )

                        ChannelPut("bn10_act_layer3_memtile", activations_out)

                        DeallocOp(activations_in)
                        DeallocOp(activations_out)
                        yield_([])

                    bytes_output = np.prod(bn10_layer3_out_ty.shape) * bn10_InH3
                    expected_bytes_output = np.prod(activationsOutL3_ty.shape)
                    print(
                        f"bn10_layer3 TOTAL OUTPUT: {bytes_output}, EXPECTED OUTPUT: {expected_bytes_output} TILES: {bn10_InH3}"
                    )
                    assert bytes_output == expected_bytes_output

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

    mlir_module = build_module()

    if args.print_module_only:
        print(mlir_module)
        exit(0)

    activationsIn = np.full(shape=(activationsInSize,), fill_value=1, dtype=np.int8)
    weightsIn = np.zeros(shape=(bn10_totalWeightsSize,), dtype=np.int8)
    activationsOut = np.full(shape=(activationsOutSize,), fill_value=1, dtype=np.int8)

    for i in range(bn10_totalWeightsSize):
        if i < bn10_layer1_wts_size:
            weightsIn[i] = 1
        elif i < bn10_layer1_wts_size + bn10_layer2_wts_size:
            weightsIn[i] = 3
        else:
            weightsIn[i] = 7

    for i in range(activationsOutSize):
        activationsOut[i] += 1 + 3 + 7

    runner = XRTRunner(verbose=args.verbose, experimental_passes=True)
    exit(
        runner.run_test(
            mlir_module,
            inputs=[activationsIn, weightsIn],
            expected_outputs=[activationsOut],
        )
    )
