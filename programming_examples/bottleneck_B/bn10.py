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

WEIGHTS_SIZE_LAYER1 = 8
WEIGHTS_SIZE_LAYER2 = 16
WEIGHTS_SIZE_LAYER3 = 8
# so we can convert to int32 for channel offset
assert WEIGHTS_SIZE_LAYER1 % 4 == 0
assert WEIGHTS_SIZE_LAYER2 % 4 == 0
assert WEIGHTS_SIZE_LAYER3 % 4 == 0

WEIGHTS_SIZE = WEIGHTS_SIZE_LAYER1 + WEIGHTS_SIZE_LAYER2 + WEIGHTS_SIZE_LAYER3
WEIGHTS_SIZE_32B = WEIGHTS_SIZE // 4

ACTIVATIONS_IN_SIZE = 16
ACTIVATIONS_OUT_SIZE = 16
assert ACTIVATIONS_IN_SIZE == ACTIVATIONS_OUT_SIZE


@module_builder
def build_module():
    activationsInL3_ty = MemRefType.get((ACTIVATIONS_IN_SIZE,), T.i8())
    weightsInL3_ty = MemRefType.get((WEIGHTS_SIZE_32B,), T.i32())
    activationsOutL3_ty = MemRefType.get((ACTIVATIONS_OUT_SIZE,), T.i8())

    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    activationsInL1_ty = MemRefType.get(
        (ACTIVATIONS_IN_SIZE,), T.i8(), memory_space=mem_space_l1
    )
    weightsInLayer1L1_ty = MemRefType.get(
        (WEIGHTS_SIZE_LAYER1,), T.i8(), memory_space=mem_space_l1
    )
    weightsInLayer2L1_ty = MemRefType.get(
        (WEIGHTS_SIZE_LAYER2,), T.i8(), memory_space=mem_space_l1
    )
    weightsInLayer3L1_ty = MemRefType.get(
        (WEIGHTS_SIZE_LAYER3,), T.i8(), memory_space=mem_space_l1
    )
    activationsOutL1_ty = MemRefType.get(
        (ACTIVATIONS_OUT_SIZE,), T.i8(), memory_space=mem_space_l1
    )

    ChannelOp("ActivationsIn")
    ChannelOp("ActivationsLayer1Layer2")
    ChannelOp("ActivationsLayer2Layer3")
    ChannelOp("ActivationsOut")

    ChannelOp("WeightsInLayer1")
    ChannelOp("WeightsInLayer2")
    ChannelOp("WeightsInLayer3")

    @FuncOp.from_py_func(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
    def sequence(activationsIn, weightsIn, activationsOut):

        @launch(operands=[activationsIn, weightsIn, activationsOut])
        def launch_body(activationsInL3, weightsInL3, activationsOutL3):

            ChannelPut("ActivationsIn", activationsInL3)

            ChannelPut(
                "WeightsInLayer1",
                weightsInL3,
                sizes=(WEIGHTS_SIZE_LAYER1 // 4,),
                offsets=(0,),
                strides=(1,),
            )
            ChannelPut(
                "WeightsInLayer2",
                weightsInL3,
                sizes=(WEIGHTS_SIZE_LAYER2 // 4,),
                offsets=(WEIGHTS_SIZE_LAYER1 // 4,),
                strides=(1,),
            )
            ChannelPut(
                "WeightsInLayer3",
                weightsInL3,
                sizes=(WEIGHTS_SIZE_LAYER3 // 4,),
                offsets=((WEIGHTS_SIZE_LAYER1 + WEIGHTS_SIZE_LAYER2) // 4,),
                strides=(1,),
            )

            ChannelGet("ActivationsOut", activationsOutL3)

            @segment(name="seg")
            def segment_body():

                @herd(name="layer1", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):
                    activations_in = AllocOp(activationsInL1_ty, [], [])
                    weights_in = AllocOp(weightsInLayer1L1_ty, [], [])
                    activations_out = AllocOp(activationsOutL1_ty, [], [])

                    ChannelGet("ActivationsIn", activations_in)
                    ChannelGet("WeightsInLayer1", weights_in)

                    # Do something with the first bit of data
                    for i in range_(WEIGHTS_SIZE_LAYER1):
                        val = load(activations_in, [i])
                        weight_val = load(weights_in, [i])
                        val_out = arith.AddIOp(val, weight_val)
                        store(val_out, activations_out, [i])
                        yield_([])

                    # Passthrough the rest
                    for i in range_(ACTIVATIONS_IN_SIZE - WEIGHTS_SIZE_LAYER1):
                        index = arith.AddIOp(
                            i, arith.ConstantOp.create_index(WEIGHTS_SIZE_LAYER1)
                        )
                        val = load(activations_in, [index])
                        store(val, activations_out, [index])
                        yield_([])

                    # Send transformed activations to the next layer
                    ChannelPut("ActivationsLayer1Layer2", activations_out)

                    # Cleanup
                    DeallocOp(activations_in)
                    DeallocOp(weights_in)
                    DeallocOp(activations_out)

                @herd(name="layer2", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):
                    activations_in = AllocOp(activationsInL1_ty, [], [])
                    weights_in = AllocOp(weightsInLayer2L1_ty, [], [])
                    activations_out = AllocOp(activationsOutL1_ty, [], [])

                    ChannelGet("ActivationsLayer1Layer2", activations_in)
                    ChannelGet("WeightsInLayer2", weights_in)

                    # Do something with the first bit of data
                    assert ACTIVATIONS_IN_SIZE == WEIGHTS_SIZE_LAYER2
                    for i in range_(ACTIVATIONS_IN_SIZE):
                        val = load(activations_in, [i])
                        weight_val = load(weights_in, [i])
                        val_out = arith.AddIOp(val, weight_val)
                        store(val_out, activations_out, [i])
                        yield_([])

                    # Send transformed activations to the next layer
                    ChannelPut("ActivationsLayer2Layer3", activations_out)

                    # Cleanup
                    DeallocOp(activations_in)
                    DeallocOp(weights_in)
                    DeallocOp(activations_out)

                @herd(name="layer3", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):
                    activations_in = AllocOp(activationsInL1_ty, [], [])
                    weights_in = AllocOp(weightsInLayer3L1_ty, [], [])
                    activations_out = AllocOp(activationsOutL1_ty, [], [])

                    ChannelGet("ActivationsLayer2Layer3", activations_in)
                    ChannelGet("WeightsInLayer3", weights_in)

                    # Passthrough the beginning
                    for i in range_(ACTIVATIONS_IN_SIZE - WEIGHTS_SIZE_LAYER3):
                        val = load(activations_in, [i])
                        store(val, activations_out, [i])
                        yield_([])

                    # Do something with the second bit of data
                    for i in range_(WEIGHTS_SIZE_LAYER3):
                        index = arith.AddIOp(
                            i,
                            arith.ConstantOp.create_index(
                                ACTIVATIONS_IN_SIZE - WEIGHTS_SIZE_LAYER3
                            ),
                        )
                        val = load(activations_in, [index])
                        weight_val = load(weights_in, [i])
                        val_out = arith.AddIOp(val, weight_val)
                        store(val_out, activations_out, [index])
                        yield_([])

                    # Send transformed activations to the next layer
                    ChannelPut("ActivationsOut", activations_out)

                    # Cleanup
                    DeallocOp(activations_in)
                    DeallocOp(weights_in)
                    DeallocOp(activations_out)


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

    activationsIn = np.full(shape=(ACTIVATIONS_IN_SIZE,), fill_value=1, dtype=np.int8)
    weightsIn = np.zeros(shape=(WEIGHTS_SIZE,), dtype=np.int8)
    activationsOut = np.full(shape=(ACTIVATIONS_IN_SIZE,), fill_value=1, dtype=np.int8)

    for i in range(WEIGHTS_SIZE):
        if i < WEIGHTS_SIZE_LAYER1:
            weightsIn[i] = 1
        elif i < WEIGHTS_SIZE_LAYER1 + WEIGHTS_SIZE_LAYER2:
            weightsIn[i] = 3
        else:
            weightsIn[i] = 7

    for i in range(WEIGHTS_SIZE_LAYER1):
        activationsOut[i] = (activationsOut[i] + weightsIn[i]) % 0xF
    for i in range(WEIGHTS_SIZE_LAYER2):
        activationsOut[i] = (
            activationsOut[i] + weightsIn[WEIGHTS_SIZE_LAYER1 + i]
        ) % 0xF
    for i in range(WEIGHTS_SIZE_LAYER3):
        activationsOut[(ACTIVATIONS_IN_SIZE - WEIGHTS_SIZE_LAYER3) + i] = (
            activationsOut[(ACTIVATIONS_IN_SIZE - WEIGHTS_SIZE_LAYER3) + i]
            + weightsIn[(WEIGHTS_SIZE - WEIGHTS_SIZE_LAYER3) + i]
        ) % 0xF

    runner = XRTRunner(verbose=args.verbose, experimental_passes=True)
    exit(
        runner.run_test(
            mlir_module,
            inputs=[activationsIn, weightsIn],
            expected_outputs=[activationsOut],
        )
    )
