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

WEIGHTS_SIZE_LAYER1 = 8
WEIGHTS_SIZE_LAYER2 = 16
WEIGHTS_SIZE_LAYER3 = 8
WEIGHTS_SIZE = WEIGHTS_SIZE_LAYER1 + WEIGHTS_SIZE_LAYER2 + WEIGHTS_SIZE_LAYER3

ACTIVATIONS_IN_SIZE = 16
ACTIVATIONS_OUT_SIZE = 16


@module_builder
def build_module():
    activationsInL3_ty = MemRefType.get((ACTIVATIONS_IN_SIZE,), T.i32())
    weightsInL3_ty = MemRefType.get((WEIGHTS_SIZE,), T.i32())
    activationsOutL3_ty = MemRefType.get((ACTIVATIONS_OUT_SIZE,), T.i32())

    mem_space_l1 = IntegerAttr.get(T.i32(), MemorySpace.L1)
    activationsInL1_ty = MemRefType.get(
        (ACTIVATIONS_IN_SIZE,), T.i32(), memory_space=mem_space_l1
    )
    # weightsInL1_ty = MemRefType.get((WEIGHTS_SIZE,), T.i32(), memory_space=mem_space_l1)
    # activationsOutL1_ty = MemRefType.get((ACTIVATIONS_OUT_SIZE,), T.i32(), memory_space=mem_space_l1)

    ChannelOp("ActivationsIn")
    ChannelOp("ActivationsLayer1Layer2")
    ChannelOp("ActivationsLayer2Layer3")
    ChannelOp("ActivationsOut")

    # ChannelOp("WeightsInLayer1")
    # ChannelOp("WeightsInLayer2")
    # ChannelOp("WeightsInLayer3")

    @FuncOp.from_py_func(activationsInL3_ty, weightsInL3_ty, activationsOutL3_ty)
    def sequence(activationsIn, weightsIn, activationsOut):

        @launch(operands=[activationsIn, weightsIn, activationsOut])
        def launch_body(activationsInL3, weightsInL3, activationsOutL3):

            ChannelPut("ActivationsIn", activationsInL3)

            # ChannelPut("WeightsInLayer1", weightsInL3)

            ChannelGet("ActivationsOut", activationsOutL3)

            @segment(name="seg")
            def segment_body():

                @herd(name="layer1", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):
                    # Get the input activations to this layer
                    activations_in = AllocOp(activationsInL1_ty, [], [])
                    ChannelGet("ActivationsIn", activations_in)

                    # Send transformed activations to the next layer
                    ChannelPut("ActivationsLayer1Layer2", activations_in)
                    DeallocOp(activations_in)

                @herd(name="layer2", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):
                    # Get the input activations to this layer
                    activations_in = AllocOp(activationsInL1_ty, [], [])
                    ChannelGet("ActivationsLayer1Layer2", activations_in)

                    # Send transformed activations to the next layer
                    ChannelPut("ActivationsLayer2Layer3", activations_in)
                    DeallocOp(activations_in)

                @herd(name="layer3", sizes=[1, 1])
                def herd_body(tx, ty, sx, sy):
                    # Get the input activations to this layer
                    activations_in = AllocOp(activationsInL1_ty, [], [])
                    ChannelGet("ActivationsLayer2Layer3", activations_in)

                    # Send transformed activations to the next layer
                    ChannelPut("ActivationsOut", activations_in)
                    DeallocOp(activations_in)


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

    activationsIn = np.zeros(shape=(ACTIVATIONS_IN_SIZE,), dtype=np.int32)
    weightsIn = np.zeros(shape=(WEIGHTS_SIZE,), dtype=np.int32)
    activationsOut = np.zeros(shape=(ACTIVATIONS_OUT_SIZE,), dtype=np.int32)

    runner = XRTRunner(verbose=args.verbose, experimental_passes=True)
    exit(
        runner.run_test(
            mlir_module,
            inputs=[activationsIn, weightsIn],
            expected_outputs=[activationsOut],
        )
    )
