#map = affine_map<()[s0] -> (s0 * 128)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
module {
  func.func private @linalg_fill_f32_view64x64xf32as2(f32, memref<64x64xf32, 2>) attributes {link_with = "kernel.o", llvm.emit_c_interface}
  func.func private @linalg_matmul_view64x64xbf16as2_view64x64xbf16as2_view64x64xf32as2(memref<64x64xbf16, 2>, memref<64x64xbf16, 2>, memref<64x64xf32, 2>) attributes {link_with = "kernel.o", llvm.emit_c_interface}
  func.func @forward(%arg0: memref<512x1024xbf16>, %arg1: memref<1024x512xbf16>, %arg2: memref<512x512xf32>) -> memref<512x512xf32> {
    %c4 = arith.constant 4 : index
    air.launch (%arg3, %arg4) in (%arg5=%c4, %arg6=%c4) args(%arg7=%arg2, %arg8=%arg0, %arg9=%arg1) : memref<512x512xf32>, memref<512x1024xbf16>, memref<1024x512xbf16> {
      air.segment @forward_0  args(%arg10=%arg3, %arg11=%arg4, %arg12=%arg7, %arg13=%arg8, %arg14=%arg9) : index, index, memref<512x512xf32>, memref<512x1024xbf16>, memref<1024x512xbf16> {
        %c2 = arith.constant 2 : index
        %c512 = arith.constant 512 : index
        %c1 = arith.constant 1 : index
        %c128 = arith.constant 128 : index
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c256 = arith.constant 256 : index
        %0 = affine.apply #map()[%arg10]
        %1 = affine.apply #map()[%arg11]
        %alloc = memref.alloc() : memref<128x1024xbf16, 1>
        scf.for %arg15 = %c0 to %c1024 step %c256 {
          air.dma_memcpy_nd (%alloc[%c0, %arg15] [%c128, %c256] [%c1024, %c1], %arg13[%0, %arg15] [%c128, %c256] [%c1024, %c1]) {id = 1 : i32} : (memref<128x1024xbf16, 1>, memref<512x1024xbf16>)
        }
        %alloc_0 = memref.alloc() : memref<1024x128xbf16, 1>
        scf.for %arg15 = %c0 to %c1024 step %c256 {
          air.dma_memcpy_nd (%alloc_0[%arg15, %c0] [%c256, %c128] [%c128, %c1], %arg14[%arg15, %1] [%c256, %c128] [%c512, %c1]) {id = 2 : i32} : (memref<1024x128xbf16, 1>, memref<1024x512xbf16>)
        }
        %alloc_1 = memref.alloc() : memref<128x128xf32, 1>
        air.herd @herd_0  tile (%arg15, %arg16) in (%arg17=%c2, %arg18=%c2) args(%arg19=%alloc_1, %arg20=%alloc, %arg21=%alloc_0) : memref<128x128xf32, 1>, memref<128x1024xbf16, 1>, memref<1024x128xbf16, 1> attributes {link_with = "kernel.o"} {
          %c128_2 = arith.constant 128 : index
          %c1_3 = arith.constant 1 : index
          %cst = arith.constant 0.000000e+00 : f32
          %c0_4 = arith.constant 0 : index
          %c1024_5 = arith.constant 1024 : index
          %c64 = arith.constant 64 : index
          %2 = affine.apply #map1()[%arg15]
          %3 = affine.apply #map1()[%arg16]
          %alloc_6 = memref.alloc() : memref<64x64xf32, 2>
          func.call @linalg_fill_f32_view64x64xf32as2(%cst, %alloc_6) : (f32, memref<64x64xf32, 2>) -> ()
          scf.for %arg22 = %c0_4 to %c1024_5 step %c64 {
            %alloc_7 = memref.alloc() : memref<64x64xbf16, 2>
            %alloc_8 = memref.alloc() : memref<64x64xbf16, 2>
            air.dma_memcpy_nd (%alloc_7[] [] [], %arg20[%2, %arg22] [%c64, %c64] [%c1024_5, %c1_3]) {id = 3 : i32} : (memref<64x64xbf16, 2>, memref<128x1024xbf16, 1>)
            air.dma_memcpy_nd (%alloc_8[] [] [], %arg21[%arg22, %3] [%c64, %c64] [%c128_2, %c1_3]) {id = 4 : i32} : (memref<64x64xbf16, 2>, memref<1024x128xbf16, 1>)
            func.call @linalg_matmul_view64x64xbf16as2_view64x64xbf16as2_view64x64xf32as2(%alloc_7, %alloc_8, %alloc_6) : (memref<64x64xbf16, 2>, memref<64x64xbf16, 2>, memref<64x64xf32, 2>) -> ()
            memref.dealloc %alloc_7 : memref<64x64xbf16, 2>
            memref.dealloc %alloc_8 : memref<64x64xbf16, 2>
          }
          air.dma_memcpy_nd (%arg19[%2, %3] [%c64, %c64] [%c128_2, %c1_3], %alloc_6[] [] []) {id = 5 : i32} : (memref<128x128xf32, 1>, memref<64x64xf32, 2>)
          memref.dealloc %alloc_6 : memref<64x64xf32, 2>
        }
        air.dma_memcpy_nd (%arg12[%0, %1] [%c128, %c128] [%c512, %c1], %alloc_1[] [] []) {id = 6 : i32} : (memref<512x512xf32>, memref<128x128xf32, 1>)
        memref.dealloc %alloc : memref<128x1024xbf16, 1>
        memref.dealloc %alloc_0 : memref<1024x128xbf16, 1>
        memref.dealloc %alloc_1 : memref<128x128xf32, 1>
      }
    }
    return %arg2 : memref<512x512xf32>
  }
}
