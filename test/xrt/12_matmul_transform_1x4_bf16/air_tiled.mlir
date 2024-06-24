#map = affine_map<(d0) -> (d0 * 128)>
#map1 = affine_map<(d0) -> (d0 * 64)>
module {
  func.func @forward(%arg0: memref<512x1024xbf16>, %arg1: memref<1024x512xbf16>, %arg2: memref<512x512xf32>) -> memref<512x512xf32> {
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    scf.forall (%arg3, %arg4) in (4, 4) {
      %0 = affine.apply #map(%arg3)
      %1 = affine.apply #map(%arg4)
      %subview = memref.subview %arg2[%0, %1] [128, 128] [1, 1] : memref<512x512xf32> to memref<128x128xf32, strided<[512, 1], offset: ?>>
      %alloc = memref.alloc() : memref<128x1024xbf16, 1>
      scf.for %arg5 = %c0 to %c1024 step %c256 {
        %subview_2 = memref.subview %arg0[%0, %arg5] [128, 256] [1, 1] : memref<512x1024xbf16> to memref<128x256xbf16, strided<[1024, 1], offset: ?>>
        %subview_3 = memref.subview %alloc[0, %arg5] [128, 256] [1, 1] : memref<128x1024xbf16, 1> to memref<128x256xbf16, strided<[1024, 1], offset: ?>, 1>
        linalg.copy ins(%subview_2 : memref<128x256xbf16, strided<[1024, 1], offset: ?>>) outs(%subview_3 : memref<128x256xbf16, strided<[1024, 1], offset: ?>, 1>)
      }
      %alloc_0 = memref.alloc() : memref<1024x128xbf16, 1>
      scf.for %arg5 = %c0 to %c1024 step %c256 {
        %subview_2 = memref.subview %arg1[%arg5, %1] [256, 128] [1, 1] : memref<1024x512xbf16> to memref<256x128xbf16, strided<[512, 1], offset: ?>>
        %subview_3 = memref.subview %alloc_0[%arg5, 0] [256, 128] [1, 1] : memref<1024x128xbf16, 1> to memref<256x128xbf16, strided<[128, 1], offset: ?>, 1>
        linalg.copy ins(%subview_2 : memref<256x128xbf16, strided<[512, 1], offset: ?>>) outs(%subview_3 : memref<256x128xbf16, strided<[128, 1], offset: ?>, 1>)
      }
      %alloc_1 = memref.alloc() : memref<128x128xf32, 1>
      scf.forall (%arg5, %arg6) in (2, 2) {
        %2 = affine.apply #map1(%arg5)
        %3 = affine.apply #map1(%arg6)
        %subview_2 = memref.subview %alloc_1[%2, %3] [64, 64] [1, 1] : memref<128x128xf32, 1> to memref<64x64xf32, strided<[128, 1], offset: ?>, 1>
        %alloc_3 = memref.alloc() : memref<64x64xf32, 2>
        linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<64x64xf32, 2>)
        scf.for %arg7 = %c0 to %c1024 step %c64 {
          %subview_4 = memref.subview %alloc[%2, %arg7] [64, 64] [1, 1] : memref<128x1024xbf16, 1> to memref<64x64xbf16, strided<[1024, 1], offset: ?>, 1>
          %subview_5 = memref.subview %alloc_0[%arg7, %3] [64, 64] [1, 1] : memref<1024x128xbf16, 1> to memref<64x64xbf16, strided<[128, 1], offset: ?>, 1>
          %alloc_6 = memref.alloc() : memref<64x64xbf16, 2>
          %alloc_7 = memref.alloc() : memref<64x64xbf16, 2>
          memref.copy %subview_4, %alloc_6 : memref<64x64xbf16, strided<[1024, 1], offset: ?>, 1> to memref<64x64xbf16, 2>
          memref.copy %subview_5, %alloc_7 : memref<64x64xbf16, strided<[128, 1], offset: ?>, 1> to memref<64x64xbf16, 2>
          linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%alloc_6, %alloc_7 : memref<64x64xbf16, 2>, memref<64x64xbf16, 2>) outs(%alloc_3 : memref<64x64xf32, 2>)
          memref.dealloc %alloc_6 : memref<64x64xbf16, 2>
          memref.dealloc %alloc_7 : memref<64x64xbf16, 2>
        }
        memref.copy %alloc_3, %subview_2 : memref<64x64xf32, 2> to memref<64x64xf32, strided<[128, 1], offset: ?>, 1>
        memref.dealloc %alloc_3 : memref<64x64xf32, 2>
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      linalg.copy ins(%alloc_1 : memref<128x128xf32, 1>) outs(%subview : memref<128x128xf32, strided<[512, 1], offset: ?>>)
      memref.dealloc %alloc : memref<128x1024xbf16, 1>
      memref.dealloc %alloc_0 : memref<1024x128xbf16, 1>
      memref.dealloc %alloc_1 : memref<128x128xf32, 1>
    }
    return %arg2 : memref<512x512xf32>
  }
}
