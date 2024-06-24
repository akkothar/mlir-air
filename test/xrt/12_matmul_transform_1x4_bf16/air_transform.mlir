#map = affine_map<(d0) -> (d0 * 128)>
#map1 = affine_map<(d0) -> (d0 * 64)>
module {
  func.func @forward(%arg0: tensor<512x1024xbf16>, %arg1: tensor<1024x512xbf16>, %arg2: tensor<512x512xf32>) -> tensor<512x512xf32> {
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = scf.forall (%arg3, %arg4) in (4, 4) shared_outs(%arg5 = %arg2) -> (tensor<512x512xf32>) {
      %1 = affine.apply #map(%arg3)
      %2 = affine.apply #map(%arg4)
      %extracted_slice = tensor.extract_slice %arg0[%1, 0] [128, 1024] [1, 1] : tensor<512x1024xbf16> to tensor<128x1024xbf16>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %2] [1024, 128] [1, 1] : tensor<1024x512xbf16> to tensor<1024x128xbf16>
      %extracted_slice_1 = tensor.extract_slice %arg5[%1, %2] [128, 128] [1, 1] : tensor<512x512xf32> to tensor<128x128xf32>
      %3 = bufferization.alloc_tensor() : tensor<128x1024xbf16>
      %alloc = memref.alloc() : memref<128x1024xbf16, 1>
      %4 = bufferization.to_tensor %alloc restrict writable : memref<128x1024xbf16, 1>
      %5 = scf.for %arg6 = %c0 to %c1024 step %c256 iter_args(%arg7 = %4) -> (tensor<128x1024xbf16>) {
        %extracted_slice_4 = tensor.extract_slice %extracted_slice[0, %arg6] [128, 256] [1, 1] : tensor<128x1024xbf16> to tensor<128x256xbf16>
        %extracted_slice_5 = tensor.extract_slice %arg7[0, %arg6] [128, 256] [1, 1] : tensor<128x1024xbf16> to tensor<128x256xbf16>
        %13 = linalg.copy ins(%extracted_slice_4 : tensor<128x256xbf16>) outs(%extracted_slice_5 : tensor<128x256xbf16>) -> tensor<128x256xbf16>
        %inserted_slice = tensor.insert_slice %13 into %arg7[0, %arg6] [128, 256] [1, 1] : tensor<128x256xbf16> into tensor<128x1024xbf16>
        scf.yield %inserted_slice : tensor<128x1024xbf16>
      }
      %6 = bufferization.alloc_tensor() : tensor<1024x128xbf16>
      %alloc_2 = memref.alloc() : memref<1024x128xbf16, 1>
      %7 = bufferization.to_tensor %alloc_2 restrict writable : memref<1024x128xbf16, 1>
      %8 = scf.for %arg6 = %c0 to %c1024 step %c256 iter_args(%arg7 = %7) -> (tensor<1024x128xbf16>) {
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_0[%arg6, 0] [256, 128] [1, 1] : tensor<1024x128xbf16> to tensor<256x128xbf16>
        %extracted_slice_5 = tensor.extract_slice %arg7[%arg6, 0] [256, 128] [1, 1] : tensor<1024x128xbf16> to tensor<256x128xbf16>
        %13 = linalg.copy ins(%extracted_slice_4 : tensor<256x128xbf16>) outs(%extracted_slice_5 : tensor<256x128xbf16>) -> tensor<256x128xbf16>
        %inserted_slice = tensor.insert_slice %13 into %arg7[%arg6, 0] [256, 128] [1, 1] : tensor<256x128xbf16> into tensor<1024x128xbf16>
        scf.yield %inserted_slice : tensor<1024x128xbf16>
      }
      %9 = bufferization.alloc_tensor() : tensor<128x128xf32>
      %alloc_3 = memref.alloc() : memref<128x128xf32, 1>
      %10 = bufferization.to_tensor %alloc_3 restrict writable : memref<128x128xf32, 1>
      %11 = scf.forall (%arg6, %arg7) in (2, 2) shared_outs(%arg8 = %10) -> (tensor<128x128xf32>) {
        %13 = affine.apply #map1(%arg6)
        %14 = affine.apply #map1(%arg7)
        %extracted_slice_4 = tensor.extract_slice %5[%13, 0] [64, 1024] [1, 1] : tensor<128x1024xbf16> to tensor<64x1024xbf16>
        %extracted_slice_5 = tensor.extract_slice %8[0, %14] [1024, 64] [1, 1] : tensor<1024x128xbf16> to tensor<1024x64xbf16>
        %extracted_slice_6 = tensor.extract_slice %arg8[%13, %14] [64, 64] [1, 1] : tensor<128x128xf32> to tensor<64x64xf32>
        %15 = linalg.fill ins(%cst : f32) outs(%extracted_slice_6 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %16 = linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%extracted_slice_4, %extracted_slice_5 : tensor<64x1024xbf16>, tensor<1024x64xbf16>) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %16 into %arg8[%13, %14] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<128x128xf32>
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      %12 = linalg.copy ins(%11 : tensor<128x128xf32>) outs(%extracted_slice_1 : tensor<128x128xf32>) -> tensor<128x128xf32>
      memref.dealloc %alloc : memref<128x1024xbf16, 1>
      memref.dealloc %alloc_2 : memref<1024x128xbf16, 1>
      memref.dealloc %alloc_3 : memref<128x128xf32, 1>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %12 into %arg5[%1, %2] [128, 128] [1, 1] : tensor<128x128xf32> into tensor<512x512xf32>
      }
    }
    return %0 : tensor<512x512xf32>
  }
}
