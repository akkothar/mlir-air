#map = affine_map<()[s0] -> (s0 * 128)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
module {
  air.channel @channel_7 [1, 1]
  air.channel @channel_6 [2, 2]
  air.channel @channel_5 [1, 1]
  air.channel @channel_4 [1, 1]
  air.channel @channel_3 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_2 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_1 [1, 1] {broadcast_shape = [1, 2]}
  air.channel @channel_0 [1, 1] {broadcast_shape = [1, 2]}
  func.func private @linalg_fill_f32_view64x64xf32as2(f32, memref<64x64xf32, 2>) attributes {link_with = "kernel.o", llvm.emit_c_interface}
  func.func private @linalg_matmul_view64x64xbf16as2_view64x64xbf16as2_view64x64xf32as2(memref<64x64xbf16, 2>, memref<64x64xbf16, 2>, memref<64x64xf32, 2>) attributes {link_with = "kernel.o", llvm.emit_c_interface}
  func.func @forward(%arg0: memref<512x1024xbf16>, %arg1: memref<1024x512xbf16>, %arg2: memref<512x512xf32>) -> memref<512x512xf32> {
    %c4 = arith.constant 4 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c4, %arg6=%c4) args(%arg7=%arg2, %arg8=%arg0, %arg9=%arg1) : memref<512x512xf32>, memref<512x1024xbf16>, memref<1024x512xbf16> attributes {id = 1 : i32} {
      %c512 = arith.constant 512 : index
      %c1 = arith.constant 1 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c256 = arith.constant 256 : index
      %async_token, %results = air.execute -> (index) {
        %5 = affine.apply #map()[%arg3]
        air.execute_terminator %5 : index
      }
      %1 = scf.for %arg10 = %c0 to %c1024 step %c256 iter_args(%arg11 = %async_token) -> (!air.async.token) {
        %5 = air.channel.put async [%arg11]  @channel_4[] (%arg8[%results, %arg10] [%c128, %c256] [%c1024, %c1]) {id = 1 : i32} : (memref<512x1024xbf16>)
        scf.yield %5 : !air.async.token
      }
      %async_token_0, %results_1 = air.execute -> (index) {
        %5 = affine.apply #map()[%arg4]
        air.execute_terminator %5 : index
      }
      %2 = scf.for %arg10 = %c0 to %c1024 step %c256 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
        %5 = air.channel.put async [%arg11]  @channel_5[] (%arg9[%arg10, %results_1] [%c256, %c128] [%c512, %c1]) {id = 2 : i32} : (memref<1024x512xbf16>)
        scf.yield %5 : !air.async.token
      }
      %async_token_2, %results_3 = air.execute -> (index) {
        %5 = affine.apply #map()[%arg3]
        air.execute_terminator %5 : index
      }
      %async_token_4, %results_5 = air.execute -> (index) {
        %5 = affine.apply #map()[%arg4]
        air.execute_terminator %5 : index
      }
      %3 = air.channel.get async [%async_token_2, %async_token_4]  @channel_7[] (%arg7[%results_3, %results_5] [%c128, %c128] [%c512, %c1]) {id = 3 : i32} : (memref<512x512xf32>)
      %4 = air.segment @forward_0 async  attributes {id = 2 : i32} {
        %c64 = arith.constant 64 : index
        %c2 = arith.constant 2 : index
        %c1_6 = arith.constant 1 : index
        %c128_7 = arith.constant 128 : index
        %c0_8 = arith.constant 0 : index
        %c1024_9 = arith.constant 1024 : index
        %c256_10 = arith.constant 256 : index
        %5 = air.wait_all async
        %6 = air.wait_all async
        %7 = scf.for %arg10 = %c0_8 to %c1024_9 step %c256_10 iter_args(%arg11 = %6) -> (!air.async.token) {
          %async_token_14, %results_15 = air.execute -> (memref<128x256xbf16, 1>) {
            %alloc = memref.alloc() {hoist_alloc = true} : memref<128x256xbf16, 1>
            air.execute_terminator %alloc : memref<128x256xbf16, 1>
          }
          %async_token_16, %results_17 = air.execute -> (memref<256x128xbf16, 1>) {
            %alloc = memref.alloc() {hoist_alloc = true} : memref<256x128xbf16, 1>
            air.execute_terminator %alloc : memref<256x128xbf16, 1>
          }
          %14 = air.channel.get async [%async_token_14]  @channel_4[] (%results_15[%c0_8, %arg10] [%c128_7, %c256_10] [%c256_10, %c1_6]) {id = 4 : i32} : (memref<128x256xbf16, 1>)
          %15 = scf.for %arg12 = %c0_8 to %c256_10 step %c64 iter_args(%arg13 = %14) -> (!air.async.token) {
            %21 = air.channel.put async [%arg13]  @channel_0[] (%results_15[%c0_8, %arg12] [%c64, %c64] [%c256_10, %c1_6]) {id = 6 : i32} : (memref<128x256xbf16, 1>)
            scf.yield %21 : !air.async.token
          }
          %16 = scf.for %arg12 = %c0_8 to %c256_10 step %c64 iter_args(%arg13 = %14) -> (!air.async.token) {
            %21 = air.channel.put async [%arg13]  @channel_1[] (%results_15[%c64, %arg12] [%c64, %c64] [%c256_10, %c1_6]) {id = 7 : i32} : (memref<128x256xbf16, 1>)
            scf.yield %21 : !air.async.token
          }
          %17 = air.channel.get async [%async_token_16]  @channel_5[] (%results_17[%arg10, %c0_8] [%c256_10, %c128_7] [%c128_7, %c1_6]) {id = 5 : i32} : (memref<256x128xbf16, 1>)
          %18 = scf.for %arg12 = %c0_8 to %c256_10 step %c64 iter_args(%arg13 = %17) -> (!air.async.token) {
            %21 = air.channel.put async [%arg13]  @channel_2[] (%results_17[%arg12, %c0_8] [%c64, %c64] [%c128_7, %c1_6]) {id = 8 : i32} : (memref<256x128xbf16, 1>)
            scf.yield %21 : !air.async.token
          }
          %19 = scf.for %arg12 = %c0_8 to %c256_10 step %c64 iter_args(%arg13 = %17) -> (!air.async.token) {
            %21 = air.channel.put async [%arg13]  @channel_3[] (%results_17[%arg12, %c64] [%c64, %c64] [%c128_7, %c1_6]) {id = 9 : i32} : (memref<256x128xbf16, 1>)
            scf.yield %21 : !air.async.token
          }
          %async_token_18 = air.execute {
            memref.dealloc %results_15 : memref<128x256xbf16, 1>
          }
          %async_token_19 = air.execute {
            memref.dealloc %results_17 : memref<256x128xbf16, 1>
          }
          %20 = air.wait_all async [%14, %15, %16, %17, %18, %19, %async_token_18, %async_token_19]
          scf.yield %20 : !air.async.token
        } {unroll = 2 : i32}
        %8 = air.wait_all async
        %async_token_11, %results_12 = air.execute -> (memref<128x128xf32, 1>) {
          %alloc = memref.alloc() : memref<128x128xf32, 1>
          air.execute_terminator %alloc : memref<128x128xf32, 1>
        }
        %9 = scf.parallel (%arg10, %arg11) = (%c0_8, %c0_8) to (%c2, %c2) step (%c1_6, %c1_6) init (%async_token_11) -> !air.async.token {
          %async_token_14, %results_15 = air.execute -> (index) {
            %15 = affine.apply #map1()[%arg10]
            air.execute_terminator %15 : index
          }
          %async_token_16, %results_17 = air.execute -> (index) {
            %15 = affine.apply #map1()[%arg11]
            air.execute_terminator %15 : index
          }
          %14 = air.channel.get async [%async_token_11, %async_token_16, %async_token_14]  @channel_6[%arg10, %arg11] (%results_12[%results_15, %results_17] [%c64, %c64] [%c128_7, %c1_6]) {id = 10 : i32} : (memref<128x128xf32, 1>)
          scf.reduce(%14 : !air.async.token) {
          ^bb0(%arg12: !air.async.token, %arg13: !air.async.token):
            %15 = air.wait_all async [%arg12, %arg13]
            scf.reduce.return %15 : !air.async.token
          }
        }
        %10 = air.herd @herd_0 async [%7, %7, %async_token_11]  tile (%arg10, %arg11) in (%arg12=%c2, %arg13=%c2) attributes {id = 3 : i32, link_with = "kernel.o"} {
          %cst = arith.constant 0.000000e+00 : f32
          %c0_14 = arith.constant 0 : index
          %c1024_15 = arith.constant 1024 : index
          %c64_16 = arith.constant 64 : index
          %async_token_17, %results_18 = air.execute -> (memref<64x64xf32, 2>) {
            %alloc = memref.alloc() : memref<64x64xf32, 2>
            air.execute_terminator %alloc : memref<64x64xf32, 2>
          }
          %async_token_19 = air.execute [%async_token_17] {
            func.call @linalg_fill_f32_view64x64xf32as2(%cst, %results_18) : (f32, memref<64x64xf32, 2>) -> ()
          }
          %14 = scf.for %arg14 = %c0_14 to %c1024_15 step %c64_16 iter_args(%arg15 = %async_token_19) -> (!air.async.token) {
            %async_token_21, %results_22 = air.execute -> (memref<64x64xbf16, 2>) {
              %alloc = memref.alloc() {hoist_alloc = true} : memref<64x64xbf16, 2>
              air.execute_terminator %alloc : memref<64x64xbf16, 2>
            }
            %async_token_23, %results_24 = air.execute -> (memref<64x64xbf16, 2>) {
              %alloc = memref.alloc() {hoist_alloc = true} : memref<64x64xbf16, 2>
              air.execute_terminator %alloc : memref<64x64xbf16, 2>
            }
            %16 = affine.if #set()[%arg10, %arg11] -> !air.async.token {
              %18 = air.channel.get async [%arg15, %async_token_21]  @channel_0[%arg10, %arg11] (%results_22[] [] []) {id = 11 : i32} : (memref<64x64xbf16, 2>)
              affine.yield %18 : !air.async.token
            } else {
              %18 = air.channel.get async [%arg15, %async_token_21]  @channel_1[%arg10, %arg11] (%results_22[] [] []) {id = 12 : i32} : (memref<64x64xbf16, 2>)
              affine.yield %18 : !air.async.token
            }
            %17 = affine.if #set1()[%arg10, %arg11] -> !air.async.token {
              %18 = air.channel.get async [%arg15, %async_token_23]  @channel_2[%arg10, %arg11] (%results_24[] [] []) {id = 13 : i32} : (memref<64x64xbf16, 2>)
              affine.yield %18 : !air.async.token
            } else {
              %18 = air.channel.get async [%arg15, %async_token_23]  @channel_3[%arg10, %arg11] (%results_24[] [] []) {id = 14 : i32} : (memref<64x64xbf16, 2>)
              affine.yield %18 : !air.async.token
            }
            %async_token_25 = air.execute [%17, %16] {
              func.call @linalg_matmul_view64x64xbf16as2_view64x64xbf16as2_view64x64xf32as2(%results_22, %results_24, %results_18) : (memref<64x64xbf16, 2>, memref<64x64xbf16, 2>, memref<64x64xf32, 2>) -> ()
            }
            %async_token_26 = air.execute [%async_token_25] {
              memref.dealloc %results_22 : memref<64x64xbf16, 2>
            }
            %async_token_27 = air.execute [%async_token_25] {
              memref.dealloc %results_24 : memref<64x64xbf16, 2>
            }
            scf.yield %async_token_25 : !air.async.token
          } {unroll = 2 : i32}
          %15 = air.channel.put async [%14]  @channel_6[%arg10, %arg11] (%results_18[] [] []) {id = 15 : i32} : (memref<64x64xf32, 2>)
          %async_token_20 = air.execute [%15] {
            memref.dealloc %results_18 : memref<64x64xf32, 2>
          }
        }
        %11 = air.channel.put async [%10]  @channel_7[] (%results_12[] [] []) {id = 16 : i32} : (memref<128x128xf32, 1>)
        %12 = air.wait_all async
        %13 = air.wait_all async
        %async_token_13 = air.execute [%11] {
          memref.dealloc %results_12 : memref<128x128xf32, 1>
        }
      }
    }
    return %arg2 : memref<512x512xf32>
  }
}
