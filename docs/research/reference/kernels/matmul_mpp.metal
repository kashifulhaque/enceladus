// Metal 4 Metal Performance Primitives matmul2d inside a compute shader.
// Tensors are built in-shader from plain device buffers (tensor_inline), so the host
// binds ordinary MTLBuffers -- no MTLTensor objects or MTL4 argument tables needed.
// Macros: DT (in type), DTO (out type)  -- not T/TO: single-letter macros break MPP/metal_tensor headers, BM, BN, NSG (simdgroups), RELAXED (0/1), KT (0 = dynamic K).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void mpp_mm(device DT* A [[buffer(0)]], device DT* B [[buffer(1)]], device DTO* C [[buffer(2)]],
                   constant uint3& MNK [[buffer(3)]], uint2 tgid [[threadgroup_position_in_grid]]) {
  const int M = MNK.x, N = MNK.y, K = MNK.z;
  // extents are innermost-first: a row-major M x K matrix has extents (K, M).
  tensor<device DT, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(K, M));
  tensor<device DT, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(N, K));
  tensor<device DTO, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(N, M));

  // KT > 0 loops over K manually, so the op must accumulate into its destination
  // (the default mode is `multiply`, which overwrites it).
  constexpr auto desc = matmul2d_descriptor(BM, BN, KT == 0 ? static_cast<int>(dynamic_extent) : KT,
                                            false, false, RELAXED,
                                            KT == 0 ? matmul2d_descriptor::mode::multiply
                                                    : matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  const int row0 = tgid.y * BM, col0 = tgid.x * BN;
#if KT == 0
  auto mA = tA.slice(0, row0);
  auto mB = tB.slice(col0, 0);
  auto mC = tC.slice(col0, row0);
  op.run(mA, mB, mC);
#else
  // Manual K loop with a cooperative (register) accumulator; the epilogue walks the
  // thread-owned elements, which is where a fused bias/activation would go.
  auto mA0 = tA.slice(0, row0);
  auto mB0 = tB.slice(col0, 0);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA0), decltype(mB0), float>();
  #pragma unroll
  for (uint16_t i = 0; i < cT.get_capacity(); ++i)
    if (cT.is_valid_element(i)) cT[i] = 0;
  for (int k = 0; k < K; k += KT) {
    auto mA = tA.slice(k, row0);
    auto mB = tB.slice(col0, k);
    op.run(mA, mB, cT);
  }
  #pragma unroll
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    if (cT.is_valid_element(i)) {
      auto idx = cT.get_multidimensional_index(i);   // (col, row) inside the BM x BN tile
      const int r = row0 + idx[1], c = col0 + idx[0];
      if (r < M && c < N) C[r * N + c] = DTO(cT[i]);
    }
  }
#endif
}
