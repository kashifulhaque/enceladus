// Best hand-written simdgroup_matrix GEMM measured on M4 Pro (16-core GPU).
//   C[M,N] = A[M,K] @ B[K,N], row-major, fp32 accumulation, T in {float, half, bfloat}.
//   Threadgroup tile 64x64, 4 simdgroups stacked along M (each owns a 16x64 strip = 2x8
//   fragments of 8x8), K stepped by 8, fragments loaded straight from device memory.
//   Interior tiles take an unmasked fast path; in edge tiles only the fragments that straddle
//   M/N use masked loads; the K % 8 tail is masked.
// Measured: fp32 5.4 TFLOPS, fp16/bf16 5.9 TFLOPS at 2048^3 and 4096^3
// (~96-98% of MPSMatrixMultiplication, ~95% of MPP matmul2d).
//
// This is the reference shape of what a Forge `tl.dot` lowering should emit. Each constant
// below is a compile-time specialization a compiler would bake in.
#include <metal_stdlib>
using namespace metal;

template <typename T, int BM, int BN, int WM, int WN>
struct GemmCfg {
  static constexpr constant int SM = BM / WM, SN = BN / WN;   // per-simdgroup C tile
  static constexpr constant int TM = SM / 8, TN = SN / 8;     // 8x8 fragments per simdgroup
  static constexpr constant int NT = WM * WN * 32;            // threads per threadgroup
};

// Masked 8x8 fragment load: every lane fetches the 2 elements it owns; OOB -> 0.
template <typename T>
inline void load_frag_masked(thread simdgroup_matrix<T, 8, 8>& f, device const T* p, uint ld,
                             int rows_left, int cols_left, uint fm, uint fn) {
  thread auto& e = f.thread_elements();
  const bool rok = int(fm) < rows_left;
  e[0] = (rok && int(fn) < cols_left) ? p[fm * ld + fn] : T(0);
  e[1] = (rok && int(fn) + 1 < cols_left) ? p[fm * ld + fn + 1] : T(0);
}

template <typename T, int BM, int BN, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]]
void gemm(device const T* A [[buffer(0)]], device const T* B [[buffer(1)]],
          device T* C [[buffer(2)]], constant uint3& MNK [[buffer(3)]],
          uint2 tgid [[threadgroup_position_in_grid]],
          uint sg [[simdgroup_index_in_threadgroup]],
          uint lane [[thread_index_in_simdgroup]]) {
  using Cfg = GemmCfg<T, BM, BN, WM, WN>;
  constexpr int TM = Cfg::TM, TN = Cfg::TN;
  const uint M = MNK.x, N = MNK.y, K = MNK.z;

  // --- program-id -> tile origin (Forge: tl.program_id(0/1)) ---
  const uint row0 = tgid.y * BM, col0 = tgid.x * BN;
  const uint sr = (sg / WN) * Cfg::SM, sc = (sg % WN) * Cfg::SN;

  // --- lane -> (row, col) of its two elements in an 8x8 fragment ---
  const uint qid = lane / 4;
  const uint fm = (qid & 4) + ((lane / 2) % 4);
  const uint fn = (qid & 2) * 2 + (lane % 2) * 2;

  // --- accumulator = tl.zeros((BM, BN), tl.float32), distributed over simdgroups ---
  simdgroup_matrix<float, 8, 8> acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  device const T* Ap = A + (row0 + sr) * K;   // this simdgroup's A strip
  device const T* Bp = B + col0 + sc;         // this simdgroup's B strip

  // --- uniform (per-threadgroup) interior/edge split ---
  const bool edge = (row0 + BM > M) || (col0 + BN > N);
  const uint kmain = (K / 8) * 8;

  if (!edge) {
    // --- fast path: for k in range(0, K, 8): acc += tl.dot(a_frag, b_frag) ---
    for (uint k = 0; k < kmain; k += 8) {
      simdgroup_matrix<T, 8, 8> a[TM], b[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) simdgroup_load(a[i], Ap + i * 8 * K + k, K);
      #pragma unroll
      for (int j = 0; j < TN; ++j) simdgroup_load(b[j], Bp + k * N + j * 8, N);
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
  } else {
    // --- edge tile: only fragments that straddle M/N use masked loads; the per-fragment
    //     test is uniform across the simdgroup, so there is no divergence ---
    for (uint k = 0; k < kmain; k += 8) {
      simdgroup_matrix<T, 8, 8> a[TM], b[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) {
        const int rows_left = int(M) - int(row0 + sr + i * 8);
        if (rows_left >= 8) simdgroup_load(a[i], Ap + i * 8 * K + k, K);
        else load_frag_masked(a[i], Ap + i * 8 * K + k, K, rows_left, 8, fm, fn);
      }
      #pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int cols_left = int(N) - int(col0 + sc + j * 8);
        if (cols_left >= 8) simdgroup_load(b[j], Bp + k * N + j * 8, N);
        else load_frag_masked(b[j], Bp + k * N + j * 8, N, 8, cols_left, fm, fn);
      }
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
  }
  // --- K tail (K % 8 != 0): masked, zero-filled fragment loads ---
  for (uint k = kmain; k < K; k += 8) {
    simdgroup_matrix<T, 8, 8> a[TM], b[TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
      load_frag_masked(a[i], Ap + i * 8 * K + k, K, int(M) - int(row0 + sr + i * 8), int(K - k), fm, fn);
    #pragma unroll
    for (int j = 0; j < TN; ++j)
      load_frag_masked(b[j], Bp + k * N + j * 8, N, int(K - k), int(N) - int(col0 + sc + j * 8), fm, fn);
    #pragma unroll
    for (int i = 0; i < TM; ++i)
      #pragma unroll
      for (int j = 0; j < TN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
  }

  // --- epilogue: registers -> C (fused bias/activation/cast would go here) ---
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      const uint r = row0 + sr + i * 8 + fm;
      const uint c = col0 + sc + j * 8 + fn;
      thread auto& e = acc[i][j].thread_elements();
      if (!edge) {
        *(device vec<T, 2>*)(C + r * N + c) = vec<T, 2>(T(e[0]), T(e[1]));
      } else if (r < M) {
        if (c < N) C[r * N + c] = T(e[0]);
        if (c + 1 < N) C[r * N + c + 1] = T(e[1]);
      }
    }
  }
}

// Explicit specializations (Forge would emit one per (dtype, config) it autotunes).
#define INST(T, NAME)                                                                          \
  template [[host_name(NAME)]] [[kernel]] void gemm<T, 64, 64, 4, 1>(                          \
      device const T*, device const T*, device T*, constant uint3&, uint2, uint, uint);
INST(float, "gemm_f32")
INST(half, "gemm_f16")
INST(bfloat, "gemm_bf16")
