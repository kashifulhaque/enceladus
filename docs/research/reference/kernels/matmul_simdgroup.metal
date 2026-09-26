// simdgroup_matrix GEMM template: C[M,N] = A[M,K] @ B[K,N], row-major, NN.
// This is the codegen template a Forge `tl.dot` lowering should emit.
//
// Compile-time macros (Forge would bake these as constexpr):
//   T      element type of A/B/C (float, half, bfloat)
//   ACC    accumulator type (float, or T)
//   BM,BN,BK  threadgroup tile; WM x WN simdgroups per threadgroup
//   PAD    extra elements per threadgroup-memory row (bank-conflict avoidance)
//   MASK   0 = no bounds checks (M,N % tile == 0, K % BK == 0)
//          1 = every load/store bounds-checked
//          2 = bounds-check only edge tiles (uniform branch) + masked K tail
//   SWZ    log2 threadgroup swizzle for L2 locality (0 = off)
//   STREAM 1 = stream B fragments one at a time in the inner loop (fewer live registers)
//   MAXT   1 = emit [[max_total_threads_per_threadgroup(NT)]] (critical, see below)
//   PREFETCH 1 = (direct path) register double-buffering of the next k-step fragments
//   TGMEM  1 = stage tiles through threadgroup memory, 0 = simdgroup_load straight from device
#include <metal_stdlib>
using namespace metal;

#define NSG (WM * WN)
#define NT (NSG * 32)
#define SM (BM / WM)   // rows of C per simdgroup
#define SN (BN / WN)   // cols of C per simdgroup
#define TM (SM / 8)    // 8x8 fragments per simdgroup along M
#define TN (SN / 8)    // along N
#define LDA (BK + PAD)
#define LDB (BN + PAD)

typedef vec<T, 4> T4;

// Cooperative global -> threadgroup copy of a ROWS x COLS tile with 4-wide vectors.
template <int ROWS, int COLS, int LD, bool CHECK>
inline void load_tile(threadgroup T* dst, device const T* src, uint ld_src, uint tid,
                      uint rows_left, uint cols_left) {
  constexpr int V = ROWS * COLS / 4;
  #pragma unroll
  for (int i = tid; i < V; i += NT) {
    const int r = i / (COLS / 4), c = (i % (COLS / 4)) * 4;
    T4 v;
    if (CHECK) {
      v = (uint(r) < rows_left && uint(c) < cols_left) ? *(device const T4*)(src + r * ld_src + c) : T4(0);
    } else {
      v = *(device const T4*)(src + r * ld_src + c);
    }
    *(threadgroup T4*)(dst + r * LD + c) = v;
  }
}

// Masked 8x8 fragment load straight from device memory: each lane fetches the two
// elements it owns; out-of-bounds elements become 0.
inline void load_frag_masked(thread simdgroup_matrix<T, 8, 8>& f, device const T* p, uint ld,
                             int rows_left, int cols_left, uint fm, uint fn) {
  thread auto& e = f.thread_elements();
  const bool rok = int(fm) < rows_left;
  e[0] = (rok && int(fn) < cols_left) ? p[fm * ld + fn] : T(0);
  e[1] = (rok && int(fn) + 1 < cols_left) ? p[fm * ld + fn + 1] : T(0);
}

#ifndef PREFETCH
#define PREFETCH 0
#endif
#ifndef MAXT
#define MAXT 1
#endif
#if MAXT
// Tell the compiler the real threadgroup size so it can budget more registers per thread.
// Without this the pipeline must support 1024 threads/TG and the compiler caps registers,
// which makes 32x32-per-simdgroup fp32 tiles spill (~10x slowdown).
[[max_total_threads_per_threadgroup(NT)]]
#endif
kernel void sgmm(device const T* A [[buffer(0)]], device const T* B [[buffer(1)]],
                 device T* C [[buffer(2)]], constant uint3& MNK [[buffer(3)]],
                 uint2 tgid [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
  const uint M = MNK.x, N = MNK.y, K = MNK.z;
#if SWZ > 0
  // Swizzle: consecutive threadgroups walk a (1<<SWZ)-tall column of tiles, so they share B tiles in L2.
  const uint bx = tgid.x >> SWZ;
  const uint by = (tgid.y << SWZ) + (tgid.x & ((1u << SWZ) - 1));
  if (by * BM >= M || bx * BN >= N) return;
#else
  const uint bx = tgid.x, by = tgid.y;
#endif
  const uint row0 = by * BM, col0 = bx * BN;
  const uint sr = (sg / WN) * SM, sc = (sg % WN) * SN;   // simdgroup's sub-tile origin
  // Lane -> (row, col) of the 2 elements it owns in an 8x8 simdgroup_matrix (thread_elements()).
  const uint qid = lane / 4;
  const uint fm = (qid & 4) + ((lane / 2) % 4);
  const uint fn = (qid & 2) * 2 + (lane % 2) * 2;

  simdgroup_matrix<ACC, 8, 8> acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = make_filled_simdgroup_matrix<ACC, 8, 8>(0);

#if TGMEM
  threadgroup T As[BM * LDA];
  threadgroup T Bs[BK * LDB];
  device const T* Ab = A + row0 * K;
  device const T* Bb = B + col0;
#if MASK == 1
  const bool edge = true;
#elif MASK == 2
  const bool edge = (row0 + BM > M) || (col0 + BN > N);
#else
  const bool edge = false;
#endif
  for (uint k0 = 0; k0 < K; k0 += BK) {
#if MASK
    if (edge || k0 + BK > K) {
      load_tile<BM, BK, LDA, true>(As, Ab + k0, K, tid, M - row0, K - k0);
      load_tile<BK, BN, LDB, true>(Bs, Bb + k0 * N, N, tid, K - k0, N - col0);
    } else
#endif
    {
      load_tile<BM, BK, LDA, false>(As, Ab + k0, K, tid, 0, 0);
      load_tile<BK, BN, LDB, false>(Bs, Bb + k0 * N, N, tid, 0, 0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
#if STREAM
    // Keep only TM A-fragments + 1 B-fragment live (lower register pressure).
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_matrix<T, 8, 8> a[TM];
      #pragma unroll
      for (int i = 0; i < TM; ++i) simdgroup_load(a[i], As + (sr + i * 8) * LDA + kk, LDA);
      #pragma unroll
      for (int j = 0; j < TN; ++j) {
        simdgroup_matrix<T, 8, 8> b;
        simdgroup_load(b, Bs + kk * LDB + sc + j * 8, LDB);
        #pragma unroll
        for (int i = 0; i < TM; ++i) simdgroup_multiply_accumulate(acc[i][j], a[i], b, acc[i][j]);
      }
    }
#else
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_matrix<T, 8, 8> a[TM], b[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) simdgroup_load(a[i], As + (sr + i * 8) * LDA + kk, LDA);
      #pragma unroll
      for (int j = 0; j < TN; ++j) simdgroup_load(b[j], Bs + kk * LDB + sc + j * 8, LDB);
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#else
  // Direct-from-device fragments: no threadgroup memory, no barriers; the GPU caches
  // provide the reuse. Fastest variant on M4 for interior tiles.
  device const T* Ap = A + (row0 + sr) * K;
  device const T* Bp = B + col0 + sc;
#if MASK == 1
  const uint kmain = 0;                                        // everything masked
#elif MASK == 2
  const bool edge = (row0 + BM > M) || (col0 + BN > N);        // uniform per threadgroup
  const uint kmain = edge ? 0 : (K / BK) * BK;                 // interior: fast path for full K blocks
#else
  const uint kmain = K;
#endif
#if PREFETCH
  // Software pipelining: fragments for step k+8 are loaded before the MMAs of step k.
  if (kmain > 0) {
    simdgroup_matrix<T, 8, 8> a[TM], b[TN], an[TM], bn[TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) simdgroup_load(a[i], Ap + i * 8 * K, K);
    #pragma unroll
    for (int j = 0; j < TN; ++j) simdgroup_load(b[j], Bp + j * 8, N);
    for (uint k = 8; k < kmain; k += 8) {
      #pragma unroll
      for (int i = 0; i < TM; ++i) simdgroup_load(an[i], Ap + i * 8 * K + k, K);
      #pragma unroll
      for (int j = 0; j < TN; ++j) simdgroup_load(bn[j], Bp + k * N + j * 8, N);
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
      #pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = an[i];
      #pragma unroll
      for (int j = 0; j < TN; ++j) b[j] = bn[j];
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i)
      #pragma unroll
      for (int j = 0; j < TN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
  }
#else
  for (uint k0 = 0; k0 < kmain; k0 += BK) {
    #pragma unroll
    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_matrix<T, 8, 8> a[TM], b[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) simdgroup_load(a[i], Ap + i * 8 * K + k0 + kk, K);
      #pragma unroll
      for (int j = 0; j < TN; ++j) simdgroup_load(b[j], Bp + (k0 + kk) * N + j * 8, N);
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
  }
#endif
#if MASK
  // Slow path (edge tiles and K tail): per-lane masked fragment loads, zero-filled.
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
#endif
#endif

  // Epilogue: each lane owns 2 adjacent elements (fm, fn), (fm, fn+1) of every 8x8 fragment.
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      const uint r = row0 + sr + i * 8 + fm;
      const uint c = col0 + sc + j * 8 + fn;
      thread auto& e = acc[i][j].thread_elements();
      // (a fused epilogue -- bias, activation, residual -- goes here, in registers)
#if MASK
      if (r < M) {
        if (c < N) C[r * N + c] = T(e[0]);
        if (c + 1 < N) C[r * N + c + 1] = T(e[1]);
      }
#else
      *(device vec<T, 2>*)(C + r * N + c) = vec<T, 2>(T(e[0]), T(e[1]));
#endif
    }
  }
}
