// Enceladus prelude: helpers included in every generated kernel.

// erf with absolute error below 1.5e-7 (Abramowitz and Stegun 7.1.26).
static inline float tg_erf(float x) {
  float s = x < 0.0f ? -1.0f : 1.0f;
  float a = fabs(x);
  float t = 1.0f / fma(0.3275911f, a, 1.0f);
  float y = t * fma(t, fma(t, fma(t, fma(t, 1.061405429f, -1.453152027f), 1.421413741f),
                           -0.284496736f), 0.254829592f);
  return s * (1.0f - y * exp(-a * a));
}

static inline float tg_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// tanh within 1.4 ulp for every float, as in Cephes `tanhf`: an odd polynomial below
// 0.625, and 1 - 2 / (exp(2|x|) + 1) above. It replaces Metal's fast `tanh`, which
// returns 0 at 44 and NaN from 45, and runs about 4x faster than `precise::tanh`.
static inline float tg_tanh(float x) {
  const float z = fabs(x);
  float r;
  if (z < 0.625f) {
    const float s = z * z;
    const float p = fma(fma(fma(fma(-5.70498872745e-3f, s, 2.06390887954e-2f), s,
                                -5.37397155531e-2f), s, 1.33314422036e-1f), s,
                        -3.33332819422e-1f);
    r = fma(p * s, z, z);
  } else {
    // exp(88) is finite, and tanh(44) rounds to 1.
    r = 1.0f - 2.0f / (exp(2.0f * min(z, 44.0f)) + 1.0f);
  }
  return isnan(x) ? x : copysign(r, x);
}

// Returns a threadgroup-uniform x unchanged, in a form that Metal's optimizer can't see
// through. A loop over a 64-bit range takes its trip count from it: closed forms that the
// optimizer derives from a known 64-bit trip count crash Metal's compiler service
// (XPC_ERROR_CONNECTION_INTERRUPTED).
static inline ulong tg_opaque(ulong x) {
  return as_type<ulong>(simd_broadcast_first(as_type<uint2>(x)));
}

// simd_shuffle_xor for every register type. bfloat, bool, and 64-bit integers aren't in
// the native type set, so they shuffle through bit-compatible types.
template <typename T> static inline T tg_shfl_xor(T x, ushort m) { return simd_shuffle_xor(x, m); }
static inline bool tg_shfl_xor(bool x, ushort m) { return simd_shuffle_xor(ushort(x), m) != 0; }
static inline bfloat tg_shfl_xor(bfloat x, ushort m) {
  return as_type<bfloat>(simd_shuffle_xor(as_type<ushort>(x), m));
}
static inline long tg_shfl_xor(long x, ushort m) {
  return as_type<long>(simd_shuffle_xor(as_type<uint2>(x), m));
}
static inline ulong tg_shfl_xor(ulong x, ushort m) {
  return as_type<ulong>(simd_shuffle_xor(as_type<uint2>(x), m));
}

// Loads an 8x8 fragment whose logical (row, col) lives at p[row * ld + col], or at
// p[col * ld + row] when transposed. The fragment starts at logical (row0, col0) of a
// rows x cols tensor, and elements outside it read as zero. Fragments that fit load with
// simdgroup_load; the test is uniform across the SIMD group. Straddling fragments fall back
// to per-lane masked loads of the two elements each lane owns, (fm, fn) and (fm, fn + 1).
template <typename T, typename P>
static inline void tg_load_frag(thread simdgroup_matrix<T, 8, 8>& f, P p, int ld, int row0,
                                int rows, int col0, int cols, bool transpose, int fm, int fn) {
  if (row0 >= 0 && col0 >= 0 && row0 + 8 <= rows && col0 + 8 <= cols) {
    simdgroup_load(f, p, ulong(ld), ulong2(0, 0), transpose);
    return;
  }
  thread auto& e = f.thread_elements();
  const bool rok = uint(row0 + fm) < uint(rows);
  for (int i = 0; i < 2; ++i) {
    const int c = fn + i;
    const bool ok = rok && uint(col0 + c) < uint(cols);
    e[i] = ok ? T(transpose ? p[c * ld + fm] : p[fm * ld + c]) : T(0);
  }
}
