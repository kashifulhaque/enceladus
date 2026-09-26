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
// p[col * ld + row] when transposed. Fragments that fit load with simdgroup_load; the
// test is uniform across the SIMD group. Straddling fragments fall back to per-lane
// masked loads of the two elements each lane owns, (fm, fn) and (fm, fn + 1).
template <typename T, typename P>
static inline void tg_load_frag(thread simdgroup_matrix<T, 8, 8>& f, P p, int ld,
                                int rows_left, int cols_left, bool transpose, int fm, int fn) {
  if (rows_left >= 8 && cols_left >= 8) {
    simdgroup_load(f, p, ulong(ld), ulong2(0, 0), transpose);
    return;
  }
  thread auto& e = f.thread_elements();
  const bool rok = fm < rows_left;
  for (int i = 0; i < 2; ++i) {
    const int c = fn + i;
    const bool ok = rok && c < cols_left;
    e[i] = ok ? T(transpose ? p[c * ld + fm] : p[fm * ld + c]) : T(0);
  }
}
