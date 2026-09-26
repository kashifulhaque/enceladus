// Tegula prelude: helpers included in every generated kernel.

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

// Python-style floor division and modulo for floats (Triton semantics).
static inline float tg_floordiv(float a, float b) { return floor(a / b); }
static inline float tg_fmod(float a, float b) { return a - b * floor(a / b); }
