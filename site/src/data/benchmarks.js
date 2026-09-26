// Apple M4 Pro results from benchmarks/results/2026-09-26-applegpu_g16s-m9.md.
// Each row is [workload, config, Enceladus throughput, MLX throughput], using minimum times.
export const BENCHMARKS = [
  ["Memory-bound, GB/s", [
    ["Vector add", "float32 · BLOCK=1024", 224.1, 206.2],
    ["Vector add", "float16 · BLOCK=1024", 221.4, 206.6],
    ["Row softmax 4096×4096", "float32 · 4 warps", 231.1, 193.3],
    ["Row softmax 4096×4096", "float16 · 16 warps", 237.9, 170.9]]],
  ["Matmul, TFLOPS", [
    ["4096³", "float32 · simdgroup 64×64×32", 5.09, 5.19],
    ["4096³", "float16 · mpp 64×64", 5.70, 5.77],
    ["4096³", "bfloat16 · mpp 64×64", 5.73, 5.59],
    ["2000³", "float32 · tuned mpp 64×64", 4.77, 4.81],
    ["2000³", "float16 · mpp 64×64", 5.67, 5.34],
    ["1024×4096×1024", "float16 · tuned mpp 64×32", 6.09, 5.53],
    ["513³", "float16 · tuned mpp 32×32", 4.32, 1.39]]],
  ["Flash attention float16 (B=1, H=16), TFLOPS", [
    ["S=2048, D=64", "non-causal · 64×32 w4", 5.18, 5.05],
    ["S=2048, D=64", "causal · 32×32 w4", 4.84, 4.70],
    ["S=2048, D=128", "non-causal · 64×32 w8", 4.81, 4.94],
    ["S=2048, D=128", "causal · 64×32 w8", 4.74, 4.79],
    ["S=4096, D=64", "non-causal · 128×32 w8", 4.91, 5.03],
    ["S=4096, D=64", "causal · 64×32 w8", 4.54, 4.77],
    ["S=4096, D=128", "non-causal · 64×16 w8", 4.32, 4.66],
    ["S=4096, D=128", "causal · 64×32 w8", 4.19, 4.39]]]
];
