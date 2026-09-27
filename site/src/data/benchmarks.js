// Apple M4 Pro results from benchmarks/results/2026-09-27-applegpu_g16s-2.md.
// Each row is [workload, config, Enceladus throughput, MLX throughput], using minimum times.
export const BENCHMARKS = [
  ["Memory-bound, GB/s", [
    ["Vector add", "float32 · BLOCK=1024", 236.6, 227.4],
    ["Vector add", "float16 · BLOCK=1024", 238.1, 213.0],
    ["Row softmax 4096×4096", "float32 · 4 warps", 241.3, 193.7],
    ["Row softmax 4096×4096", "float16 · 8 warps", 256.3, 171.3]]],
  ["Matmul, TFLOPS", [
    ["4096³", "float32 · simdgroup 64×64×32", 5.27, 5.21],
    ["4096³", "float16 · mpp 64×64", 6.04, 5.88],
    ["4096³", "bfloat16 · mpp 64×64", 6.06, 5.86],
    ["2000³", "float32 · mpp 64×64", 5.14, 4.75],
    ["2000³", "float16 · tuned mpp 64×32", 5.86, 5.40],
    ["1024×4096×1024", "float16 · mpp 64×64", 6.05, 5.47],
    ["513³", "float16 · tuned mpp 32×32", 4.31, 1.53]]],
  ["Flash attention float16 (B=1, H=16), TFLOPS", [
    ["S=2048, D=64", "non-causal · 64×16 w4", 5.23, 5.23],
    ["S=2048, D=64", "causal · 64×32 w8", 4.96, 4.75],
    ["S=2048, D=128", "non-causal · 64×32 w8", 5.07, 5.20],
    ["S=2048, D=128", "causal · 64×32 w8", 4.83, 4.87],
    ["S=4096, D=64", "non-causal · 64×32 w4", 5.26, 5.37],
    ["S=4096, D=64", "causal · 64×32 w8", 5.06, 5.16],
    ["S=4096, D=128", "non-causal · 64×32 w8", 5.04, 5.29],
    ["S=4096, D=128", "causal · 64×32 w8", 4.93, 5.08]]]
];
