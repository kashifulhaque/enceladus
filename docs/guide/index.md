# Enceladus user guide

Enceladus is a Python language for writing GPU kernels for Apple silicon Macs. You
write a kernel as a Python function that operates on tiles, the way you write Triton
kernels, and Enceladus compiles it to Metal Shading Language (MSL) and runs it on the
GPU. Kernels accept NumPy arrays, PyTorch tensors on the `mps` device, and MLX arrays
without copying them.

The guide has the following pages:

- [Quickstart](quickstart.md): install Enceladus and run a vector add, a row softmax,
  and a matrix multiplication.
- [Programming model](programming-model.md): programs and the grid, tiles, masks,
  pointers and tensor descriptors, constexprs and specialization, and synchronization.
- [Language reference](language-reference.md): every function in
  `enceladus.language`, generated from the builtin docstrings.
- [Debugging](debugging.md): compilation errors, the CPU interpreter, IR and MSL dumps,
  `kernel.explain`, printing and asserts on the GPU, and GPU capture.
- [Framework interop](interop.md): PyTorch, MLX, NumPy, and DLPack.
- [Porting from Triton](porting-from-triton.md): what carries over, what differs, and a
  ported matmul.
- [Performance tips](performance.md): measuring, tensor descriptors, autotuning,
  asynchronous launches, and `tl.dot` backends.

The kernels in the repository's `examples/` directory show complete programs, from a
vector add to flash attention. Each one runs with `uv run python examples/FILE.py`.

## Check the guide's code

The Python code blocks in these pages run as part of the documentation checks. To run
them yourself from the repository root, run the following command:

```bash
uv run python docs/guide/check_snippets.py
```

The language reference is generated from the builtin docstrings. After you change a
builtin, regenerate it with the following command:

```bash
uv run python docs/guide/gen_reference.py
```
