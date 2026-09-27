// Builtins and host API shown on the reference page. Each group is
// [namespace, title, note, [[signature, description], ...]], and namespace is "tl" or "host".
// Keep it in sync with docs/guide/language-reference.md and src/enceladus/__init__.py.
export const REFERENCE = [
  ["tl", "Programs", "", [
    ["program_id(axis)", "Returns the index of the current program along axis 0, 1, or 2."],
    ["num_programs(axis)", "Returns the number of programs in the grid along axis 0, 1, or 2."]]],
  ["tl", "Tile creation", "Shapes must be compile-time powers of two.", [
    ["arange(start, end)", "Returns an int32 tile of [start, end). The bounds are compile-time, and end - start is a power of two."],
    ["full(shape, value, dtype)", "Returns a tile filled with a scalar, which can be a literal or a runtime value."],
    ["zeros(shape, dtype)", "Returns a tile filled with zeros."],
    ["full_like(input, value, dtype=None)", "Returns a tile with the shape of input, filled with value."],
    ["zeros_like(input)", "Returns zeros with the shape and dtype of input."]]],
  ["tl", "Memory", "", [
    ["load(pointer, mask=None, other=None, boundary_check=(), padding_option='', cache_modifier='', eviction_policy='', volatile=False)", "Loads through a pointer or pointer tile. Masked-off lanes return other, or 0. Cache and eviction hints are accepted and ignored."],
    ["store(pointer, value, mask=None, boundary_check=(), cache_modifier='', eviction_policy='')", "Stores where mask is true. The value is converted to the pointee dtype and broadcast."],
    ["debug_barrier()", "Waits for every thread of the program and makes their earlier device memory stores visible to all of them."]]],
  ["tl", "Tensor descriptors", "", [
    ["make_tensor_descriptor(base, shape, strides, block_shape, padding_option='zero')", "Creates a bounds-checked block descriptor over a pointer. The last stride must be 1."],
    ["desc.load(offsets)", "Loads the block at element offsets. Elements outside the array are zero."],
    ["desc.store(offsets, value)", "Stores the block at element offsets and skips elements outside the array."],
    ["desc.dtype, desc.block_shape", "The element type and the block shape of the descriptor."]]],
  ["tl", "Elementwise", "Arguments broadcast to one shape.", [
    ["where(condition, x, y)", "Picks x where condition is true and y elsewhere."],
    ["maximum(x, y)", "Returns the elementwise maximum. A NaN operand yields the other operand."],
    ["minimum(x, y)", "Returns the elementwise minimum. A NaN operand yields the other operand."],
    ["clamp(x, min, max)", "Returns minimum(maximum(x, min), max)."],
    ["fma(x, y, z)", "Returns x * y + z with a single rounding."],
    ["abs(x)", "Returns the elementwise absolute value."],
    ["cdiv(x, div)", "Returns (x + div - 1) // div, the number of blocks that cover x."]]],
  ["tl", "Math", "Float inputs only. 16-bit floats compute in float32 and round back. Each is also a tile method.", [
    ["exp(x), exp2(x)", "Returns e or 2 raised to the power x."],
    ["log(x), log2(x)", "Returns the natural or base-2 logarithm."],
    ["sqrt(x), rsqrt(x)", "Returns the square root or its reciprocal."],
    ["sin(x), cos(x)", "Returns the sine or cosine of x in radians."],
    ["tanh(x)", "Returns the hyperbolic tangent."],
    ["sigmoid(x)", "Returns 1 / (1 + exp(-x))."],
    ["erf(x)", "Returns the Gauss error function."],
    ["floor(x), ceil(x)", "Rounds down or up to an integer value."]]],
  ["tl", "Type conversion", "", [
    ["cast(x, dtype, bitcast=False, fp_downcast_rounding=None)", "Converts x to dtype, or reinterprets its bits when bitcast is true. Float to integer truncates toward zero."],
    ["x.to(dtype)", "Same as cast. Converting to int1 gives x != 0."]]],
  ["tl", "Shape", "", [
    ["broadcast_to(input, *shape)", "Broadcasts input to shape."],
    ["expand_dims(input, axis)", "Inserts size-1 dimensions at axis, an int or a tuple of ints."],
    ["reshape(input, *shape, can_reorder=False)", "Reshapes in row-major order. Every dimension must be a power of two."],
    ["trans(input, *dims)", "Permutes dimensions, or reverses them when no dims are given. Same as x.T."],
    ["permute(input, *dims)", "Permutes dimensions into the order dims."]]],
  ["tl", "Reductions", "", [
    ["sum(input, axis=None, keep_dims=False, dtype=None)", "Sums along axis, or over all elements. 16-bit floats accumulate in float32."],
    ["max(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False)", "Returns the maximum, and optionally the int32 index of the first maximum. NaNs are ignored."],
    ["min(input, axis=None, return_indices=False, return_indices_tie_break_left=True, keep_dims=False)", "Returns the minimum, and optionally the int32 index of the first minimum."],
    ["argmax(input, axis, tie_break_left=True, keep_dims=False)", "Returns the int32 index of the first maximum."],
    ["argmin(input, axis, tie_break_left=True, keep_dims=False)", "Returns the int32 index of the first minimum."],
    ["reduce(input, axis, combine_fn, keep_dims=False)", "Reduces a tile or tuple of tiles with an associative, commutative @jit combine_fn."]]],
  ["tl", "Scans", "Inclusive prefix scans.", [
    ["cumsum(input, axis=0, reverse=False, dtype=None)", "Returns the inclusive cumulative sum, or the suffix sum when reverse is true."],
    ["associative_scan(input, axis, combine_fn, reverse=False)", "Scans a tile or tuple of tiles with an associative @jit combine_fn."]]],
  ["tl", "Matrix multiplication", "", [
    ["dot(input, other, acc=None, input_precision=None, allow_tf32=None, max_num_imprecise_acc=None, out_dtype=tl.float32)", "Returns input @ other + acc for 2D tiles of the same float dtype. The accumulator is float32 or float16, and K must be a multiple of 8. Launch options dot_warps and dot_backend control the lowering."]]],
  ["tl", "Atomics", "Each returns the previous values, or 0 where mask is false. Ordering is relaxed.", [
    ["atomic_add(pointer, val, mask=None, sem=None, scope=None)", "Adds val atomically. 16-bit floats aren't supported."],
    ["atomic_max(pointer, val, mask=None, sem=None, scope=None)", "Stores the maximum atomically. A NaN operand is ignored."],
    ["atomic_min(pointer, val, mask=None, sem=None, scope=None)", "Stores the minimum atomically. A NaN operand is ignored."],
    ["atomic_xchg(pointer, val, mask=None, sem=None, scope=None)", "Stores val atomically and returns the old values."],
    ["atomic_and(pointer, val, mask=None, sem=None, scope=None)", "Applies a bitwise AND atomically. Integers only."],
    ["atomic_or(pointer, val, mask=None, sem=None, scope=None)", "Applies a bitwise OR atomically. Integers only."],
    ["atomic_xor(pointer, val, mask=None, sem=None, scope=None)", "Applies a bitwise XOR atomically. Integers only."],
    ["atomic_cas(pointer, cmp, val, sem=None, scope=None)", "Stores val where memory equals cmp bitwise, and returns the old values. Takes no mask."]]],
  ["tl", "Loops and compile time", "", [
    ["range(start, end=None, step=None, num_stages=None, ...)", "A runtime for-loop range. Triton pipelining hints are accepted and ignored."],
    ["static_range(start, end=None, step=None)", "A loop range that the compiler unrolls. The bounds must be compile-time values."],
    ["static_assert(cond, msg='')", "Raises CompilationError if the compile-time cond is false."],
    ["static_print(*values, sep=' ')", "Prints compile-time values while the kernel compiles."],
    ["constexpr", "Annotates a parameter as a compile-time constant, as in BLOCK: tl.constexpr."]]],
  ["tl", "Debugging", "", [
    ["device_print(prefix, *args, hex=False)", "Prints runtime values, one line per element, to stderr when the stream synchronizes."],
    ["device_assert(cond, msg='', mask=None)", "Checks cond on the GPU when ENCELADUS_DEBUG=1 and raises DeviceAssertionError at the next sync."]]],
  ["host", "Compile and launch", "", [
    ["jit(fn=None, *, interpret=None, do_not_specialize=(), math_mode='relaxed')", "Decorates a kernel or helper. Launch a kernel with kernel[grid](*args, **constexprs, num_warps=4)."],
    ["kernel.warmup(*args, grid=None, num_warps=4, dot_warps=None, dot_backend='auto', **constexprs)", "Compiles one specialization without launching it. The result has msl, ir, num_warps, and threadgroup_memory_bytes."],
    ["kernel.explain(*args, grid=None, num_warps=4, dot_warps=None, dot_backend='auto', **constexprs)", "Prints and returns the compiler's layout, register, and backend decisions."],
    ["metal_kernel(source, name, language_version=None, math_mode='relaxed')", "Compiles hand-written MSL. Launch with kernel[grid, threads_per_group](*args)."],
    ["cdiv(x, div)", "Returns ceil(x / div), the number of blocks that cover x."],
    ["next_power_of_2(n)", "Returns the smallest power of two that is at least n."]]],
  ["host", "Autotuning", "", [
    ["autotune(configs, key, prune_configs_by=None, reset_to_zero=None, restore_value=None, warmup_ms=50, rep=20)", "Picks the fastest config for each distinct set of key values and argument dtypes."],
    ["Config(kwargs, num_warps=4, dot_warps=None, dot_backend='auto', pre_hook=None)", "One autotuning candidate: constexpr values plus launch options."],
    ["heuristics(values)", "Computes constexpr values from the kernel's arguments."],
    ["configs.matmul_configs(dtype='float16')", "Returns the shipped matmul candidates for float32, float16, or bfloat16."],
    ["configs.attention_configs(dtype='float16', head_dim=64)", "Returns the shipped flash attention candidates."],
    ["testing.do_bench(fn, warmup_ms=50.0, rep=20, return_mode='median')", "Times fn with GPU timestamps and returns milliseconds."]]],
  ["host", "Streams and interop", "", [
    ["synchronize()", "Waits for all work on the default stream."],
    ["async_numpy(enabled=True)", "Lets launches with NumPy arrays return before the GPU finishes."],
    ["element_strides(obj)", "Returns strides in elements for Enceladus, NumPy, PyTorch, or MLX arrays."],
    ["element_dtype(obj)", "Returns the element type of an Enceladus, NumPy, PyTorch, or MLX array as a NumPy dtype."],
    ["new_empty(like, shape=None, dtype=None)", "Allocates an array of the same framework and device as like. dtype can be a NumPy dtype."],
    ["new_zeros(like, shape=None, dtype=None)", "Like new_empty, and fills the array with zeros."],
    ["get_device()", "Returns the process-wide Metal device and its capabilities."],
    ["capture(path)", "A context manager that records GPU work to a .gputrace bundle."]]],
  ["host", "Tensors", "An enceladus.Tensor lives in memory that the CPU and GPU share. Slicing returns a view.", [
    ["empty(shape, dtype='float32')", "Returns an uninitialized tensor."],
    ["zeros(shape, dtype='float32'), ones(shape, dtype='float32')", "Returns a tensor filled with zeros or ones."],
    ["full(shape, value, dtype='float32')", "Returns a tensor filled with value."],
    ["randn(*shape, dtype='float32', seed=None), rand(*shape, dtype='float32', seed=None)", "Returns standard normal or uniform [0, 1) samples."],
    ["arange(start, end=None, step=1, dtype='int32')", "Returns a range tensor, like np.arange."],
    ["empty_like(t, dtype=None), zeros_like(t, dtype=None)", "Returns a tensor with the shape of t."],
    ["from_numpy(a)", "Shares a's memory when it's C-contiguous and 16 KB page-aligned, and copies otherwise."],
    ["Tensor.numpy(), Tensor.tolist()", "Waits for pending GPU work and returns the data on the host."]]],
  ["host", "Errors", "", [
    ["CompilationError", "Raised for a construct that Enceladus can't compile, with the source location and a fix."],
    ["DeviceAssertionError", "An AssertionError subclass raised when tl.device_assert fails."],
    ["MetalError", "Raised for a Metal runtime error, such as a failed command buffer."]]]
];
