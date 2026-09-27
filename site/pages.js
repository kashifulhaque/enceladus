// The site's sections and pages, in reading order. The layout plugin builds the
// header, each section's sidebar, the previous and next links, and the card grids
// from this list, and vite.config.js builds one HTML entry per page.
//
// Each section has its own sidebar. Within a section, `group` sets the sidebar
// heading a page sits under.

export const HOME = {
  file: "index.html",
  url: "/",
  title: "Enceladus",
  description: "Triton-style GPU kernels in Python, compiled to Metal for Apple silicon.",
};

export const SECTIONS = [
  {
    id: "docs",
    title: "Docs",
    description: "Install Enceladus, learn the programming model, and read the guides.",
    pages: [
      {
        file: "docs/index.html",
        url: "/docs/",
        group: "Get started",
        title: "Introduction",
        description: "What Enceladus is, what it does, and the state of the project.",
      },
      {
        file: "docs/quickstart/index.html",
        url: "/docs/quickstart/",
        group: "Get started",
        title: "Quickstart",
        description: "Install Enceladus and run a vector add, a softmax, a matmul, and flash attention.",
      },
      {
        file: "docs/programming-model/index.html",
        url: "/docs/programming-model/",
        group: "Concepts",
        title: "Programming model",
        description: "Programs, the grid, tiles and their layout across SIMD groups, and specialization.",
      },
      {
        file: "docs/memory/index.html",
        url: "/docs/memory/",
        group: "Concepts",
        title: "Memory and synchronization",
        description: "Masks, pointer tiles, tensor descriptors, streams, and when results are visible.",
      },
      {
        file: "docs/debugging/index.html",
        url: "/docs/debugging/",
        group: "Guides",
        title: "Debugging",
        description: "Compilation errors, the CPU interpreter, MSL dumps, kernel.explain, and GPU capture.",
      },
      {
        file: "docs/interop/index.html",
        url: "/docs/interop/",
        group: "Guides",
        title: "Framework interop",
        description: "Pass PyTorch, MLX, and NumPy arrays to kernels, and share memory through DLPack.",
      },
      {
        file: "docs/performance/index.html",
        url: "/docs/performance/",
        group: "Guides",
        title: "Performance",
        description: "Block sizes, tl.dot backends, autotuning, and asynchronous launches.",
      },
      {
        file: "docs/porting/index.html",
        url: "/docs/porting/",
        group: "Guides",
        title: "Porting from Triton",
        description: "The steps to port a Triton kernel, what differs, and what isn't supported.",
      },
    ],
  },
  {
    id: "reference",
    title: "Reference",
    description: "Every tl builtin, the host API, the data types, and the environment variables.",
    pages: [
      {
        file: "reference/index.html",
        url: "/reference/",
        group: "Reference",
        title: "Language builtins",
        description: "Every builtin in enceladus.language, imported as tl.",
      },
      {
        file: "reference/host-api/index.html",
        url: "/reference/host-api/",
        group: "Reference",
        title: "Host API",
        description: "Compile, launch, autotune, and allocate from Python with the enceladus module.",
      },
      {
        file: "reference/data-types/index.html",
        url: "/reference/data-types/",
        group: "Reference",
        title: "Data types and tiles",
        description: "The element types, and the attributes, operators, and methods of a tile.",
      },
      {
        file: "reference/environment/index.html",
        url: "/reference/environment/",
        group: "Reference",
        title: "Environment variables",
        description: "The variables that control the interpreter, the cache, dumps, and autotuning.",
      },
    ],
  },
  {
    id: "examples",
    title: "Examples",
    description: "Ten complete kernels, from a vector add to flash attention, each checked against NumPy.",
    pages: [
      {
        file: "examples/index.html",
        url: "/examples/",
        group: "Overview",
        title: "All examples",
        description: "Ten complete kernels, from a vector add to flash attention.",
      },
      {
        file: "examples/vector-add/index.html",
        url: "/examples/vector-add/",
        group: "Elementwise",
        title: "Vector add",
        description: "Add two arrays elementwise with a 1D grid, block offsets, and a mask.",
      },
      {
        file: "examples/softmax/index.html",
        url: "/examples/softmax/",
        group: "Reductions and normalization",
        title: "Row softmax",
        description: "Compute a softmax over each row, with one program per row and the whole row in one tile.",
      },
      {
        file: "examples/layernorm/index.html",
        url: "/examples/layernorm/",
        group: "Reductions and normalization",
        title: "LayerNorm",
        description: "Normalize each row with a learned scale and shift, looping over the row in blocks.",
      },
      {
        file: "examples/matmul/index.html",
        url: "/examples/matmul/",
        group: "Matrix multiplication",
        title: "Matrix multiplication",
        description: "Multiply matrices with pointer tiles or tensor descriptors, and autotune the block sizes.",
      },
      {
        file: "examples/fused-gelu/index.html",
        url: "/examples/fused-gelu/",
        group: "Elementwise",
        title: "Fused GELU",
        description: "Fuse a scale, a per-column bias add, and a GELU into one elementwise kernel.",
      },
      {
        file: "examples/rmsnorm/index.html",
        url: "/examples/rmsnorm/",
        group: "Reductions and normalization",
        title: "RMSNorm",
        description: "Divide each row by its root mean square and scale it, with tl.rsqrt.",
      },
      {
        file: "examples/matmul-bias-gelu/index.html",
        url: "/examples/matmul-bias-gelu/",
        group: "Matrix multiplication",
        title: "Matmul with bias and GELU",
        description: "Fuse a bias add and a GELU into a descriptor matmul's epilogue, on the accumulator registers.",
      },
      {
        file: "examples/flash-attention/index.html",
        url: "/examples/flash-attention/",
        group: "Attention",
        title: "Flash attention",
        description: "Compute attention with an online softmax and an optional causal mask, and autotune it.",
      },
      {
        file: "examples/histogram/index.html",
        url: "/examples/histogram/",
        group: "Atomics and scans",
        title: "Histogram",
        description: "Count float values into equal bins with tl.atomic_add on a shared histogram.",
      },
      {
        file: "examples/cumsum/index.html",
        url: "/examples/cumsum/",
        group: "Atomics and scans",
        title: "Cumulative sum",
        description: "Compute prefix or suffix sums along each row with tl.cumsum.",
      },
    ],
  },
  {
    id: "benchmarks",
    title: "Benchmarks",
    description: "Throughput against MLX and PyTorch on an M4 Pro, launch overhead, and how to reproduce it.",
    pages: [
      {
        file: "benchmarks/index.html",
        url: "/benchmarks/",
        group: "Overview",
        title: "Overview",
        description: "Enceladus throughput relative to MLX on one M4 Pro.",
      },
      {
        file: "benchmarks/methodology/index.html",
        url: "/benchmarks/methodology/",
        group: "Overview",
        title: "Methodology",
        description: "How the scripts time each framework, the machine, and how to run the benchmarks yourself.",
      },
      {
        file: "benchmarks/elementwise/index.html",
        url: "/benchmarks/elementwise/",
        group: "Memory-bound",
        title: "Elementwise",
        description: "Vector add bandwidth on 256 MB arrays against MLX, and compile time to MSL.",
      },
      {
        file: "benchmarks/softmax/index.html",
        url: "/benchmarks/softmax/",
        group: "Memory-bound",
        title: "Softmax and reductions",
        description: "Row softmax against MLX and PyTorch, and a row sum across threadgroup sizes.",
      },
      {
        file: "benchmarks/normalization/index.html",
        url: "/benchmarks/normalization/",
        group: "Memory-bound",
        title: "Normalization",
        description: "LayerNorm and RMSNorm forward bandwidth on a 4096 × 4096 matrix.",
      },
      {
        file: "benchmarks/scan-atomics/index.html",
        url: "/benchmarks/scan-atomics/",
        group: "Memory-bound",
        title: "Scans and atomics",
        description: "Row cumsum against MLX, and a histogram built with global atomics.",
      },
      {
        file: "benchmarks/matmul/index.html",
        url: "/benchmarks/matmul/",
        group: "Compute-bound",
        title: "Matmul",
        description: "Both tl.dot backends and the autotuned kernel across four shapes and three dtypes.",
      },
      {
        file: "benchmarks/attention/index.html",
        url: "/benchmarks/attention/",
        group: "Compute-bound",
        title: "Flash attention",
        description: "Causal and non-causal float16 attention against MLX's fused kernel.",
      },
      {
        file: "benchmarks/dispatch/index.html",
        url: "/benchmarks/dispatch/",
        group: "Overhead",
        title: "Dispatch overhead",
        description: "The host cost of a launch on each Enceladus path, next to PyTorch and MLX.",
      },
    ],
  },
];

// Built, but left out of the navigation. Cloudflare Pages serves it for unknown URLs.
export const NOT_FOUND = {
  file: "404.html",
  url: "/404.html",
  title: "Page not found",
  description: "This page doesn't exist.",
};

export const PAGES = SECTIONS.flatMap((s) => s.pages.map((p) => ({ ...p, section: s })));
