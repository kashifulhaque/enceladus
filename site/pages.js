// The site's pages, in reading order. The layout plugin builds the sidebar, the
// previous and next links, and the overview's page cards from this list, and
// vite.config.js builds one HTML entry per page.
export const PAGES = [
  {
    file: "index.html",
    url: "/",
    group: "Get started",
    title: "Overview",
    description: "What Enceladus is, what it does, and the state of the project.",
  },
  {
    file: "quickstart/index.html",
    url: "/quickstart/",
    group: "Get started",
    title: "Quickstart",
    description: "Install Enceladus and run a vector add, a softmax, a matmul, and flash attention.",
  },
  {
    file: "programming-model/index.html",
    url: "/programming-model/",
    group: "Concepts",
    title: "Programming model",
    description: "Programs, the grid, tiles and their layout across SIMD groups, and specialization.",
  },
  {
    file: "memory/index.html",
    url: "/memory/",
    group: "Concepts",
    title: "Memory and synchronization",
    description: "Masks, pointer tiles, tensor descriptors, streams, and when results are visible.",
  },
  {
    file: "reference/index.html",
    url: "/reference/",
    group: "Reference",
    title: "Language reference",
    description: "Every tl builtin and host function, the data types, and the environment variables.",
  },
  {
    file: "debugging/index.html",
    url: "/debugging/",
    group: "Guides",
    title: "Debugging",
    description: "Compilation errors, the CPU interpreter, MSL dumps, kernel.explain, and GPU capture.",
  },
  {
    file: "interop/index.html",
    url: "/interop/",
    group: "Guides",
    title: "Framework interop",
    description: "Pass PyTorch, MLX, and NumPy arrays to kernels, and share memory through DLPack.",
  },
  {
    file: "performance/index.html",
    url: "/performance/",
    group: "Guides",
    title: "Performance",
    description: "Block sizes, tl.dot backends, autotuning, and asynchronous launches.",
  },
  {
    file: "benchmarks/index.html",
    url: "/benchmarks/",
    group: "Guides",
    title: "Benchmarks",
    description: "Throughput against MLX on an M4 Pro, and launch overhead.",
  },
  {
    file: "porting/index.html",
    url: "/porting/",
    group: "Guides",
    title: "Porting from Triton",
    description: "The steps to port a Triton kernel, what differs, and what isn't supported.",
  },
];

// Built, but left out of the navigation. Cloudflare Pages serves it for unknown URLs.
export const NOT_FOUND = {
  file: "404.html",
  url: "/404.html",
  title: "Page not found",
  description: "This page doesn't exist.",
};

// Pages linked from the header, by URL.
export const HEADER_LINKS = ["/quickstart/", "/reference/", "/benchmarks/"];
