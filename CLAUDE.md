# Tegula

Tegula is a Triton-like Python language for writing GPU kernels for Apple M-series GPUs.
It compiles Python tile programs to Metal Shading Language (MSL) and runs them through
Metal. The distribution and import names are both `tegula`, and the language namespace is
`import tegula.language as tl`.

## Start here

Before you write code, read
[Part 2: Implementation guide](PLAN.md#part-2-implementation-guide) in `PLAN.md`. It
defines the modules, interfaces, algorithms, milestones (M0-M9), and acceptance criteria.
Work through the milestones in order. Read [Part 1](PLAN.md#part-1-design-overview) when
you need the reasoning behind a decision.

Before each milestone, read the research report that the milestone names. The reports and
the verified reference code are in [`docs/research/`](docs/research/). They use the
working name Forge, and the prototype files keep `forge_` prefixes. Both refer to this
project.

## Rules

The full rules are in [Working rules](PLAN.md#working-rules). These are the ones that
matter most:

- Use `uv` for everything (`uv sync`, `uv run pytest`, `uv add`). Don't call `pip`.
- Don't download the offline Metal Toolchain. Compile MSL at run time with
  `newLibraryWithSource`.
- Write meaningful tests, not many verbose ones. Every test must be able to catch a real
  bug. Prefer one parametrized differential test (compiled, interpreter, and NumPy) over
  many near-identical tests. Don't snapshot generated MSL. Keep the default suite under
  60 seconds.
- Keep performance checks in `benchmarks/`, not in `pytest`. Time with GPU timestamps,
  warm up first, and compare against MLX.
- Keep generated MSL deterministic. Refuse unsupported constructs with a
  source-located `tegula.CompilationError`. Never miscompile silently.
- After each milestone, append results, benchmark numbers, known gaps, and any deviations
  from the plan to `docs/progress.md`.

## Machine

The development machine is a MacBook Pro with an M4 Pro (16-core GPU, Apple9 family,
24 GB), macOS 27, and Xcode 27. Its GPU isn't Apple10, so Metal 4 `matmul2d` runs on
regular shader cores here.
