// Minimal C ABI over Metal for the Forge runtime benchmark.
// All object handles are retained Objective-C objects cast to void*;
// release them with fr_release().
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double gpu_start;  // seconds (CFTimeInterval, host time base)
    double gpu_end;
    int status;        // MTLCommandBufferStatus
} fr_cb_result;

void *fr_device_default(void);
void fr_device_name(void *dev, char *out, size_t n);
void *fr_queue_new(void *dev);
void fr_release(void *obj);
void fr_retain(void *obj);

// lang_version: MTLLanguageVersion raw value, 0 = compiler default.
void *fr_library_new(void *dev, const char *src, uint32_t lang_version,
                     int fast_math, char *err, size_t errlen);
void *fr_function_new(void *lib, const char *name);
void *fr_pipeline_new(void *dev, void *fn, char *err, size_t errlen);
uint32_t fr_pipeline_simd_width(void *pso);
uint32_t fr_pipeline_max_threads(void *pso);
uint32_t fr_pipeline_static_tg_mem(void *pso);

void *fr_buffer_new(void *dev, size_t nbytes);
void *fr_buffer_nocopy(void *dev, void *ptr, size_t nbytes);
void *fr_buffer_contents(void *buf);
size_t fr_buffer_length(void *buf);

// One dispatch in its own command buffer. grid/tg are in threadgroups/threads.
// If wait != 0 blocks until completion and fills *res (may be NULL).
void fr_dispatch(void *queue, void *pso, void *const *bufs, int nbufs,
                 const void *bytes, size_t nbytes, int bytes_index,
                 const uint32_t grid[3], const uint32_t tg[3], int wait,
                 fr_cb_result *res);

// Batched dispatches: one command buffer, one encoder.
void *fr_batch_begin(void *queue, int unretained);
void fr_batch_dispatch(void *batch, void *pso, void *const *bufs, int nbufs,
                       const void *bytes, size_t nbytes, int bytes_index,
                       const uint32_t grid[3], const uint32_t tg[3]);
void fr_batch_end(void *batch, int wait, fr_cb_result *res);

// Pure-native loop: iters command buffers, each containing per_cb dispatches,
// each committed and waited on. Writes per-command-buffer wall time (ns).
void fr_bench_loop(void *queue, void *pso, void *const *bufs, int nbufs,
                   const void *bytes, size_t nbytes, int bytes_index,
                   const uint32_t grid[3], const uint32_t tg[3], int iters,
                   int per_cb, double *out_wall_ns, double *out_gpu_ns);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
// Experiments on completion-wait strategies (sync round trip of a single dispatch).
// mode: 0 waitUntilCompleted, 1 spin on cb.status, 2 completion handler + semaphore,
//       3 encodeSignalEvent + spin on MTLSharedEvent.signaledValue,
//       4 unretained-references CB + waitUntilCompleted.
void fr_bench_wait_mode(void *queue, void *pso, void *const *bufs, int nbufs,
                        const void *bytes, size_t nbytes, int bytes_index,
                        const uint32_t grid[3], const uint32_t tg[3], int iters,
                        int mode, double *out_wall_ns);
// Async: iters command buffers (1 dispatch each) committed without waiting,
// then one final wait. Returns total ns.
double fr_bench_async(void *queue, void *pso, void *const *bufs, int nbufs,
                      const void *bytes, size_t nbytes, int bytes_index,
                      const uint32_t grid[3], const uint32_t tg[3], int iters);
// Encodes n dispatches in one encoder (concurrent != 0 -> MTLDispatchTypeConcurrent),
// returns encode+commit ns (before waiting); fills gpu ns after waiting.
double fr_bench_encode(void *queue, void *pso, void *const *bufs, int nbufs,
                       const void *bytes, size_t nbytes, int bytes_index,
                       const uint32_t grid[3], const uint32_t tg[3], int n,
                       int concurrent, double *out_gpu_ns);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
// Packed launch record, built in Python with one struct.pack call:
//   u64 pso; u32 grid[3]; u32 tg[3]; u32 nbufs; u32 nbytes; u32 bytes_index; u32 pad;
//   u64 bufs[nbufs]; u8 bytes[nbytes]
void fr_batch_dispatch_packed(void *batch, const void *rec);
void fr_dispatch_packed(void *queue, const void *rec, int wait, fr_cb_result *res);
#ifdef __cplusplus
}
#endif
