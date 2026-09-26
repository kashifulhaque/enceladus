// C ABI over Metal for the Enceladus runtime.
//
// Object handles are retained Objective-C objects cast to void*. Release them with
// fr_release(). Streams are C++ objects; free them with fr_stream_free().
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FR_MAX_BINDINGS 31
#define FR_ERR_LEN 65536

// ---- Device ----------------------------------------------------------------

typedef struct {
    uint64_t max_threadgroup_memory;
    uint64_t max_threads_per_threadgroup;
    uint64_t max_buffer_length;
    uint64_t recommended_max_working_set;
    uint64_t max_concurrent_compilations;
    int supports_apple[11];  // index i -> MTLGPUFamilyApple(i), 0 unused
    int supports_metal3;
    int supports_metal4;
} fr_device_info;

void *fr_device_default(void);
void fr_device_name(void *dev, char *out, size_t n);
void fr_device_architecture(void *dev, char *out, size_t n);
void fr_device_query(void *dev, fr_device_info *out);
void *fr_queue_new(void *dev);
void fr_release(void *obj);
void fr_retain(void *obj);

// ---- Shader logging --------------------------------------------------------

// Creates a command queue with an MTLLogState that collects `os_log` messages from
// kernels compiled with logging enabled. *sink_out receives a log sink that owns the
// collected messages; free it with fr_log_sink_free() after releasing the queue. A
// message equal to `sentinel` isn't collected; it only counts as a sentinel (see
// fr_stream_set_log_sentinel). Returns NULL with a message in err on failure.
void *fr_queue_new_logging(void *dev, uint64_t buffer_size, const char *sentinel,
                           void **sink_out, char *err, size_t errlen);
// Calls fn(ctx, message, length) for each collected message, oldest first, and removes
// them. Returns the number of messages delivered. Doesn't block.
int fr_log_sink_drain(void *sink, void (*fn)(void *ctx, const char *msg, size_t len), void *ctx);
// Returns the number of sentinel messages the sink has received.
uint64_t fr_log_sink_sentinels(void *sink);
// Waits until the sink has received at least `count` sentinels or `timeout_s` passes.
// Returns 1 if the count was reached.
int fr_log_sink_wait_sentinels(void *sink, uint64_t count, double timeout_s);
void fr_log_sink_free(void *sink);

// ---- GPU capture -----------------------------------------------------------

// Starts capturing all GPU work on `dev` into a GPU trace document at `path`. Metal
// refuses once the process has created a command queue with an MTLLogState.
// Returns 0 on success, or nonzero with a message in err.
int fr_capture_start(void *dev, const char *path, char *err, size_t errlen);
// Stops the capture that fr_capture_start started.
void fr_capture_stop(void);

// ---- Compilation -----------------------------------------------------------

enum { FR_MATH_SAFE = 0, FR_MATH_RELAXED = 1, FR_MATH_FAST = 2 };
enum { FR_FP32_FAST = 0, FR_FP32_PRECISE = 1 };

typedef struct {
    uint32_t language_version;  // raw MTLLanguageVersion; 0 means 3.2
    int math_mode;              // FR_MATH_*
    int math_fp32_functions;    // FR_FP32_*
    int preserve_invariance;
    int enable_logging;
} fr_compile_options;

// Returns NULL on failure with the full diagnostics in err. On success, err holds
// any warnings (possibly empty).
void *fr_library_new(void *dev, const char *src, const fr_compile_options *opts, char *err,
                     size_t errlen);
void *fr_function_new(void *lib, const char *name);
void *fr_pipeline_new(void *dev, void *fn, char *err, size_t errlen);

typedef struct {
    uint32_t thread_execution_width;
    uint32_t max_total_threads_per_threadgroup;
    uint32_t static_threadgroup_memory;
} fr_pipeline_info;
void fr_pipeline_query(void *pso, fr_pipeline_info *out);

// Buffer bindings from pipeline reflection. kind: 0 buffer, 1 other.
// data_type is the raw MTLDataType, data_size is sizeof(T) for `T *arg`,
// access: 0 read-only, 1 read-write, 2 write-only.
typedef struct {
    uint32_t index;
    uint32_t kind;
    uint32_t data_type;
    uint32_t data_size;
    uint32_t access;
    char name[64];
} fr_binding;
int fr_pipeline_bindings(void *pso, fr_binding *out, int cap);

// ---- Buffers ---------------------------------------------------------------

void *fr_buffer_new(void *dev, size_t nbytes);
void *fr_buffer_nocopy(void *dev, void *ptr, size_t nbytes);
// Retains an existing id<MTLBuffer> (for example from DLPack). Returns ptr.
void *fr_buffer_from_mtl(void *mtl_buffer);
void *fr_buffer_contents(void *buf);
size_t fr_buffer_length(void *buf);

// ---- Streams ---------------------------------------------------------------

typedef struct {
    uint32_t index;   // [[buffer(index)]]
    uint32_t offset;  // byte offset into scalar_bytes
    uint32_t size;    // byte size
} fr_scalar_slot;

// Precomputed per-specialization binding plan.
typedef struct {
    int nbufs;
    uint32_t buf_index[FR_MAX_BINDINGS];
    int nscalars;
    fr_scalar_slot scalars[FR_MAX_BINDINGS];
} fr_launch_plan;

void *fr_stream_new(void *queue);
void fr_stream_free(void *stream);
// Encodes one dispatch into the stream's open command buffer. grid is in
// threadgroups, tg in threads. name identifies the kernel in error messages; the
// stream copies it.
void fr_stream_dispatch(void *stream, void *pso, const char *name, const fr_launch_plan *plan,
                        void *const *bufs, const uint64_t *offsets, const void *scalar_bytes,
                        const uint32_t grid[3], const uint32_t tg[3]);
// Ends the encoder and commits the command buffer, then signals the shared event.
void fr_stream_flush(void *stream);
// Flushes and waits. Returns 0 on success; nonzero with a message in err if any
// command buffer since the previous sync failed.
int fr_stream_sync(void *stream, char *err, size_t errlen);
int fr_stream_pending(void *stream);
// Shader log messages arrive after their command buffer completes, in order within a
// command buffer but not across command buffers. With a sentinel pipeline set, the
// stream ends each command buffer that holds a marked dispatch with one dispatch of the
// sentinel kernel, which logs the queue's sentinel message. When the sink has counted
// fr_stream_log_sentinels() sentinels, every earlier message has arrived.
void fr_stream_set_log_sentinel(void *stream, void *pso);
// Marks the open (or next) command buffer as holding a dispatch that logs.
void fr_stream_mark_logging(void *stream);
// Returns the number of sentinel dispatches the stream has committed.
uint64_t fr_stream_log_sentinels(void *stream);
// Flushes, then runs one dispatch in its own command buffer and waits. Writes
// GPUStartTime and GPUEndTime in seconds. Returns 0 on success.
int fr_stream_timed_run(void *stream, void *pso, const char *name, const fr_launch_plan *plan,
                        void *const *bufs, const uint64_t *offsets, const void *scalar_bytes,
                        const uint32_t grid[3], const uint32_t tg[3], double *gpu_start,
                        double *gpu_end, char *err, size_t errlen);
// Commits whatever the open command buffer holds, waits for it, and writes its GPU
// start and end times. Returns 0 on success. Used to time a group of launches.
int fr_stream_flush_timed(void *stream, double *gpu_start, double *gpu_end, char *err,
                          size_t errlen);

#ifdef __cplusplus
}
#endif
