// Metal implementation of the Enceladus runtime C ABI. Compile with -fobjc-arc.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <string.h>
#include <unistd.h>

#include <string>
#include <utility>
#include <vector>

#include "enceladus_rt.h"

#define RETAIN(x) ((__bridge_retained void *)(x))
#define BORROW(T, p) ((__bridge T)(p))

static void copy_str(NSString *s, char *out, size_t n) {
    if (!out || n == 0) return;
    out[0] = 0;
    if (s) strlcpy(out, s.UTF8String, n);
}

static void copy_std(const std::string &s, char *out, size_t n) {
    if (!out || n == 0) return;
    strlcpy(out, s.c_str(), n);
}

// ---- Device ----------------------------------------------------------------

void *fr_device_default(void) {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    return d ? RETAIN(d) : NULL;
}

void fr_device_name(void *dev, char *out, size_t n) {
    copy_str(BORROW(id<MTLDevice>, dev).name, out, n);
}

void fr_device_architecture(void *dev, char *out, size_t n) {
    @autoreleasepool {
        copy_str(BORROW(id<MTLDevice>, dev).architecture.name, out, n);
    }
}

void fr_device_query(void *dev, fr_device_info *out) {
    id<MTLDevice> d = BORROW(id<MTLDevice>, dev);
    memset(out, 0, sizeof *out);
    out->max_threadgroup_memory = d.maxThreadgroupMemoryLength;
    MTLSize t = d.maxThreadsPerThreadgroup;
    out->max_threads_per_threadgroup = t.width;
    out->max_buffer_length = d.maxBufferLength;
    out->recommended_max_working_set = d.recommendedMaxWorkingSetSize;
    out->max_concurrent_compilations = d.maximumConcurrentCompilationTaskCount;
    for (int i = 1; i <= 10; ++i)
        out->supports_apple[i] = [d supportsFamily:(MTLGPUFamily)(1000 + i)] ? 1 : 0;
    out->supports_metal3 = [d supportsFamily:(MTLGPUFamily)5001] ? 1 : 0;
    out->supports_metal4 = [d supportsFamily:(MTLGPUFamily)5002] ? 1 : 0;
}

void *fr_queue_new(void *dev) {
    return RETAIN([BORROW(id<MTLDevice>, dev) newCommandQueue]);
}

void fr_retain(void *obj) {
    if (obj) CFRetain(obj);
}

void fr_release(void *obj) {
    if (obj) CFRelease(obj);
}

// ---- Compilation -----------------------------------------------------------

void *fr_library_new(void *dev, const char *src, const fr_compile_options *o, char *err,
                     size_t errlen) {
    @autoreleasepool {
        MTLCompileOptions *opts = [MTLCompileOptions new];
        uint32_t lv = o && o->language_version ? o->language_version : ((3u << 16) + 2);
        opts.languageVersion = (MTLLanguageVersion)lv;
        if (o) {
            opts.mathMode = (MTLMathMode)o->math_mode;
            opts.mathFloatingPointFunctions = (MTLMathFloatingPointFunctions)o->math_fp32_functions;
            opts.preserveInvariance = o->preserve_invariance ? YES : NO;
            opts.enableLogging = o->enable_logging ? YES : NO;
        } else {
            opts.mathMode = MTLMathModeRelaxed;
        }
        NSError *e = nil;
        id<MTLLibrary> lib = [BORROW(id<MTLDevice>, dev) newLibraryWithSource:@(src)
                                                                      options:opts
                                                                        error:&e];
        // Warnings also come back through `e` with a non-nil library.
        copy_str(e ? e.localizedDescription : nil, err, errlen);
        return lib ? RETAIN(lib) : NULL;
    }
}

void *fr_function_new(void *lib, const char *name) {
    @autoreleasepool {
        id<MTLFunction> fn = [BORROW(id<MTLLibrary>, lib) newFunctionWithName:@(name)];
        return fn ? RETAIN(fn) : NULL;
    }
}

// Reflection is kept alongside the pipeline so that bindings can be queried later.
static NSMapTable *reflection_table(void) {
    static NSMapTable *t;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
      t = [NSMapTable mapTableWithKeyOptions:NSPointerFunctionsWeakMemory |
                                             NSPointerFunctionsObjectPointerPersonality
                                valueOptions:NSPointerFunctionsStrongMemory];
    });
    return t;
}

void *fr_pipeline_new(void *dev, void *fn, char *err, size_t errlen) {
    @autoreleasepool {
        NSError *e = nil;
        MTLComputePipelineReflection *refl = nil;
        id<MTLComputePipelineState> pso = [BORROW(id<MTLDevice>, dev)
            newComputePipelineStateWithFunction:BORROW(id<MTLFunction>, fn)
                                        options:MTLPipelineOptionBindingInfo
                                     reflection:&refl
                                          error:&e];
        copy_str(pso ? nil : e.localizedDescription, err, errlen);
        if (!pso) return NULL;
        if (refl) {
            NSMapTable *t = reflection_table();
            @synchronized(t) {
                [t setObject:refl forKey:pso];
            }
        }
        return RETAIN(pso);
    }
}

void fr_pipeline_query(void *p, fr_pipeline_info *out) {
    id<MTLComputePipelineState> pso = BORROW(id<MTLComputePipelineState>, p);
    out->thread_execution_width = (uint32_t)pso.threadExecutionWidth;
    out->max_total_threads_per_threadgroup = (uint32_t)pso.maxTotalThreadsPerThreadgroup;
    out->static_threadgroup_memory = (uint32_t)pso.staticThreadgroupMemoryLength;
}

int fr_pipeline_bindings(void *p, fr_binding *out, int cap) {
    @autoreleasepool {
        id<MTLComputePipelineState> pso = BORROW(id<MTLComputePipelineState>, p);
        NSMapTable *t = reflection_table();
        MTLComputePipelineReflection *refl;
        @synchronized(t) {
            refl = [t objectForKey:pso];
        }
        if (!refl) return 0;
        int n = 0;
        for (id<MTLBinding> b in refl.bindings) {
            if (n >= cap) break;
            if (!b.used && b.type != MTLBindingTypeBuffer) continue;
            fr_binding *o = &out[n];
            memset(o, 0, sizeof *o);
            o->index = (uint32_t)b.index;
            if (b.type == MTLBindingTypeBuffer) {
                id<MTLBufferBinding> bb = (id<MTLBufferBinding>)b;
                o->kind = 0;
                o->data_type = (uint32_t)bb.bufferDataType;
                o->data_size = (uint32_t)bb.bufferDataSize;
            } else {
                o->kind = 1;
            }
            o->access = b.access == MTLBindingAccessReadOnly ? 0
                        : b.access == MTLBindingAccessReadWrite ? 1
                                                                : 2;
            copy_str(b.name, o->name, sizeof o->name);
            ++n;
        }
        return n;
    }
}

// ---- Buffers ---------------------------------------------------------------

void *fr_buffer_new(void *dev, size_t nbytes) {
    id<MTLBuffer> b = [BORROW(id<MTLDevice>, dev) newBufferWithLength:(nbytes ? nbytes : 16)
                                                              options:MTLResourceStorageModeShared];
    return b ? RETAIN(b) : NULL;
}

void *fr_buffer_nocopy(void *dev, void *ptr, size_t nbytes) {
    id<MTLBuffer> b = [BORROW(id<MTLDevice>, dev) newBufferWithBytesNoCopy:ptr
                                                                    length:nbytes
                                                                   options:MTLResourceStorageModeShared
                                                               deallocator:nil];
    return b ? RETAIN(b) : NULL;
}

void *fr_buffer_from_mtl(void *mtl_buffer) {
    if (!mtl_buffer) return NULL;
    CFRetain(mtl_buffer);
    return mtl_buffer;
}

void *fr_buffer_contents(void *buf) { return BORROW(id<MTLBuffer>, buf).contents; }
size_t fr_buffer_length(void *buf) { return BORROW(id<MTLBuffer>, buf).length; }

// ---- Streams ---------------------------------------------------------------

struct fr_committed {
    id<MTLCommandBuffer> cb;
    std::string kernels;  // comma-separated kernel names, for error messages
};

struct fr_stream {
    id<MTLCommandQueue> queue;
    id<MTLSharedEvent> event;
    MTLCommandBufferDescriptor *desc;
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    uint64_t next_value = 0;
    int pending = 0;
    std::vector<const char *> names;  // kernels in the open command buffer
    std::vector<fr_committed> committed;  // since the previous sync
};

void *fr_stream_new(void *queue) {
    @autoreleasepool {
        fr_stream *s = new fr_stream;
        s->queue = BORROW(id<MTLCommandQueue>, queue);
        s->event = [s->queue.device newSharedEvent];
        s->desc = [MTLCommandBufferDescriptor new];
        s->desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
        return s;
    }
}

static void stream_open(fr_stream *s) {
    @autoreleasepool {
        s->cb = [s->queue commandBufferWithDescriptor:s->desc];
        s->enc = [s->cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    }
}

static inline void encode(id<MTLComputeCommandEncoder> enc, void *pso, const fr_launch_plan *plan,
                          void *const *bufs, const uint64_t *offsets, const void *scalar_bytes,
                          const uint32_t grid[3], const uint32_t tg[3]) {
    [enc setComputePipelineState:BORROW(id<MTLComputePipelineState>, pso)];
    for (int i = 0; i < plan->nbufs; ++i)
        [enc setBuffer:BORROW(id<MTLBuffer>, bufs[i])
                offset:(offsets ? offsets[i] : 0)
               atIndex:plan->buf_index[i]];
    const uint8_t *sb = (const uint8_t *)scalar_bytes;
    for (int i = 0; i < plan->nscalars; ++i) {
        const fr_scalar_slot &sl = plan->scalars[i];
        [enc setBytes:sb + sl.offset length:sl.size atIndex:sl.index];
    }
    [enc dispatchThreadgroups:MTLSizeMake(grid[0], grid[1], grid[2])
        threadsPerThreadgroup:MTLSizeMake(tg[0], tg[1], tg[2])];
}

void fr_stream_dispatch(void *stream, void *pso, const char *name, const fr_launch_plan *plan,
                        void *const *bufs, const uint64_t *offsets, const void *scalar_bytes,
                        const uint32_t grid[3], const uint32_t tg[3]) {
    fr_stream *s = (fr_stream *)stream;
    if (!s->enc) stream_open(s);
    encode(s->enc, pso, plan, bufs, offsets, scalar_bytes, grid, tg);
    if (s->names.empty() || s->names.back() != name) s->names.push_back(name);
    ++s->pending;
}

static std::string join_names(const std::vector<const char *> &names) {
    std::string out;
    std::vector<std::string> seen;
    for (const char *n : names) {
        std::string v = n ? n : "<unnamed>";
        bool dup = false;
        for (auto &x : seen) dup |= (x == v);
        if (dup) continue;
        seen.push_back(v);
        if (!out.empty()) out += ", ";
        out += v;
    }
    return out;
}

void fr_stream_flush(void *stream) {
    fr_stream *s = (fr_stream *)stream;
    if (!s->enc) return;
    @autoreleasepool {
        [s->enc endEncoding];
        [s->cb encodeSignalEvent:s->event value:++s->next_value];
        [s->cb commit];
        s->committed.push_back({s->cb, join_names(s->names)});
        s->enc = nil;
        s->cb = nil;
        s->names.clear();
        s->pending = 0;
    }
}

static std::string describe_error(id<MTLCommandBuffer> cb, const std::string &kernels) {
    std::string msg = "Metal command buffer failed";
    if (!kernels.empty()) msg += " (kernels: " + kernels + ")";
    NSError *e = cb.error;
    if (e) {
        msg += ": ";
        msg += e.localizedDescription.UTF8String;
        NSArray *infos = e.userInfo[MTLCommandBufferEncoderInfoErrorKey];
        for (id<MTLCommandBufferEncoderInfo> info in infos) {
            if (info.errorState == MTLCommandEncoderErrorStateFaulted) {
                msg += " [faulted encoder";
                if (info.label.length) msg += std::string(" ") + info.label.UTF8String;
                msg += "]";
            }
        }
    }
    return msg;
}

// Waits for everything committed so far and collects errors. Returns 0 on success.
static int stream_wait(fr_stream *s, char *err, size_t errlen) {
    int rc = 0;
    if (!s->committed.empty()) {
        uint64_t target = s->next_value;
        id<MTLCommandBuffer> last = s->committed.back().cb;
        // Spin on the shared event, which was the fastest wait. A failed command
        // buffer may never signal, so also watch the last buffer's status.
        for (unsigned spins = 0; s->event.signaledValue < target; ++spins) {
            if (last.status >= MTLCommandBufferStatusCompleted) break;
            if (spins > 2000) usleep(spins > 20000 ? 200 : 10);
        }
        std::string msg;
        for (auto &c : s->committed) {
            // The event fires when GPU work ends; the status settles a moment later.
            while (c.cb.status < MTLCommandBufferStatusCompleted) usleep(1);
            if (c.cb.status == MTLCommandBufferStatusError && msg.empty()) {
                msg = describe_error(c.cb, c.kernels);
                rc = 1;
            }
        }
        if (rc) copy_std(msg, err, errlen);
        s->committed.clear();
    }
    return rc;
}

int fr_stream_sync(void *stream, char *err, size_t errlen) {
    fr_stream *s = (fr_stream *)stream;
    if (err && errlen) err[0] = 0;
    fr_stream_flush(s);
    return stream_wait(s, err, errlen);
}

int fr_stream_pending(void *stream) { return ((fr_stream *)stream)->pending; }

void fr_stream_free(void *stream) {
    fr_stream *s = (fr_stream *)stream;
    if (!s) return;
    char err[16];
    fr_stream_sync(s, err, sizeof err);
    delete s;
}

int fr_stream_timed_run(void *stream, void *pso, const char *name, const fr_launch_plan *plan,
                        void *const *bufs, const uint64_t *offsets, const void *scalar_bytes,
                        const uint32_t grid[3], const uint32_t tg[3], double *gpu_start,
                        double *gpu_end, char *err, size_t errlen) {
    fr_stream *s = (fr_stream *)stream;
    fr_stream_dispatch(s, pso, name, plan, bufs, offsets, scalar_bytes, grid, tg);
    // Anything already pending would be timed too; callers sync first.
    return fr_stream_flush_timed(s, gpu_start, gpu_end, err, errlen);
}

int fr_stream_flush_timed(void *stream, double *gpu_start, double *gpu_end, char *err,
                          size_t errlen) {
    fr_stream *s = (fr_stream *)stream;
    if (err && errlen) err[0] = 0;
    *gpu_start = *gpu_end = 0;
    if (!s->enc) return stream_wait(s, err, errlen);
    id<MTLCommandBuffer> cb = s->cb;
    fr_stream_flush(s);
    int rc = stream_wait(s, err, errlen);
    *gpu_start = cb.GPUStartTime;
    *gpu_end = cb.GPUEndTime;
    return rc;
}
