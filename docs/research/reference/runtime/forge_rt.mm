// Compile with -fobjc-arc.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <string.h>

#include "forge_rt.h"

#define RETAIN(x) ((__bridge_retained void *)(x))
#define BORROW(T, p) ((__bridge T)(p))

static void copy_err(NSError *e, char *err, size_t errlen) {
    if (!err || errlen == 0) return;
    err[0] = 0;
    if (e) strlcpy(err, e.localizedDescription.UTF8String, errlen);
}

static double now_ns(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom;
}

void *fr_device_default(void) { return RETAIN(MTLCreateSystemDefaultDevice()); }

void fr_device_name(void *dev, char *out, size_t n) {
    strlcpy(out, BORROW(id<MTLDevice>, dev).name.UTF8String, n);
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

void *fr_library_new(void *dev, const char *src, uint32_t lang_version,
                     int fast_math, char *err, size_t errlen) {
    @autoreleasepool {
        MTLCompileOptions *opts = [MTLCompileOptions new];
        if (lang_version) opts.languageVersion = (MTLLanguageVersion)lang_version;
        opts.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;
        NSError *e = nil;
        id<MTLLibrary> lib =
            [BORROW(id<MTLDevice>, dev) newLibraryWithSource:@(src) options:opts error:&e];
        // Warnings also come back through `e` with a non-nil library.
        copy_err(lib ? nil : e, err, errlen);
        return lib ? RETAIN(lib) : NULL;
    }
}

void *fr_function_new(void *lib, const char *name) {
    @autoreleasepool {
        id<MTLFunction> fn = [BORROW(id<MTLLibrary>, lib) newFunctionWithName:@(name)];
        return fn ? RETAIN(fn) : NULL;
    }
}

void *fr_pipeline_new(void *dev, void *fn, char *err, size_t errlen) {
    @autoreleasepool {
        NSError *e = nil;
        id<MTLComputePipelineState> pso = [BORROW(id<MTLDevice>, dev)
            newComputePipelineStateWithFunction:BORROW(id<MTLFunction>, fn)
                                          error:&e];
        copy_err(pso ? nil : e, err, errlen);
        return pso ? RETAIN(pso) : NULL;
    }
}

uint32_t fr_pipeline_simd_width(void *p) {
    return (uint32_t)BORROW(id<MTLComputePipelineState>, p).threadExecutionWidth;
}
uint32_t fr_pipeline_max_threads(void *p) {
    return (uint32_t)BORROW(id<MTLComputePipelineState>, p).maxTotalThreadsPerThreadgroup;
}
uint32_t fr_pipeline_static_tg_mem(void *p) {
    return (uint32_t)BORROW(id<MTLComputePipelineState>, p).staticThreadgroupMemoryLength;
}

void *fr_buffer_new(void *dev, size_t nbytes) {
    return RETAIN([BORROW(id<MTLDevice>, dev) newBufferWithLength:nbytes
                                                          options:MTLResourceStorageModeShared]);
}

void *fr_buffer_nocopy(void *dev, void *ptr, size_t nbytes) {
    id<MTLBuffer> b = [BORROW(id<MTLDevice>, dev)
        newBufferWithBytesNoCopy:ptr
                          length:nbytes
                         options:MTLResourceStorageModeShared
                     deallocator:nil];
    return b ? RETAIN(b) : NULL;
}

void *fr_buffer_contents(void *buf) { return BORROW(id<MTLBuffer>, buf).contents; }
size_t fr_buffer_length(void *buf) { return BORROW(id<MTLBuffer>, buf).length; }

static inline void encode_one(id<MTLComputeCommandEncoder> enc, void *pso,
                              void *const *bufs, int nbufs, const void *bytes,
                              size_t nbytes, int bytes_index, const uint32_t grid[3],
                              const uint32_t tg[3]) {
    [enc setComputePipelineState:BORROW(id<MTLComputePipelineState>, pso)];
    for (int i = 0; i < nbufs; ++i)
        [enc setBuffer:BORROW(id<MTLBuffer>, bufs[i]) offset:0 atIndex:i];
    if (nbytes) [enc setBytes:bytes length:nbytes atIndex:bytes_index];
    [enc dispatchThreadgroups:MTLSizeMake(grid[0], grid[1], grid[2])
        threadsPerThreadgroup:MTLSizeMake(tg[0], tg[1], tg[2])];
}

static inline void fill_result(id<MTLCommandBuffer> cb, fr_cb_result *res) {
    if (!res) return;
    res->gpu_start = cb.GPUStartTime;
    res->gpu_end = cb.GPUEndTime;
    res->status = (int)cb.status;
}

void fr_dispatch(void *queue, void *pso, void *const *bufs, int nbufs,
                 const void *bytes, size_t nbytes, int bytes_index,
                 const uint32_t grid[3], const uint32_t tg[3], int wait,
                 fr_cb_result *res) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [BORROW(id<MTLCommandQueue>, queue) commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        encode_one(enc, pso, bufs, nbufs, bytes, nbytes, bytes_index, grid, tg);
        [enc endEncoding];
        [cb commit];
        if (wait) {
            [cb waitUntilCompleted];
            fill_result(cb, res);
        }
    }
}

// Batch state lives in a heap struct; ObjC pointers are strong under ARC.
struct fr_batch {
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
};

void *fr_batch_begin(void *queue, int unretained) {
    @autoreleasepool {
        fr_batch *b = new fr_batch;
        id<MTLCommandQueue> q = BORROW(id<MTLCommandQueue>, queue);
        b->cb = unretained ? [q commandBufferWithUnretainedReferences] : [q commandBuffer];
        b->enc = [b->cb computeCommandEncoder];
        return b;
    }
}

void fr_batch_dispatch(void *batch, void *pso, void *const *bufs, int nbufs,
                       const void *bytes, size_t nbytes, int bytes_index,
                       const uint32_t grid[3], const uint32_t tg[3]) {
    fr_batch *b = (fr_batch *)batch;
    encode_one(b->enc, pso, bufs, nbufs, bytes, nbytes, bytes_index, grid, tg);
}

void fr_batch_end(void *batch, int wait, fr_cb_result *res) {
    @autoreleasepool {
        fr_batch *b = (fr_batch *)batch;
        [b->enc endEncoding];
        [b->cb commit];
        if (wait) {
            [b->cb waitUntilCompleted];
            fill_result(b->cb, res);
        }
        delete b;
    }
}

void fr_bench_loop(void *queue, void *pso, void *const *bufs, int nbufs,
                   const void *bytes, size_t nbytes, int bytes_index,
                   const uint32_t grid[3], const uint32_t tg[3], int iters,
                   int per_cb, double *out_wall_ns, double *out_gpu_ns) {
    id<MTLCommandQueue> q = BORROW(id<MTLCommandQueue>, queue);
    for (int it = 0; it < iters; ++it) {
        @autoreleasepool {
            double t0 = now_ns();
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            for (int j = 0; j < per_cb; ++j)
                encode_one(enc, pso, bufs, nbufs, bytes, nbytes, bytes_index, grid, tg);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            out_wall_ns[it] = now_ns() - t0;
            out_gpu_ns[it] = (cb.GPUEndTime - cb.GPUStartTime) * 1e9;
        }
    }
}

void fr_bench_wait_mode(void *queue, void *pso, void *const *bufs, int nbufs,
                        const void *bytes, size_t nbytes, int bytes_index,
                        const uint32_t grid[3], const uint32_t tg[3], int iters,
                        int mode, double *out_wall_ns) {
    id<MTLCommandQueue> q = BORROW(id<MTLCommandQueue>, queue);
    id<MTLSharedEvent> ev = [q.device newSharedEvent];
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    uint64_t val = 0;
    for (int it = 0; it < iters; ++it) {
        @autoreleasepool {
            double t0 = now_ns();
            id<MTLCommandBuffer> cb = mode == 4 ? [q commandBufferWithUnretainedReferences] : [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encode_one(enc, pso, bufs, nbufs, bytes, nbytes, bytes_index, grid, tg);
            [enc endEncoding];
            if (mode == 2)
                [cb addCompletedHandler:^(id<MTLCommandBuffer> _) { dispatch_semaphore_signal(sem); }];
            if (mode == 3) [cb encodeSignalEvent:ev value:++val];
            [cb commit];
            switch (mode) {
                case 1:
                    while (cb.status < MTLCommandBufferStatusCompleted) {}
                    break;
                case 2:
                    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
                    break;
                case 3:
                    while (ev.signaledValue < val) {}
                    break;
                default:
                    [cb waitUntilCompleted];
            }
            out_wall_ns[it] = now_ns() - t0;
        }
    }
}

double fr_bench_async(void *queue, void *pso, void *const *bufs, int nbufs,
                      const void *bytes, size_t nbytes, int bytes_index,
                      const uint32_t grid[3], const uint32_t tg[3], int iters) {
    id<MTLCommandQueue> q = BORROW(id<MTLCommandQueue>, queue);
    double t0 = now_ns();
    id<MTLCommandBuffer> last = nil;
    for (int it = 0; it < iters; ++it) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encode_one(enc, pso, bufs, nbufs, bytes, nbytes, bytes_index, grid, tg);
            [enc endEncoding];
            [cb commit];
            last = cb;
        }
    }
    [last waitUntilCompleted];
    return now_ns() - t0;
}

double fr_bench_encode(void *queue, void *pso, void *const *bufs, int nbufs,
                       const void *bytes, size_t nbytes, int bytes_index,
                       const uint32_t grid[3], const uint32_t tg[3], int n,
                       int concurrent, double *out_gpu_ns) {
    @autoreleasepool {
        id<MTLCommandQueue> q = BORROW(id<MTLCommandQueue>, queue);
        double t0 = now_ns();
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc =
            [cb computeCommandEncoderWithDispatchType:concurrent ? MTLDispatchTypeConcurrent
                                                                 : MTLDispatchTypeSerial];
        for (int j = 0; j < n; ++j) {
            // concurrent == 2: concurrent encoder plus a buffer barrier between
            // dispatches, i.e. what a runtime that tracks its own hazards emits
            // for a fully dependent chain.
            if (concurrent == 2 && j > 0) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            encode_one(enc, pso, bufs, nbufs, bytes, nbytes, bytes_index, grid, tg);
        }
        [enc endEncoding];
        [cb commit];
        double t = now_ns() - t0;
        [cb waitUntilCompleted];
        if (out_gpu_ns) *out_gpu_ns = (cb.GPUEndTime - cb.GPUStartTime) * 1e9;
        return t;
    }
}

struct fr_rec_hdr {
    uint64_t pso;
    uint32_t grid[3], tg[3];
    uint32_t nbufs, nbytes, bytes_index, pad;
};

static inline void encode_packed(id<MTLComputeCommandEncoder> enc, const void *rec) {
    const fr_rec_hdr *h = (const fr_rec_hdr *)rec;
    void *const *bufs = (void *const *)(h + 1);
    const uint8_t *bytes = (const uint8_t *)(bufs + h->nbufs);
    encode_one(enc, (void *)h->pso, bufs, (int)h->nbufs, bytes, h->nbytes, (int)h->bytes_index,
               h->grid, h->tg);
}

void fr_batch_dispatch_packed(void *batch, const void *rec) {
    encode_packed(((fr_batch *)batch)->enc, rec);
}

void fr_dispatch_packed(void *queue, const void *rec, int wait, fr_cb_result *res) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [BORROW(id<MTLCommandQueue>, queue) commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        encode_packed(enc, rec);
        [enc endEncoding];
        [cb commit];
        if (wait) {
            [cb waitUntilCompleted];
            fill_result(cb, res);
        }
    }
}
