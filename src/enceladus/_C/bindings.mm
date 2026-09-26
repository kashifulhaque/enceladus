// nanobind module enceladus._C over the enceladus_rt C ABI.
// Compile with -fobjc-arc together with enceladus_rt.mm.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <string.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "enceladus_rt.h"

namespace nb = nanobind;

struct MetalError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Handle {
    void *p = nullptr;
    explicit Handle(void *p_) : p(p_) {}
    Handle(const Handle &) = delete;
    ~Handle() { fr_release(p); }
    uintptr_t addr() const { return (uintptr_t)p; }
};

struct Device : Handle {
    using Handle::Handle;
};
struct Queue : Handle {
    using Handle::Handle;
};
struct Library : Handle {
    using Handle::Handle;
};
struct Pipeline : Handle {
    std::string name;
    Pipeline(void *p_, std::string n) : Handle(p_), name(std::move(n)) {}
};
struct Buffer : Handle {
    using Handle::Handle;
    nb::object owner;  // keeps borrowed memory (numpy, DLPack) alive
};

struct LaunchPlan {
    fr_launch_plan plan{};
};

struct Stream {
    void *s = nullptr;
    int flush_every = 64;
    nb::object queue;  // keeps the queue alive
    ~Stream() { fr_stream_free(s); }
};

using Dim3 = std::tuple<uint32_t, uint32_t, uint32_t>;

static void read_dim3(nb::handle h, uint32_t out[3]) {
    PyObject *fast = PySequence_Fast(h.ptr(), "grid must be a sequence");
    if (!fast) throw nb::python_error();
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    if (n < 1 || n > 3) {
        Py_DECREF(fast);
        throw std::invalid_argument("grid must have 1 to 3 dimensions");
    }
    PyObject **items = PySequence_Fast_ITEMS(fast);
    for (int i = 0; i < 3; ++i) {
        if (i < n) {
            long long v = PyLong_AsLongLong(items[i]);
            if (v == -1 && PyErr_Occurred()) {
                Py_DECREF(fast);
                throw nb::python_error();
            }
            if (v < 0 || v > 0xFFFFFFFFll) {
                Py_DECREF(fast);
                throw std::invalid_argument("grid dimension out of range");
            }
            out[i] = (uint32_t)v;
        } else {
            out[i] = 1;
        }
    }
    Py_DECREF(fast);
}

// Collects MTLBuffer handles and byte offsets without allocating.
static int read_bufs(nb::handle bufs, nb::handle offsets, void **bp, uint64_t *op) {
    PyObject *fb = PySequence_Fast(bufs.ptr(), "bufs must be a sequence");
    if (!fb) throw nb::python_error();
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fb);
    if (n > FR_MAX_BINDINGS) {
        Py_DECREF(fb);
        throw std::invalid_argument("too many buffers");
    }
    PyObject **items = PySequence_Fast_ITEMS(fb);
    for (Py_ssize_t i = 0; i < n; ++i) {
        Buffer *b;
        if (!nb::try_cast<Buffer *>(nb::handle(items[i]), b) || !b) {
            Py_DECREF(fb);
            throw std::invalid_argument("bufs must contain enceladus._C.Buffer objects");
        }
        bp[i] = b->p;
        op[i] = 0;
    }
    Py_DECREF(fb);
    if (!offsets.is_none()) {
        PyObject *fo = PySequence_Fast(offsets.ptr(), "offsets must be a sequence");
        if (!fo) throw nb::python_error();
        Py_ssize_t m = PySequence_Fast_GET_SIZE(fo);
        if (m != n) {
            Py_DECREF(fo);
            throw std::invalid_argument("offsets and bufs differ in length");
        }
        PyObject **oi = PySequence_Fast_ITEMS(fo);
        for (Py_ssize_t i = 0; i < n; ++i) op[i] = PyLong_AsUnsignedLongLong(oi[i]);
        Py_DECREF(fo);
        if (PyErr_Occurred()) throw nb::python_error();
    }
    return (int)n;
}

struct DispatchArgs {
    void *bp[FR_MAX_BINDINGS];
    uint64_t op[FR_MAX_BINDINGS];
    uint32_t grid[3], tg[3];
    const char *scalars;
};

static void prepare(DispatchArgs &a, const LaunchPlan &plan, nb::handle bufs, nb::handle offsets,
                    nb::bytes &scalars, nb::handle grid, nb::handle tg) {
    int n = read_bufs(bufs, offsets, a.bp, a.op);
    if (n != plan.plan.nbufs)
        throw std::invalid_argument("expected " + std::to_string(plan.plan.nbufs) +
                                    " buffers, got " + std::to_string(n));
    size_t need = 0;
    for (int i = 0; i < plan.plan.nscalars; ++i)
        need = std::max(need, (size_t)plan.plan.scalars[i].offset + plan.plan.scalars[i].size);
    if (scalars.size() < need) throw std::invalid_argument("scalar bytes too short for plan");
    a.scalars = scalars.c_str();
    read_dim3(grid, a.grid);
    read_dim3(tg, a.tg);
}

// ---- DLPack ------------------------------------------------------------------
// The subset of dlpack.h (v1.0) that Enceladus needs. The layouts match the header.

enum { kDLMetal = 8 };

struct DLDevice {
    int32_t device_type;
    int32_t device_id;
};
struct DLDataType {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
};
struct DLTensor {
    void *data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t *shape;
    int64_t *strides;
    uint64_t byte_offset;
};
struct DLManagedTensor {
    DLTensor dl_tensor;
    void *manager_ctx;
    void (*deleter)(DLManagedTensor *self);
};
struct DLPackVersion {
    uint32_t major;
    uint32_t minor;
};
struct DLManagedTensorVersioned {
    DLPackVersion version;
    void *manager_ctx;
    void (*deleter)(DLManagedTensorVersioned *self);
    uint64_t flags;
    DLTensor dl_tensor;
};

// Owns the shape and strides arrays and a reference to the Python object that keeps
// the MTLBuffer alive.
struct DLContext {
    PyObject *owner = nullptr;
    std::vector<int64_t> shape, strides;
};

template <typename M>
static void dl_delete(M *m) {
    auto *ctx = static_cast<DLContext *>(m->manager_ctx);
    // Consumers can call the deleter from any thread, with or without the GIL.
    PyGILState_STATE s = PyGILState_Ensure();
    Py_XDECREF(ctx->owner);
    PyGILState_Release(s);
    delete ctx;
    delete m;
}

template <typename M>
static void dl_capsule_destructor(PyObject *cap, const char *name) {
    // A consumer renames the capsule when it takes ownership.
    if (!PyCapsule_IsValid(cap, name)) return;
    PyObject *type, *value, *tb;
    PyErr_Fetch(&type, &value, &tb);
    auto *m = static_cast<M *>(PyCapsule_GetPointer(cap, name));
    if (m && m->deleter) m->deleter(m);
    PyErr_Restore(type, value, tb);
}

static void dl_destroy_legacy(PyObject *cap) {
    dl_capsule_destructor<DLManagedTensor>(cap, "dltensor");
}
static void dl_destroy_versioned(PyObject *cap) {
    dl_capsule_destructor<DLManagedTensorVersioned>(cap, "dltensor_versioned");
}

static nb::object dlpack_export(Buffer &buf, const std::vector<int64_t> &shape,
                                const std::vector<int64_t> &strides, uint64_t byte_offset,
                                uint8_t code, uint8_t bits, nb::object owner, bool versioned) {
    if (shape.size() != strides.size())
        throw std::invalid_argument("shape and strides differ in length");
    auto *ctx = new DLContext();
    ctx->shape = shape;
    ctx->strides = strides;
    ctx->owner = owner.inc_ref().ptr();
    DLTensor t{};
    t.data = buf.p;  // kDLMetal: the id<MTLBuffer>, not its contents
    t.device = {kDLMetal, 0};
    t.ndim = (int32_t)shape.size();
    t.dtype = {code, bits, 1};
    t.shape = ctx->shape.data();
    t.strides = ctx->strides.data();
    t.byte_offset = byte_offset;
    PyObject *cap;
    if (versioned) {
        auto *m = new DLManagedTensorVersioned();
        m->version = {1, 0};
        m->manager_ctx = ctx;
        m->deleter = dl_delete<DLManagedTensorVersioned>;
        m->flags = 0;
        m->dl_tensor = t;
        cap = PyCapsule_New(m, "dltensor_versioned", dl_destroy_versioned);
        if (!cap) m->deleter(m);
    } else {
        auto *m = new DLManagedTensor();
        m->dl_tensor = t;
        m->manager_ctx = ctx;
        m->deleter = dl_delete<DLManagedTensor>;
        cap = PyCapsule_New(m, "dltensor", dl_destroy_legacy);
        if (!cap) m->deleter(m);
    }
    if (!cap) throw nb::python_error();
    return nb::steal(cap);
}

// Reads a DLPack capsule without consuming it.
static nb::tuple dlpack_inspect(nb::handle cap) {
    const DLTensor *t = nullptr;
    if (PyCapsule_IsValid(cap.ptr(), "dltensor_versioned")) {
        auto *m = static_cast<DLManagedTensorVersioned *>(
            PyCapsule_GetPointer(cap.ptr(), "dltensor_versioned"));
        if (m->version.major != 1)
            throw std::invalid_argument("unsupported DLPack major version " +
                                        std::to_string(m->version.major));
        t = &m->dl_tensor;
    } else if (PyCapsule_IsValid(cap.ptr(), "dltensor")) {
        t = &static_cast<DLManagedTensor *>(PyCapsule_GetPointer(cap.ptr(), "dltensor"))
                 ->dl_tensor;
    } else {
        throw std::invalid_argument("expected an unconsumed DLPack capsule");
    }
    nb::list shape, strides;
    for (int i = 0; i < t->ndim; ++i) shape.append(t->shape[i]);
    nb::object st = nb::none();
    if (t->strides) {
        for (int i = 0; i < t->ndim; ++i) strides.append(t->strides[i]);
        st = nb::tuple(strides);
    }
    return nb::make_tuple((uintptr_t)t->data, t->device.device_type, t->device.device_id,
                          t->byte_offset, nb::tuple(shape), st,
                          nb::make_tuple(t->dtype.code, t->dtype.bits, t->dtype.lanes));
}

NB_MODULE(_C, m) {
    nb::exception<MetalError>(m, "MetalError");

    nb::class_<Device>(m, "Device")
        .def_prop_ro("handle", &Device::addr)
        .def_prop_ro("name",
                     [](Device &d) {
                         char buf[256];
                         fr_device_name(d.p, buf, sizeof buf);
                         return std::string(buf);
                     })
        .def_prop_ro("architecture",
                     [](Device &d) {
                         char buf[256];
                         fr_device_architecture(d.p, buf, sizeof buf);
                         return std::string(buf);
                     })
        .def("query", [](Device &d) {
            fr_device_info i;
            fr_device_query(d.p, &i);
            nb::dict out;
            out["max_threadgroup_memory"] = i.max_threadgroup_memory;
            out["max_threads_per_threadgroup"] = i.max_threads_per_threadgroup;
            out["max_buffer_length"] = i.max_buffer_length;
            out["recommended_max_working_set"] = i.recommended_max_working_set;
            out["max_concurrent_compilations"] = i.max_concurrent_compilations;
            nb::list fams;
            for (int k = 1; k <= 10; ++k)
                if (i.supports_apple[k]) fams.append(k);
            out["apple_families"] = fams;
            out["metal3"] = (bool)i.supports_metal3;
            out["metal4"] = (bool)i.supports_metal4;
            return out;
        });

    nb::class_<Queue>(m, "Queue").def_prop_ro("handle", &Queue::addr);
    nb::class_<Library>(m, "Library").def_prop_ro("handle", &Library::addr);

    nb::class_<Pipeline>(m, "Pipeline")
        .def_prop_ro("handle", &Pipeline::addr)
        .def_ro("name", &Pipeline::name)
        .def_prop_ro("thread_execution_width",
                     [](Pipeline &p) {
                         fr_pipeline_info i;
                         fr_pipeline_query(p.p, &i);
                         return i.thread_execution_width;
                     })
        .def_prop_ro("max_total_threads_per_threadgroup",
                     [](Pipeline &p) {
                         fr_pipeline_info i;
                         fr_pipeline_query(p.p, &i);
                         return i.max_total_threads_per_threadgroup;
                     })
        .def_prop_ro("static_threadgroup_memory",
                     [](Pipeline &p) {
                         fr_pipeline_info i;
                         fr_pipeline_query(p.p, &i);
                         return i.static_threadgroup_memory;
                     })
        .def("bindings", [](Pipeline &p) {
            fr_binding b[FR_MAX_BINDINGS + 8];
            int n = fr_pipeline_bindings(p.p, b, FR_MAX_BINDINGS + 8);
            nb::list out;
            for (int i = 0; i < n; ++i) {
                nb::dict d;
                d["index"] = b[i].index;
                d["kind"] = b[i].kind == 0 ? "buffer" : "other";
                d["data_type"] = b[i].data_type;
                d["data_size"] = b[i].data_size;
                d["access"] = b[i].access == 0 ? "read" : b[i].access == 1 ? "read_write" : "write";
                d["name"] = std::string(b[i].name);
                out.append(d);
            }
            return out;
        });

    nb::class_<Buffer>(m, "Buffer")
        .def_prop_ro("handle", &Buffer::addr)
        .def_prop_ro("ptr", [](Buffer &b) { return (uintptr_t)fr_buffer_contents(b.p); })
        .def_prop_ro("nbytes", [](Buffer &b) { return fr_buffer_length(b.p); })
        .def_prop_ro("owner", [](Buffer &b) { return b.owner; });

    nb::class_<LaunchPlan>(m, "LaunchPlan")
        .def(
            "__init__",
            [](LaunchPlan *self, const std::vector<uint32_t> &buf_index,
               const std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> &scalars) {
                new (self) LaunchPlan();
                if (buf_index.size() + scalars.size() > FR_MAX_BINDINGS)
                    throw std::invalid_argument("a kernel can bind at most 31 arguments");
                self->plan.nbufs = (int)buf_index.size();
                for (size_t i = 0; i < buf_index.size(); ++i)
                    self->plan.buf_index[i] = buf_index[i];
                self->plan.nscalars = (int)scalars.size();
                for (size_t i = 0; i < scalars.size(); ++i) {
                    auto &[idx, off, size] = scalars[i];
                    if (size > 4096) throw std::invalid_argument("scalar exceeds 4 KB");
                    self->plan.scalars[i] = {idx, off, size};
                }
            },
            nb::arg("buf_index"), nb::arg("scalars"))
        .def_prop_ro("nbufs", [](LaunchPlan &p) { return p.plan.nbufs; })
        .def_prop_ro("nscalars", [](LaunchPlan &p) { return p.plan.nscalars; });

    nb::class_<Stream>(m, "Stream")
        .def(
            "__init__",
            [](Stream *self, nb::object queue, int flush_every) {
                new (self) Stream();
                self->s = fr_stream_new(nb::cast<Queue *>(queue)->p);
                self->flush_every = flush_every < 1 ? 1 : flush_every;
                self->queue = queue;
            },
            nb::arg("queue"), nb::arg("flush_every") = 64)
        .def_rw("flush_every", &Stream::flush_every)
        .def_prop_ro("pending", [](Stream &s) { return fr_stream_pending(s.s); })
        .def(
            "dispatch",
            [](Stream &s, Pipeline &pso, LaunchPlan &plan, nb::handle bufs, nb::handle offsets,
               nb::bytes scalars, nb::handle grid, nb::handle tg) {
                DispatchArgs a;
                prepare(a, plan, bufs, offsets, scalars, grid, tg);
                fr_stream_dispatch(s.s, pso.p, pso.name.c_str(), &plan.plan, a.bp, a.op,
                                   a.scalars, a.grid, a.tg);
                if (fr_stream_pending(s.s) >= s.flush_every) fr_stream_flush(s.s);
            },
            nb::arg("pipeline"), nb::arg("plan"), nb::arg("bufs"), nb::arg("offsets").none(),
            nb::arg("scalars"), nb::arg("grid"), nb::arg("tg"))
        .def("flush", [](Stream &s) { fr_stream_flush(s.s); })
        .def("sync",
             [](Stream &s) {
                 std::unique_ptr<char[]> err(new char[FR_ERR_LEN]);
                 int rc;
                 {
                     nb::gil_scoped_release nogil;
                     rc = fr_stream_sync(s.s, err.get(), FR_ERR_LEN);
                 }
                 if (rc) throw MetalError(err.get());
             })
        .def(
            "timed_run",
            [](Stream &s, Pipeline &pso, LaunchPlan &plan, nb::handle bufs, nb::handle offsets,
               nb::bytes scalars, nb::handle grid, nb::handle tg) {
                DispatchArgs a;
                prepare(a, plan, bufs, offsets, scalars, grid, tg);
                std::unique_ptr<char[]> err(new char[FR_ERR_LEN]);
                double t0, t1;
                int rc;
                {
                    char sync_err[FR_ERR_LEN > 4096 ? 4096 : FR_ERR_LEN];
                    nb::gil_scoped_release nogil;
                    rc = fr_stream_sync(s.s, sync_err, sizeof sync_err);
                    if (rc) strlcpy(err.get(), sync_err, FR_ERR_LEN);
                    if (!rc)
                        rc = fr_stream_timed_run(s.s, pso.p, pso.name.c_str(), &plan.plan, a.bp,
                                                 a.op, a.scalars, a.grid, a.tg, &t0, &t1,
                                                 err.get(), FR_ERR_LEN);
                }
                if (rc) throw MetalError(err.get());
                return nb::make_tuple(t0, t1);
            },
            nb::arg("pipeline"), nb::arg("plan"), nb::arg("bufs"), nb::arg("offsets").none(),
            nb::arg("scalars"), nb::arg("grid"), nb::arg("tg"))
        .def("flush_timed", [](Stream &s) {
            std::unique_ptr<char[]> err(new char[FR_ERR_LEN]);
            double t0, t1;
            int rc;
            {
                nb::gil_scoped_release nogil;
                rc = fr_stream_flush_timed(s.s, &t0, &t1, err.get(), FR_ERR_LEN);
            }
            if (rc) throw MetalError(err.get());
            return nb::make_tuple(t0, t1);
        });

    m.def("default_device", [] {
        void *d = fr_device_default();
        if (!d) throw MetalError("no Metal device is available");
        return new Device(d);
    });
    m.def("new_queue", [](Device &d) { return new Queue(fr_queue_new(d.p)); });

    m.def(
        "compile",
        [](Device &d, const std::string &src, uint32_t language_version, int math_mode,
           int math_fp32_functions, bool preserve_invariance, bool enable_logging) {
            fr_compile_options o{language_version, math_mode, math_fp32_functions,
                                 preserve_invariance ? 1 : 0, enable_logging ? 1 : 0};
            std::unique_ptr<char[]> err(new char[FR_ERR_LEN]);
            void *lib;
            {
                nb::gil_scoped_release nogil;  // compilation is slow; let Python run
                lib = fr_library_new(d.p, src.c_str(), &o, err.get(), FR_ERR_LEN);
            }
            if (!lib) throw MetalError(std::string("MSL compilation failed:\n") + err.get());
            return nb::make_tuple(nb::cast(new Library(lib), nb::rv_policy::take_ownership),
                                  std::string(err.get()));
        },
        nb::arg("device"), nb::arg("source"), nb::arg("language_version") = 0,
        nb::arg("math_mode") = 1, nb::arg("math_fp32_functions") = 0,
        nb::arg("preserve_invariance") = false, nb::arg("enable_logging") = false);

    m.def("pipeline", [](Device &d, Library &lib, const std::string &name) {
        void *fn = fr_function_new(lib.p, name.c_str());
        if (!fn) throw MetalError("the library has no kernel function named '" + name + "'");
        std::unique_ptr<char[]> err(new char[FR_ERR_LEN]);
        void *pso;
        {
            nb::gil_scoped_release nogil;
            pso = fr_pipeline_new(d.p, fn, err.get(), FR_ERR_LEN);
        }
        fr_release(fn);
        if (!pso) throw MetalError(std::string("pipeline creation failed: ") + err.get());
        return new Pipeline(pso, name);
    });

    m.def("new_buffer", [](Device &d, size_t n) {
        void *b = fr_buffer_new(d.p, n);
        if (!b) throw MetalError("buffer allocation of " + std::to_string(n) + " bytes failed");
        return new Buffer(b);
    });
    m.def(
        "buffer_nocopy",
        [](Device &d, uintptr_t ptr, size_t n, nb::object owner) -> nb::object {
            void *b = fr_buffer_nocopy(d.p, (void *)ptr, n);
            if (!b) return nb::none();
            auto *buf = new Buffer(b);
            buf->owner = owner;
            return nb::cast(buf, nb::rv_policy::take_ownership);
        },
        nb::arg("device"), nb::arg("ptr"), nb::arg("nbytes"), nb::arg("owner").none());
    m.def(
        "buffer_from_mtl",
        [](uintptr_t handle, nb::object owner) {
            if (!handle) throw std::invalid_argument("null MTLBuffer handle");
            auto *buf = new Buffer(fr_buffer_from_mtl((void *)handle));
            buf->owner = owner;
            return buf;
        },
        nb::arg("handle"), nb::arg("owner").none());
    m.def("dlpack_export", &dlpack_export, nb::arg("buffer"), nb::arg("shape"),
          nb::arg("strides"), nb::arg("byte_offset"), nb::arg("code"), nb::arg("bits"),
          nb::arg("owner").none(), nb::arg("versioned") = false,
          "Returns a kDLMetal DLPack capsule whose data is the buffer's id<MTLBuffer>.");
    m.def("dlpack_inspect", &dlpack_inspect, nb::arg("capsule"),
          "Returns (data, device_type, device_id, byte_offset, shape, strides, "
          "(code, bits, lanes)) without consuming the capsule.");
}
