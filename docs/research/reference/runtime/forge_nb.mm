// nanobind extension wrapping the forge_rt C ABI with Python classes.
// Compile with -fobjc-arc together with forge_rt.mm.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <stdexcept>
#include <string>

#include "forge_rt.h"

namespace nb = nanobind;

struct Handle {
    void *p = nullptr;
    explicit Handle(void *p_) : p(p_) {}
    Handle(const Handle &) = delete;
    ~Handle() { fr_release(p); }
    uintptr_t ptr() const { return (uintptr_t)p; }
};
struct Device : Handle { using Handle::Handle; };
struct Queue : Handle { using Handle::Handle; };
struct Library : Handle { using Handle::Handle; };
struct Pipeline : Handle { using Handle::Handle; };
struct Buffer : Handle {
    using Handle::Handle;
    nb::object owner;  // keeps borrowed host memory alive for no-copy buffers
};

struct Batch {
    void *b = nullptr;
    ~Batch() {
        if (b) fr_batch_end(b, 0, nullptr);
    }
};

using Dim3 = std::tuple<uint32_t, uint32_t, uint32_t>;

// Collect MTLBuffer pointers from a Python sequence of Buffer objects without
// allocating; kernels in Forge have a bounded argument count.
static int collect_bufs(nb::handle seq, void **out, int cap) {
    PyObject *fast = PySequence_Fast(seq.ptr(), "bufs must be a sequence");
    if (!fast) throw nb::python_error();
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    if (n > cap) {
        Py_DECREF(fast);
        throw std::runtime_error("too many buffers");
    }
    PyObject **items = PySequence_Fast_ITEMS(fast);
    for (Py_ssize_t i = 0; i < n; ++i) out[i] = nb::inst_ptr<Buffer>(items[i])->p;
    Py_DECREF(fast);
    return (int)n;
}

static void to_arr(const Dim3 &d, uint32_t a[3]) {
    a[0] = std::get<0>(d);
    a[1] = std::get<1>(d);
    a[2] = std::get<2>(d);
}

static nb::tuple result_tuple(const fr_cb_result &r) {
    return nb::make_tuple(r.gpu_start, r.gpu_end, r.status);
}

NB_MODULE(forge_nb, m) {
    nb::class_<Device>(m, "Device")
        .def_prop_ro("ptr", &Device::ptr)
        .def_prop_ro("name", [](Device &d) {
            char buf[256];
            fr_device_name(d.p, buf, sizeof buf);
            return std::string(buf);
        });
    nb::class_<Queue>(m, "Queue").def_prop_ro("ptr", &Queue::ptr);
    nb::class_<Library>(m, "Library").def_prop_ro("ptr", &Library::ptr);
    nb::class_<Pipeline>(m, "Pipeline")
        .def_prop_ro("ptr", &Pipeline::ptr)
        .def_prop_ro("thread_execution_width",
                     [](Pipeline &p) { return fr_pipeline_simd_width(p.p); })
        .def_prop_ro("max_total_threads_per_threadgroup",
                     [](Pipeline &p) { return fr_pipeline_max_threads(p.p); });
    nb::class_<Buffer>(m, "Buffer")
        .def_prop_ro("ptr", &Buffer::ptr)
        .def_prop_ro("contents", [](Buffer &b) { return (uintptr_t)fr_buffer_contents(b.p); })
        .def_prop_ro("length", [](Buffer &b) { return fr_buffer_length(b.p); });

    m.def("default_device", [] { return new Device(fr_device_default()); });
    m.def("new_queue", [](Device &d) { return new Queue(fr_queue_new(d.p)); });

    m.def(
        "compile",
        [](Device &d, const std::string &src, uint32_t lang_version, bool fast_math) {
            char err[8192];
            void *lib;
            {
                nb::gil_scoped_release nogil;  // compilation is slow; let Python run
                lib = fr_library_new(d.p, src.c_str(), lang_version, fast_math, err, sizeof err);
            }
            if (!lib) throw std::runtime_error(std::string("MSL compile failed: ") + err);
            return new Library(lib);
        },
        nb::arg("device"), nb::arg("source"), nb::arg("lang_version") = 0,
        nb::arg("fast_math") = true);

    m.def("pipeline", [](Device &d, Library &lib, const std::string &name) {
        void *fn = fr_function_new(lib.p, name.c_str());
        if (!fn) throw std::runtime_error("no function named " + name);
        char err[4096];
        void *pso;
        {
            nb::gil_scoped_release nogil;
            pso = fr_pipeline_new(d.p, fn, err, sizeof err);
        }
        fr_release(fn);
        if (!pso) throw std::runtime_error(std::string("pipeline failed: ") + err);
        return new Pipeline(pso);
    });

    m.def("new_buffer", [](Device &d, size_t n) { return new Buffer(fr_buffer_new(d.p, n)); });
    m.def(
        "buffer_nocopy",
        [](Device &d, uintptr_t ptr, size_t n, nb::object owner) {
            void *b = fr_buffer_nocopy(d.p, (void *)ptr, n);
            if (!b) throw std::runtime_error("newBufferWithBytesNoCopy returned nil");
            auto *buf = new Buffer(b);
            buf->owner = owner;
            return buf;
        },
        nb::arg("device"), nb::arg("ptr"), nb::arg("nbytes"), nb::arg("owner").none());

    m.def(
        "launch",
        [](Queue &q, Pipeline &pso, nb::handle bufs, nb::bytes args, int args_index,
           const Dim3 &grid, const Dim3 &tg, bool wait) -> nb::object {
            void *bp[31];
            int nb_ = collect_bufs(bufs, bp, 31);
            uint32_t g[3], t[3];
            to_arr(grid, g);
            to_arr(tg, t);
            fr_cb_result r{};
            if (wait) {
                nb::gil_scoped_release nogil;
                fr_dispatch(q.p, pso.p, bp, nb_, args.c_str(), args.size(), args_index, g, t, 1, &r);
            } else {
                fr_dispatch(q.p, pso.p, bp, nb_, args.c_str(), args.size(), args_index, g, t, 0, nullptr);
            }
            return wait ? nb::object(result_tuple(r)) : nb::none();
        },
        nb::arg("queue"), nb::arg("pipeline"), nb::arg("bufs"), nb::arg("args"),
        nb::arg("args_index"), nb::arg("grid"), nb::arg("tg"), nb::arg("wait") = true);

    nb::class_<Batch>(m, "Batch")
        .def("dispatch",
             [](Batch &b, Pipeline &pso, nb::handle bufs, nb::bytes args, int args_index,
                const Dim3 &grid, const Dim3 &tg) {
                 void *bp[31];
                 int nb_ = collect_bufs(bufs, bp, 31);
                 uint32_t g[3], t[3];
                 to_arr(grid, g);
                 to_arr(tg, t);
                 fr_batch_dispatch(b.b, pso.p, bp, nb_, args.c_str(), args.size(), args_index, g, t);
             })
        .def("end", [](Batch &b, bool wait) -> nb::object {
            fr_cb_result r{};
            {
                nb::gil_scoped_release nogil;
                fr_batch_end(b.b, wait, &r);
            }
            b.b = nullptr;
            return wait ? nb::object(result_tuple(r)) : nb::none();
        });
    m.def("batch_begin", [](Queue &q, bool unretained) {
        auto *b = new Batch;
        b->b = fr_batch_begin(q.p, unretained);
        return b;
    }, nb::arg("queue"), nb::arg("unretained") = false);

    // Wrap a foreign id<MTLBuffer> (for example from torch MPS) without taking
    // ownership semantics beyond a +1 retain.
    m.def("buffer_from_handle", [](uintptr_t h, nb::object owner) {
        fr_retain((void *)h);
        auto *buf = new Buffer((void *)h);
        buf->owner = owner;
        return buf;
    });
}
