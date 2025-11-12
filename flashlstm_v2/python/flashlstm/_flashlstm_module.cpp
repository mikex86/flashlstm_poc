#include <Python.h>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

#include "lstm.hpp"

namespace {

template <typename T>
T PtrFromUnsigned(unsigned long long value) {
    return reinterpret_cast<T>(static_cast<uintptr_t>(value));
}

PyObject *StreamingLstmForward(PyObject *, PyObject *args) {
    unsigned long long time_steps{};
    unsigned long long batch_size{};
    unsigned long long input_size{};
    unsigned long long hidden_size{};
    unsigned long long recompute_interval{};
    unsigned long long x_tensor_host{};
    unsigned long long h0_device{};
    unsigned long long c0_device{};
    unsigned long long weights_ih{};
    unsigned long long weights_hh{};
    unsigned long long bias_ih{};
    unsigned long long bias_hh{};
    unsigned long long y_tensor_host{};
    unsigned long long gate_cache_host{};
    unsigned long long hy_device{};
    unsigned long long cy_device{};
    unsigned long long compute_stream{};
    unsigned long long h2d_stream{};
    unsigned long long d2h_stream{};

    if (!PyArg_ParseTuple(
            args,
            "KKKKKKKKKKKKKKKKKKK",
            &time_steps,
            &batch_size,
            &input_size,
            &hidden_size,
            &recompute_interval,
            &x_tensor_host,
            &h0_device,
            &c0_device,
            &weights_ih,
            &weights_hh,
            &bias_ih,
            &bias_hh,
            &y_tensor_host,
            &gate_cache_host,
            &hy_device,
            &cy_device,
            &compute_stream,
            &h2d_stream,
            &d2h_stream)) {
        return nullptr;
    }

    Py_BEGIN_ALLOW_THREADS
    flstm_StreamingLstmForward(
        static_cast<size_t>(time_steps),
        static_cast<size_t>(batch_size),
        static_cast<size_t>(input_size),
        static_cast<size_t>(hidden_size),
        static_cast<size_t>(recompute_interval),
        PtrFromUnsigned<const __half *>(x_tensor_host),
        PtrFromUnsigned<const __half *>(h0_device),
        PtrFromUnsigned<const __half *>(c0_device),
        PtrFromUnsigned<const float *>(weights_ih),
        PtrFromUnsigned<const float *>(weights_hh),
        PtrFromUnsigned<const float *>(bias_ih),
        PtrFromUnsigned<const float *>(bias_hh),
        PtrFromUnsigned<__half *>(y_tensor_host),
        PtrFromUnsigned<__half *>(gate_cache_host),
        PtrFromUnsigned<__half *>(hy_device),
        PtrFromUnsigned<__half *>(cy_device),
        PtrFromUnsigned<cudaStream_t>(compute_stream),
        PtrFromUnsigned<cudaStream_t>(h2d_stream),
        PtrFromUnsigned<cudaStream_t>(d2h_stream)
    );
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

PyObject *StreamingLstmBackward(PyObject *, PyObject *args) {
    unsigned long long time_steps{};
    unsigned long long batch_size{};
    unsigned long long input_size{};
    unsigned long long hidden_size{};
    unsigned long long recompute_interval{};
    unsigned long long x_tensor_host{};
    unsigned long long y_tensor_host{};
    unsigned long long gate_cache_host{};
    unsigned long long dY_tensor_host{};
    unsigned long long d_hn_device{};
    unsigned long long d_cn_device{};
    unsigned long long h0_device{};
    unsigned long long c0_device{};
    unsigned long long weights_ih{};
    unsigned long long weights_hh{};
    unsigned long long bias_ih{};
    unsigned long long bias_hh{};
    unsigned long long dx_tensor_host{};
    unsigned long long dW_ih{};
    unsigned long long dW_hh{};
    unsigned long long db_ih{};
    unsigned long long db_hh{};
    unsigned long long dh0_out{};
    unsigned long long dc0_out{};
    unsigned long long compute_stream{};
    unsigned long long h2d_stream{};
    unsigned long long d2h_stream{};

    if (!PyArg_ParseTuple(
            args,
            "KKKKKKKKKKKKKKKKKKKKKKKKKKK",
            &time_steps,
            &batch_size,
            &input_size,
            &hidden_size,
            &recompute_interval,
            &x_tensor_host,
            &y_tensor_host,
            &gate_cache_host,
            &dY_tensor_host,
            &d_hn_device,
            &d_cn_device,
            &h0_device,
            &c0_device,
            &weights_ih,
            &weights_hh,
            &bias_ih,
            &bias_hh,
            &dx_tensor_host,
            &dW_ih,
            &dW_hh,
            &db_ih,
            &db_hh,
            &dh0_out,
            &dc0_out,
            &compute_stream,
            &h2d_stream,
            &d2h_stream)) {
        return nullptr;
    }

    Py_BEGIN_ALLOW_THREADS
    flstm_StreamingLstmBackward(
        static_cast<size_t>(time_steps),
        static_cast<size_t>(batch_size),
        static_cast<size_t>(input_size),
        static_cast<size_t>(hidden_size),
        static_cast<size_t>(recompute_interval),
        PtrFromUnsigned<const __half *>(x_tensor_host),
        PtrFromUnsigned<const __half *>(y_tensor_host),
        PtrFromUnsigned<const __half *>(gate_cache_host),
        PtrFromUnsigned<const __half *>(dY_tensor_host),
        PtrFromUnsigned<const __half *>(d_hn_device),
        PtrFromUnsigned<const __half *>(d_cn_device),
        PtrFromUnsigned<const __half *>(h0_device),
        PtrFromUnsigned<const __half *>(c0_device),
        PtrFromUnsigned<const float *>(weights_ih),
        PtrFromUnsigned<const float *>(weights_hh),
        PtrFromUnsigned<const float *>(bias_ih),
        PtrFromUnsigned<const float *>(bias_hh),
        PtrFromUnsigned<__half *>(dx_tensor_host),
        PtrFromUnsigned<float *>(dW_ih),
        PtrFromUnsigned<float *>(dW_hh),
        PtrFromUnsigned<float *>(db_ih),
        PtrFromUnsigned<float *>(db_hh),
        PtrFromUnsigned<float *>(dh0_out),
        PtrFromUnsigned<float *>(dc0_out),
        PtrFromUnsigned<cudaStream_t>(compute_stream),
        PtrFromUnsigned<cudaStream_t>(h2d_stream),
        PtrFromUnsigned<cudaStream_t>(d2h_stream)
    );
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

PyMethodDef ModuleMethods[] = {
    {"streaming_lstm_forward", StreamingLstmForward, METH_VARARGS, "Invoke the streaming LSTM forward pass."},
    {"streaming_lstm_backward", StreamingLstmBackward, METH_VARARGS, "Invoke the streaming LSTM backward pass."},
    {nullptr, nullptr, 0, nullptr}
};

PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT,
    "_flashlstm",
    "Minimal bindings for the flash LSTM kernels.",
    -1,
    ModuleMethods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

} // namespace

PyMODINIT_FUNC PyInit__flashlstm(void) {
    return PyModule_Create(&ModuleDef);
}
