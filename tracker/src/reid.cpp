#include "reid.h"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <cstring>
#include <cstdio>
#include <cmath>

using namespace nvinfer1;

namespace {
class Logger : public ILogger {
    void log(Severity s, const char *msg) noexcept override {
        if (s <= Severity::kWARNING) printf("[reid][trt] %s\n", msg);
    }
};
Logger gLogger;
}

bool ReID::init(const std::string &enginePath, int maxBatch)
{
    maxBatch_ = maxBatch;

    std::ifstream f(enginePath, std::ios::binary);
    if (!f.good()) { printf("[reid] cannot open %s\n", enginePath.c_str()); return false; }
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(sz);
    f.read(blob.data(), sz);
    f.close();
    printf("[reid] engine %zu bytes\n", sz);

    IRuntime *rt = createInferRuntime(gLogger);
    if (!rt) { printf("[reid] createInferRuntime failed\n"); return false; }
    ICudaEngine *eng = rt->deserializeCudaEngine(blob.data(), sz);
    if (!eng) { printf("[reid] deserializeCudaEngine failed\n"); return false; }
    IExecutionContext *ctx = eng->createExecutionContext();
    if (!ctx) { printf("[reid] createExecutionContext failed\n"); return false; }

    runtime_ = rt; engine_ = eng; context_ = ctx;

    // print all tensors, to see the names and shapes
    int nb = eng->getNbIOTensors();
    printf("[reid] tensors: %d\n", nb);
    for (int i = 0; i < nb; i++) {
        const char *nm = eng->getIOTensorName(i);
        Dims d = eng->getTensorShape(nm);
        bool in = (eng->getTensorIOMode(nm) == TensorIOMode::kINPUT);
        printf("[reid]   %-10s %s  dims=[", nm, in ? "IN " : "OUT");
        for (int k = 0; k < d.nbDims; k++) printf("%s%d", k ? "," : "", (int)d.d[k]);
        printf("]\n");
        if (!in && d.nbDims >= 2) featSize_ = (int)d.d[d.nbDims - 1];
    }
    printf("[reid] featSize = %d\n", featSize_);

    size_t inBytes  = (size_t)maxBatch_ * 3 * 256 * 128 * sizeof(float);
    size_t outBytes = (size_t)maxBatch_ * featSize_ * sizeof(float);
    if (cudaMalloc(&dIn_,  inBytes)  != cudaSuccess) { printf("[reid] cudaMalloc in failed\n");  return false; }
    if (cudaMalloc(&dOut_, outBytes) != cudaSuccess) { printf("[reid] cudaMalloc out failed\n"); return false; }
    cudaStream_t st;
    cudaStreamCreate(&st);
    stream_ = st;

    ready_ = true;
    printf("[reid] init OK\n");
    fflush(stdout);
    return true;
}

bool ReID::infer(const float *inputNCHW, int n, float *outFeat)
{
    if (!ready_ || n <= 0) return false;
    if (n > maxBatch_) n = maxBatch_;

    ICudaEngine *eng = (ICudaEngine*)engine_;
    IExecutionContext *ctx = (IExecutionContext*)context_;
    cudaStream_t st = (cudaStream_t)stream_;

    size_t inBytes  = (size_t)n * 3 * 256 * 128 * sizeof(float);
    size_t outBytes = (size_t)n * featSize_ * sizeof(float);

    if (cudaMemcpyAsync(dIn_, inputNCHW, inBytes, cudaMemcpyHostToDevice, st) != cudaSuccess)
        return false;

    // set the shapes and addresses
    int nb = eng->getNbIOTensors();
    for (int i = 0; i < nb; i++) {
        const char *nm = eng->getIOTensorName(i);
        bool in = (eng->getTensorIOMode(nm) == TensorIOMode::kINPUT);
        if (in) {
            Dims d = eng->getTensorShape(nm);
            if (d.d[0] == -1) { d.d[0] = n; ctx->setInputShape(nm, d); }
            ctx->setTensorAddress(nm, dIn_);
        } else {
            ctx->setTensorAddress(nm, dOut_);
        }
    }

    if (!ctx->enqueueV3(st)) return false;
    if (cudaMemcpyAsync(outFeat, dOut_, outBytes, cudaMemcpyDeviceToHost, st) != cudaSuccess)
        return false;
    cudaStreamSynchronize(st);

    // L2 normalisation
    for (int i = 0; i < n; i++) {
        float *v = outFeat + (size_t)i * featSize_;
        double s = 0.0;
        for (int k = 0; k < featSize_; k++) s += (double)v[k] * v[k];
        s = std::sqrt(s);
        if (s > 1e-9) for (int k = 0; k < featSize_; k++) v[k] = (float)(v[k] / s);
    }
    return true;
}

void ReID::deinit()
{
    if (dIn_)  cudaFree(dIn_);
    if (dOut_) cudaFree(dOut_);
    if (stream_) cudaStreamDestroy((cudaStream_t)stream_);
    if (context_) delete (IExecutionContext*)context_;
    if (engine_)  delete (ICudaEngine*)engine_;
    if (runtime_) delete (IRuntime*)runtime_;
    dIn_=dOut_=stream_=context_=engine_=runtime_=nullptr;
    ready_ = false;
}
