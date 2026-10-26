#ifndef MYTRACKER_REID_H
#define MYTRACKER_REID_H
#include <vector>
#include <string>
#include <cstdint>

// Wrapper around the OSNet TensorRT engine.
// Input:  N x 3 x 256 x 128, already normalised, NCHW, RGB
// Output: N x 512, L2-normalised
class ReID {
public:
    bool  init(const std::string &enginePath, int maxBatch = 16);
    void  deinit();
    // returns false if inference fails
    bool  infer(const float *inputNCHW, int n, float *outFeat);
    int   featSize() const { return featSize_; }
    bool  ready()    const { return ready_; }
private:
    void *runtime_ = nullptr;
    void *engine_  = nullptr;
    void *context_ = nullptr;
    void *dIn_     = nullptr;
    void *dOut_    = nullptr;
    void *stream_  = nullptr;
    int   featSize_ = 512;
    int   maxBatch_ = 16;
    bool  ready_    = false;
};

#endif
