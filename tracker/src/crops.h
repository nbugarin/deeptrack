#ifndef MYTRACKER_CROPS_H
#define MYTRACKER_CROPS_H
#include <nvbufsurface.h>
#include <vector>

struct CropRect { int x, y, w, h; };

// Crops rectangles from one frame of the GPU surface, scales them to 256x128 RGB,
// normalises them and packs them into an NCHW float buffer ready for OSNet.
// The normalisation is the same as in config_tracker_maritime.yml:
//   offsets [123.675, 116.280, 103.530], netScaleFactor 0.01735207
class CropExtractor {
public:
    bool init(int maxBatch = 16, int gpuId = 0);
    void deinit();
    // out must have room for n*3*256*128 floats
    // returns the number of crops actually processed
    // srcParams is frame->bufferList[i] from NvMOTFrame
    int  extract(NvBufSurfaceParams *srcParams, NvBufSurfaceMemType srcMem,
                 const std::vector<CropRect> &rects, float *out);
    bool ready() const { return ready_; }
private:
    NvBufSurface *dst_ = nullptr;   // RGBA 256x128 batch on the GPU
    int   maxBatch_ = 16;
    int   gpuId_ = 0;
    bool  ready_ = false;
};

#endif
