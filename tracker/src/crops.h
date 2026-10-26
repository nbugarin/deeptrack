#ifndef MYTRACKER_CROPS_H
#define MYTRACKER_CROPS_H
#include <nvbufsurface.h>
#include <vector>

struct CropRect { int x, y, w, h; };

// Izrezuje pravokutnike iz jednog frejma GPU surfacea, skalira na 256x128 RGB,
// normalizira i slaze u NCHW float buffer spreman za OSNet.
// Normalizacija je ista kao u config_tracker_maritime.yml:
//   offsets [123.675, 116.280, 103.530], netScaleFactor 0.01735207
class CropExtractor {
public:
    bool init(int maxBatch = 16, int gpuId = 0);
    void deinit();
    // out mora imati mjesta za n*3*256*128 float
    // vraca broj stvarno obradenih isjecaka
    // srcParams je frame->bufferList[i] iz NvMOTFrame
    int  extract(NvBufSurfaceParams *srcParams, NvBufSurfaceMemType srcMem,
                 const std::vector<CropRect> &rects, float *out);
    bool ready() const { return ready_; }
private:
    NvBufSurface *dst_ = nullptr;   // batch RGBA 256x128 na GPU
    int   maxBatch_ = 16;
    int   gpuId_ = 0;
    bool  ready_ = false;
};

#endif
