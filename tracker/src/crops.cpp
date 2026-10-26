#include "crops.h"
#include <nvbufsurftransform.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstring>
#include <algorithm>

static const int CW = 128;   // sirina isjecka
static const int CH = 256;   // visina isjecka

bool CropExtractor::init(int maxBatch, int gpuId)
{
    maxBatch_ = maxBatch;
    gpuId_    = gpuId;

    NvBufSurfaceCreateParams p;
    memset(&p, 0, sizeof(p));
    p.gpuId       = gpuId;
    p.width       = CW;
    p.height      = CH;
    p.size        = 0;
    p.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    p.layout      = NVBUF_LAYOUT_PITCH;
    p.memType     = NVBUF_MEM_CUDA_PINNED;   // dostupno i CPU-u i GPU-u

    if (NvBufSurfaceCreate(&dst_, maxBatch_, &p) != 0) {
        printf("[crops] NvBufSurfaceCreate pao\n");
        return false;
    }
    dst_->numFilled = maxBatch_;

    ready_ = true;
    printf("[crops] init OK  batch=%d  %dx%d RGBA  pitch=%u\n",
           maxBatch_, CW, CH, dst_->surfaceList[0].pitch);
    fflush(stdout);
    return true;
}

int CropExtractor::extract(NvBufSurfaceParams *srcParams, NvBufSurfaceMemType srcMem,
                           const std::vector<CropRect> &rects, float *out)
{
    if (!ready_ || !srcParams || rects.empty()) return 0;
    const int n = std::min((int)rects.size(), maxBatch_);

    NvBufSurfTransformConfigParams cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.compute_mode = NvBufSurfTransformCompute_Default;
    cfg.gpu_id       = gpuId_;
    cfg.cuda_stream  = 0;
    if (NvBufSurfTransformSetSessionParams(&cfg) != NvBufSurfTransformError_Success) {
        printf("[crops] SetSessionParams pao\n"); fflush(stdout);
        return 0;
    }

    const int SW = (int)srcParams->width;
    const int SH = (int)srcParams->height;

    // Omotaj jedan NvBufSurfaceParams u minimalni NvBufSurface.
    NvBufSurface one;
    memset(&one, 0, sizeof(one));
    one.gpuId        = gpuId_;
    one.memType      = srcMem;
    one.batchSize    = 1;
    one.numFilled    = 1;
    one.isContiguous = false;
    one.surfaceList  = srcParams;

    // Jedan poziv po isjecku; dst je uvijek jedan element batcha.
    int done = 0;
    for (int i = 0; i < n; i++)
    {
        int x = std::max(0, rects[i].x);
        int y = std::max(0, rects[i].y);
        int w = std::min(rects[i].w, SW - x);
        int h = std::min(rects[i].h, SH - y);
        if (w < 4 || h < 4) continue;          // premalo, preskoci

        NvBufSurfTransformRect sr, dr;
        sr.left = (uint32_t)x; sr.top = (uint32_t)y;
        sr.width = (uint32_t)w; sr.height = (uint32_t)h;
        dr.left = 0; dr.top = 0;
        dr.width = (uint32_t)CW; dr.height = (uint32_t)CH;

        NvBufSurface d1;
        memset(&d1, 0, sizeof(d1));
        d1.gpuId       = dst_->gpuId;
        d1.memType     = dst_->memType;
        d1.batchSize   = 1;
        d1.numFilled   = 1;
        d1.isContiguous = false;
        d1.surfaceList = &dst_->surfaceList[done];

        NvBufSurfTransformParams tp;
        memset(&tp, 0, sizeof(tp));
        tp.transform_flag   = NVBUFSURF_TRANSFORM_CROP_SRC |
                              NVBUFSURF_TRANSFORM_CROP_DST |
                              NVBUFSURF_TRANSFORM_FILTER;
        tp.transform_filter = NvBufSurfTransformInter_Bilinear;
        tp.src_rect = &sr;
        tp.dst_rect = &dr;

        if (NvBufSurfTransform(&one, &d1, &tp) != NvBufSurfTransformError_Success)
            continue;                          // preskoci neuspjeli isjecak
        done++;
    }
    if (done == 0) return 0;

    // RGBA -> normaliziran NCHW float
    const float off[3] = { 123.675f, 116.280f, 103.530f };
    const float sc     = 0.01735207f;

    for (int i = 0; i < done; i++)
    {
        NvBufSurfaceParams &s = dst_->surfaceList[i];
        const uint8_t *p = (const uint8_t*)s.dataPtr;
        if (!p) return i;
        const int pitch = (int)s.pitch;

        float *o = out + (size_t)i * 3 * CH * CW;
        for (int yy = 0; yy < CH; yy++) {
            const uint8_t *row = p + (size_t)yy * pitch;
            for (int xx = 0; xx < CW; xx++) {
                const uint8_t *px = row + (size_t)xx * 4;
                o[(size_t)0*CH*CW + yy*CW + xx] = ((float)px[0] - off[0]) * sc;
                o[(size_t)1*CH*CW + yy*CW + xx] = ((float)px[1] - off[1]) * sc;
                o[(size_t)2*CH*CW + yy*CW + xx] = ((float)px[2] - off[2]) * sc;
            }
        }
    }
    return done;
}

void CropExtractor::deinit()
{
    if (dst_) { NvBufSurfaceDestroy(dst_); dst_ = nullptr; }
    ready_ = false;
}
