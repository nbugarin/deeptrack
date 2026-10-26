// ============================================================
//  mytracker  --  development step 2: track structure and management
//  IoU + size similarity, Hungarian matching.
//  (At that step without a Kalman filter; kalman.h and reid.h were added in later steps.)
// ============================================================
#include <nvdstracker.h>
#include "tracker.h"
#include "reid.h"
#include "crops.h"
#include <cstdio>
#include <cstring>
#include <map>
#include <algorithm>

struct NvMOTContext {
    std::map<NvMOTStreamId, Tracker> trk;
    uint32_t frames = 0;
    ReID          reid;
    CropExtractor crops;
    std::vector<float> cropBuf;    // n*3*256*128
    std::vector<float> featBuf;    // n*512
    bool reidOn = false;
};

extern "C" {

NvMOTStatus NvMOT_Query(uint16_t sz, char *path, NvMOTQuery *pQuery)
{
    if (!pQuery) return NvMOTStatus_Error;
    memset(pQuery, 0, sizeof(NvMOTQuery));
    pQuery->computeConfig          = NVMOTCOMP_GPU;
    pQuery->numTransforms          = 1;
    pQuery->colorFormats[0]        = NVBUF_COLOR_FORMAT_RGBA;
    pQuery->memType                = NVBUF_MEM_CUDA_UNIFIED;
    pQuery->batchMode              = NvMOTBatchMode_Batch;
    pQuery->maxTargetsPerStream    = 150;
    pQuery->maxShadowTrackingAge   = 30;
    pQuery->outputReidTensor       = false;
    pQuery->reidFeatureSize        = 0;
    pQuery->supportPastFrame       = false;
    printf("[mytracker] Query OK\n"); fflush(stdout);
    return NvMOTStatus_OK;
}

NvMOTStatus NvMOT_Init(NvMOTConfig *cfg, NvMOTContextHandle *ph,
                       NvMOTConfigResponse *resp)
{
    if (!ph) return NvMOTStatus_Error;
    NvMOTContext *ctx = new NvMOTContext();
    *ph = ctx;

    const char *engPath = "/workspace/osnet_vessel.engine";
    const int   MAXB    = 16;
    if (ctx->reid.init(engPath, MAXB) && ctx->crops.init(MAXB, 0)) {
        ctx->cropBuf.resize((size_t)MAXB * 3 * 256 * 128);
        ctx->featBuf.resize((size_t)MAXB * ctx->reid.featSize());
        ctx->reidOn = true;
        printf("[mytracker] Re-ID enabled\n");
    } else {
        printf("[mytracker] Re-ID disabled -- running without the appearance term\n");
    }
    if (resp) {
        memset(resp, 0, sizeof(NvMOTConfigResponse));
        resp->summaryStatus = NvMOTConfigStatus_OK;
    }
    printf("[mytracker] Init OK\n"); fflush(stdout);
    return NvMOTStatus_OK;
}

void NvMOT_DeInit(NvMOTContextHandle h)
{
    NvMOTContext *ctx = (NvMOTContext*)h;
    if (!ctx) return;
    ctx->reid.deinit();
    ctx->crops.deinit();
    uint64_t total = 0;
    for (auto &kv : ctx->trk) total += kv.second.tracks.size();
    printf("[mytracker] DeInit  frames: %u  live tracks: %lu\n",
           ctx->frames, (unsigned long)total);
    fflush(stdout);
    delete ctx;
}

NvMOTStatus NvMOT_Process(NvMOTContextHandle h,
                          NvMOTProcessParams *pParams,
                          NvMOTTrackedObjBatch *pBatch)
{
    NvMOTContext *ctx = (NvMOTContext*)h;
    if (!ctx || !pParams || !pBatch) return NvMOTStatus_Error;

    pBatch->numFilled = 0;

    for (uint32_t i = 0; i < pParams->numFrames; i++)
    {
        NvMOTFrame *frame = &pParams->frameList[i];
        if (i >= pBatch->numAllocated) break;

        NvMOTTrackedObjList *outList = &pBatch->list[i];
        outList->streamID  = frame->streamID;
        outList->frameNum  = frame->frameNum;
        outList->valid     = true;
        outList->numFilled = 0;

        Tracker &T = ctx->trk[frame->streamID];

        // input -> Detection
        std::vector<Detection> dets;
        dets.reserve(frame->objectsIn.numFilled);
        for (uint32_t j = 0; j < frame->objectsIn.numFilled; j++) {
            NvMOTObjToTrack *in = &frame->objectsIn.list[j];
            Detection d;
            d.box = { (float)in->bbox.x, (float)in->bbox.y,
                      (float)in->bbox.width, (float)in->bbox.height };
            d.classId      = in->classId;
            d.confidence   = in->confidence;
            d.matchedTrack = -1;
            d.srcIndex     = (int)j;
            dets.push_back(d);
        }

        // Re-ID features for all detections of this frame
        if (T.P.useReid && ctx->reidOn && !dets.empty() && frame->numBuffers > 0 && frame->bufferList) {
            std::vector<CropRect> rects;
            rects.reserve(dets.size());
            for (const auto &d : dets)
                rects.push_back({ (int)d.box.x, (int)d.box.y, (int)d.box.w, (int)d.box.h });
            int nc = ctx->crops.extract(frame->bufferList[0],
                                        NVBUF_MEM_CUDA_UNIFIED,
                                        rects, ctx->cropBuf.data());
            if (nc > 0 && ctx->reid.infer(ctx->cropBuf.data(), nc, ctx->featBuf.data())) {
                int F = ctx->reid.featSize();
                int lim = std::min(nc, (int)dets.size());
                for (int q = 0; q < lim; q++)
                    dets[q].feature.assign(ctx->featBuf.begin() + (size_t)q*F,
                                           ctx->featBuf.begin() + (size_t)(q+1)*F);
            }
        }

        std::vector<std::pair<uint64_t,bool>> ids = T.update(dets);

        // output
        uint32_t k = 0;
        for (size_t j = 0; j < dets.size() && k < outList->numAllocated; j++)
        {
            if (!ids[j].second) continue;      // track still in probation
            NvMOTTrackedObj *out = &outList->list[k];
            memset(out, 0, sizeof(NvMOTTrackedObj));
            out->classId            = dets[j].classId;
            out->bbox               = frame->objectsIn.list[dets[j].srcIndex].bbox;
            out->confidence         = dets[j].confidence;
            out->trackingId         = ids[j].first;
            out->age                = 1;
            out->associatedObjectIn = &frame->objectsIn.list[dets[j].srcIndex];
            k++;
        }
        outList->numFilled = k;
        ctx->frames++;
    }

    pBatch->numFilled = pParams->numFrames;
    return NvMOTStatus_OK;
}

NvMOTStatus NvMOT_RetrieveMiscData(NvMOTContextHandle, NvMOTProcessParams*,
                                   NvMOTTrackerMiscData*)
{ return NvMOTStatus_OK; }

NvMOTStatus NvMOT_RemoveStreams(NvMOTContextHandle h, NvMOTStreamId mask)
{
    NvMOTContext *ctx = (NvMOTContext*)h;
    if (ctx) ctx->trk.erase(mask);
    return NvMOTStatus_OK;
}

} // extern "C"
