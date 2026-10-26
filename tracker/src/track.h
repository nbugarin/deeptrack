#ifndef MYTRACKER_TRACK_H
#define MYTRACKER_TRACK_H

#include "kalman.h"
#include <vector>
#include <deque>
#include <cstdint>
#include <cmath>

struct Box {
    float x, y, w, h;
    float cx() const { return x + w * 0.5f; }
    float cy() const { return y + h * 0.5f; }
    float area() const { return w * h; }
};

inline float iou(const Box &a, const Box &b)
{
    float x1 = std::fmax(a.x, b.x);
    float y1 = std::fmax(a.y, b.y);
    float x2 = std::fmin(a.x + a.w, b.x + b.w);
    float y2 = std::fmin(a.y + a.h, b.y + b.h);
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    float inter = (x2 - x1) * (y2 - y1);
    float uni   = a.area() + b.area() - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

inline float sizeSim(const Box &a, const Box &b)
{
    float dw = std::fabs(a.w - b.w) / std::fmax(a.w + b.w, 1e-6f);
    float dh = std::fabs(a.h - b.h) / std::fmax(a.h + b.h, 1e-6f);
    return std::exp(-2.0f * (dw + dh));
}

struct Track {
    uint64_t id;
    uint16_t classId;
    Box      box;          // prediction for the current frame
    float    confidence;
    Kalman   kf;

    int framesSinceSeen;
    int age;
    int hits;
    bool confirmed;
    Box lastMatched;      // box of the detection it held in the previous frame
    bool hadMatch;        // whether it was matched at all in the previous frame
    std::deque<std::vector<float>> gallery;   // stored Re-ID features

    Track(uint64_t id_, uint16_t cls, const Box &b, float conf, const Kalman &proto)
        : id(id_), classId(cls), box(b), confidence(conf), kf(proto),
          framesSinceSeen(0), age(1), hits(1), confirmed(false),
          lastMatched(b), hadMatch(true)
    {
        kf.init(b.cx(), b.cy(), b.w / std::fmax(b.h, 1e-6f), b.h);
    }

    void predict()
    {
        kf.predict();
        double bx, by, bw, bh;
        kf.toBox(bx, by, bw, bh);
        box = { (float)bx, (float)by, (float)bw, (float)bh };
    }

    void correct(const Box &m)
    {
        kf.update(m.cx(), m.cy(), m.w / std::fmax(m.h, 1e-6f), m.h);
        double bx, by, bw, bh;
        kf.toBox(bx, by, bw, bh);
        box = { (float)bx, (float)by, (float)bw, (float)bh };
    }
};

struct Params {
    float wIou    = 1.0f;
    float wSize   = 0.3f;
    float maxCost = 0.75f;
    int   maxAge  = 100;
    int   minHits = 20;
    int   minHitsHigh = 2;      // probation for confident detections (adaptive probation)
    float confHigh    = 2.0f;   // confident-detection threshold; > 1 = off
    float maxAgeSec   = -1.0f;  // > 0: maxAge in seconds, converted using the FPS
    float minHitsSec  = -1.0f;  // > 0: minHits in seconds
    float dupIou  = 0.75f;   // two tracks above this = duplicate
    float detNms  = 0.70f;    // 1.0 = off (measured: makes it worse)
    float minConf = 0.0f;    // 0.0 = no filter (measured: makes it worse)
    float inertia = 0.10f;   // discount for the track that held this detection in the previous frame

    // --- Re-ID ---
    bool  useReid    = true;
    int   classGate    = 1;     // 1 = only the same classes are matched (previous behaviour); 0 = the class does not block
    float classPenalty = 0.0f;  // with classGate 0: cost added for a different class
    int   selfCal  = 0;      // self-calibration, bit mask: 1 lifetime, 2 appearance, 4 probation, 8 class; 0 = off
    float calAgeK  = 2.0f;   // maxAge = calAgeK * Q90(gap)
    float calDRef  = 2.0f;   // separability d' at which the appearance weight is full
    float calAlpha = 0.05f;  // SPRT: allowed error rate
    float calKappa = 0.11f;  // scale of the class penalty
    int   calMinN  = 30;     // minimum number of samples before the estimate is used
    int   probRule = 0;      // 1 = probation from the typical track duration; 0 = off
    float probFrac = 0.01f;  // probation = probFrac * typical track duration (frames)
    float wReid      = 0.8f;   // weight of the appearance term in the cost
    int   reidAfterN = 0;      // Re-ID is used only for tracks unmatched for >= N frames
                               //   0  = always (as NvDeepSORT)
                               //   1+ = selectively
                               //   999 = never (as NvSORT)
    int   galleryMax = 5;     // how many features per track
    // noise -- passed to every new track
    double wPos  = 1.0/20.0;
    double wVel  = 1.0/160.0;
    double wMeas = 0.15;
};

#endif
