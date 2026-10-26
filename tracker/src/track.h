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
    Box      box;          // predikcija za tekuci frejm
    float    confidence;
    Kalman   kf;

    int framesSinceSeen;
    int age;
    int hits;
    bool confirmed;
    Box lastMatched;      // okvir detekcije koju je drzao proslog frejma
    bool hadMatch;        // je li uopce bio uparen proslog frejma
    std::deque<std::vector<float>> gallery;   // zapamcene Re-ID znacajke

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
    int   minHitsHigh = 2;      // proba za sigurne detekcije (prilagodljiva proba)
    float confHigh    = 2.0f;   // prag sigurne detekcije; > 1 = iskljuceno
    float maxAgeSec   = -1.0f;  // > 0: maxAge u sekundama, preracunava se prema FPS-u
    float minHitsSec  = -1.0f;  // > 0: minHits u sekundama
    float dupIou  = 0.75f;   // dva traga iznad ovoga = duplikat
    float detNms  = 0.70f;    // 1.0 = iskljuceno (mjereno: pogorsava)
    float minConf = 0.0f;    // 0.0 = bez filtra (mjereno: pogorsava)
    float inertia = 0.10f;   // popust tragu koji je tu detekciju drzao proslog frejma

    // --- Re-ID ---
    bool  useReid    = true;
    int   classGate    = 1;     // 1 = uparuju se samo iste klase (dosadasnje); 0 = klasa ne blokira
    float classPenalty = 0.0f;  // uz classGate 0: dodatak cijeni za razlicitu klasu
    int   selfCal  = 0;      // samokalibracija, bitmaska: 1 zivotni vijek, 2 izgled, 4 proba, 8 klasa; 0 = iskljuceno
    float calAgeK  = 2.0f;   // maxAge = calAgeK * Q90(praznina)
    float calDRef  = 2.0f;   // razdvojivost d' pri kojoj je tezina izgleda puna
    float calAlpha = 0.05f;  // SPRT: dopustena stopa pogreske
    float calKappa = 0.11f;  // skala kazne za klasu
    int   calMinN  = 30;     // najmanje uzoraka prije koristenja procjene
    int   probRule = 0;      // 1 = proba iz tipicnog trajanja traga; 0 = iskljuceno
    float probFrac = 0.01f;  // proba = probFrac * tipicno trajanje traga (frejmovi)
    float wReid      = 0.8f;   // tezina appearance clana u cijeni
    int   reidAfterN = 0;      // Re-ID ulazi tek za tragove neuparene >= N frejmova
                               //   0  = uvijek (kao NvDeepSORT)
                               //   1+ = selektivno
                               //   999 = nikad (kao NvSORT)
    int   galleryMax = 5;     // koliko znacajki po tragu
    // sum -- prosljeduje se svakom novom tragu
    double wPos  = 1.0/20.0;
    double wVel  = 1.0/160.0;
    double wMeas = 0.15;
};

#endif
