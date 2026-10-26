// Checks of the two rules of the article on the tracker core, without DeepStream.
// Build and run (Eigen 3 required):
//   g++ -std=c++14 -O2 -Isrc -I/usr/include/eigen3 tests/test_rules.cpp src/tracker.cpp src/hungarian.cpp src/kalman.cpp -o test_rules
//   MYTRACKER_FPS=25 ./test_rules
#include "tracker.h"
#include <cstdio>
#include <cstdlib>

static int fails = 0;
static void check(bool ok, const char *what) { printf("%s  %s\n", ok ? "OK  " : "FAIL", what); if (!ok) fails++; }

static Detection det(float x, uint16_t cls, float conf) {
    Detection d; d.box = { x, 100.f, 80.f, 40.f }; d.classId = cls; d.confidence = conf;
    d.matchedTrack = -1; d.srcIndex = 0; return d;
}
static void setFinal(Tracker &T) {          // final variant of the article (Table II)
    T.P.classGate = 0; T.P.classPenalty = 0.5f; T.P.confHigh = 0.8f; T.P.minHitsSec = 0.6667f; T.P.maxAgeSec = 3.3333f;
    T.P.useReid = false;                     // no descriptors in this test
}
// frame at which the first detection's track is confirmed (1-based), or -1
static int confirmFrame(Tracker &T, float conf, int frames) {
    for (int f = 1; f <= frames; f++) {
        std::vector<Detection> d{ det(100.f + f, 0, conf) };
        auto out = T.update(d);
        if (out[0].second) return f;
    }
    return -1;
}

int main() {
    if (!getenv("MYTRACKER_FPS")) { printf("set MYTRACKER_FPS=25\n"); return 2; }

    // 1. Baseline: a class change opens a new track.
    { Tracker T; T.P.useReid = false;
      std::vector<Detection> a{ det(100, 0, 0.9f) }; auto o1 = T.update(a);
      std::vector<Detection> b{ det(101, 1, 0.9f) }; auto o2 = T.update(b);
      check(o1[0].first != o2[0].first, "baseline: class change gives a new identifier"); }

    // 2. Class-tolerant cost: the same vessel keeps its identifier across a class change.
    { Tracker T; setFinal(T);
      std::vector<Detection> a{ det(100, 0, 0.9f) }; auto o1 = T.update(a);
      std::vector<Detection> b{ det(101, 1, 0.9f) }; auto o2 = T.update(b);
      check(o1[0].first == o2[0].first, "penalty 0.5: class change keeps the identifier"); }

    // 3. Confidence-gated probation at 25 FPS: confident detections from the 2nd frame, others after round(0.6667*25) = 17.
    { Tracker T; setFinal(T); int f = confirmFrame(T, 0.9f, 40);
      printf("      confident track confirmed at frame %d\n", f); check(f == 2, "probation: p >= 0.8 confirmed at the second frame"); }
    { Tracker T; setFinal(T); int f = confirmFrame(T, 0.5f, 40);
      printf("      uncertain track confirmed at frame %d\n", f); check(f == 17, "probation: p < 0.8 confirmed after 17 frames at 25 FPS"); }

    // 4. Baseline confirmation after 20 frames regardless of confidence.
    { Tracker T; T.P.useReid = false; int f = confirmFrame(T, 0.9f, 40);
      printf("      baseline track confirmed at frame %d\n", f); check(f == 20, "baseline: confirmed after 20 frames"); }

    printf(fails ? "\n%d check(s) failed\n" : "\nall checks passed\n", fails);
    return fails ? 1 : 0;
}
