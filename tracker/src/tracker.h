#ifndef MYTRACKER_TRACKER_H
#define MYTRACKER_TRACKER_H
#include "track.h"
#include <vector>
#include <cmath>
#include <cstdint>
#include <utility>

struct Detection {
    Box      box;
    uint16_t classId;
    float    confidence;
    int      matchedTrack;
    int      srcIndex;             // index in the original list from the plugin
    std::vector<float> feature;    // Re-ID feature, empty if not computed
};

class Tracker {
public:
    Params P;
    std::vector<Track> tracks;
    // returns (id, whether the track is confirmed) per detection
    std::vector<std::pair<uint64_t,bool>> update(std::vector<Detection> &dets);
private:
    uint64_t nextId = 1;
    void associate(std::vector<Detection> &dets);
};
#endif
