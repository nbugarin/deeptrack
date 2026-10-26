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
    int      srcIndex;             // indeks u izvornoj listi iz plugina
    std::vector<float> feature;    // Re-ID znacajka, prazna ako nije racunata
};

class Tracker {
public:
    Params P;
    std::vector<Track> tracks;
    // vraca (id, je li trag potvrden) po detekciji
    std::vector<std::pair<uint64_t,bool>> update(std::vector<Detection> &dets);
private:
    uint64_t nextId = 1;
    void associate(std::vector<Detection> &dets);
};
#endif
