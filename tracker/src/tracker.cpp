#include <cstdio>
#include "tracker.h"
#include "hungarian.h"
#include <algorithm>

// najveca kosinusna slicnost prema galeriji traga
static float bestSim(const std::deque<std::vector<float>> &g,
                        const std::vector<float> &f)
{
    if (g.empty() || f.empty()) return -1.0f;
    float best = -1.0f;
    for (const auto &v : g) {
        if (v.size() != f.size()) continue;
        float s = 0.0f;
        for (size_t k = 0; k < f.size(); k++) s += v[k] * f[k];
        if (s > best) best = s;
    }
    return best;
}

// ======================= SAMOKALIBRIRAJUCA ASOCIJACIJA =======================
// Tracker iz vlastitih potvrdenih tragova procjenjuje svojstva scene i iz njih izvodi parametre.
// P.selfCal je bitmaska: 1 zivotni vijek, 2 tezina izgleda, 4 proba (SPRT), 8 kazna za klasu. 0 = iskljuceno.
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <deque>
#include <map>
#include <unordered_map>
#include <vector>
// FPS videa iz varijable okoline MYTRACKER_FPS (zadano 30)
static float envFps(){ static float f=-1.f; if(f<0.f){ const char*e=getenv("MYTRACKER_FPS"); f=e?(float)atof(e):30.f; if(f<=0.f) f=30.f; } return f; }
namespace cal {
    static std::deque<int>   gaps;                 // trajanja praznina potvrdenih tragova (frejmovi)
    static std::deque<float> simSame, simDiff;     // slicnosti izgleda: isti trag / razliciti tragovi
    static std::map<std::pair<int,int>,float> frameSim;   // slicnosti izracunate u ovom frejmu (i,j)
    static const int NB=5;                          // razredi pouzdanosti detekcije
    static double confTrue[NB]={0}, confFalse[NB]={0};
    static std::unordered_map<uint64_t,std::vector<float>> early;   // pouzdanosti prvih detekcija traga
    static std::map<std::pair<int,int>,double> clsCount; static std::map<int,double> clsTotal;
    static int   maxAgeFrames=-1;                  // izracunato na pocetku frejma
    static float wReidNow=-1.f;
    template<class D> static void push(D &d, typename D::value_type v, size_t cap){ d.push_back(v); while(d.size()>cap) d.pop_front(); }
    static int bin(float c){ int b=(int)(c*NB); return b<0?0:(b>=NB?NB-1:b); }
}
template<class PP> static int baseMaxAge(const PP &p){ return p.maxAgeSec > 0.f ? (int)std::lround(p.maxAgeSec*envFps()) : p.maxAge; }
template<class PP> static int effMaxAge (const PP &p){ return ((p.selfCal & 1) && cal::maxAgeFrames > 0) ? cal::maxAgeFrames : baseMaxAge(p); }
// ---- pravilo probe v2: proba = probFrac * tipicno trajanje traga (iz vlastitih tragova) ----
// Tipicno trajanje = medijan RASPONA tragova PONDERIRAN brojem uparivanja (procjenitelj E6):
// neosjetljiv na mnogo kratkih laznih tragova i na povratnu spregu kratke probe.
namespace cal {
    static std::deque<std::pair<int,int>> doneLen;   // (raspon, broj uparivanja) zavrsenih potvrdenih tragova
    static int probNow=-1, probL=-1;
}
static void calDone(int span,int hits){ if (span>0 && hits>0) cal::push(cal::doneLen,std::make_pair(span,hits),300); }
template<class PP,class TV> static void calTrackStats(const PP &p,const TV &tracks){
    cal::probNow=-1; cal::probL=-1;
    if (!p.probRule) return;
    std::vector<std::pair<int,int>> L(cal::doneLen.begin(), cal::doneLen.end());
    for (const auto &t: tracks) if (t.confirmed && t.hits>0) L.push_back(std::make_pair(t.age - t.framesSinceSeen + 1, t.hits));
    if ((int)L.size() < 5) { cal::probNow = 2; return; }            // dok nema statistike: najkraca proba
    std::sort(L.begin(), L.end());
    double W=0; for (auto &x: L) W+=x.second;
    double c=0; int med=L.back().first;
    for (auto &x: L){ c+=x.second; if (c >= W/2){ med=x.first; break; } }
    cal::probL=med;
    cal::probNow = std::max(2, (int)std::lround(p.probFrac * med));
}
template<class PP> static int effMinHits(const PP &p){ if (p.probRule && cal::probNow > 0) return cal::probNow; return p.minHitsSec > 0.f ? std::max(1,(int)std::lround(p.minHitsSec*envFps())) : p.minHits; }
template<class PP> static float calWReid(const PP &p){ return ((p.selfCal & 2) && cal::wReidNow >= 0.f) ? cal::wReidNow : p.wReid; }
// ---- dijagnosticki zapis samokalibracije (samo cita statistiku, ne mijenja pracenje) ----
// Ukljucuje se varijablom okoline MYTRACKER_CALLOG=<putanja>; zapisuje se svakih 50 frejmova.
#include <cstdio>
template<class PP> static void calDump(const PP &p){
    static int frame=0; static const char *path=getenv("MYTRACKER_CALLOG");
    frame++; if (!path || (frame % 50)) return;
    FILE *f=fopen(path,"w"); if (!f) return;
    std::vector<int> g(cal::gaps.begin(), cal::gaps.end()); std::sort(g.begin(), g.end());
    auto q=[](const std::vector<int>&v,double a){ return v.empty()? -1 : v[(size_t)std::floor(a*(v.size()-1))]; };
    std::vector<int> lg; for (int x: g) if (x>5) lg.push_back(x);
    auto ms=[](const std::deque<float>&d,double &m,double &s){ m=0; s=0; if(d.empty()) return; for(float x:d) m+=x; m/=d.size(); for(float x:d) s+=(x-m)*(x-m); s=std::sqrt(s/std::max<size_t>(1,d.size()-1)); };
    double m1,s1,m2,s2; ms(cal::simSame,m1,s1); ms(cal::simDiff,m2,s2);
    double dp=(cal::simSame.size()>1 && cal::simDiff.size()>1)? (m1-m2)/std::sqrt(std::max(1e-9,(s1*s1+s2*s2)/2.0)) : -1;
    double nT=0,nF=0; for(int b=0;b<cal::NB;b++){ nT+=cal::confTrue[b]; nF+=cal::confFalse[b]; }
    fprintf(f,"{\"frame\":%d,\"fps\":%.3f,\"selfCal\":%d,",frame,envFps(),p.selfCal);
    fprintf(f,"\"gaps\":{\"n\":%zu,\"q50\":%d,\"q90\":%d,\"max\":%d,\"n_long\":%zu,\"q90_long\":%d},",g.size(),q(g,.5),q(g,.9),g.empty()?-1:g.back(),lg.size(),q(lg,.9));
    fprintf(f,"\"maxAge\":{\"base\":%d,\"used\":%d},",baseMaxAge(p),effMaxAge(p));
    fprintf(f,"\"prob\":{\"L\":%d,\"minHits\":%d},",cal::probL,effMinHits(p));
    fprintf(f,"\"reid\":{\"n_same\":%zu,\"n_diff\":%zu,\"mean_same\":%.3f,\"sd_same\":%.3f,\"mean_diff\":%.3f,\"sd_diff\":%.3f,\"dprime\":%.3f,\"w_used\":%.3f},",
            cal::simSame.size(),cal::simDiff.size(),m1,s1,m2,s2,dp,calWReid(p));
    fprintf(f,"\"sprt\":{\"n_true\":%.0f,\"n_false\":%.0f,\"true\":[",nT,nF);
    for(int b=0;b<cal::NB;b++) fprintf(f,"%s%.0f",b?",":"",cal::confTrue[b]);
    fprintf(f,"],\"false\":["); for(int b=0;b<cal::NB;b++) fprintf(f,"%s%.0f",b?",":"",cal::confFalse[b]);
    fprintf(f,"],\"active\":%s},",(nT>=p.calMinN && nF>=p.calMinN)?"true":"false");
    fprintf(f,"\"cls\":{\"total\":{"); bool first=true;
    for (auto &kv: cal::clsTotal){ fprintf(f,"%s\"%d\":%.0f",first?"":",",kv.first,kv.second); first=false; }
    fprintf(f,"},\"changes\":["); first=true;
    for (auto &kv: cal::clsCount) if (kv.first.first!=kv.first.second){ fprintf(f,"%s[%d,%d,%.0f,%.3f]",first?"":",",kv.first.first,kv.first.second,kv.second,calClassPen(p,kv.first.first,kv.first.second)); first=false; }
    fprintf(f,"]}}\n"); fclose(f);
}

// poziva se na pocetku svakog frejma: osvjezava izvedene parametre iz nakupljene statistike
template<class PP> static void calBeginFrame(const PP &p){
    calDump(p);
    cal::frameSim.clear();
    cal::maxAgeFrames=-1; cal::wReidNow=-1.f;
    if ((p.selfCal & 1) && (int)cal::gaps.size() >= p.calMinN) {
        std::vector<int> g(cal::gaps.begin(), cal::gaps.end());
        size_t k=(size_t)std::floor(0.9*(g.size()-1)); std::nth_element(g.begin(), g.begin()+k, g.end());
        int lo=(int)std::lround(1.0f*envFps()), hi=(int)std::lround(10.0f*envFps());
        cal::maxAgeFrames = std::min(hi, std::max(lo, (int)std::lround(p.calAgeK * g[k])));
    }
    if ((p.selfCal & 2) && (int)cal::simSame.size() >= p.calMinN && (int)cal::simDiff.size() >= p.calMinN) {
        auto ms=[](const std::deque<float>&d,double &m,double &v){ m=0; for(float x:d) m+=x; m/=d.size(); v=0; for(float x:d) v+=(x-m)*(x-m); v/=std::max<size_t>(1,d.size()-1); };
        double m1,v1,m2,v2; ms(cal::simSame,m1,v1); ms(cal::simDiff,m2,v2);
        double dprime=(m1-m2)/std::sqrt(std::max(1e-9,(v1+v2)/2.0));
        cal::wReidNow = (float)(p.wReid * std::min(1.0, std::max(0.0, dprime / p.calDRef)));
    }
}
static void calSim(int i,int j,float s){ cal::frameSim[{i,j}]=s; }
// nakon uparivanja: slicnost s vlastitim tragom -> "isti", s drugim potvrdenim tragovima -> "razliciti"
template<class TV> static void calCollect(const TV &tracks,int i,int j){
    if (!tracks[i].confirmed) return;
    auto it=cal::frameSim.find({i,j}); if (it!=cal::frameSim.end()) cal::push(cal::simSame,it->second,2000);
    int n=0;
    for (int k=0;k<(int)tracks.size() && n<3;k++){
        if (k==i || !tracks[k].confirmed) continue;
        auto jt=cal::frameSim.find({k,j}); if (jt!=cal::frameSim.end()){ cal::push(cal::simDiff,jt->second,4000); n++; }
    }
}
// praznina potvrdenog traga koji je ponovno uparen
static void calGap(bool confirmed,int framesSinceSeen){ if (confirmed && framesSinceSeen>1) cal::push(cal::gaps,framesSinceSeen-1,500); }
// pouzdanosti prvih 20 detekcija traga (za SPRT)
static void calEarly(uint64_t id,int hits,float conf){ if (hits<20) cal::early[id].push_back(conf); }
// oznaka traga: stvaran (dosegao 30 uparivanja) ili lazan (obrisan prije potvrde)
static void calLabel(uint64_t id,bool real){
    auto it=cal::early.find(id); if (it==cal::early.end()) return;
    for (float c: it->second) (real?cal::confTrue:cal::confFalse)[cal::bin(c)] += 1.0;
    cal::early.erase(it);
}
// SPRT: potvrdi kad je nakupljeni omjer izglednosti iznad praga A = log((1-alpha)/alpha)
template<class PP> static bool calConfirm(const PP &p,uint64_t id){
    if (!(p.selfCal & 4)) return false;
    double nT=0,nF=0; for (int b=0;b<cal::NB;b++){ nT+=cal::confTrue[b]; nF+=cal::confFalse[b]; }
    if (nT < p.calMinN || nF < p.calMinN) return false;
    auto it=cal::early.find(id); if (it==cal::early.end()) return false;
    double A=std::log((1.0-p.calAlpha)/p.calAlpha), s=0;
    for (float c: it->second){ int b=cal::bin(c);
        s += std::log(((cal::confTrue[b]+1.0)/(nT+cal::NB)) / ((cal::confFalse[b]+1.0)/(nF+cal::NB))); }
    return s >= A;
}
// prijelazi klase unutar potvrdenog traga uparenog i u proslom frejmu
static void calClass(bool use,int a,int b){ if (!use) return; cal::clsCount[{a,b}]+=1.0; cal::clsTotal[a]+=1.0; }
template<class PP> static float calClassPen(const PP &p,int a,int b){
    if (!(p.selfCal & 8)) return p.classPenalty;
    auto t=cal::clsTotal.find(a); if (t==cal::clsTotal.end() || t->second < p.calMinN) return p.classPenalty;
    auto c=cal::clsCount.find({a,b}); double n=(c==cal::clsCount.end()?0.0:c->second);
    double pr=(n+1.0)/(t->second+8.0);
    return (float)std::min(1.0, p.calKappa * (-std::log(pr)));
}
// ===================== kraj samokalibrirajuce asocijacije =====================


void Tracker::associate(std::vector<Detection> &dets)
{
    calBeginFrame(P);
    for (auto &d : dets) d.matchedTrack = -1;
    if (tracks.empty() || dets.empty()) return;

    int R = (int)tracks.size(), C = (int)dets.size();
    std::vector<std::vector<float>> cost(R, std::vector<float>(C, 1.0f));

    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            float clsAdd = 0.0f;
            if (tracks[i].classId != dets[j].classId) { if (P.classGate) { cost[i][j] = 1.0f; continue; } clsAdd = calClassPen(P, tracks[i].classId, dets[j].classId); }
            float c = P.wIou  * (1.0f - iou(tracks[i].box, dets[j].box))
                    + P.wSize * (1.0f - sizeSim(tracks[i].box, dets[j].box));
            c /= (P.wIou + P.wSize);
            // appearance clan -- ulazi ovisno o reidAfterN
            if (P.useReid && !dets[j].feature.empty() &&
                tracks[i].framesSinceSeen >= P.reidAfterN)
            {
                float sim = bestSim(tracks[i].gallery, dets[j].feature);
                if (sim >= -0.5f) {
                    float wR = calWReid(P);
                    c = (c * (P.wIou + P.wSize) + wR * (1.0f - sim))
                        / (P.wIou + P.wSize + wR);
                    calSim(i, j, sim);
                }
            }
            // temporalna inercija: popust ako je ovaj trag drzao ovu detekciju
            if (tracks[i].hadMatch &&
                iou(tracks[i].lastMatched, dets[j].box) > 0.5f)
                c -= P.inertia;
            cost[i][j] = c + clsAdd;
        }

    std::vector<int> assign = hungarian(cost);
    for (int i = 0; i < R; i++) {
        int j = assign[i];
        if (j < 0 || cost[i][j] > P.maxCost) continue;
        dets[j].matchedTrack = i;
        calCollect(tracks, i, j);
    }
}

// Ukloni detekcije koje se medusobno jako preklapaju (fragmentacija detektora).
// Zadrzava se ona s visom pouzdanoscu.
static void suppressOverlapping(std::vector<Detection> &dets, float thr)
{
    if (thr >= 1.0f || dets.size() < 2) return;
    std::vector<char> drop(dets.size(), 0);
    for (size_t i = 0; i < dets.size(); i++) {
        if (drop[i]) continue;
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (drop[j]) continue;
            if (iou(dets[i].box, dets[j].box) < thr) continue;
            if (dets[i].confidence >= dets[j].confidence) drop[j] = 1;
            else { drop[i] = 1; break; }
        }
    }
    size_t k = 0;
    for (size_t i = 0; i < dets.size(); i++)
        if (!drop[i]) dets[k++] = dets[i];
    dets.erase(dets.begin() + k, dets.end());
}

std::vector<std::pair<uint64_t,bool>> Tracker::update(std::vector<Detection> &dets)
{
    // 0. spoji fragmentirane detekcije
    dets.erase(std::remove_if(dets.begin(), dets.end(),
        [&](const Detection &d){ return d.confidence < P.minConf; }), dets.end());
    suppressOverlapping(dets, P.detNms);

    // 1. predikcija
    for (auto &t : tracks) {
        t.age++; t.framesSinceSeen++;
        t.hadMatch = (t.framesSinceSeen == 1);   // uparen bas proslog frejma
        t.predict();
    }

    calTrackStats(P, tracks);
    // 2. uparivanje na predikcijama
    associate(dets);

    // 3. korekcija uparenih
    std::vector<std::pair<uint64_t,bool>> out(dets.size(), {0,false});
    for (size_t j = 0; j < dets.size(); j++) {
        int i = dets[j].matchedTrack;
        if (i < 0) continue;
        Track &t = tracks[i];
        t.correct(dets[j].box);
        t.lastMatched     = dets[j].box;
        if (!dets[j].feature.empty()) {
            t.gallery.push_back(dets[j].feature);
            while ((int)t.gallery.size() > P.galleryMax) t.gallery.pop_front();
        }
        t.confidence      = dets[j].confidence;
        calClass(t.confirmed && t.hadMatch, t.classId, dets[j].classId);
        t.classId         = dets[j].classId;
        calGap(t.confirmed, t.framesSinceSeen);
        calEarly(t.id, t.hits, dets[j].confidence);
        t.framesSinceSeen = 0;
        t.hits++;
        if (t.hits == 30) calLabel(t.id, true);
        if (!t.confirmed && (calConfirm(P, t.id) || t.hits >= (t.confidence >= P.confHigh ? P.minHitsHigh : effMinHits(P)))) t.confirmed = true;
        out[j] = { t.id, t.confirmed };
    }

    // 4. novi tragovi
    Kalman proto;
    proto.wPos = P.wPos; proto.wVel = P.wVel; proto.wMeas = P.wMeas;
    for (size_t j = 0; j < dets.size(); j++) {
        if (dets[j].matchedTrack >= 0) continue;
        tracks.emplace_back(nextId++, dets[j].classId, dets[j].box,
                            dets[j].confidence, proto);
        if (!dets[j].feature.empty()) tracks.back().gallery.push_back(dets[j].feature);
        calEarly(tracks.back().id, 0, dets[j].confidence);
        if (calConfirm(P, tracks.back().id) || (dets[j].confidence >= P.confHigh ? P.minHitsHigh : effMinHits(P)) <= 1) tracks.back().confirmed = true;
        out[j] = { tracks.back().id, tracks.back().confirmed };
    }

    // 5. uklanjanje duplikata -- dva traga na istom objektu
    //    Zadrzava se onaj s vise uparivanja; kod izjednacenja stariji.
    if (P.dupIou < 1.0f) {
        std::vector<char> drop(tracks.size(), 0);
        for (size_t i = 0; i < tracks.size(); i++) {
            if (drop[i]) continue;
            for (size_t j = i + 1; j < tracks.size(); j++) {
                if (drop[j]) continue;
                if (P.classGate && tracks[i].classId != tracks[j].classId) continue;
                if (iou(tracks[i].box, tracks[j].box) < P.dupIou) continue;
                bool keepI = (tracks[i].hits > tracks[j].hits) ||
                             (tracks[i].hits == tracks[j].hits &&
                              tracks[i].age  >= tracks[j].age);
                if (keepI) drop[j] = 1;
                else      { drop[i] = 1; break; }
            }
        }
        size_t k = 0;
        for (size_t i = 0; i < tracks.size(); i++)
            if (!drop[i]) tracks[k++] = tracks[i];
        tracks.erase(tracks.begin() + k, tracks.end());
    }

    // samokalibracija: trag obrisan prije potvrde je lazan
    for (auto &t : tracks) if (t.framesSinceSeen > effMaxAge(P)) { if (!t.confirmed) calLabel(t.id, false); else { cal::early.erase(t.id); calDone(t.age - t.framesSinceSeen + 1, t.hits); } }

    // 6. brisanje starih
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
        [&](const Track &t){ return t.framesSinceSeen > effMaxAge(P); }), tracks.end());

    return out;
}
