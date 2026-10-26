// Samostalna provjera: ucitaj engine, pokreni na lazenim podacima.
// Ne dira DeepStream ni GPU surface.
#include "reid.h"
#include <cstdio>
#include <vector>
#include <cmath>

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "/workspace/osnet_vessel.engine";
    ReID r;
    if (!r.init(path, 8)) { printf("INIT PAO\n"); return 1; }

    int N = 4, F = r.featSize();
    std::vector<float> in((size_t)N*3*256*128, 0.0f);
    // dvije razlicite "slike": prva polovica 0.5, druga -0.5
    for (size_t i = 0; i < in.size()/2; i++) in[i] = 0.5f;
    for (size_t i = in.size()/2; i < in.size(); i++) in[i] = -0.5f;

    std::vector<float> out((size_t)N*F, 0.0f);
    if (!r.infer(in.data(), N, out.data())) { printf("INFER PAO\n"); return 1; }

    printf("\nizlaz %d x %d\n", N, F);
    for (int i = 0; i < N; i++) {
        double norm = 0.0;
        for (int k = 0; k < F; k++) norm += (double)out[i*F+k]*out[i*F+k];
        printf("  vektor %d:  norma=%.4f   prvih 5: %.3f %.3f %.3f %.3f %.3f\n",
               i, std::sqrt(norm), out[i*F+0], out[i*F+1], out[i*F+2], out[i*F+3], out[i*F+4]);
    }
    // kosinus izmedu 0 i 1 (isti ulaz) te 0 i 2 (razlicit ulaz)
    auto cos=[&](int a,int b){ double s=0; for(int k=0;k<F;k++) s+=(double)out[a*F+k]*out[b*F+k]; return s; };
    printf("\ncos(0,1) = %.4f   (isti ulaz, ocekivano ~1)\n", cos(0,1));
    printf("cos(0,3) = %.4f   (razlicit ulaz)\n", cos(0,3));
    r.deinit();
    return 0;
}
