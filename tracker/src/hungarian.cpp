// Hungarian / Jonker-Volgenant, O(n^3) version with potentials.
// Works on a rectangular matrix; the missing entries are filled with a large cost.
#include "hungarian.h"
#include <limits>
#include <algorithm>

static const float INF = std::numeric_limits<float>::max() / 4;

std::vector<int> hungarian(const std::vector<std::vector<float>> &costIn)
{
    int rows = (int)costIn.size();
    if (rows == 0) return {};
    int cols = (int)costIn[0].size();
    if (cols == 0) return std::vector<int>(rows, -1);

    int n = std::max(rows, cols);
    std::vector<std::vector<float>> a(n + 1, std::vector<float>(n + 1, 0.0f));
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            a[i][j] = (i <= rows && j <= cols) ? costIn[i-1][j-1] : 0.0f;

    std::vector<float> u(n + 1, 0.0f), v(n + 1, 0.0f);
    std::vector<int>   p(n + 1, 0), way(n + 1, 0);

    for (int i = 1; i <= n; i++) {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(n + 1, INF);
        std::vector<char>  used(n + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            float delta = INF;
            for (int j = 1; j <= n; j++) if (!used[j]) {
                float cur = a[i0][j] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= n; j++) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else         { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    std::vector<int> assign(rows, -1);
    for (int j = 1; j <= n; j++) {
        int i = p[j];
        if (i >= 1 && i <= rows && j <= cols) assign[i-1] = j-1;
    }
    return assign;
}
