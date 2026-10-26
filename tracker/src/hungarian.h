#ifndef MYTRACKER_HUNGARIAN_H
#define MYTRACKER_HUNGARIAN_H
#include <vector>
// Vraca za svaki redak indeks stupca, ili -1 ako nije uparen.
// cost je matrica [rows][cols]. Minimizira ukupnu cijenu.
std::vector<int> hungarian(const std::vector<std::vector<float>> &cost);
#endif
