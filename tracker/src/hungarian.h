#ifndef MYTRACKER_HUNGARIAN_H
#define MYTRACKER_HUNGARIAN_H
#include <vector>
// Returns the column index for each row, or -1 if the row is unmatched.
// cost is a [rows][cols] matrix. Minimises the total cost.
std::vector<int> hungarian(const std::vector<std::vector<float>> &cost);
#endif
