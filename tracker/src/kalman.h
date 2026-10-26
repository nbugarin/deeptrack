#ifndef MYTRACKER_KALMAN_H
#define MYTRACKER_KALMAN_H
#include <Eigen/Dense>

// Stanje: [x, y, a, h, vx, vy, va, vh]
//   x, y  sredisnja tocka
//   a     omjer stranica  w/h
//   h     visina
// Mjerenje: [x, y, a, h]
// Sum se skalira visinom okvira, kao u NVIDIA-inoj implementaciji.
class Kalman {
public:
    Eigen::Matrix<double,8,1> x;
    Eigen::Matrix<double,8,8> P;

    // Tezine suma -- ovo su kandidati za razlaganje u Fazi 1.
    double wPos = 1.0/20.0;   // proces, polozaj
    double wVel = 1.0/160.0;  // proces, brzina
    double wMeas = 1.0/20.0;  // mjerenje

    void init(double cx, double cy, double a, double h);
    void predict();
    void update(double cx, double cy, double a, double h);

    void toBox(double &bx, double &by, double &bw, double &bh) const;
};

#endif
