#ifndef MYTRACKER_KALMAN_H
#define MYTRACKER_KALMAN_H
#include <Eigen/Dense>

// State: [x, y, a, h, vx, vy, va, vh]
//   x, y  centre point
//   a     aspect ratio  w/h
//   h     height
// Measurement: [x, y, a, h]
// The noise is scaled by the box height, as in the NVIDIA implementation.
class Kalman {
public:
    Eigen::Matrix<double,8,1> x;
    Eigen::Matrix<double,8,8> P;

    // Noise weights -- these are the candidates for the decomposition in Phase 1.
    double wPos = 1.0/20.0;   // process, position
    double wVel = 1.0/160.0;  // process, velocity
    double wMeas = 1.0/20.0;  // measurement

    void init(double cx, double cy, double a, double h);
    void predict();
    void update(double cx, double cy, double a, double h);

    void toBox(double &bx, double &by, double &bw, double &bh) const;
};

#endif
