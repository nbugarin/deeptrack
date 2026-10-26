#include "kalman.h"

void Kalman::init(double cx, double cy, double a, double h)
{
    x.setZero();
    x(0)=cx; x(1)=cy; x(2)=a; x(3)=h;

    P.setZero();
    double s1 = 2.0 * wMeas * h;
    double s2 = 10.0 * wVel * h;
    P(0,0)=s1*s1; P(1,1)=s1*s1; P(2,2)=1e-2; P(3,3)=s1*s1;
    P(4,4)=s2*s2; P(5,5)=s2*s2; P(6,6)=1e-5; P(7,7)=s2*s2;
}

void Kalman::predict()
{
    double h = x(3);
    // F: constant velocity
    Eigen::Matrix<double,8,8> F = Eigen::Matrix<double,8,8>::Identity();
    for (int i = 0; i < 4; i++) F(i, i+4) = 1.0;

    double sp = wPos * h, sv = wVel * h;
    Eigen::Matrix<double,8,8> Q = Eigen::Matrix<double,8,8>::Zero();
    Q(0,0)=sp*sp; Q(1,1)=sp*sp; Q(2,2)=1e-4; Q(3,3)=sp*sp;
    Q(4,4)=sv*sv; Q(5,5)=sv*sv; Q(6,6)=1e-10; Q(7,7)=sv*sv;

    x = F * x;
    P = F * P * F.transpose() + Q;
}

void Kalman::update(double cx, double cy, double a, double h)
{
    Eigen::Matrix<double,4,8> H = Eigen::Matrix<double,4,8>::Zero();
    for (int i = 0; i < 4; i++) H(i,i) = 1.0;

    double sm = wMeas * x(3);
    Eigen::Matrix<double,4,4> R = Eigen::Matrix<double,4,4>::Zero();
    R(0,0)=sm*sm; R(1,1)=sm*sm; R(2,2)=1e-2; R(3,3)=sm*sm;

    Eigen::Matrix<double,4,1> z; z << cx, cy, a, h;
    Eigen::Matrix<double,4,1> y = z - H * x;
    Eigen::Matrix<double,4,4> S = H * P * H.transpose() + R;
    Eigen::Matrix<double,8,4> K = P * H.transpose() * S.inverse();

    x = x + K * y;
    Eigen::Matrix<double,8,8> I = Eigen::Matrix<double,8,8>::Identity();
    P = (I - K * H) * P;

    if (x(3) < 1.0)  x(3) = 1.0;    // the height must not drop to zero
    if (x(2) < 0.05) x(2) = 0.05;
}

void Kalman::toBox(double &bx, double &by, double &bw, double &bh) const
{
    bh = x(3);
    bw = x(2) * bh;
    bx = x(0) - bw * 0.5;
    by = x(1) - bh * 0.5;
}
