#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
    
    //predict
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
    
    //update
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;
    
    x_ = x_ + (K * y);
    long x_size = x_.size();
    
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    
    //update extended kalman filter
    long double px = x_[0];
    long double py = x_[1];
    long double vx = x_[2];
    long double vy = x_[3];
    
    //check for zeros
    if (fabs(px) < 0.0001) {
        px = 0.0001;
    }
    
    if (fabs(py) < 0.0001) {
        py = 0.0001;
    }
    
    if (fabs(vx) < 0.0001) {
        vx = 0.0001;
    }
    
    if (fabs(vy) < 0.0001) {
        vy = 0.0001;
    }
    
    long double rho = sqrt(px * px + py * py);
    
    if (fabs(rho) < 0.0001) {
        rho = 0.0001;
    }
    
    long double phi = atan(py / px);
    
    if (fabs(phi) < 0.0001) {
        phi = 0.0001;
    }
    long double rho_dot = (px * vx + py * vy) / rho;
    
    if (fabs(rho_dot) < 0.0001) {
        rho_dot = 0.0001;
    }
    
    //z vector created from rho, phi, rho_dot
    VectorXd z_pred(3);
    z_pred << rho, phi, rho_dot;
    
    //update
    VectorXd y = z - z_pred;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;
    

    x_ = x_ + (K * y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
}
