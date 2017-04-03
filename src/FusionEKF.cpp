#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
    
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);
  MatrixXd P_ = MatrixXd(4, 4);
  MatrixXd F_ = MatrixXd(4, 4);
  MatrixXd Q_ = MatrixXd(4, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
  
  //measurement matrix - laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  
  //state covariance matrix P
  P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;
    
    
  VectorXd x_ = VectorXd(4);
  
  x_ << 1, 1, 1, 1;
    
  ekf_.Init(x_, P_, F_, H_laser_, R_laser_, Q_);

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;
      
    long double px = 0;
    long double py = 0;
    long double vx = 0;
    long double vy = 0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      
      //initialize rho, phi, rho_dot and calculate px,py,vx,vy
      long double rho = measurement_pack.raw_measurements_[0];
      long double phi = measurement_pack.raw_measurements_[1];
      long double rho_dot = measurement_pack.raw_measurements_[2];
        
      px = rho * cos(phi);
      py = rho * sin(phi);
      vx = rho_dot * cos(phi);
      vy = rho_dot * sin(phi);
      
      ekf_.x_ << px, py, vx, vy;
        
      //check if values are zero and set to 0.0001 if it's the case
      if(fabs(px) < 0.0001){
          px = 0.0001;

      }
      if(fabs(py) < 0.0001){
          py = 0.0001;
          
      }
      if(fabs(vx) < 0.0001){
          vx = 0.0001;
          
      }
      if(fabs(vy) < 0.0001){
          vy = 0.0001;
          
      }
      
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
       
      //initialize px,py from laser data
      px = measurement_pack.raw_measurements_[0];
      py = measurement_pack.raw_measurements_[1];
      
      ekf_.x_ << px, py, 0, 0;
    }

    previous_timestamp_ = measurement_pack.timestamp_;
    
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  long double current_timestamp = measurement_pack.timestamp_;
  long double time_diff = (current_timestamp - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
    
  if (time_diff > 0.001) {
  //initial transition matrix F with time difference
    ekf_.F_ << 1, 0, time_diff, 0,
               0, 1, 0, time_diff,
               0, 0, 1, 0,
               0, 0, 0, 1;
  
  //set noise for Q matrix to 9 as recommended
    long double noise_ax = 9;
    long double noise_ay = 9;
    
    long double dt_4 = time_diff*time_diff*time_diff*time_diff;
    long double dt_3 = time_diff*time_diff*time_diff;
    long double dt_2 = time_diff*time_diff;
    
    ekf_.Q_ << dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
               0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
               dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
               0, dt_3/2*noise_ay, 0, dt_2*noise_ax;

    ekf_.Predict();
  }
    
  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      
      // Radar updates
      ekf_.R_ = R_radar_;
      Tools tools;
      ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
      ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
      
      // Laser updates
      ekf_.R_ = R_laser_;
      ekf_.H_ = H_laser_;
      ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
