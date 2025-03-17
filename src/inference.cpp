#include <cstdio>
#include <trajectory.hpp>
//#include <utils.hpp>
#include <config.hpp>

int main(int argc, char ** argv)
{
  (void) argc;
  (void) argv;

  //Config 
  Config config;
  
  config.state_func_covariance = 1;
  config.measure_func_covariance = 0.01;
  config.prediction_score_decay = 0.02;
  config.LiDAR_scanning_frequency = 10;

  config.max_prediction_num = 12;
  config.max_prediction_num_for_new_object = 10;

  config.input_score = 0;
  config.init_score = 0;
  config.update_score = 0;
  config.post_score = 0.55;

  config.latency = 0;

  Eigen::VectorXd init_bb(7);
  init_bb << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0;

  Trajectory traj(init_bb, nullptr, 0.9, 0, 1, true, false, config);

  traj.state_prediction(1);

  Eigen::VectorXd bb(7);
  bb << 1.1, 2.1, 3.1, 4.0, 5.0, 6.0, 0.0;
  traj.state_update(bb, nullptr, 0.95, 1);

  return 0;
}
