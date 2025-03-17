#include <cstdio>
#include <trajectory.hpp>

int main(int argc, char ** argv)
{
  (void) argc;
  (void) argv;

  Config config;
  config.LiDAR_scanning_frequency = 10.0;
  config.state_func_covariance = 0.1;
  config.measure_func_covariance = 0.1;
  config.prediction_score_decay = 0.01;
  config.latency = 0.1;

  Eigen::VectorXd init_bb(7);
  init_bb << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0;

  Trajectory traj(init_bb, nullptr, 0.9, 0, 1, true, false, config);

  traj.state_prediction(1);

  Eigen::VectorXd bb(7);
  bb << 1.1, 2.1, 3.1, 4.0, 5.0, 6.0, 0.0;
  traj.state_update(bb, nullptr, 0.95, 1);

  return 0;
}
