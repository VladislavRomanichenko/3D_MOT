#include <cstdio>
#include<tracker.hpp>
//#include <utils.hpp>

#include <iostream>

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

    Tracker3D tracker("default_box_type", config);

    Eigen::MatrixXd init_bbs(1, 7);  
    init_bbs << 1.0, 1.0, 1.0, 4.0, 5.0, 6.0, 0.0;

    Eigen::VectorXd init_scores(1);  
    init_scores << 0.9;

    std::cout << "Tracking at timestamp 0:\n";
    auto [bbs_t0, ids_t0] = tracker.tracking(init_bbs, &init_scores, nullptr, 0);
    std::cout << "Bounding boxes:\n" << bbs_t0 << "\n";
    std::cout << "IDs: ";
    for (int id : ids_t0) std::cout << id << " ";
    std::cout << "\n\n";

    Eigen::MatrixXd new_bbs(1, 7);
    new_bbs << 1.1, 2.1, 3.1, 4.0, 5.0, 6.0, 0.0;

    Eigen::VectorXd new_scores(1);
    new_scores << 0.95;

    std::cout << "Tracking at timestamp 1:\n";
    auto [bbs_t1, ids_t1] = tracker.tracking(new_bbs, &new_scores, nullptr, 1);
    std::cout << "Bounding boxes:\n" << bbs_t1 << "\n";
    std::cout << "IDs: ";
    for (int id : ids_t1) std::cout << id << " ";
    std::cout << "\n\n";

    std::cout << "Predicting future trajectories for 2 steps:\n";
    auto future_trajs = tracker.predict_future_trajectories(2);
    for (const auto& [id, states] : future_trajs) {
        std::cout << "Trajectory ID " << id << ":\n";
        for (const auto& state : states) {
            std::cout << "  " << state.transpose() << "\n";
        }
    }

    return 0;
}
