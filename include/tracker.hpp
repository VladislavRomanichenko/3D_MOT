#pragma once

#include <trajectory.hpp>
#include <config.hpp>

#include <Eigen/Dense>

#include <vector>
#include <map>
#include <memory>

class Tracker3D {
public:
    Tracker3D(bool tracking_features = false,
              bool bb_as_features = false,
              const std::string& box_type = "Centerpoint",
              const Config& config = Config());

    std::pair<Eigen::MatrixXd, std::vector<int>> tracking(
        const Eigen::MatrixXd& bbs_3D = Eigen::MatrixXd(),
        const Eigen::MatrixXd* features = nullptr,
        const Eigen::VectorXd* scores = nullptr,
        const Eigen::Matrix4d* pose = nullptr,
        int timestamp = 0);

    void trajectories_prediction();

    std::map<int, std::vector<Eigen::VectorXd>> predict_future_trajectories(int steps = 0);

    std::map<int, Trajectory> post_processing(const Config& config);

private:
    Config config_;
    int current_timestamp_;
    std::unique_ptr<Eigen::MatrixXd> current_bbs_;
    std::unique_ptr<Eigen::MatrixXd> current_features_;
    std::unique_ptr<Eigen::VectorXd> current_scores_;
    std::unique_ptr<Eigen::Matrix4d> current_pose_;
    bool tracking_features_;
    bool bb_as_features_;
    std::string box_type_;
    int label_seed_;
    std::map<int, Trajectory> active_trajectories_;
    std::map<int, Trajectory> dead_trajectories_;

    std::pair<Eigen::MatrixXd, std::vector<int>> compute_cost_map();
    std::vector<int> association();
    std::pair<Eigen::MatrixXd, std::vector<int>> trajectories_update_init(const std::vector<int>& ids);

    Eigen::MatrixXd convert_bbs_type(const Eigen::MatrixXd& bbs, const std::string& box_type);
    Eigen::MatrixXd register_bbs(const Eigen::MatrixXd& bbs, const Eigen::Matrix4d* pose);
};
