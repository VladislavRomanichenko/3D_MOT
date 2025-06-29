#include <tracker.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>

Tracker3D::Tracker3D(const std::string& box_type,
                    const Config& config)
                    :
                    box_type_(box_type),
                    config_(config),
                    current_timestamp_(0),
                    label_seed_(0) {}


std::pair<Eigen::MatrixXd, std::vector<int>> Tracker3D::tracking(
    const Eigen::MatrixXd& bbs_3D,
    const Eigen::VectorXd* scores,
    const Eigen::Matrix4d* pose,
    int timestamp)
{
    current_bbs_ = std::make_unique<Eigen::MatrixXd>(bbs_3D);
    current_scores_ = scores ? std::make_unique<Eigen::VectorXd>(*scores) : nullptr;
    current_pose_ = pose ? std::make_unique<Eigen::Matrix4d>(*pose) : nullptr;
    current_timestamp_ = timestamp;

    trajectories_prediction();

    if (!current_bbs_ || current_bbs_->rows() == 0) {
        return {Eigen::MatrixXd::Zero(0, 7), std::vector<int>()};
    }

    std::vector<int> ids = association();
    auto [bbs, valid_ids] = trajectories_update_init(ids);

    return {bbs, valid_ids};
}


void Tracker3D::trajectories_prediction()
{
    if (active_trajectories_.empty()) {
        return;
    }

    std::vector<int> dead_track_ids;

    for (auto& [id, traj] : active_trajectories_) {
        if (traj.consecutive_missed_num >= config_.max_prediction_num) {
            dead_track_ids.push_back(id);
            continue;
        }

        if (traj.size() - traj.consecutive_missed_num == 1 &&
            traj.size() >= config_.max_prediction_num_for_new_object) {
            dead_track_ids.push_back(id);
            continue;
        }

        traj.state_prediction(current_timestamp_);
    }

    for (int id : dead_track_ids) {
        active_trajectories_.erase(id);
    }
}


std::map<int, std::vector<Eigen::VectorXd>> Tracker3D::predict_future_trajectories(int steps)
{
    std::map<int, std::vector<Eigen::VectorXd>> future_predictions;
    int last_timestamp = current_timestamp_;

    for (auto& [track_id, trajectory] : active_trajectories_) {
        std::vector<Eigen::VectorXd> future_states;
        Trajectory temp_trajectory = trajectory;

        for (int step = 1; step <= steps; ++step) {
            int next_timestamp = last_timestamp + step;
            temp_trajectory.state_prediction(next_timestamp);
            Eigen::VectorXd predicted_state = temp_trajectory.trajectory[next_timestamp].predicted_state;
            future_states.push_back(predicted_state.transpose());
        }

        future_predictions[track_id] = future_states;
    }

    return future_predictions;
}


std::pair<Eigen::MatrixXd, std::vector<int>> Tracker3D::compute_cost_map()
{
    std::vector<int> all_ids;
    std::vector<Eigen::VectorXd> all_predictions;
    std::vector<Eigen::VectorXd> all_detections;

    for (auto& [key, traj] : active_trajectories_) {
        all_ids.push_back(key);
        Eigen::VectorXd state = traj.trajectory[current_timestamp_].predicted_state;
        double pred_score = traj.trajectory[current_timestamp_].prediction_score;
        Eigen::VectorXd state_with_score(state.size() + 1);
        state_with_score << state, pred_score;
        all_predictions.push_back(state_with_score);
    }

    for (int i = 0; i < current_bbs_->rows(); ++i) {
        Eigen::VectorXd box = current_bbs_->row(i);
        double score = current_scores_ ? (*current_scores_)[i] : 0.0;

        Trajectory new_tra(box, score, current_timestamp_, 1, config_);
        Eigen::VectorXd state = new_tra.trajectory[current_timestamp_].predicted_state;
        all_detections.push_back(state);
    }

    int det_len = all_detections.size();
    int pred_len = all_predictions.size();

    if (det_len == 0 || pred_len == 0) {
        return {Eigen::MatrixXd::Zero(det_len, pred_len), all_ids};
    }

    Eigen::MatrixXd all_detections_mat(det_len, all_detections[0].size());
    Eigen::MatrixXd all_predictions_mat(pred_len, all_predictions[0].size());
    for (int i = 0; i < det_len; ++i) all_detections_mat.row(i) = all_detections[i];
    for (int i = 0; i < pred_len; ++i) all_predictions_mat.row(i) = all_predictions[i];

    Eigen::MatrixXd det_positions = all_detections_mat.leftCols(3);
    Eigen::MatrixXd pred_positions = all_predictions_mat.leftCols(3);

    Eigen::MatrixXd dis(det_len, pred_len);
    for (int i = 0; i < det_len; ++i) {
        for (int j = 0; j < pred_len; ++j) {
            dis(i, j) = (det_positions.row(i) - pred_positions.row(j)).norm();
        }
    }

    Eigen::VectorXd pred_scores(pred_len);
    for (int j = 0; j < pred_len; ++j) {
        pred_scores(j) = all_predictions[j].tail(1)(0);
    }
    assert(pred_scores.size() == dis.cols());
    Eigen::MatrixXd cost = dis * pred_scores.asDiagonal();

    return {cost, all_ids};
}


std::vector<int> Tracker3D::association()
{
    if (active_trajectories_.empty()) {
        std::vector<int> ids(current_bbs_->rows());
        for (int i = 0; i < ids.size(); ++i) {
            ids[i] = label_seed_++;
        }
        return ids;
    }

    auto [cost_map, all_ids] = compute_cost_map();
    std::vector<int> ids(current_bbs_->rows(), -1);

    for (int i = 0; i < current_bbs_->rows(); ++i) {
        double min_cost = cost_map.row(i).minCoeff();
        Eigen::Index arg_min;
        cost_map.row(i).minCoeff(&arg_min);

        if (min_cost < config_.association_threshold) {
            ids[i] = all_ids[arg_min];
            cost_map.col(arg_min).setConstant(100000);
        } else {
            ids[i] = label_seed_++;
        }
    }

    return ids;
}


std::pair<Eigen::MatrixXd, std::vector<int>> Tracker3D::trajectories_update_init(const std::vector<int>& ids)
{
    assert(ids.size() == current_bbs_->rows());
    std::vector<Eigen::VectorXd> valid_bbs;
    std::vector<int> valid_ids;

    for (int i = 0; i < current_bbs_->rows(); ++i) {
        int label = ids[i];
        Eigen::VectorXd box = current_bbs_->row(i);
        double score = current_scores_ ? (*current_scores_)[i] : 0.0;

        auto it = active_trajectories_.find(label);
        if (it != active_trajectories_.end() && score > config_.update_score) {
            it->second.state_update(box, score, current_timestamp_);
            valid_bbs.push_back(box);
            valid_ids.push_back(label);
        } else if (it == active_trajectories_.end() && score > config_.init_score) {
            active_trajectories_.emplace(label, Trajectory(box, score, current_timestamp_, label, config_));
            valid_bbs.push_back(box);
            valid_ids.push_back(label);
        }
    }

    if (valid_bbs.empty()) {
        return {Eigen::MatrixXd::Zero(0, 7), std::vector<int>()};
    }

    Eigen::MatrixXd bbs(valid_bbs.size(), 7);
    for (int i = 0; i < valid_bbs.size(); ++i) {
        bbs.row(i) = valid_bbs[i];
    }
    return {bbs, valid_ids};
}


std::map<int, Trajectory> Tracker3D::post_processing(const Config& config)
{
    std::map<int, Trajectory> tra;

    for (auto& [key, track] : active_trajectories_) {
        track.filtering(config);
        tra.emplace(key, track);
    }

    return tra;
}

bool is_static_trajectory(const Trajectory& traj, double position_threshold, double speed_threshold, double frame_time) {
    if (traj.trajectory.size() < 2) return false;
    auto first = traj.trajectory.begin();
    auto last = std::prev(traj.trajectory.end());
    if (first == last) return false;
    if (first->second.updated_state.size() < 3 || last->second.updated_state.size() < 3) return false;
    Eigen::Vector3d first_pos = first->second.updated_state.head(3);
    Eigen::Vector3d last_pos = last->second.updated_state.head(3);
    double dist = (last_pos - first_pos).norm();
    if (dist < position_threshold) return true;
    double total_speed = 0.0;
    int count = 0;
    auto it = traj.trajectory.begin();
    auto prev = it++;
    for (; it != traj.trajectory.end(); ++it, ++prev) {
        if (prev->second.updated_state.size() < 3 || it->second.updated_state.size() < 3) continue;
        Eigen::Vector3d p1 = prev->second.updated_state.head(3);
        Eigen::Vector3d p2 = it->second.updated_state.head(3);
        double d = (p2 - p1).norm();
        double speed = d / frame_time;
        total_speed += speed;
        ++count;
    }
    if (count == 0) return false;
    double avg_speed = total_speed / count;
    return avg_speed < speed_threshold;
}