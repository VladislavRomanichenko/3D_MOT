#include <trajectory.hpp>
#include <stdexcept>
#include <iostream>

Trajectory::Trajectory(const Eigen::VectorXd& init_bb,
                       double init_score,
                       int init_timestamp,
                       int label,
                       const Config& config)
                       :
                       init_bb(init_bb),
                       init_score(init_score),
                       init_timestamp(init_timestamp),
                       label(label),
                       config(config),
                       scanning_interval(1.0 / config.LiDAR_scanning_frequency),
                       consecutive_missed_num(0),
                       first_updated_timestamp(init_timestamp),
                       last_updated_timestamp(init_timestamp),
                       tracking_bb_size(true)
{
    if (init_bb.size() < 7) {
        throw std::invalid_argument("init_bb must have at least 7 elements");
    }

    track_dim = compute_track_dim();
    init_parameters();
    init_trajectory();
}


double Trajectory::sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

int Trajectory::size() const

{
    return trajectory.size();
}


void Trajectory::state_prediction(int timestamp)
{
    int previous_timestamp = timestamp - 1;
    auto it = trajectory.find(previous_timestamp);
    if (it == trajectory.end()) {
        throw std::runtime_error("Previous timestamp not found in trajectory");
    }

    Object& prev_ob = it->second;
    Eigen::VectorXd prev_state = (prev_ob.updated_state.size() > 0) ? prev_ob.updated_state : prev_ob.predicted_state;
    Eigen::MatrixXd prev_cov = (prev_ob.updated_covariance.size() > 0) ? prev_ob.updated_covariance : prev_ob.predicted_covariance;
    double prev_score = prev_ob.prediction_score;

    double current_score = prev_score * (1 - config.prediction_score_decay);
    if (trajectory.find(timestamp - 1) != trajectory.end() && trajectory[timestamp - 1].updated_state.size() > 0) {
        current_score = prev_score * (1 - config.prediction_score_decay * 15);
    }

    Eigen::VectorXd predicted_state = A * prev_state;
    Eigen::MatrixXd predicted_cov = A * prev_cov * A.transpose() + Q;

    Object new_ob;
    new_ob.predicted_state = predicted_state;
    new_ob.predicted_covariance = predicted_cov;
    new_ob.prediction_score = current_score;
    trajectory[timestamp] = new_ob;
    consecutive_missed_num++;
}


void Trajectory::state_update(const Eigen::VectorXd& bb,
                              double score,
                              int timestamp)
{
    if (bb.size() < 7) {
        throw std::invalid_argument("bb must have at least 7 elements");
    }
    auto it = trajectory.find(timestamp);
    if (it == trajectory.end()) {
        throw std::runtime_error("Timestamp not found in trajectory");
    }

    Eigen::VectorXd detected_state = Eigen::VectorXd::Zero(track_dim - 6);
    detected_state.head(3) = bb.head(3);
    if (tracking_bb_size) {
        detected_state.segment(3, 4) = bb.segment(3, 4);
    }

    Object& curr_ob = it->second;
    Eigen::VectorXd pred_state = curr_ob.predicted_state;
    Eigen::MatrixXd pred_cov = curr_ob.predicted_covariance;

    Eigen::MatrixXd temp = B * pred_cov * B.transpose() + P;
    Eigen::MatrixXd KF_gain = pred_cov * B.transpose() * temp.inverse();

    Eigen::VectorXd updated_state = pred_state + KF_gain * (detected_state - B * pred_state);
    Eigen::MatrixXd updated_cov = (Eigen::MatrixXd::Identity(track_dim, track_dim) - KF_gain * B) * pred_cov;

    if (size() == 2) {
        auto prev_it = trajectory.find(timestamp - 1);
        if (prev_it != trajectory.end()) {
            updated_state = H * detected_state + K * (H * detected_state - prev_it->second.updated_state);
        }
    }

    curr_ob.updated_state = updated_state;
    curr_ob.updated_covariance = updated_cov;
    curr_ob.detected_state = detected_state;

    if (consecutive_missed_num > 1) {
        curr_ob.prediction_score = 1.0;
    } else if (trajectory.find(timestamp - 1) != trajectory.end() && trajectory[timestamp - 1].updated_state.size() > 0) {
        curr_ob.prediction_score += config.prediction_score_decay * 10 * sigmoid(score);
    } else {
        curr_ob.prediction_score += config.prediction_score_decay * sigmoid(score);
    }

    curr_ob.score = score;
    consecutive_missed_num = 0;
    last_updated_timestamp = timestamp;
}


void Trajectory::filtering(const Config& config)
{
    int wind_size = static_cast<int>(config.LiDAR_scanning_frequency * config.latency);

    if (wind_size < 0) {
        double detected_num = 0.00001;
        double score_sum = 0.0;

        for (auto& pair : trajectory) {
            Object& ob = pair.second;
            if (ob.score >= 0) {
                detected_num += 1;
                score_sum += ob.score;
            }
            if (pair.first >= first_updated_timestamp && pair.first <= last_updated_timestamp && ob.updated_state.size() == 0) {
                ob.updated_state = ob.predicted_state;
            }
        }

        double score = score_sum / detected_num;

        for (auto& pair : trajectory) {
            pair.second.score = score;
        }
    } else {
        for (auto it = trajectory.begin(); it != trajectory.end(); ++it) {
            int key = it->first;
            int min_key = key - wind_size;
            int max_key = key + wind_size;
            double detected_num = 0.00001;
            double score_sum = 0.0;

            for (int key_i = min_key; key_i <= max_key; ++key_i) {
                auto it_inner = trajectory.find(key_i);
                if (it_inner != trajectory.end()) {
                    Object& ob = it_inner->second;
                    if (ob.score >= 0) {
                        detected_num += 1;
                        score_sum += ob.score;
                    }
                    if (key_i >= first_updated_timestamp && key_i <= last_updated_timestamp && ob.updated_state.size() == 0) {
                        ob.updated_state = ob.predicted_state;
                    }
                }
            }

            double score = score_sum / detected_num;

            if (wind_size != 0) {
                it->second.score = score;
            }
        }
    }
}


int Trajectory::compute_track_dim()
{
    int dim = 9; // x, y, z, vx, vy, vz, ax, ay, az
    if (tracking_bb_size) {
        dim += 4; // w, h, l, yaw
    }
    return dim;
}


void Trajectory::init_parameters()
{
    A = Eigen::MatrixXd::Identity(track_dim, track_dim);
    Q = Eigen::MatrixXd::Identity(track_dim, track_dim) * config.state_func_covariance;
    P = Eigen::MatrixXd::Identity(track_dim - 6, track_dim - 6) * config.measure_func_covariance;
    B = Eigen::MatrixXd::Zero(track_dim - 6, track_dim);

    B.topRows(3) = A.topRows(3);
    if (track_dim > 9) {
        B.bottomRows(track_dim - 9) = A.bottomRows(track_dim - 9);
    }

    Eigen::Matrix3d velo = Eigen::Matrix3d::Identity() * scanning_interval;
    Eigen::Matrix3d acce = Eigen::Matrix3d::Identity() * 0.5 * scanning_interval * scanning_interval;

    A.block<3, 3>(0, 3) = velo;
    A.block<3, 3>(3, 6) = velo;
    A.block<3, 3>(0, 6) = acce;

    H = B.transpose();
    K = Eigen::MatrixXd::Zero(track_dim, track_dim);
    K(3, 0) = scanning_interval;
    K(4, 1) = scanning_interval;
    K(5, 2) = scanning_interval;
}


void Trajectory::init_trajectory()
{
    Eigen::VectorXd detected_state = Eigen::VectorXd::Zero(track_dim - 6);
    detected_state.head(3) = init_bb.head(3);
    if (tracking_bb_size) {
        detected_state.segment(3, 4) = init_bb.segment(3, 4);
    }

    Eigen::MatrixXd update_cov = Eigen::MatrixXd::Identity(track_dim, track_dim) * 0.01;
    Eigen::VectorXd update_state = H * detected_state;

    Object obj;
    obj.updated_state = update_state;
    obj.predicted_state = update_state;
    obj.detected_state = detected_state;
    obj.updated_covariance = update_cov;
    obj.predicted_covariance = update_cov;
    obj.prediction_score = 1.0;
    obj.score = init_score;

    trajectory[init_timestamp] = obj;
}