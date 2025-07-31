#pragma once

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <Eigen/Dense>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include <diagnostic_updater/diagnostic_updater.hpp>
#include <diagnostic_updater/publisher.hpp>

#include "tracker.hpp"
#include "utils.hpp"

//----------------CUSTOM MESSAGE-----------------------
#include "objects_msgs/msg/object.hpp"
#include "objects_msgs/msg/object_array.hpp"
#include <objects_msgs/msg/dynamic_object.hpp>
#include <objects_msgs/msg/dynamic_object_array.hpp>


class Tracker : public rclcpp::Node {
public:
    Tracker();

private:
    diagnostic_updater::Updater diag_updater;
    std::unique_ptr<diagnostic_updater::TopicDiagnostic> diag_input;
    std::unique_ptr<diagnostic_updater::TopicDiagnostic> diag_output;
    double diag_input_min_freq = 0;
    double diag_input_max_freq = 0;
    double diag_output_min_freq = 0;
    double diag_output_max_freq = 0;
    int diag_lvl = diagnostic_msgs::msg::DiagnosticStatus::OK;
    std::string diag_msg = "OK";

    void update_diagnostic_status(diagnostic_updater::DiagnosticStatusWrapper& stat);


    void tracker_callback(const objects_msgs::msg::ObjectArray::SharedPtr objects);
    Eigen::VectorXd dynamic_msg_to_eigen_array(const objects_msgs::msg::Object& object);
    objects_msgs::msg::DynamicObject eigen_array_to_dynamic_msg(const Eigen::VectorXd& array);

    Config config_;
    bool tracker_flag_;
    Tracker3D tracker_;

    rclcpp::Subscription<objects_msgs::msg::ObjectArray>::SharedPtr subscriber_;
    rclcpp::Publisher<objects_msgs::msg::DynamicObjectArray>::SharedPtr publisher_;

    rclcpp::Duration timeout_{rclcpp::Duration::from_seconds(0.01)};
    std::string target_frame_;
    std::string frame_odom_;
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Time prev_time_;
    int num_future_states_;
    int min_trajectory_history_;
    bool tracking_in_odom_;
    int timestamp_for_tracker_;
    int current_timestamp_;
    std::string prev_frame_id_;  

    bool evaluation_mode_;
    std::string frame_id_param_;

    void save_result(const std::string& filename, int frame, int track_id, const std::string& type,
                                double truncation, double occlusion, double alpha,
                                double h, double w, double l,
                                double x, double y, double z, double ry, double score);
};

bool is_static_trajectory(const Trajectory& traj, double position_threshold, double speed_threshold, double frame_time);
