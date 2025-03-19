#pragma once

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Dense>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include "tracker.hpp"  // Assumes Tracker3D is defined here
#include "utils.hpp"   // Assumes transform_object is defined here

//----------------CUSTOM MESSAGE-----------------------
#include "objects_msgs/msg/object.hpp"
#include "objects_msgs/msg/object_array.hpp"
#include <objects_msgs/msg/dynamic_object.hpp>
#include <objects_msgs/msg/dynamic_object_array.hpp>


class Tracker : public rclcpp::Node {
public:
    Tracker();

private:
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
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Time prev_time_;
    int timestamp_for_tracker_;
    std::string tf_error_;
};
