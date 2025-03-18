#pragma once

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <tf2/transform_datatypes.h>

#include <cmath>
//----------------CUSTOM MESSAGE----------------------------
#include "objects_msgs/msg/object.hpp"
#include "objects_msgs/msg/object_array.hpp"


typedef Eigen::Matrix4d Matrix4;
typedef Eigen::Quaterniond Quaterniond;
typedef Eigen::Vector3d Vector3d;


/**
 * @brief Constructs a 4x4 transformation matrix from translation and quaternion.
 * 
 * @param t The translation component as geometry_msgs::msg::Point.
 * @param q The rotation component as geometry_msgs::msg::Quaternion.
 * @return Eigen::Matrix4d The resulting 4x4 transformation matrix.
 */
Eigen::Matrix4d Rt_from_tq(const geometry_msgs::msg::Point &t, const geometry_msgs::msg::Quaternion &q);


/**
 * @brief Constructs a 4x4 transformation matrix from the pose.
 * 
 * @param pose The input pose as geometry_msgs::msg::Pose.
 * @return Matrix4 The resulting 4x4 transformation matrix.
 */
Matrix4 Rt_from_Pose(const geometry_msgs::msg::Pose &pose);


/**
 * @brief Constructs a 4x4 transformation matrix from a TransformStamped message.
 * 
 * @param tf_stamped The TransformStamped message containing transform information.
 * @return Eigen::Matrix4d The resulting 4x4 transformation matrix.
 */
Eigen::Matrix4d Rt_from_Transform(const geometry_msgs::msg::TransformStamped &tf_stamped);


/**
 * @brief Constructs a Pose from a 4x4 transformation matrix.
 * 
 * @param Rt The input 4x4 transformation matrix.
 * @return geometry_msgs::msg::Pose The resulting pose.
 */
geometry_msgs::msg::Pose Pose_from_Rt(const Matrix4 &Rt);


/**
 * @brief Transforms an object by applying a given transformation.
 * 
 * @param object The object to be transformed as objects_msgs::msg::Object.
 * @param tf The transformation to be applied as geometry_msgs::msg::TransformStamped.
 */
void transform_object(objects_msgs::msg::Object object, const geometry_msgs::msg::TransformStamped &tf);


/**
 * @brief Sets the yaw of an object.
 * 
 * @param object The object whose yaw is to be set as objects_msgs::msg::Object or geometry_msgs::msg::PoseStamped.
 * @param yaw The desired yaw angle.
 * @param camera_frame A flag indicating if the frame is that of a camera.
 */
void set_object_yaw(objects_msgs::msg::Object object, double yaw);
void set_object_yaw(geometry_msgs::msg::PoseStamped object, double yaw);


/**
 * @brief Gets the yaw of an object.
 * 
 * @param object The object whose yaw is to be computed as objects_msgs::msg::Object.
 * @param camera_frame A flag indicating if the frame is that of a camera.
 * @return double The computed yaw angle.
 */
double get_object_yaw(objects_msgs::msg::Object object);

