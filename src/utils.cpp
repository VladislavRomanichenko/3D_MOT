#include <utils.hpp>

Eigen::Matrix4d Rt_from_tq(const geometry_msgs::msg::Point &t, const geometry_msgs::msg::Quaternion &q) 
{
    Eigen::Matrix4d Rt = Eigen::Matrix4d::Zero();
    Eigen::Quaterniond quat(q.w, q.x, q.y, q.z);
    Eigen::Matrix3d R = quat.toRotationMatrix();

    Rt.block<3, 3>(0, 0) = R;
    Rt.block<3, 1>(0, 3) << t.x, t.y, t.z;
    Rt(3, 3) = 1.0;
    
    return Rt;
}


Matrix4 Rt_from_Pose(const geometry_msgs::msg::Pose &pose) 
{
    return Rt_from_tq(pose.position, pose.orientation);
}


Eigen::Matrix4d Rt_from_Transform(const geometry_msgs::msg::TransformStamped &tf_stamped) 
{
    const auto& tf = tf_stamped.transform;
    geometry_msgs::msg::Point t;
    t.x = tf.translation.x;
    t.y = tf.translation.y;
    t.z = tf.translation.z;

    geometry_msgs::msg::Quaternion q;
    q.x = tf.rotation.x;
    q.y = tf.rotation.y;
    q.z = tf.rotation.z;
    q.w = tf.rotation.w;

    return Rt_from_tq(t, q);
}


geometry_msgs::msg::Pose Pose_from_Rt(const Matrix4 &Rt) 
{
    geometry_msgs::msg::Pose pose;
    Quaterniond quat(Rt.block<3, 3>(0, 0));
    pose.orientation.x = quat.x();
    pose.orientation.y = quat.y();
    pose.orientation.z = quat.z();
    pose.orientation.w = quat.w();
    pose.position.x = Rt(0, 3);
    pose.position.y = Rt(1, 3);
    pose.position.z = Rt(2, 3);
    return pose;
}


void transform_object(objects_msgs::msg::Object &object, const geometry_msgs::msg::TransformStamped &tf) 
{
    Matrix4 Rt_tf = Rt_from_Transform(tf);
    Matrix4 Rt_pose = Rt_from_Pose(object.pose);
    
    object.pose = Pose_from_Rt(Rt_tf * Rt_pose);
}


void set_object_yaw(objects_msgs::msg::Object &object, double yaw) 
{
    Quaterniond quat;
    quat = Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));

    object.pose.orientation.x = quat.x();
    object.pose.orientation.y = quat.y();
    object.pose.orientation.z = quat.z();
    object.pose.orientation.w = quat.w();
}

void set_object_yaw(geometry_msgs::msg::PoseStamped &object, double yaw) 
{
    Quaterniond quat;
    quat = Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));

    object.pose.orientation.x = quat.x();
    object.pose.orientation.y = quat.y();
    object.pose.orientation.z = quat.z();
    object.pose.orientation.w = quat.w();
}


double get_object_yaw(const objects_msgs::msg::Object &object) 
{
    double x = object.pose.orientation.x;
    double y = object.pose.orientation.y;
    double z = object.pose.orientation.z;
    double w = object.pose.orientation.w;
    double yaw = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
    return yaw;
}