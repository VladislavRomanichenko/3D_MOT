#include "inference.hpp"

Tracker::Tracker() : Node("tracker_node")
{
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    RCLCPP_INFO(this->get_logger(), "Initializing Tracker");

    //Params
    this->declare_parameter("tracker_flag", false);
    this->declare_parameter("timeout", 0.01);
    this->declare_parameter("target_frame", "local_map");

    //Declare params for config
    this->declare_parameter("state_func_covariance", 1.0);
    this->declare_parameter("measure_func_covariance", 0.01);
    this->declare_parameter("prediction_score_decay", 0.02);
    this->declare_parameter("LiDAR_scanning_frequency", 10.0);
    this->declare_parameter("max_prediction_num", 12);
    this->declare_parameter("max_prediction_num_for_new_object", 10);
    this->declare_parameter("input_score", 0.0);
    this->declare_parameter("init_score", 0.0);
    this->declare_parameter("update_score", 0.0);
    this->declare_parameter("post_score", 0.55);
    this->declare_parameter("latency", 0.0);

    this->declare_parameter("num_future_states", 15);

    tracker_flag_ = this->get_parameter("tracker_flag").as_bool();
    timeout_ = rclcpp::Duration::from_seconds(this->get_parameter("timeout").as_double());
    target_frame_ = this->get_parameter("target_frame").as_string();

    Config config_;
    config_.state_func_covariance = this->get_parameter("state_func_covariance").as_double();
    config_.measure_func_covariance = this->get_parameter("measure_func_covariance").as_double();
    config_.prediction_score_decay = this->get_parameter("prediction_score_decay").as_double();
    config_.LiDAR_scanning_frequency = this->get_parameter("LiDAR_scanning_frequency").as_double();
    config_.max_prediction_num = this->get_parameter("max_prediction_num").as_int();
    config_.max_prediction_num_for_new_object = this->get_parameter("max_prediction_num_for_new_object").as_int();
    config_.input_score = this->get_parameter("input_score").as_double();
    config_.init_score = this->get_parameter("init_score").as_double();
    config_.update_score = this->get_parameter("update_score").as_double();
    config_.post_score = this->get_parameter("post_score").as_double();
    config_.latency = this->get_parameter("latency").as_double();

    num_future_states_ = this->get_parameter("num_future_states").as_int();

    timestamp_for_tracker_ = 0;
    tracker_ = Tracker3D("Centerpoint", config_);

    //Subscriber
    subscriber_ = this->create_subscription<objects_msgs::msg::ObjectArray>("objects", 10, std::bind(&Tracker::tracker_callback, this, std::placeholders::_1));

    //Publisher
    publisher_ = this->create_publisher<objects_msgs::msg::DynamicObjectArray>("tracks", 10);

    RCLCPP_INFO(this->get_logger(), "Initializing is OK");
}


Eigen::VectorXd Tracker::dynamic_msg_to_eigen_array(const objects_msgs::msg::Object& object)
{
    Eigen::VectorXd bbox(7);

    bbox << object.pose.position.x, object.pose.position.y, object.pose.position.z, object.size.x, object.size.y, object.size.z, get_object_yaw(object);

    return bbox;
}

void Tracker::eigen_array_to_dynamic_msg(objects_msgs::msg::DynamicObject& dynamic_object, const Eigen::VectorXd& array)
{
    dynamic_object.object.pose.position.x = array(0);
    dynamic_object.object.pose.position.y = array(1);
    dynamic_object.object.pose.position.z = array(2);

    dynamic_object.object.size.x = array(3);
    dynamic_object.object.size.y = array(4);
    dynamic_object.object.size.z = array(5);

    set_object_yaw(dynamic_object.object, array(6));

}


void Tracker::tracker_callback(const objects_msgs::msg::ObjectArray::SharedPtr objects)
{
    rclcpp::Time msg_time = objects->header.stamp;

    //TODO добавить диагностику по примеру
    if (prev_time_.nanoseconds() != 0 && prev_time_ > msg_time) {
        tf_buffer_->clear();
        prev_time_ = msg_time;
    return;
    }

    geometry_msgs::msg::TransformStamped tf;
    try {
        tf_error_.clear();
        tf = tf_buffer_->lookupTransform(target_frame_, objects->header.frame_id, objects->header.stamp, timeout_);
    } catch (const tf2::TransformException& ex) {
        tf_error_ = std::string("tf lookup error: ") + ex.what();
        RCLCPP_WARN(this->get_logger(), "%s", tf_error_.c_str());
        return;
    }

    objects_msgs::msg::DynamicObjectArray dynamic_objects;
    dynamic_objects.header = objects->header;
    dynamic_objects.header.frame_id = target_frame_;

    objects_msgs::msg::DynamicObject tracked_object;

    std::vector<Eigen::VectorXd> bbox_list;
    std::vector<double> score_list;

    for (auto& obj : objects->objects) {
        transform_object(obj, tf);

        if (tracker_flag_) {
            Eigen::VectorXd bbox = dynamic_msg_to_eigen_array(obj);

            tracked_object.object.score = obj.score;
            tracked_object.object.label = obj.label;

            bbox_list.push_back(bbox);
            score_list.push_back(obj.score);
        } else {
            objects_msgs::msg::DynamicObject dynamic_object;
            dynamic_object.object = obj;
            dynamic_objects.objects.push_back(dynamic_object);
        }
    }

    if (!bbox_list.empty() && tracker_flag_) {
        Eigen::MatrixXd bbox_array(bbox_list.size(), 7);

        for (size_t i = 0; i < bbox_list.size(); ++i) {
            bbox_array.row(i) = bbox_list[i];
        }

        Eigen::VectorXd score_array = Eigen::Map<Eigen::VectorXd>(score_list.data(), score_list.size());

        auto [tracked_bboxes, track_ids] = tracker_.tracking(bbox_array, &score_array, nullptr, timestamp_for_tracker_);
        timestamp_for_tracker_++;

        auto tracks = tracker_.post_processing(config_);

        std::map<int, std::map<int, std::pair<Eigen::VectorXd, double>>> frame_first_dict;

        for (const auto& [ob_id, track] : tracks) {
            for (const auto& [frame_id, ob] : track.trajectory) {
                if (ob.updated_state.size() == 0 || ob.score < config_.post_score) {
                    continue;
                }
                frame_first_dict[frame_id][ob_id] = {ob.updated_state.transpose(), ob.score};
            }
        }

        auto future_predictions = tracker_.predict_future_trajectories(num_future_states_);

        for (int i = 0; i < tracked_bboxes.rows(); ++i) {
            eigen_array_to_dynamic_msg(tracked_object, tracked_bboxes.row(i));

            tracked_object.object.id = static_cast<int>(track_ids[i]);

            if (future_predictions.count(track_ids[i])) {
                for (const auto& preds : future_predictions[track_ids[i]]) {
                    geometry_msgs::msg::PoseStamped preds_pose;
                    preds_pose.header = objects->header;
                    preds_pose.header.frame_id = target_frame_;

                    if (preds.size() >= 13) {
                        preds_pose.pose.position.x = preds(0);
                        preds_pose.pose.position.y = preds(1);
                        preds_pose.pose.position.z = preds(2);
                        set_object_yaw(preds_pose, preds(12));
                        tracked_object.prediction.push_back(preds_pose);
                    } else {
                        RCLCPP_WARN(this->get_logger(), "Traj size error");
                    }
                }
            }

            dynamic_objects.objects.push_back(tracked_object);
        }
    }

    prev_time_ = msg_time;
    publisher_->publish(dynamic_objects);
}


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Tracker>();

    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_INFO(node->get_logger(), "Node shutdown :(");
    }

    rclcpp::shutdown();
    return 0;
}