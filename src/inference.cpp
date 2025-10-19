#include "inference.hpp"
#include <fstream>
#include <iomanip>
#include <filesystem>

Tracker::Tracker() : Node("tracker_node"), diag_updater(this) {
    diag_updater.setHardwareID("none");
    diag_updater.setPeriod(1.0);

    //ROS2 transformation
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    RCLCPP_INFO(this->get_logger(), "Initializing Tracker");

    //Params
    tracker_flag_ = this->declare_parameter("tracker_flag", false);
    timeout_ = rclcpp::Duration::from_seconds(this->declare_parameter("timeout", 0.01));
    target_frame_ = this->declare_parameter("target_frame", "local_map");
    frame_odom_ = this->declare_parameter("frame_odom", "odom");

    //Params for evaluation mode
    evaluation_mode_ = this->declare_parameter("evaluation_mode", false);
    frame_id_param_ = this->declare_parameter("frame_id_param", "0021");
    prev_frame_id_ = "";
    current_timestamp_ = 0;

    if(evaluation_mode_){
        std::string result_file_path = "results/tracker_results/data/" + frame_id_param_ + ".txt";
        std::ofstream clear_file(result_file_path, std::ios::trunc);
        clear_file.close();
    }

    //Declare params for tracker config
    Config config_;

    config_.state_func_covariance = this->declare_parameter("state_func_covariance", 1.0);
    config_.measure_func_covariance = this->declare_parameter("measure_func_covariance", 0.01);
    config_.prediction_score_decay = this->declare_parameter("prediction_score_decay", 0.02);
    config_.LiDAR_scanning_frequency = this->declare_parameter("LiDAR_scanning_frequency", 10.0);
    config_.max_prediction_num = this->declare_parameter("max_prediction_num", 12);
    config_.max_prediction_num_for_new_object = this->declare_parameter("max_prediction_num_for_new_object", 10);
    config_.association_threshold = this->declare_parameter("association_threshold", 2.0);
    config_.input_score = this->declare_parameter("input_score", 0.0);
    config_.init_score = this->declare_parameter("init_score", 0.0);
    config_.update_score = this->declare_parameter("update_score", 0.0);
    config_.post_score = this->declare_parameter("post_score", 0.55);
    config_.latency = this->declare_parameter("latency", 0.0);

    //Declare params for prediction states
    num_future_states_ = this->declare_parameter("num_future_states", 15);

    //Minimum trajectory history for prediction
    min_trajectory_history_ = this->declare_parameter("min_trajectory_history", 5);

    tracking_in_odom_ = this->declare_parameter("tracking_in_odom", true);

    timestamp_for_tracker_ = 0;
    tracker_ = Tracker3D("Centerpoint", config_);

    //Diagnostic params
    double input_freq = declare_parameter("input_main_freq", 10.0);
    double output_freq = declare_parameter("output_freq", 10.0);
    double diag_min_time = declare_parameter("diag_min_time", 0.0);
    double diag_max_time = declare_parameter("diag_max_time", 0.3);

    //Diagnostic init
    diag_updater.add("Status", this, &Tracker::update_diagnostic_status);

    diag_input_min_freq = input_freq;
    diag_input_max_freq = input_freq;
    diag_input = std::make_unique<diagnostic_updater::TopicDiagnostic>(
        "Input", diag_updater,
        diagnostic_updater::FrequencyStatusParam(&diag_input_min_freq, &diag_input_max_freq, 0.1, 10),
        diagnostic_updater::TimeStampStatusParam(diag_min_time, diag_max_time), this->get_clock());

    diag_output_min_freq = output_freq;
    diag_output_max_freq = output_freq;
    diag_output = std::make_unique<diagnostic_updater::TopicDiagnostic>(
        "Output", diag_updater,
        diagnostic_updater::FrequencyStatusParam(&diag_output_min_freq, &diag_output_max_freq, 0.1, 10),
        diagnostic_updater::TimeStampStatusParam(diag_min_time, diag_max_time), this->get_clock());

    //Subscriber
    subscriber_ = this->create_subscription<objects_msgs::msg::ObjectArray>("objects", 10, std::bind(&Tracker::tracker_callback, this, std::placeholders::_1));

    //Publisher
    publisher_ = this->create_publisher<objects_msgs::msg::DynamicObjectArray>("tracks", 10);

    RCLCPP_INFO(this->get_logger(), "Initializing is OK");
}


void Tracker::update_diagnostic_status(diagnostic_updater::DiagnosticStatusWrapper& stat) {
    stat.summary(diag_lvl, diag_msg);
    diag_lvl = diagnostic_msgs::msg::DiagnosticStatus::OK;
    diag_msg = "OK";
}


Eigen::VectorXd Tracker::dynamic_msg_to_eigen_array(const objects_msgs::msg::Object& object)
{
    Eigen::VectorXd bbox(7);

    bbox << object.pose.position.x, object.pose.position.y, object.pose.position.z, object.size.x, object.size.y, object.size.z, get_object_yaw(object);

    return bbox;
}


objects_msgs::msg::DynamicObject Tracker::eigen_array_to_dynamic_msg(const Eigen::VectorXd& array)
{
    objects_msgs::msg::DynamicObject dynamic_object;

    dynamic_object.object.pose.position.x = array(0);
    dynamic_object.object.pose.position.y = array(1);
    dynamic_object.object.pose.position.z = array(2);

    dynamic_object.object.size.x = array(3);
    dynamic_object.object.size.y = array(4);
    dynamic_object.object.size.z = array(5);

    set_object_yaw(dynamic_object.object, array(6));
    return dynamic_object;
}


void Tracker::save_result(const std::string& filename, int frame, int track_id, const std::string& type,
                                double truncation, double occlusion, double alpha,
                                double h, double w, double l,
                                double x, double y, double z, double ry, double score) {
    std::ofstream out(filename, std::ios::app);
    out << frame << " " << track_id << " " << type << " "
        << truncation << " " << occlusion << " " << alpha << " "
        << -1 << " " << -1 << " " << -1 << " " << -1 << " " //2D box (x1 y1 x2 y2), заполняем как -1 для отключения подсчёта метрик для 2D
        << std::fixed << std::setprecision(6)
        << h << " " << w << " " << l << " "
        << x << " " << y << " " << z << " "
        << ry << " " << score << std::endl;
    out.close();
}


//Centerpoint labels
std::string label_to_type(int label) {
    switch (label) {
        case -1: return "DontCare";
        case 0: return "Car";
        case 1: return "Pedestrian";
        case 2: return "Cyclist";
        default: return "Unknown";
    }
}


void Tracker::tracker_callback(const objects_msgs::msg::ObjectArray::SharedPtr objects)
{
    rclcpp::Time msg_time = objects->header.stamp;

    diag_input->tick(objects->header.stamp);

    if (prev_time_.nanoseconds() != 0 && prev_time_ > msg_time) {
        diag_updater.broadcast(diagnostic_msgs::msg::DiagnosticStatus::WARN, "Detect jump in time: reset tf buffer");
        tf_buffer_->clear();
        prev_time_ = msg_time;
        return;
    }

    //lidar_frame → odom (для трекинга)
    geometry_msgs::msg::TransformStamped tf_odom;

    //odom → local_map (для публикации)
    geometry_msgs::msg::TransformStamped tf_odom_to_target;

    //lidar_frame → target_frame (для прямого трекинга)
    geometry_msgs::msg::TransformStamped tf;

    try {
        tf = tf_buffer_->lookupTransform(target_frame_, objects->header.frame_id, objects->header.stamp, timeout_);
        if(tracking_in_odom_){
            tf_odom = tf_buffer_->lookupTransform(frame_odom_, objects->header.frame_id, objects->header.stamp, timeout_);
            tf_odom_to_target = tf_buffer_->lookupTransform(target_frame_, frame_odom_, objects->header.stamp, timeout_);
        }
    } catch (const tf2::TransformException& ex) {
        RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "tf lookup error: %s", ex.what());
        diag_lvl = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
        diag_msg = std::string("Can't get robot coordinates");
        diag_updater.force_update();
        return;
    }

    objects_msgs::msg::DynamicObjectArray dynamic_objects;
    dynamic_objects.header = objects->header;
    //Трекинг и предикшен будут производиться в одометрии, но в итоге будут трансформироваться обратно в target_frame
    dynamic_objects.header.frame_id = target_frame_;

    std::vector<Eigen::VectorXd> bbox_list;
    std::vector<double> score_arr;
    std::vector<int> label_arr;

    for (auto& obj : objects->objects) {
        if(tracking_in_odom_){
            transform_object(obj, tf_odom);
        } else {
            transform_object(obj, tf);
        }
        if (tracker_flag_) {
            Eigen::VectorXd bbox = dynamic_msg_to_eigen_array(obj);
            bbox_list.push_back(bbox);
            score_arr.push_back(obj.score);
            label_arr.push_back(obj.label);
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

        Eigen::VectorXd score_array = Eigen::Map<Eigen::VectorXd>(score_arr.data(), score_arr.size());

        auto [tracked_bboxes, track_ids] = tracker_.tracking(bbox_array, &score_array, nullptr, timestamp_for_tracker_);
        timestamp_for_tracker_++;

        /* TODO: Фильтрация пропущенных детекций
        auto previous_trajectories = tracker_.post_processing(config_);

        std::map<int, std::map<int, std::pair<Eigen::VectorXd, double>>> frame_first_dict;

        for (const auto& [ob_id, track] : previous_trajectories) {
            for (const auto& [frame_id, ob] : track.trajectory) {
                if (ob.updated_state.size() == 0 || ob.score < config_.post_score) {
                    continue;
                }
                frame_first_dict[frame_id][ob_id] = {ob.updated_state.transpose(), ob.score};
            }
        }
        */

        auto future_predictions = tracker_.predict_future_trajectories(num_future_states_);

        for (int i = 0; i < tracked_bboxes.rows(); ++i) {
            objects_msgs::msg::DynamicObject tracked_object = eigen_array_to_dynamic_msg(tracked_bboxes.row(i));

            tracked_object.object.label = label_arr[i];
            tracked_object.object.score = score_arr[i];

            tracked_object.object.id = static_cast<int>(track_ids[i]);
            rclcpp::Time base_time = objects->header.stamp;
            double dt = 0.1;
            int pred_idx = 1;

            bool is_static = false;
            auto it_traj = tracker_.active_trajectories_.find(track_ids[i]);
            if (it_traj != tracker_.active_trajectories_.end()) {
                is_static = is_static_trajectory(it_traj->second, 0.15, 0.1, 0.1);

                if (it_traj->second.size() > min_trajectory_history_ && !is_static && future_predictions.count(track_ids[i])) {
                    for (const auto& preds : future_predictions[track_ids[i]]) {
                        geometry_msgs::msg::PoseStamped preds_pose;
                        preds_pose.header = objects->header;
                        preds_pose.header.frame_id = tracking_in_odom_ ? frame_odom_ : target_frame_;
                        preds_pose.header.stamp = base_time + rclcpp::Duration::from_seconds(dt * pred_idx);

                        if (preds.size() >= 10) {
                            preds_pose.pose.position.x = preds(0);
                            preds_pose.pose.position.y = preds(1);
                            preds_pose.pose.position.z = tracked_object.object.pose.position.z; //Так как фильтр калмана реагирует и на эти изменения
                            set_object_yaw(preds_pose, preds(9));

                            //Трансформируем предикшен в target_frame только если трекинг был в odom
                            if (tracking_in_odom_) {
                                geometry_msgs::msg::PoseStamped preds_pose_transformed;
                                try {
                                    tf2::doTransform(preds_pose, preds_pose_transformed, tf_odom_to_target);
                                    //Сохраняем header, так как doTransform его убирает
                                    preds_pose_transformed.header = preds_pose.header;
                                    preds_pose_transformed.header.frame_id = target_frame_;
                                    tracked_object.prediction.push_back(preds_pose_transformed);
                                } catch (const tf2::TransformException& ex) {
                                    RCLCPP_WARN(this->get_logger(), "Failed to transform prediction to target frame: %s", ex.what());
                                }
                            } else {
                                tracked_object.prediction.push_back(preds_pose);
                            }
                        } else {
                            RCLCPP_WARN(this->get_logger(), "Traj size error: expected 10, got %ld", preds.size());
                        }
                        pred_idx++;
                    }
                }
            }

            //Трансформируем основной объект в target_frame перед публикацией
            if (tracking_in_odom_) {
                transform_object(tracked_object.object, tf_odom_to_target);
            }

            dynamic_objects.objects.push_back(tracked_object);
        }
    }

    prev_time_ = msg_time;
    diag_output->tick(objects->header.stamp);
    publisher_->publish(dynamic_objects);

    //Часть кода сохраняющая результаты, требуемая для оценки трекера
    if (evaluation_mode_) {
        std::string frame_id = frame_id_param_;

        std::string filename = frame_id;
        if (filename.length() < 4) {
            filename = std::string(4 - filename.length(), '0') + filename;
        }

        std::string save_path = "results/tracker_results/data/" + filename + ".txt";

        if (frame_id != prev_frame_id_) {
            current_timestamp_ = 0;
            prev_frame_id_ = frame_id;
        }

        int frame = current_timestamp_;

        for (const auto& tracked_object : dynamic_objects.objects) {
            int track_id = tracked_object.object.id;
            std::string type = label_to_type(tracked_object.object.label);
            double truncation = -1;  //нужно только для 2D
            double occlusion = -1;   //нужно только для 2D
            double alpha = -10;      //нужно только для 2D
            double h = tracked_object.object.size.z;
            double w = tracked_object.object.size.x;
            double l = tracked_object.object.size.y;
            double x = tracked_object.object.pose.position.x;
            double y = tracked_object.object.pose.position.y;
            double z = tracked_object.object.pose.position.z;
            double yaw = get_object_yaw(tracked_object.object);
            double score = tracked_object.object.score;
            save_result(save_path, frame, track_id, type, truncation, occlusion, alpha, h, w, l, x, y, z, yaw, score);
        }

        current_timestamp_++;
    }
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