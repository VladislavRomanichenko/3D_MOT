from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tracker_prediction',
            executable='evaluate_node.py',
            name='evaluate_node',
            parameters=[{
                'test_mode': True,
                'label_dir': 'evaluation/label'
            }]
        ),
        Node(
            package='tracker_prediction',
            executable='tracker_node',
            name='tracker_node',
            namespace='centerpoint',
            parameters=[
                # Node parameters
                {'target_frame': 'local_map'},
                {'timeout': 0.01},
                {'tracker_flag': True},
                {'save_results_for_evaluation': True},

                # KF parameters
                {'state_func_covariance': 50.0},
                {'measure_func_covariance': 0.001},
                {'prediction_score_decay': 0.01},
                {'LiDAR_scanning_frequency': 10.0},

                # Trajectory prediction
                {'num_future_states': 10},

                # Max prediction number of state function
                {'max_prediction_num': 20},
                {'max_prediction_num_for_new_object': 8},
                {'association_threshold': 1.5},

                # Detection score threshold
                {'input_score': 0.0},
                {'init_score': 0.15},
                {'update_score': -0.3},
                {'post_score': 0.55},

                # Tracking latency (s)
                # -1: global tracking
                # 0.->500 : online -> near online tracking
                {'latency': 0.0}
            ],
            remappings=[
                ('objects', '/objects3d'),
                ('tracks', '/tracking/tracking_objects'),
            ]
        )
    ]) 