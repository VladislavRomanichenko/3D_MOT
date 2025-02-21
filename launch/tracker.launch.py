import launch
from launch import LaunchDescription
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='tracker_prediction',
            executable='tracker_node',
            namespace='centerpoint',
            parameters=[
                {'target_frame': 'local_map'},
                {'config': '/home/vlad/Desktop/waymo_tracker/tracker_prediction/config/online/centerpoint_mot.yaml'},
                {'timeout': 0.01},
            ],
            remappings=[
                ('objects', '/centerpoint/objects3d'),
                ('tracks', '/tracking/tracking_objects'),
            ],
        ),
    ])
