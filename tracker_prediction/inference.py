import numpy as np
from collections import namedtuple
from scipy.spatial.transform import Rotation

from .tracker.tracker import Tracker3D
from .tracker.config import cfg, cfg_from_yaml_file
from .tracker.box_op import *

import rclpy
import rclpy.clock
import rclpy.logging

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from objects_msgs.msg import ObjectArray, DynamicObjectArray, DynamicObject
from geometry_msgs.msg import Pose

import diagnostic_updater
from diagnostic_msgs.msg import DiagnosticStatus


class Tracker(Node):

    def __init__(self):
        super().__init__('tracker_node')

        yaml_file = self.declare_parameter('config', 'centerpoint_mot.yaml').value
        self.config = cfg_from_yaml_file(yaml_file, cfg)

        self.get_logger().info('Initializing Tracker')
        self.tracker = Tracker3D(box_type="Centerpoint", tracking_features=False, config = self.config)

        #Create subscriber and publisher
        self.subscriber = self.create_subscription(ObjectArray, "objects", self.tracker_callback, 10)
        self.publisher = self.create_publisher(DynamicObjectArray, "tracks", 10)

        #Create parameter for transformation
        self.timeout = Duration(seconds=self.declare_parameter("timeout", 0.3).value)

        self.target_frame = self.declare_parameter('target_frame', 'hdl32').value

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        #Create diagnostic
        self.tf_error = None
        self.prev_time = None

        #TODO: переделать timestamp под непоследовательные числа, а под значения из ROS хедера
        self.timestamp_for_tracker = 0

        self.updater = diagnostic_updater.Updater(self)
        self.updater.setHardwareID('none')
        self.freq_bounds = {'min': 5.0, 'max': 30.0}

        self.input_diag = diagnostic_updater.TopicDiagnostic(
            'input',
            self.updater,
            diagnostic_updater.FrequencyStatusParam(self.freq_bounds, 0.1, 10),
            diagnostic_updater.TimeStampStatusParam(min_acceptable=0.0,
                                                    max_acceptable=0.5),
        )

        self.output_diag = diagnostic_updater.TopicDiagnostic(
            'output',
            self.updater,
            diagnostic_updater.FrequencyStatusParam(self.freq_bounds, 0.1, 10),
            diagnostic_updater.TimeStampStatusParam(min_acceptable=0.0,
                                                    max_acceptable=0.5),
        )

        self.updater.add('Self check', self.self_check)


        self.get_logger().info('Initializing is OK')


    def self_check(self, stat):
        if self.tf_error:
            stat.summary(DiagnosticStatus.ERROR, self.tf_error)
        else:
            stat.summary(DiagnosticStatus.OK, 'TF is OK')
        return stat


    def Rt_from_tq(self, t, q):
        Rt = np.eye(4)
        Rt[:3, :3] = Rotation.from_quat((q.x, q.y, q.z, q.w)).as_matrix()
        Rt[:3, 3] = t.x, t.y, t.z
        return Rt


    def Rt_from_Pose(self, pose):
        return self.Rt_from_tq(pose.position, pose.orientation)


    def Rt_from_Transform(self, tf):
        return self.Rt_from_tq(tf.translation, tf.rotation)


    def Pose_from_Rt(self, Rt):
        pose = Pose()
        q = pose.orientation
        t = pose.position
        q.x, q.y, q.z, q.w = Rotation.from_matrix(Rt[:3, :3]).as_quat()
        t.x, t.y, t.z = Rt[:3, 3]
        return pose


    def set_object_yaw(self, obj, yaw):
            obj.pose.orientation.x = 0.0
            obj.pose.orientation.y = 0.0
            obj.pose.orientation.z = np.sin(yaw / 2)
            obj.pose.orientation.w = np.cos(yaw / 2)


    def get_object_yaw(self, obj):
        x = obj.pose.orientation.x
        y = obj.pose.orientation.y
        z = obj.pose.orientation.z
        w = obj.pose.orientation.w

        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    def transform_object(self, obj, tf):
        obj.pose = self.Pose_from_Rt(
            np.dot(self.Rt_from_Transform(tf), self.Rt_from_Pose(obj.pose)))


    def dynamic_msg_to_np_array(self, object):
        """
        Convert an DynamicObject from message to numpy array format (x, y, z, l, w, h, yaw) that corresponds to Centerpoint, Waymo, OpenPCDet
        """
        x, y, z = object.pose.position.x, object.pose.position.y, object.pose.position.z
        l, w, h = object.size.x, object.size.y, object.size.z
        yaw = self.get_object_yaw(object)

        return np.array([x, y, z, l, w, h, yaw])


    def np_array_to_dynamic_msg(self, array):
        """
        Convert numpy array (x, y, z, l, w, h, yaw) -> DynamicObject
        """
        dynamic_object = DynamicObject()

        dynamic_object.object.pose.position.x, dynamic_object.object.pose.position.y, dynamic_object.object.pose.position.z = array[:3]
        dynamic_object.object.size.x, dynamic_object.object.size.y, dynamic_object.object.size.z = array[3:6]

        yaw = array[6]

        self.set_object_yaw(dynamic_object.object, yaw)

        return dynamic_object


    def track_one_seq(self, tracker, objects, config):
        """
        tracking one sequence

        Args:
            config: config

        Returns: dataset:
            tracker: Tracker3D
        """

        #TODO: перенести всё лишнее из callback в одну функцию


    def tracker_callback(self, objects):
        msg_time = Time.from_msg(objects.header.stamp)

        self.input_diag.tick(msg_time.nanoseconds / 1e9)

        if self.prev_time is not None and self.prev_time > msg_time:
            self.updater.broadcast(DiagnosticStatus.WARN,
                                   'Detect jump in time: reset tf buffer')
            self.buffer.clear()
            self.prev_time = msg_time
            return

        try:
            self.tf_error = None
            tf = self.buffer.lookup_transform(
                self.target_frame, objects.header.frame_id,
                objects.header.stamp, self.timeout)
        except TransformException as ex:
            self.tf_error = f'tf lookup error: {ex}'
            #self.get_logger().warn(f'tf lookup error: {ex}')
            self.updater.force_update()
            return

        dynamic_objects = DynamicObjectArray()
        dynamic_objects.header = objects.header
        dynamic_objects.header.frame_id = self.target_frame

        #Objects list for tracker
        bbox_list = []
        score_list = []

        for obj in objects.objects:

            #Transform object
            self.transform_object(obj, tf.transform)

            #Convert msg to numpy array
            bbox = self.dynamic_msg_to_np_array(obj)

            bbox_list.append(bbox)
            score_list.append(obj.score)

        if len(bbox_list) > 0:
            bbox_array = np.array(bbox_list)
            score_array = np.array(score_list)

            #Track
            tracked_bboxes, track_ids = self.tracker.tracking(
                bbs_3D=bbox_array,
                features=None,
                scores=score_array,
                timestamp=self.timestamp_for_tracker #Здесь идут просто последовательные числа(TODO: надо будет переписать класс Trajectory)
                #timestamp=msg_time.nanoseconds // 1e6
            )
            self.timestamp_for_tracker += 1

            tracks = self.tracker.post_processing(self.config)

            frame_first_dict = {}
            for ob_id in tracks.keys():
                track = tracks[ob_id]

                for frame_id in track.trajectory.keys():
                    ob = track.trajectory[frame_id]

                    if ob.updated_state is None:
                        continue

                    if ob.score < self.config.post_score:
                        continue

                    if frame_id in frame_first_dict.keys():
                        frame_first_dict[frame_id][ob_id] = (np.array(ob.updated_state.T), ob.score)
                        #self.get_logger().info(f'{frame_first_dict[frame_id][ob_id]}')
                    else:
                        frame_first_dict[frame_id] = {ob_id:(np.array(ob.updated_state.T), ob.score)}
                        #self.get_logger().info(f'{frame_first_dict[frame_id]}')

#----------------------------------TODO------------------------------------------
            future_predictions = self.tracker.predict_future_trajectories(steps=5)

            for track_id, predictions in future_predictions.items():
                self.get_logger().info(f"ID: {track_id}")
                for state, timestamp in predictions:
                    self.get_logger().info(f"Predicted state at timestamp {timestamp}: {state[:3]}") #State до 3, это x, y, z, далее идут скорости, размеры и т.д.
#--------------------------------------------------------------------------------

            #Convert numpy array to msg
            for i in range(len(tracked_bboxes)):
                tracked_object = self.np_array_to_dynamic_msg(tracked_bboxes[i])
                tracked_object.object.id = int(track_ids[i])
                dynamic_objects.objects.append(tracked_object)

        self.prev_time = msg_time
        self.output_diag.tick(msg_time.nanoseconds / 1e9)
        self.publisher.publish(dynamic_objects)


def main(args=None):
    rclpy.init(args=args)
    node = Tracker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node shutdown')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
