import numpy
from collections import namedtuple
from scipy.spatial.transform import Rotation

import rclpy
import rclpy.clock
import rclpy.logging
import tf2_ros

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from objects_msgs.msg import ObjectArray, DynamicObjectArray, DynamicObject
from geometry_msgs.msg import Pose

import diagnostic_updater
from diagnostic_msgs.msg import DiagnosticStatus


class Tracker(Node):

    def __init__(self):
        super().__init__('tracker_node')

        self.get_logger().info('Initializing Tracker')

        #QOS options
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        #Create subscriber and publisher
        self.subscriber = self.create_subscription(ObjectArray, "objects3d", self.tracker_callback, qos)
        self.publisher = self.create_publisher(DynamicObjectArray, "tracker", qos)

        #Create parameter for transformation
        self.timeout = Duration(seconds=self.declare_parameter("timeout", 0.3).value)

        self.target_frame = self.declare_parameter('target_frame', 'hdl32').value

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)

        #Create diagnostic
        self.tf_error = None
        self.prev_time = None

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
        Rt = numpy.eye(4)
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


    def transform_object(self, obj, tf):
        obj.pose = self.Pose_from_Rt(
            numpy.dot(self.Rt_from_Transform(tf), self.Rt_from_Pose(obj.pose)))


    def tracker_callback(self, objects):
        msg_time = Time.from_msg(objects.header.stamp)
        self.input_diag.tick(msg_time.nanoseconds / 1e9)

        if self.prev_time is not None and self.prev_time > msg_time:
            self.updater.broadcast(DiagnosticStatus.WARN,
                                   'Detect jump in time: reset tf buffer')
            self.tf_buffer.clear()
            self.prev_time = msg_time
            return

        self.prev_time = msg_time

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

        for obj in objects.objects:
            dynamic_object = DynamicObject()

            #Transform object
            self.transform_object(obj, tf.transform)

            #Track object
            #...

            dynamic_object.object = obj
            dynamic_objects.objects.append(dynamic_object)

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
