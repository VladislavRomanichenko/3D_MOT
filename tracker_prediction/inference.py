import numpy
from collections import namedtuple
from scipy.spatial.transform import Rotation

import rclpy
import rclpy.clock
import rclpy.logging
import rclpy.time
import tf2_ros

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from objects_msgs.msg import ObjectArray, DynamicObjectArray, DynamicObject
from geometry_msgs.msg import Pose


class Tracker(Node):

    def __init__(self):
        super().__init__('tracker_node')

        self.get_logger().info('Initializing Tracker')

        #QOS options
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        #Declare parameter
        self.declare_parameter('subscriber_topic', '/centerpoint/objects3d')
        self.subscriber_topic = self.get_parameter('subscriber_topic').value

        self.declare_parameter('publisher_topic', '/centerpoint/dynamic_objects3d')
        self.publisher_topic = self.get_parameter('publisher_topic').value

        #Create subscriber and publisher
        self.subscriber = self.create_subscription(ObjectArray, self.subscriber_topic, self.tracker_callback, qos)
        self.publisher = self.create_publisher(DynamicObjectArray, self.publisher_topic, qos)

        #Create parameter for transformation
        self.fixed_frame = self.declare_parameter('fixed_frame', 'hdl32').value

        self.queue_size = self.declare_parameter('queue_size', 5).value


        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        #self.prev_transform = None

        self.get_logger().info('Initializing is OK')


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

        try:
            tf = self.buffer.lookup_transform(
                self.fixed_frame, objects.header.frame_id, objects.header.stamp)
        except TransformException as ex:
            self.get_logger().warn(f'tf lookup error: {ex}')
            return

        dynamic_objects = DynamicObjectArray()
        dynamic_object = DynamicObject()

        dynamic_objects.header = objects.header

        for obj in objects.objects:
            self.transform_object(obj, tf.transform)
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
