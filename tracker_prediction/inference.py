import numpy
import math

import rclpy
import rclpy.clock
import rclpy.logging
import rclpy.time
import tf2_ros

from rclpy.duration import Duration
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from objects_msgs.msg import ObjectArray, DynamicObjectArray

#from src import ObjectTfConverter


#TODO: разобраться с cache_time в buffer, который инициализируется в конструкторе
#      разобраться с математикой в quaternion_matrix, euler_from_matrix_vec, euler_to_matrix

class ObjectTfConverter:

    def __init__(self, node):
        #_EPS = numpy.finfo(float).eps * 4.0

        self.node = node
        self.fixed_frame = 'local_map'

        self.buffer = Buffer()

        self.listener = TransformListener(self.buffer, self.node)
        self.prev_transform = None


    def get_transform(self, frame_id, stamp):
        try:
            transform = self.buffer.lookup_transform(
                target_frame=self.fixed_frame, source_frame=frame_id, time=stamp, Duration=Duration(seconds=0.001)
                )
            self.prev_transform = transform
            return transform

        #Обработка ошибок
        except(
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException
        ) as error:
            self.node.get_logger().warn("TF wait-time limit reached, using previous tf")
            self.node.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
            self.node.get_logger().warn(f"TF Error: {str(error)}, using previous tf")
            return self.prev_transform #Используем предыдущую трансформацию


    def quaternion_matrix(self, quaternion):
        q = numpy.array(quaternion[:4], dtype=numpy.float64, copy=True)
        nq = numpy.dot(q, q)

        if nq < self._EPS:
            return numpy.identity(4)

        q *= math.sqrt(2.0 / nq)
        q = numpy.outer(q, q)

        return numpy.array((
            (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
            (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
            (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
            (                0.0,                 0.0,                 0.0, 1.0)
            ), dtype=numpy.float64)


    def euler_from_matrix_vec(self, matrix):
        pitch = numpy.arctan2(matrix[:, 2, 1], numpy.sqrt(numpy.power(matrix[:, 0, 0], 2) + numpy.power(matrix[:, 1, 0], 2)))

        deg_pos_90 = numpy.full(pitch.shape, 1.5707, dtype=numpy.float16)
        deg_neg_90 = numpy.full(pitch.shape, 1.5707, dtype=numpy.float16)

        pos_90_mask = deg_pos_90 == pitch
        neg_90_mask = deg_neg_90 == pitch

        not_90 = numpy.logical_and(~pos_90_mask, ~neg_90_mask)

        yaw_not_90 = numpy.arctan2(matrix[:, 1, 0], matrix[:, 0, 0])
        yaw_pos_90 = numpy.arctan2(matrix[:, 1, 2], matrix[:, 0, 2])
        yaw_neg_90 = numpy.arctan2(-matrix[:, 1, 2], -matrix[:, 0, 2])

        yaw = numpy.zeros(pitch.shape, dtype=numpy.float16)

        yaw[pos_90_mask] = yaw_pos_90[pos_90_mask]
        yaw[not_90] = yaw_not_90[not_90]
        yaw[neg_90_mask] = yaw_neg_90[neg_90_mask]

        return yaw


    def euler_to_matrix(self, angles, coords):
        assert len(angles) == len(coords)

        num_samples = len(angles)

        coss = numpy.cos(angles)
        sins = numpy.sin(angles)
        nels = numpy.zeros(num_samples, dtype=numpy.float16)
        ones = numpy.ones(num_samples, dtype=numpy.float16)

        Rz = numpy.stack((
            numpy.stack((coss, sins, nels)),
            numpy.stack((-sins, coss, nels)),
            numpy.stack((nels, nels, ones))
        ), axis=0)  # Стек наших массивов по оси 0

        return Rz


    def transform_pose(self, box3d, header):
        transform = self.get_transform(header.frame_id, header.stamp)

        #Если нужен будет вывод центра робота
        #robot_center = numpy.array([[0, 0, 0, 0]], dtype=numpy.float16) # X, Y, Z, yaw

        if transform is None:
            self.node.get_logger().warn("No transform received! ")
            return box3d

        #Матирца ротаций
        rotation = numpy.array([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ], dtype=numpy.float16)

        #Матрица трансляций
        translation = numpy.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ], dtype=numpy.float16)

        M = self.quaternion_matrix(rotation)
        P = translation

        other_P = box3d[:, :3]
        other_M = self.euler_to_matrix(box3d[:, -1], box3d[:, :3])

        other_P = (M[:3, :3] @ other_P.T).T + P
        other_M = M[:3, :3] @ other_M.astype(numpy.float32)

        box3d[:, -1] = self.euler_from_matrix_vec(other_M).astype(numpy.float16)
        box3d[:, :3] = other_P.astype(numpy.float16)

        return box3d


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

        self.object_tranformer = ObjectTfConverter(self)
        self.get_logger().info('Initializing is OK')


    def tracker_callback(self, objects):
        self.box3d = []
        self.scores = []
        self.label_preds = []

        for obj in objects.objects:
            orientation = obj.pose.orientation

            x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
            self.yaw = numpy.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            self.theta = -self.yaw - numpy.pi / 2

            self.box3d.append([obj.pose.position.x,
                                obj.pose.position.y,
                                obj.pose.position.z,
                                obj.size.y,
                                obj.size.x,
                                obj.size.z,
                                self.theta])

        self.box3d = numpy.array(self.box3d, dtype=numpy.float16)

        if(len(self.box3d) > 0):
            self.box3d[:, -1] = -self.box3d[:, -1] - numpy.pi / 2
            self.box3d = self.object_tranformer.transform_pose(self.box3d, self.header)


def main(args=None):
    rclpy.init(args=args)
    node = Tracker()

    # executor = rclpy.executors.MultiThreadedExecutor()
    # executor.add_node(node)

    # try:
    #     executor.spin()
    # except KeyboardInterrupt:
    #     pass

    # while rclpy.ok():
    #     rclpy.spin_once(node)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Остановка узла')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
