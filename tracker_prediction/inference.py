import numpy
import math
from collections import namedtuple
from packaging import version
from scipy.spatial.transform import Rotation
import scipy

import rclpy
import rclpy.clock
import rclpy.logging
import rclpy.time
import tf2_ros

from rclpy.duration import Duration
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from objects_msgs.msg import ObjectArray, DynamicObjectArray, DynamicObject
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, Vector3
from sensor_msgs.msg import CameraInfo

# class ObjectTfConverter:

#     def __init__(self, node):
#         self._EPS = numpy.finfo(float).eps * 4.0

#         self.node = node
#         self.fixed_frame = 'local_map'

#         self.buffer = Buffer()

#         self.listener = TransformListener(self.buffer, self.node)
#         self.prev_transform = None


#     def quaternion_matrix(self, quaternion):
#         q = numpy.array(quaternion[:4], dtype=numpy.float64, copy=True)
#         nq = numpy.dot(q, q)

#         if nq < self._EPS:
#             return numpy.identity(4)

#         q *= math.sqrt(2.0 / nq)
#         q = numpy.outer(q, q)

#         return numpy.array((
#             (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
#             (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
#             (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
#             (                0.0,                 0.0,                 0.0, 1.0)
#             ), dtype=numpy.float64)


#     def euler_from_matrix_vec(self, matrix):
#         if matrix.ndim == 2:
#             matrix = matrix[numpy.newaxis, :, :]  # Добавляем новую ось, если матрица двумерная
#         elif matrix.ndim != 3:
#             raise ValueError("Матрица должна быть трехмерной (N, 3, 3)")
#         pitch = numpy.arctan2(matrix[:, 2, 1], numpy.sqrt(numpy.power(matrix[:, 0, 0], 2) + numpy.power(matrix[:, 1, 0], 2)))

#         deg_pos_90 = numpy.full(pitch.shape, 1.5707, dtype=numpy.float16)
#         deg_neg_90 = numpy.full(pitch.shape, 1.5707, dtype=numpy.float16)

#         pos_90_mask = deg_pos_90 == pitch
#         neg_90_mask = deg_neg_90 == pitch

#         not_90 = numpy.logical_and(~pos_90_mask, ~neg_90_mask)

#         yaw_not_90 = numpy.arctan2(matrix[:, 1, 0], matrix[:, 0, 0])
#         yaw_pos_90 = numpy.arctan2(matrix[:, 1, 2], matrix[:, 0, 2])
#         yaw_neg_90 = numpy.arctan2(-matrix[:, 1, 2], -matrix[:, 0, 2])

#         yaw = numpy.zeros(pitch.shape, dtype=numpy.float16)

#         yaw[pos_90_mask] = yaw_pos_90[pos_90_mask]
#         yaw[not_90] = yaw_not_90[not_90]
#         yaw[neg_90_mask] = yaw_neg_90[neg_90_mask]

#         return yaw


#     def euler_to_matrix(self, angles, coords):
#         assert len(angles) == len(coords)

#         num_samples = len(angles)

#         coss = numpy.cos(angles)
#         sins = numpy.sin(angles)
#         nels = numpy.zeros(num_samples, dtype=numpy.float16)
#         ones = numpy.ones(num_samples, dtype=numpy.float16)

#         Rz = numpy.stack((
#             numpy.stack((coss, sins, nels)),
#             numpy.stack((-sins, coss, nels)),
#             numpy.stack((nels, nels, ones))
#         ), axis=0)  # Стек наших массивов по оси 0

#         return Rz


#     def transform_pose(self, box3d, header):
#         transform = self.get_transform(header.frame_id, header.stamp)

#         #Если нужен будет вывод центра робота
#         #robot_center = numpy.array([[0, 0, 0, 0]], dtype=numpy.float16) # X, Y, Z, yaw

#         if transform is None:
#             self.node.get_logger().warn("No transform received! ")
#             return box3d

#         #Матирца ротаций
#         rotation = numpy.array([
#             transform.transform.rotation.x,
#             transform.transform.rotation.y,
#             transform.transform.rotation.z,
#             transform.transform.rotation.w,
#         ], dtype=numpy.float16)

#         #Матрица трансляций
#         translation = numpy.array([
#             transform.transform.translation.x,
#             transform.transform.translation.y,
#             transform.transform.translation.z,
#         ], dtype=numpy.float16)

#         M = self.quaternion_matrix(rotation)
#         P = translation

#         other_P = box3d[:, :3]
#         other_M = self.euler_to_matrix(box3d[:, -1], box3d[:, :3])

#         other_P = (M[:3, :3] @ other_P.T).T + P
#         other_M = M[:3, :3] @ other_M.astype(numpy.float32)
#         other_M = other_M.reshape(-1, 3, 3)
#         print(other_M.shape)
#         box3d[:, -1] = self.euler_from_matrix_vec(other_M).astype(numpy.float16)
#         box3d[:, :3] = other_P.astype(numpy.float16)
#         return box3d


# if version.parse(scipy.__version__) < version.parse('1.4'):
#     class RotationMod(Rotation):
#         def as_matrix(self):
#             return self.as_dcm()
#     Rotation = RotationMod


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


    # def quaternion_from_matrix(self, matrix):
    #     m = matrix
    #     t = numpy.trace(m)

    #     if t > 0:
    #         s = numpy.sqrt(t + 1.0) * 2  # s=4*qw
    #         qw = 0.25 * s
    #         qx = (m[2][1] - m[1][2]) / s
    #         qy = (m[0][2] - m[2][0]) / s
    #         qz = (m[1][0] - m[0][1]) / s
    #     else:
    #         if (m[0][0] > m[1][1]) and (m[0][0] > m[2][2]):
    #             s = numpy.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2  # s=4*qx
    #             qw = (m[2][1] - m[1][2]) / s
    #             qx = 0.25 * s
    #             qy = (m[0][1] + m[1][0]) / s
    #             qz = (m[0][2] + m[2][0]) / s
    #         elif m[1][1] > m[2][2]:
    #             s = numpy.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2  # s=4*qy
    #             qw = (m[0][2] - m[2][0]) / s
    #             qx = (m[0][1] + m[1][0]) / s
    #             qy = 0.25 * s
    #             qz = (m[1][2] + m[2][1]) / s
    #         else:
    #             s = numpy.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2  # s=4*qz
    #             qw = (m[1][0] - m[0][1]) / s
    #             qx = (m[0][2] + m[2][0]) / s
    #             qy = (m[1][2] + m[2][1]) / s
    #             qz = 0.25 * s

    #     norm = numpy.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    #     return numpy.array([qx, qy, qz, qw]) / norm



    # def Pose_from_Rt(self, Rt):
    #     pose = Pose()
    #     q = pose.orientation
    #     t = pose.position

    #     quat = self.quaternion_from_matrix(Rt[:3, :3])
    #     q.x, q.y, q.z, q.w = quat[0], quat[1], quat[2], quat[3]

    #     t.x, t.y, t.z = Rt[:3, 3]
    #     return pose


    def transform_object(self, obj, tf):
        obj.pose = self.Pose_from_Rt(
            numpy.dot(self.Rt_from_Transform(tf), self.Rt_from_Pose(obj.pose)))


    def set_object_yaw(self, obj, yaw, camera_frame=False):
        if camera_frame:
            R = numpy.array([
                [-numpy.sin(yaw), -numpy.cos(yaw), 0.0],
                [0.0, 0.0, -1.0],
                [numpy.cos(yaw), -numpy.sin(yaw), 0.0],
            ])
            q = Rotation.from_matrix(R).as_quat()
            obj.pose.orientation.x = q[0]
            obj.pose.orientation.y = q[1]
            obj.pose.orientation.z = q[2]
            obj.pose.orientation.w = q[3]
        else:
            obj.pose.orientation.x = 0
            obj.pose.orientation.y = 0
            obj.pose.orientation.z = numpy.sin(yaw / 2)
            obj.pose.orientation.w = numpy.cos(yaw / 2)


    def get_object_yaw(self, obj, camera_frame=False):
        if camera_frame:
            q = obj.pose.orientation
            R = Rotation.from_quat((q.x, q.y, q.z, q.w)).as_matrix()
            yaw = numpy.arctan2(-R[0, 0], -R[0, 1])
        else:
            yaw = numpy.arctan2(obj.pose.orientation.z, obj.pose.orientation.w)
        return yaw


    def object_points(self, obj, Rt=None):
        dx, dy, dz = obj.size.x / 2, obj.size.y / 2, obj.size.z / 2
        # additional axis with ones for matrix multiplication
        pts = numpy.array([
            [-dx, -dy, -dz, 1],  # 0: back bottom right
            [+dx, -dy, -dz, 1],  # 1: front bottom right
            [+dx, +dy, -dz, 1],  # 2: front bottom left
            [-dx, +dy, -dz, 1],  # 3: back bottom left
            [-dx, -dy, +dz, 1],  # 4: back top right
            [+dx, -dy, +dz, 1],  # 5: front top right
            [+dx, +dy, +dz, 1],  # 6: front top left
            [-dx, +dy, +dz, 1],  # 7: back top left
        ])
        if Rt is None:
            Rt = numpy.eye(4)
        res = numpy.dot(numpy.dot(Rt, self.Rt_from_Pose(obj.pose)), pts.T).T[:, :3]
        return res


    EDGES_INDICES = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6),
                    (3, 7), (4, 5), (5, 6), (6, 7), (7, 4), (1, 6), (2, 5)]


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
