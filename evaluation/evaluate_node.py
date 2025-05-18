import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from objects_msgs.msg import Object, ObjectArray
import os
import numpy as np
label = {
    'car': 1,
    'van': 2,
    'pedestrian': 3,
    'cyclist': 4,
    'dontcare': 0
}

class LabelPublisher(Node):
    def __init__(self):
        super().__init__('label_publisher')
        self.publisher_ = self.create_publisher(ObjectArray, '/centerpoint/objects3d', 10)
        self.label_dir = self.declare_parameter('label_dir', 'evaluation/label_kitti').get_parameter_value().string_value
        
        self.files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.txt')])
        self.frame = 0
        self.timer = self.create_timer(0.1, self.publish_labels)  

    def publish_labels(self):
        if self.frame >= len(self.files):
            return
        msg = ObjectArray()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        file_path = os.path.join(self.label_dir, self.files[self.frame])
        with open(file_path, 'r') as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) < 17:
                    continue
                obj = Object()
                obj.id = int(fields[1])
                obj.label = label.get(fields[2].lower(), 0)
                obj.score = 1.0
                obj.pose.position.x = float(fields[13])
                obj.pose.position.y = float(fields[14])
                obj.pose.position.z = float(fields[15])
                obj.size.x = float(fields[11])
                obj.size.y = float(fields[12])
                obj.size.z = float(fields[10])
                yaw = float(fields[16])
                obj.pose.orientation.x = 0.0
                obj.pose.orientation.y = 0.0
                obj.pose.orientation.z = np.sin(yaw / 2)
                obj.pose.orientation.w = np.cos(yaw / 2)
                msg.objects.append(obj)
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published frame {self.frame} with {len(msg.objects)} objects')
        self.frame += 1

def main(args=None):
    rclpy.init(args=args)
    node = LabelPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()