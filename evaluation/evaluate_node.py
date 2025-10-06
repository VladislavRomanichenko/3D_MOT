#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from objects_msgs.msg import Object, ObjectArray
import os
import numpy as np
import csv
from typing import Dict, List, Optional
import subprocess

class LabelPublisher(Node):
    def __init__(self):
        super().__init__('label_publisher')
        
        #Declare parameters
        self.label_dir = self.declare_parameter('label_dir', 'evaluation/label').get_parameter_value().string_value
        self.default_score = self.declare_parameter('default_score', 1.0).get_parameter_value().double_value
        self.publish_rate = self.declare_parameter('publish_rate', 0.1).get_parameter_value().double_value
        self.test_mode = self.declare_parameter('test_mode', False).get_parameter_value().bool_value
        
        #Publisher
        self.publisher_ = self.create_publisher(ObjectArray, 'objects3d', 10)
        
        #TODO СДЕЛАТЬ УЧЁТ ОТ КАКОГО ДО КАКОГО ФРЕЙМА МЫ СЧИТЫВАЕМ ДАННЫЕ
        #Evaluate_tracking.seqmap.val
        self.sequences = self._load_sequences()
        if not self.sequences:
            self.get_logger().error('No sequences loaded from seqmap file')
            return
            
        self.files = self._load_files()
        if not self.files:
            self.get_logger().error(f'No matching files found in {self.label_dir} for sequences: {self.sequences}')
            return
        
        unique_labels = set()
        for file in self.files:
            file_path = os.path.join(self.label_dir, file)
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        fields = line.strip().split()
                        if len(fields) >= 3:
                            unique_labels.add(fields[2].lower())
            except Exception as e:
                self.get_logger().error(f'Error reading file {file_path}: {e}')
        
        self.get_logger().info(f'Found unique labels in dataset: {sorted(unique_labels)}')
        unknown_or_dontcare_labels = unique_labels - set(self.label.keys())
        if unknown_or_dontcare_labels:
            self.get_logger().warning(f'Unknown labels in dataset: {sorted(unknown_or_dontcare_labels)}')
            
        #Tracking variables
        self.frame = 0
        self.current_file_objects = {}  
        self.current_timestamp = None
        self.stats = {
            'total_objects': 0,
            'skipped_objects': 0,
            'unknown_or_dontcare_labels': 0,
            'invalid_lines': 0,
            'per_label': {label_name: 0 for label_name in self.label.keys()}
        }
        
        
        self._load_next_file()
        
        #timer
        self.timer = self.create_timer(self.publish_rate, self.publish_labels)
        self.get_logger().info(f'Found {len(self.files)} files for sequences: {self.sequences}')
        
        if self.test_mode:
            self.get_logger().info('Running in test mode - will print statistics after all files are processed')

    def _load_sequences(self) -> List[str]:
        sequences = []
        seqmap_path = os.path.join('evaluation', 'evaluate_tracking.seqmap.val')
        try:
            with open(seqmap_path, 'r') as f:
                for line in f:
                    fields = line.strip().split()
                    if len(fields) >= 4:
                        sequence_id = "%04d" % int(fields[0])
                        sequences.append(sequence_id)
            self.get_logger().info(f'Loaded {len(sequences)} sequences from seqmap')
        except Exception as e:
            self.get_logger().error(f'Error loading seqmap file: {e}')
        return sequences

    def _load_files(self) -> List[str]:
        try:
            all_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.txt')])
            files = [f for f in all_files if os.path.splitext(f)[0] in self.sequences]
            return files
        except Exception as e:
            self.get_logger().error(f'Error loading files: {e}')
            return []

    def _create_object(self, fields: List[str]) -> Optional[Object]:
        try:
            #ROS2 message
            obj = Object()
            obj.id = int(fields[1])
            
            #Label
            label_name = fields[2].lower()
                
            obj.label = self.label.get(label_name, 0)
            if obj.label == -1:
                self.stats['unknown_or_dontcare_labels'] += 1
            else:
                self.stats['per_label'][label_name] += 1
            
            obj.score = self.default_score
            
            #Pose
            obj.pose.position.x = float(fields[13])
            obj.pose.position.y = float(fields[14])
            obj.pose.position.z = float(fields[15])
            
            #Size
            obj.size.x = float(fields[11])  # l
            obj.size.y = float(fields[12])  # w
            obj.size.z = float(fields[10])  # h
            
            #Orientation
            yaw = float(fields[16])
            obj.pose.orientation.x = 0.0
            obj.pose.orientation.y = 0.0
            obj.pose.orientation.z = np.sin(yaw / 2)
            obj.pose.orientation.w = np.cos(yaw / 2)
            
            return obj
        except (ValueError, IndexError) as e:
            self.stats['invalid_lines'] += 1
            self.get_logger().error(f'Error processing line: {fields}, error: {e}')
            return None

    def _print_stats(self):
        self.get_logger().info(
            f'Final Statistics:\n'
            f'  Total objects: {self.stats["total_objects"]}\n'
            f'  Skipped objects: {self.stats["skipped_objects"]}\n'
            f'  Unknown or DontCare labels: {self.stats["unknown_or_dontcare_labels"]}\n'
            f'  Invalid lines: {self.stats["invalid_lines"]}\n'
            f'  Per label statistics:'
        )
        for label_name, count in self.stats['per_label'].items():
            if count > 0:
                self.get_logger().info(f'    {label_name}: {count}')

    def _load_next_file(self):
        if self.frame >= len(self.files):
            return False
            
        file_path = os.path.join(self.label_dir, self.files[self.frame])
        self.get_logger().info(f'Loading file: {file_path}')
        
        try:
            self.current_file_objects.clear()
            with open(file_path, 'r') as f:
                for line in f:
                    fields = line.strip().split()
                    if len(fields) < 17:
                        self.stats['skipped_objects'] += 1
                        self.get_logger().warning(f'Skipping invalid line: {line}')
                        continue
                    
                    timestamp = int(fields[0])
                    obj = self._create_object(fields)
                    if obj is not None:
                        if timestamp not in self.current_file_objects:
                            self.current_file_objects[timestamp] = []
                        self.current_file_objects[timestamp].append(obj)
                        self.stats['total_objects'] += 1
            
            self.current_timestamp = min(self.current_file_objects.keys()) if self.current_file_objects else None
            return True
            
        except Exception as e:
            self.get_logger().error(f'Error loading file {file_path}: {e}')
            return False

    def publish_labels(self):
        if not self.current_file_objects:
            if not self._load_next_file():
                if self.test_mode:
                    self._print_stats()
                self.timer.destroy()
                self.destroy_node()
                rclpy.shutdown()
                return
        
        if self.current_timestamp is None:
            return
            
        msg = ObjectArray()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'sequence_' + os.path.splitext(self.files[self.frame])[0]
        msg.objects = self.current_file_objects[self.current_timestamp]
        
        self.publisher_.publish(msg)
        
        self.current_file_objects.pop(self.current_timestamp)
        if self.current_file_objects:
            self.current_timestamp = min(self.current_file_objects.keys())
        else:
            self.frame += 1
            self._load_next_file()

    label = {
        'car': 1,
        'van': 2,
        'pedestrian': 3,
        'cyclist': 4,
        'truck': 5,
        'bus': 6,
        'trailer': 7,
        'construction_vehicle': 8,
        'traffic_cone': 9,
        'barrier': 10,
        'motorcycle': 11,
        'bicycle': 12,
        'scooter': 12,  # Treat scooter as bicycle
        'misc': 13,
        'person': 14,  
        'tram': 15,    
        'dontcare': -1
    }

def main(args=None):
    rclpy.init(args=args)
    node = LabelPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()