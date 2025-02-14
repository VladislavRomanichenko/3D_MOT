from tracker.tracker import Tracker3D
import time

import os
from tracker.config import cfg, cfg_from_yaml_file
from tracker.box_op import *
import numpy as np
import argparse


def track_one_seq(seq_id,config):

    """
    tracking one sequence
    Args:
        seq_id: int, the sequence id
        config: config
    Returns: dataset: KittiTrackingDataset
             tracker: Tracker3D
             all_time: float, all tracking time
             frame_num: int, num frames
    """

    tracker = Tracker3D(box_type="Kitti", tracking_features=False, config = config)

    dataset = KittiTrackingDataset(dataset_path, seq_id=seq_id, ob_path=detections_path,type=[tracking_type])


    for i in range(len(dataset)):
        objects, det_scores, pose = dataset[i]

        mask = det_scores > config.input_score
        objects = objects[mask]
        det_scores = det_scores[mask]

        tracker.tracking(objects[:,:7],
                             features=None,
                             scores=det_scores,
                             pose=pose,
                             timestamp=i)

    return tracker



