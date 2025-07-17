import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

EGO_POSE_DIR = "/home/vlad/workspace/sweeper-detection-dataset/data/mark30/dataset/mark3-2025-04-09-10-01-40/ego_pose"
LABEL_DIR = "/home/vlad/workspace/sweeper-detection-dataset/data/mark30/dataset/mark3-2025-04-09-10-01-40/label"

OUT_DIR = "./value/GT_data" #Для перевода SUSTech в нужную трансформацию, нужна лишь для промежуточного значения, после конвертации в Kitti удаляется
RESULT_DIR = "./result"  

def clear_dir(path):
    if os.path.exists(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

clear_dir('./value')
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(RESULT_DIR, "mark3-2025-04-09-10-01-40.txt") 


def convert_all_json_to_txt(json_dir, output_txt):
    files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    with open(output_txt, 'w') as out:
        for fname in files:
            frame = int(os.path.splitext(fname)[0])
            with open(os.path.join(json_dir, fname), 'r') as f:
                data = json.load(f)
            for obj in data:
                track_id = obj.get('obj_id', -1)
                obj_type = obj.get('obj_type', 'Unknown')
                truncation = 0
                occlusion = 0
                psr = obj.get('psr', {})
                pos = psr.get('position', {})
                rot = psr.get('rotation', {})
                scale = psr.get('scale', {})
                alpha = -10 
                x1 = y1 = x2 = y2 = -1
                h = scale.get('z', 0)
                w = scale.get('x', 0)
                l = scale.get('y', 0)
                x = pos.get('x', 0)
                y = pos.get('y', 0)
                z = pos.get('z', 0)
                ry = rot.get('z', 0)
                #Отнимаю единицы, так как разметчики сделали с единицы timestamp и id, а нужно было с нуля
                line = f"{int(frame) - 1} {int(track_id) - 1} {obj_type} {truncation} {occlusion} {alpha} {x1} {y1} {x2} {y2} {h:.6f} {w:.6f} {l:.6f} {x:.6f} {y:.6f} {z:.6f} {ry:.6f}\n"
                out.write(line)


def ego_pose_to_transform(ego_pose):
    x = float(ego_pose["x"])
    y = float(ego_pose["y"])
    z = float(ego_pose["z"])
    roll = float(ego_pose["roll"])
    pitch = float(ego_pose["pitch"])
    azimuth = float(ego_pose["azimuth"])

    roll_rad = np.deg2rad(roll)
    pitch_rad = np.deg2rad(pitch)
    azimuth_rad = np.deg2rad(azimuth)

    rot = R.from_euler('ZXY', [-azimuth_rad, pitch_rad, roll_rad])
    rot_matrix = rot.as_matrix()
    quat = rot.as_quat()  # [x, y, z, w]

    Rt = np.eye(4)
    Rt[:3, :3] = rot_matrix
    Rt[:3, 3] = [x, y, z]

    return Rt, quat, azimuth_rad


def process_one_pair(ego_pose_path, label_path, out_path):
    with open(ego_pose_path, 'r') as f:
        ego_pose = json.load(f)
    Rt, quat, azimuth_rad = ego_pose_to_transform(ego_pose)

    with open(label_path, 'r') as f:
        objects = json.load(f)

    transformed_objects = []
    for obj in objects:
        pos = obj['psr']['position']
        rot_obj = obj['psr']['rotation']
        scale = obj['psr']['scale']

        pos_vec = np.array([pos['x'], pos['y'], pos['z'], 1.0])
        new_pos = Rt @ pos_vec
        new_rot = dict(rot_obj)  
        new_rot['z'] = rot_obj['z'] - azimuth_rad  #вычитаем azimuth, т.к. в обратном преобразовании был минус

        new_obj = {
            "obj_id": obj["obj_id"],
            "obj_type": obj["obj_type"],
            "psr": {
                "position": {
                    "x": float(new_pos[0]),
                    "y": float(new_pos[1]),
                    "z": float(new_pos[2])
                },
                "rotation": new_rot,
                "scale": scale
            }
        }
        transformed_objects.append(new_obj)

    with open(out_path, 'w') as f:
        json.dump(transformed_objects, f, indent=2, ensure_ascii=False)


def main():
    ego_pose_files = sorted([f for f in os.listdir(EGO_POSE_DIR) if f.endswith('.json')])
    
    for ego_pose_file in ego_pose_files:
        base = os.path.splitext(ego_pose_file)[0]
        label_file = f"{base}.json"
        ego_pose_path = os.path.join(EGO_POSE_DIR, ego_pose_file)
        label_path = os.path.join(LABEL_DIR, label_file)
        out_path = os.path.join(OUT_DIR, label_file)
        if os.path.exists(label_path):
            process_one_pair(ego_pose_path, label_path, out_path)
            print(f"Processed {label_file} -> {out_path}")
        else:
            print(f"Label file not found for {ego_pose_file}, skipped.")
            
    convert_all_json_to_txt(OUT_DIR, OUT_FILE)

if __name__ == "__main__":
    main() 