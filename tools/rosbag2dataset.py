# pasrse rosbag and create dataset
# Usage: python rosbag2dataset.py --rosbag_path /path/to/rosbag --output_path /path/to/output --dataset_name dataset_name

import argparse
import numpy as np
import rosbag2_py
import zarr

from pathlib import Path
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from zmq_info.msg import RobotArm
# from realsense2_camera_msgs.msg import Metadata, IMUInfo
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from common.replay_buffer import ReplayBuffer
from common.imagecodecs_numcodecs import register_codecs, JpegXl

import cv2

register_codecs()

def get_rosbag_options(path, storage_id, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(
        uri=path, storage_id=storage_id)

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options

def parse_rosbag(rosbag_path):
    # Open rosbag
    bag_reader = rosbag2_py.SequentialReader()
    storage_options, converter_options = get_rosbag_options(str(rosbag_path), 'sqlite3')
    bag_reader.open(storage_options, converter_options)

    # Parse rosbag
    joints_pos_list, joints_vel_list, fisheye1_img_list, fisheye2_img_list = [], [], [], []
    joints_pos, joints_vel, fisheye1_img, fisheye2_img = None, None, None, None

    fisheye1_stamp, fisheye2_stamp = None, None

    updated = False
    while bag_reader.has_next():
        topic, msg, t = bag_reader.read_next()
        if topic == '/arm_info':
            data = deserialize_message(msg, RobotArm)
            joints_pos, joints_vel = np.array(data.pos), np.array(data.velocity)
        elif topic == '/camera/fisheye1/image_raw':
            data = deserialize_message(msg, Image)
            fisheye1_stamp = data.header.stamp
            fisheye1_img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width)
            updated = True
        elif topic == '/camera/fisheye2/image_raw':
            data = deserialize_message(msg, Image)
            fisheye2_stamp = data.header.stamp
            fisheye2_img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width)
            updated = True
        elif topic == '/camera/odom/sample':
            data = deserialize_message(msg, Odometry)
            # print(data.pose.pose.position)
            # print(data.pose.pose.orientation)
        if updated and joints_pos is not None and joints_vel is not None and fisheye1_stamp is not None and fisheye1_stamp == fisheye2_stamp:
            updated = False
            joints_pos_list.append(joints_pos)
            joints_vel_list.append(joints_vel)
            fisheye1_img_list.append(fisheye1_img)
            fisheye2_img_list.append(fisheye2_img)
            # cv2.imshow('fisheye', np.concat([fisheye1_img, fisheye2_img], axis=1))
            # cv2.waitKey(1)
    return {
        'joints_pos': np.stack(joints_pos_list),
        'joints_vel': np.stack(joints_vel_list),
        'fisheye1_img': np.stack(fisheye1_img_list),
        'fisheye2_img': np.stack(fisheye2_img_list),
    }

def main(args):
    # Create a zarr dataset
    rosbag_dir = Path(args.rosbag_dir)
    output_path = Path(args.output_path) if args.output_path else rosbag_dir.parent
    dataset_name = args.dataset_name if args.dataset_name else rosbag_dir.stem
    zarr_path = (output_path / dataset_name).with_suffix('.zarr.zip')
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Create dataset at: {str(zarr_path)}')
    img_crompressor = JpegXl(level=99, numthreads=8)
    dataset = ReplayBuffer.create_empty_zarr(storage=zarr.ZipStore(str(zarr_path), mode='w'))

    # Parse all the rosbag in rosbag dir
    for rosbag_path in rosbag_dir.rglob('*.db3'):
        print(f'Parsing {str(rosbag_path)}')
        episode_data = parse_rosbag(str(rosbag_path))
        print(f'Add to dataset ...')
        chunks={
            'fisheye1_img': episode_data['fisheye1_img'][:1].shape,
            'fisheye2_img': episode_data['fisheye2_img'][:1].shape
            }
        compressors={
            'fisheye1_img': img_crompressor,
            'fisheye2_img': img_crompressor
            }
        dataset.add_episode(data=episode_data, chunks=chunks, compressors=compressors)
    print(dataset.root.info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rosbag_dir', '-d', type=str, help='Path to the rosbag dir')
    parser.add_argument('--output_path', '-o', type=str, help='Path to the output directory', default=None)
    parser.add_argument('--dataset_name', '-n', type=str, help='Name of the dataset', default=None)
    args = parser.parse_args()

    main(args)
