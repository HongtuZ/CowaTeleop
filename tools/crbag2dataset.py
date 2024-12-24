#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# zarr version 2.17.0

import cv2
import sys
import numpy as np
import zarr
import argparse
import pycrmw
from common.replay_buffer import ReplayBuffer
from common.imagecodecs_numcodecs import register_codecs, JpegXl
from pathlib import Path

register_codecs()

pycrmw.Init(sys.argv)


def parse_crbag(bag_path):
    print(f'Processing {bag_path}')
    record = pycrmw.Record()
    record.OpenRead([bag_path])
    # Create an episode
    img_episode = []
    joints_pos_episode = []
    joints_vel_episode = []
    # Camera img timestep as base timestep
    can_record_joints = False
    for info in record:
        if info.channel_name == '/camera/surround/front/image_raw':
            img = np.array(info.data()[0].image, dtype=np.uint8)
            img = cv2.resize(img, (224, 224))
            img_episode.append(img)
            can_record_joints = True
        if info.channel_name == '/RL/base_info/arm':
            if can_record_joints:
                joints_pos = np.array(info.data().joint_state.pos)
                joints_vel = np.array(info.data().joint_state.speed)
                joints_pos_episode.append(joints_pos)
                joints_vel_episode.append(joints_vel)
                can_record_joints = False
    return {
        'camera_img': np.stack(img_episode),
        'joints_pos': np.stack(joints_pos_episode),
        'joints_vel': np.stack(joints_vel_episode),
    }


def main(args):
    # Create a zarr dataset
    bag_dir = Path(args.bag_dir)
    output_path = Path(
        args.output_path) if args.output_path else bag_dir.parent
    dataset_name = args.dataset_name if args.dataset_name else bag_dir.stem
    zarr_path = (output_path / dataset_name).with_suffix('.zarr.zip')
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Create dataset at: {str(zarr_path)}')
    img_crompressor = JpegXl(level=99, numthreads=8)
    dataset = ReplayBuffer.create_empty_zarr(
        storage=zarr.ZipStore(str(zarr_path), mode='w'))

    # Parse all the bag in bag dir
    for bag_path in bag_dir.rglob('*.record*'):
        print(f'Parsing {str(bag_path)}')
        episode_data = parse_crbag(str(bag_path))
        print(f'Add to dataset ...')
        chunks = {
            'camera_img': episode_data['camera_img'][:1].shape,
        }
        compressors = {
            'camera_img': img_crompressor,
        }
        dataset.add_episode(data=episode_data, chunks=chunks,
                            compressors=compressors)
    print(dataset.root.info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_dir', '-d', type=str,
                        help='Path to the bag dir')
    parser.add_argument('--output_path', '-o', type=str,
                        help='Path to the output directory', default=None)
    parser.add_argument('--dataset_name', '-n', type=str,
                        help='Name of the dataset', default=None)
    args = parser.parse_args()

    main(args)
