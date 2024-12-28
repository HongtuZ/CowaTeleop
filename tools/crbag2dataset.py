#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pip install imagecodecs
# pip install zarr==2.17.0

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
    can_record_img = False
    for info in record:
        if info.channel_name == '/camera/surround/front/image_raw':
            img = np.array(info.data()[0].image, dtype=np.uint8)
            if can_record_img:
                cv2.putText(img, 'Recording', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.imshow('img', img)
            key = cv2.waitKey(0)
            if key == ord('q'):
                exit(1)
            elif key == ord('r'):
                can_record_img = True
            elif key == ord('s'):
                break
            if not can_record_img:
                continue
            img = cv2.resize(img, (224, 224))
            img_episode.append(img)
            can_record_joints = True
            print('record img', len(img_episode))
        if info.channel_name == '/RL/base_info/arm':
            if can_record_joints:
                joints_pos = np.array(info.data().joint_state.pos)
                joints_vel = np.array(info.data().joint_state.speed)
                joints_pos_episode.append(joints_pos)
                joints_vel_episode.append(joints_vel)
                can_record_joints = False
                print('record joints', len(joints_pos_episode))
    episode_len = min(len(img_episode), len(joints_pos_episode))
    print('save img, joints len:', len(img_episode), len(joints_pos_episode))
    if episode_len == 0:
        return None
    episode_data = {
        'camera_img': np.stack(img_episode[:episode_len]),
        'joints_pos': np.stack(joints_pos_episode[:episode_len]),
        'joints_vel': np.stack(joints_vel_episode[:episode_len]),
    }
    episode_dataset = ReplayBuffer.create_empty_zarr(storage=zarr.ZipStore(str(Path(bag_path).with_suffix('.zarr.zip')), mode='w'))
    img_crompressor = JpegXl(level=99, numthreads=8)
    chunks = {'camera_img': episode_data['camera_img'][:1].shape,}
    compressors = {'camera_img': img_crompressor,}
    episode_dataset.add_episode(data=episode_data, chunks=chunks, compressors=compressors)

def create_dataset_from_dir(args):
    # Create a zarr dataset
    bag_dir = Path(args.bag_dir)
    output_path = Path(
        args.output_path) if args.output_path else bag_dir.parent
    dataset_name = args.dataset_name if args.dataset_name else bag_dir.stem
    zarr_path = (output_path / dataset_name).with_suffix('.zarr')
    if zarr_path.exists():
        print(f'Already exists: {str(zarr_path)}')
        exit(1)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Create dataset at: {str(zarr_path)}')
    dataset = ReplayBuffer.create_empty_zarr(storage=zarr.DirectoryStore(str(zarr_path)))
    for zarr_path in bag_dir.rglob('*.zarr.zip'):
        print(f'Processing {str(zarr_path)}')
        single_dataset = ReplayBuffer.copy_from_path(str(zarr_path))
        print(f'Add to dataset ...')
        dataset.add_episode(single_dataset.data)
    print(dataset.root.info)


def main(args):
    bag_dir = Path(args.bag_dir)
    # Parse all the bag in bag dir
    for bag_path in bag_dir.rglob('*.record.000*'):
        target_zarr_path = Path(bag_path).with_suffix('.zarr.zip')
        if target_zarr_path.exists():
            print(f'Exists: {str(target_zarr_path)}')
            continue
        parse_crbag(str(bag_path))


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
    create_dataset_from_dir(args)
