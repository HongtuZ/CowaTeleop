# pasrse rosbag and create dataset
# Usage: python rosbag2dataset.py --rosbag_path /path/to/rosbag --output_path /path/to/output --dataset_name dataset_name

import argparse
import rosbag
import zarr
import numpy as np
from pathlib import Path

def parse_rosbag(rosbag_path, output_path, dataset_name):
    bag = rosbag.Bag(rosbag_path)
    topics = bag.get_type_and_topic_info().topics
    topic_names = list(topics.keys())
    topic_names.sort()
    print("Topics in the rosbag: ", topic_names)
    print("Number of messages in the rosbag: ", bag.get_message_count())

    # Create a zarr dataset for each topic
    for topic_name in topic_names:
        print("Parsing topic: ", topic_name)
        data = []
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            data.append(msg)
        data = np.array(data)
        print("Shape of the data: ", data.shape)
        zarr_path = output_path / dataset_name / topic_name
        zarr_path.mkdir(parents=True, exist_ok=True)
        zarr_file = zarr_path / 'data.zarr'
        zarr.save(str(zarr_file), data)
        print("Saved data to: ", zarr_file)

    bag.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rosbag', '-b', type=str, help='Path to the rosbag file')
    parser.add_argument('--output', '-o', type=str, help='Path to the output directory')
    parser.add_argument('--name', '-n', type=str, help='Name of the dataset')
    args = parser.parse_args()

    rosbag_path = Path(args.rosbag)
    output_path = Path(args.output)
    dataset_name = args.name

    parse_rosbag(rosbag_path, output_path, dataset_name)
