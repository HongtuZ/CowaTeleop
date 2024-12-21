# CowaTeleop
Robotic arm teleoperation

## Dataset generatioon

### 1. Rosbag conversion (Optional)

If the rosbag recorded with ros1, you must convert it to ros2 version with the following conversion tool:

**a. Install conversion tool**

```bash
pip install rosbags
```

**b. Conversion**

 ```bash
 rosbags-convert --src xx.bag --dst xx_dir --src-typestore ros1_noetic --dst-typestore ros2_iron
```

### 2. Generation

**a. Create RobotArm ros msg**

```bash
cd tools/msg_ws
colcon build --symllink-install
source install/setup.bash
```

**b. Generate dataset with zarr.zip format**

```bash
python3 tools/rosbag2dataset.py -d xx_dir
```