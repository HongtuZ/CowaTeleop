# CowaTeleop
Robotic arm teleoperation

## Roadmap

- [X] Zarr dataset generation
- [X] Teleop with isomorphic exoskeleton
- [ ] Teleop with meta quest3
- [ ] Training with diffuion policy
- [ ] Training with VLA

## Teleopration

### Teleop with isomorphic exoskeleton

```bash
python3 teleop/exoskeleton_teleop.py
```

### Teleop with meta quest3

comming soon!

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

### 2. Generation with Rosbag

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

### 2. Generation with Crbag (COWA crpilot env)

**a. Enter the crpilot docker container**

first download the crpilot docker image (for help from cowa), then run

```bash
cd ~/crpilot && source /opt/cowa/crpilot-x86_v2.5/setup.bash && docker.py
```

in container run

```bash
node_server &
```

**b. Generate dataset with zarr.zip format**

```bash
python3 tools/crbag2dataset.py -d xx_dir
```