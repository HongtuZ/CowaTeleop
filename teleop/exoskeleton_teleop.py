import cv2
import time
import numpy as np 
from common.driver import DynamixelDriver
from common.config import DynamixelConfig, TeleopConfig
from common.single_hand_detector import SingleHandDetector, hand_pos
from common.msg_communicator import MsgPublisher

def teleop():
    video_cap = cv2.VideoCapture(TeleopConfig.camera_id)
    if not video_cap.isOpened():
        print(f'Can not open Camera {TeleopConfig.camera_id}!')
        video_cap.release()
        exit(1)
    dynamixel_driver = DynamixelDriver(ids=DynamixelConfig.ids, port=DynamixelConfig.port, baudrate=DynamixelConfig.baudrate)
    cmd_pub = MsgPublisher(port=22222)
    hand_detector = SingleHandDetector(hand_type='Right')
    while True:
        try:
            cmd = np.zeros(7)
            # Hand detection
            ret, frame = video_cap.read()
            fingers = hand_pos(ret, frame, hand_detector)
            griper_width = 0.1 if fingers is None else np.linalg.norm(fingers[4]-fingers[8])
            griper_cmd = (np.clip(griper_width, 0.02, 0.11)-0.02) / 0.09 * 100
            cmd[0] = griper_cmd

            # Joints reading
            joints_pos = (dynamixel_driver.get_joints() - DynamixelConfig.offsets)*DynamixelConfig.signs
            cmd[1:] = joints_pos

            # Send cmd
            msg={'action': list(cmd)}
            print(msg)
            cmd_pub.send(topic='arm_action', msg=msg)

            # Visualize
            cv2.imshow('Hand', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        except:
            dynamixel_driver.close()
            break
    dynamixel_driver.close()
    video_cap.release()

if __name__ == '__main__':
    teleop()