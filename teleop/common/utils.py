import time
import numpy as np
from .driver import DynamixelDriver
from .config import DynamixelConfig

# Make sure that your exskeleton joints are at the zero position 
def calibrate_zeros():
    dynamixel_driver = DynamixelDriver(ids=DynamixelConfig.ids, port=DynamixelConfig.port, baudrate=DynamixelConfig.baudrate)
    while True:
        try:
            joints_pos = dynamixel_driver.get_joints()
            offsets = [(joint+np.pi/4) // (np.pi/2) for joint in joints_pos]
            print(f'Zero offsets: {offsets} joints: {joints_pos}')
            time.sleep(0.1)
        except:
            dynamixel_driver.close()
            break
