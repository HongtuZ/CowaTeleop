import numpy as np

class TeleopConfig:
    hz = 200
    camera_id = 2

class DynamixelConfig:
    ids = list(range(1, 7))
    port = '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94VYOX-if00-port0'
    baudrate = 57600
    # this is calibrated
    offsets = np.array([2.0, 3.0, 2.0, 2.0, 1.0, 1.0])*(np.pi/2)
    signs = np.array([1, -1, 1, -1, -1, -1])
