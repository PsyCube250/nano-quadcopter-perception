import struct

import serial


FC_PORT = "/dev/ttyTHS1"
FC_BAUD = 115200

MSP_SET_RAW_RC = 200


def disarm_rc():
    return [1500, 1500, 1000, 1500, 1000, 1000, 1000, 1000]


class MSPFlightController:
    def __init__(self, port=FC_PORT, baud=FC_BAUD):
        self.ser = serial.Serial(port, baud, timeout=1)
        self.ser.reset_input_buffer()

    def send_rc(self, channels):
        data = bytearray()

        for ch in channels:
            data.extend(struct.pack("<H", int(ch)))

        size = len(data)
        checksum = size ^ MSP_SET_RAW_RC

        for b in data:
            checksum ^= b

        packet = bytearray(b"$M<")
        packet.append(size)
        packet.append(MSP_SET_RAW_RC)
        packet.extend(data)
        packet.append(checksum)

        self.ser.write(packet)

    def send_disarm(self):
        self.send_rc(disarm_rc())

    def close(self):
        self.ser.close()
