import serial

ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=1)

packet = bytearray(b'$M<')
packet += bytes([0, 1, 1])  # size=0, cmd=1, checksum=1

ser.write(packet)

resp = ser.read(64)
print("Response:", resp)

ser.close()