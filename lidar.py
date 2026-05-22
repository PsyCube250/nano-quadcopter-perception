import serial
import threading
import time

class LidarNode:
    def __init__(self, port='/dev/ttyUSB0'):
        try:
            self.s = serial.Serial(port, 230400, timeout=0.1)
        except Exception as e:
            print("lidar port dead", e)
            self.s = None
            
        self.dists = [0] * 360
        self.run = True
        
        t = threading.Thread(target=self.read_loop, daemon=True)
        t.start()

    def read_loop(self):
        if not self.s: return
        while self.run:
            b = self.s.read(1)
            if b == b'\x54':
                data = self.s.read(46)
                if len(data) == 46:
                    # ungarble the ldrobot protocol
                    start_angle = (data[3] | (data[4] << 8)) / 100.0
                    end_angle = (data[41] | (data[42] << 8)) / 100.0
                    
                    step = (end_angle - start_angle) / 11.0
                    if step < 0: 
                        step += 360.0 / 11.0
                    
                    for i in range(12):
                        idx = 5 + (i * 3)
                        dist = data[idx] | (data[idx+1] << 8)
                        
                        ang = int((start_angle + step * i) % 360)
                        self.dists[ang] = dist

    def close(self):
        self.run = False
        if self.s: 
            self.s.close()

if __name__ == '__main__':
    l = LidarNode('/dev/ttyUSB0')
    
    while True:
        try:
            # check distance directly in front (0 deg) and behind (180 deg)
            print("front:", l.dists[0], "mm | back:", l.dists[180], "mm")
            time.sleep(0.1)
        except KeyboardInterrupt:
            l.close()
            break