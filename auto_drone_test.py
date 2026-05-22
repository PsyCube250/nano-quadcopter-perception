import serial
import struct
import time
import threading

PORT = "/dev/ttyTHS1"
BAUD = 115200
LOOP_HZ = 50

MSP_STATUS_EX  = 150
MSP_SET_RAW_RC = 200

ARMING_FLAGS = {
    0:"NOGYRO", 1:"FAILSAFE", 2:"RXLOSS", 3:"BADVIBECOUNT",
    4:"BOXFAILSAFE", 5:"RUNAWAY_TAKEOFF", 6:"CRASH_DETECTED",
    7:"THROTTLE", 8:"ANGLE", 9:"BOOTGRACE", 10:"NOPREARM",
    11:"LOAD", 12:"CALIB", 13:"CLI", 14:"CMS_MENU", 15:"BST",
    16:"MSP", 17:"PARALYZE", 18:"GPS", 19:"RESC", 20:"RPMFILTER",
    21:"REBOOT_REQUIRED", 22:"DSHOT_TELEM", 23:"ACC_CALIB",
    24:"MOTOR_PROTOCOL", 25:"ARM_SWITCH",
}

class MSP:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.ser.flushInput()
        self._serial_lock = threading.Lock()
        self._rc_lock     = threading.Lock()
        self._rc          = [1500, 1500, 1000, 1500, 1000, 1000, 1000, 1000]
        self._paused      = False
        self._running     = True
        self._thread = threading.Thread(target=self._rc_loop, daemon=True)
        self._thread.start()

    def set_rc(self, channels):
        with self._rc_lock:
            self._rc = list(channels)

    def _rc_loop(self):
        while self._running:
            if not self._paused:
                with self._rc_lock:
                    channels = list(self._rc)
                with self._serial_lock:
                    self._write_rc(channels)
            time.sleep(1 / LOOP_HZ)

    def _write_rc(self, channels):
        data = bytearray()
        for ch in channels:
            data.extend(struct.pack('<H', int(ch)))
        size = len(data)
        checksum = size ^ MSP_SET_RAW_RC
        for b in data:
            checksum ^= b
        packet = bytearray(b'$M<')
        packet.append(size)
        packet.append(MSP_SET_RAW_RC)
        packet.extend(data)
        packet.append(checksum)
        self.ser.write(packet)

    def get_status(self):
        self._paused = True
        time.sleep(0.02)
        with self._serial_lock:
            self.ser.flushInput()
            cmd    = MSP_STATUS_EX
            packet = bytearray(b'$M<') + bytearray([0, cmd, cmd])
            self.ser.write(packet)
            flags   = None
            timeout = time.time() + 0.5
            buf     = bytearray()
            while time.time() < timeout:
                byte = self.ser.read(1)
                if not byte:
                    continue
                buf += byte
                if len(buf) >= 3 and buf[-3:] == b'$M>':
                    sz = self.ser.read(1)
                    cm = self.ser.read(1)
                    if not sz or not cm:
                        continue
                    payload = self.ser.read(sz[0])
                    self.ser.read(1)
                    if cm[0] == MSP_STATUS_EX and len(payload) >= 19:
                        flags = struct.unpack_from('<I', payload, 15)[0]
                        break
        self._paused = False
        return flags

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)

def decode_flags(flags):
    return [name for bit, name in ARMING_FLAGS.items() if flags & (1 << bit)]

def print_status(msp, label):
    flags = msp.get_status()
    print(f"\n--- {label} ---")
    if flags is not None:
        active = decode_flags(flags)
        print(f"  Raw flags : 0x{flags:08X}")
        print(f"  Blocking  : {', '.join(active) if active else 'NONE — clear!'}")
    else:
        print("  No status response received")

def neutral(): return [1500, 1500, 1000, 1500, 1000, 1000, 1000, 1000]
def armed():   return [1500, 1500, 1000, 1500, 1800, 1000, 1000, 1000]
def spin(thr): return [1500, 1500, thr,  1500, 1800, 1000, 1000, 1000]
def disarm():  return [1500, 1500, 1000, 1500, 1000, 1000, 1000, 1000]

def warmup(msp, duration=15.0):
    print(f"Warming up RC link ({duration}s) — watching flags clear:\n")
    msp.set_rc(neutral())
    start       = time.time()
    last_printed = -1
    while time.time() - start < duration:
        elapsed = int(time.time() - start)
        if elapsed % 3 == 0 and elapsed != last_printed and elapsed > 0:
            last_printed = elapsed
            flags     = msp.get_status()
            remaining = duration - (time.time() - start)
            if flags is not None:
                active = decode_flags(flags)
                s = ', '.join(active) if active else 'CLEAR — ready to arm!'
                print(f"  [{elapsed:>2}s] {s}  ({remaining:.0f}s left)")
            else:
                print(f"  [{elapsed:>2}s] No response  ({remaining:.0f}s left)")
        time.sleep(0.1)
    print("\n  Warmup done.")

def arm_sequence(msp, duration=3.0):
    print(f"\nArming ({duration}s)...")
    msp.set_rc(armed())
    time.sleep(duration)
    print("  Done.")

def spin_test(msp, throttle=1200, duration=5.0):
    print(f"\n*** PROPS OFF — Spinning at throttle {throttle} for {duration}s ***")
    msp.set_rc(spin(throttle))
    time.sleep(duration)
    print("  Done.")

def disarm_sequence(msp, duration=1.0):
    print(f"\nDisarming ({duration}s)...")
    msp.set_rc(disarm())
    time.sleep(duration)
    print("  Done.")

def main():
    print("=== Drone Motor Test ===")
    print("Steps before running:")
    print("  1. Close Betaflight Configurator completely")
    print("  2. Power cycle the FC (unplug and replug)")
    print("  3. Then press Enter\n")

    msp = MSP(PORT, BAUD)
    msp.set_rc(neutral())

    input("Press Enter once FC is powered and ready...\n")

    try:
        warmup(msp, duration=15.0)
        print_status(msp, "Status after warmup")

        arm_sequence(msp, duration=3.0)
        print_status(msp, "Status after arm attempt")

        spin_test(msp, throttle=1200, duration=5.0)

        disarm_sequence(msp, duration=1.0)
        print_status(msp, "Final status")

    finally:
        msp.set_rc(disarm())
        time.sleep(0.5)
        msp.stop()
        print("\nDone.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        try:
            msp.set_rc(disarm())
        except:
            pass