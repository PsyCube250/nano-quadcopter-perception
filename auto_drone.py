import serial
import struct
import time

# ================= CONFIG =================
PORT = "/dev/ttyTHS1"
BAUD = 115200
LOOP_HZ = 50

MSP_STATUS = 101
MSP_SET_RAW_RC = 200

# ================= ARMING FLAG DEFINITIONS =================
ARMING_FLAGS = {
    0:  "NOGYRO",
    1:  "FAILSAFE",
    2:  "RXLOSS",
    3:  "BADVIBECOUNT",
    4:  "BOXFAILSAFE",
    5:  "RUNAWAY_TAKEOFF",
    6:  "CRASH_DETECTED",
    7:  "THROTTLE",
    8:  "ANGLE",
    9:  "BOOTGRACE",
    10: "NOPREARM",
    11: "LOAD",
    12: "CALIB",
    13: "CLI",
    14: "CMS_MENU",
    15: "BST",
    16: "MSP",
    17: "PARALYZE",
    18: "GPS",
    19: "RESC",
    20: "RPMFILTER",
    21: "REBOOT_REQUIRED",
    22: "DSHOT_TELEM",
    23: "ACC_CALIB",
    24: "MOTOR_PROTOCOL",
    25: "ARM_SWITCH",
}

# ================= MSP CLASS =================
class MSP:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=1)
        self.ser.flushInput()

    def _send(self, cmd, data=bytearray()):
        size = len(data)
        checksum = size ^ cmd
        for b in data:
            checksum ^= b
        packet = bytearray(b'$M<')
        packet.append(size)
        packet.append(cmd)
        packet.extend(data)
        packet.append(checksum)
        self.ser.write(packet)

    def _read_response(self):
        """Parse a single MSP response frame from the serial buffer."""
        timeout = time.time() + 1.0
        while time.time() < timeout:
            b = self.ser.read(1)
            if b != b'$':
                continue
            if self.ser.read(1) != b'M':
                continue
            direction = self.ser.read(1)
            if direction not in (b'>', b'!'):
                continue
            size = ord(self.ser.read(1))
            cmd  = ord(self.ser.read(1))
            data = self.ser.read(size)
            _chk = self.ser.read(1)
            return cmd, data
        return None, None

    def get_status(self):
        # Flush leftover RC ack packets before requesting status
        self.ser.flushInput()
        time.sleep(0.05)
        self._send(MSP_STATUS)
        cmd, data = self._read_response()
        if data and len(data) >= 11:
            flags = struct.unpack_from('<I', data, 6)[0]
            return flags
        return None

    def send_rc(self, channels):
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

# ================= HELPERS =================
def decode_flags(flags):
    active = []
    for bit, name in ARMING_FLAGS.items():
        if flags & (1 << bit):
            active.append(name)
    return active

def print_status(msp, label):
    flags = msp.get_status()
    print(f"\n--- {label} ---")
    if flags is not None:
        active = decode_flags(flags)
        print(f"  Raw flags : 0x{flags:08X}")
        if active:
            print(f"  Blocking  : {', '.join(active)}")
        else:
            print("  Flags     : NONE — should be armed!")
    else:
        print("  No status response — FC not responding to MSP_STATUS")

def neutral_rc():
    return [1500, 1500, 1000, 1500, 1000, 1000, 1000, 1000]

def arm_rc():
    return [1500, 1500, 1000, 1500, 1800, 1000, 1000, 1000]

def spin_rc(throttle):
    return [1500, 1500, throttle, 1500, 1800, 1000, 1000, 1000]

def disarm_rc():
    return [1500, 1500, 1000, 1500, 1000, 1000, 1000, 1000]

# ================= SEQUENCES =================
def send_loop(msp, rc_fn, duration, label):
    print(f"{label} ({duration}s)...")
    start = time.time()
    count = 0
    while time.time() - start < duration:
        msp.send_rc(rc_fn())
        time.sleep(1 / LOOP_HZ)
        count += 1
    print(f"  Sent {count} packets.")

def warmup(msp, duration=15.0):
    """
    Send neutral RC long enough to:
      1. Clear RXLOSS   (~1-2s of stable packets)
      2. Outlast BOOTGRACE (Betaflight blocks arming for ~5-10s after boot)
    Prints live flag status every 3 seconds so you can watch them clear.
    """
    print(f"Warming up RC link ({duration}s)...")
    print("Watching for BOOTGRACE + RXLOSS to clear:\n")
    start = time.time()
    count = 0
    last_printed = -1

    while time.time() - start < duration:
        msp.send_rc(neutral_rc())
        time.sleep(1 / LOOP_HZ)
        count += 1

        elapsed = int(time.time() - start)
        if elapsed % 3 == 0 and elapsed != last_printed and elapsed > 0:
            last_printed = elapsed
            flags = msp.get_status()
            if flags is not None:
                active = decode_flags(flags)
                remaining = duration - (time.time() - start)
                status_str = ', '.join(active) if active else 'CLEAR — ready to arm!'
                print(f"  [{elapsed:>2}s] {status_str}  ({remaining:.0f}s left)")
            else:
                print(f"  [{elapsed:>2}s] No status response")

    print(f"\n  Warmup done. Sent {count} packets.")

def arm_sequence(msp, duration=3.0):
    send_loop(msp, arm_rc, duration, "Sending ARM signal")

def spin_test(msp, throttle=1200, duration=5.0):
    print(f"\n  *** PROPS OFF — spinning at throttle {throttle} ***")
    send_loop(msp, lambda: spin_rc(throttle), duration, "Spinning motors")

def disarm(msp, duration=1.0):
    send_loop(msp, disarm_rc, duration, "Disarming")

# ================= MAIN =================
def main():
    print("=== Drone Motor Test ===")
    print("Betaflight Configurator must be FULLY DISCONNECTED before continuing.")
    input("Press Enter to start...\n")

    msp = MSP(PORT, BAUD)

    # Step 1: Long warmup with live flag readouts every 3s
    warmup(msp, duration=15.0)
    print_status(msp, "Status after warmup")

    # Step 2: Arm
    arm_sequence(msp, duration=3.0)
    print_status(msp, "Status after arm attempt")

    # Step 3: Spin — props must be physically removed
    spin_test(msp, throttle=1200, duration=5.0)

    # Step 4: Disarm
    disarm(msp, duration=1.0)
    print_status(msp, "Final status")
    print("\nDone.")

# ================= ENTRY =================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted — disarming...")
        try:
            msp.send_rc(disarm_rc())
        except Exception:
            pass
        print("Stopped.")