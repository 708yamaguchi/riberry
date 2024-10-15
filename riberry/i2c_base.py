from filelock import FileLock, Timeout
from i2c_for_esp32 import WirePacker


class I2CBase:
    def __init__(self, i2c_addr, lock_path='/tmp/i2c-1.lock'):
        self.i2c_addr = i2c_addr
        self.bus_number = None
        self.device_type = self.identify_device()
        self.lock = FileLock(lock_path, timeout=10)
        self.setup_i2c()

    def setup_i2c(self):
        if self.device_type == 'Raspberry Pi':
            import board
            import busio
            self.i2c = busio.I2C(board.SCL, board.SDA)
            self.bus_number = 1
        elif self.device_type == 'Radxa Zero':
            import board
            import busio
            self.i2c = busio.I2C(board.SCL1, board.SDA1)
            self.bus_number = 3
        elif self.device_type == 'Khadas VIM4':
            self.i2c = i2c()
        else:
            raise ValueError('Unknown device {}'.format(
                self.device_type))

    def i2c_write(self, packet):
        try:
            self.lock.acquire()
        except Timeout as e:
            print(e)
            return
        try:
            self.i2c.writeto(self.i2c_addr, packet)
        except OSError as e:
            print(e)
        except TimeoutError as e:
            print('I2C Write error {}'.format(e))
        finally:
            try:
                self.lock.release()
            except Timeout as e:
                print(e)

    def send_string(self, sent_str):
        packer = WirePacker(buffer_size=len(sent_str) + 8)
        for s in sent_str:
            packer.write(ord(s))
        packer.end()
        if packer.available():
            self.i2c_write(packer.buffer[:packer.available()])

    def send_raw_bytes(self, raw_bytes):
        packer = WirePacker(buffer_size=len(raw_bytes) + 8)
        for r in raw_bytes:
            packer.write(r)
        packer.end()
        if packer.available():
            self.i2c_write(packer.buffer[:packer.available()])

    @staticmethod
    def identify_device():
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
            if 'Raspberry Pi' in cpuinfo:
                return 'Raspberry Pi'
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip().replace('\x00', '')
            if 'Radxa' in model\
               or 'ROCK Pi' in model\
               or model == 'Khadas VIM4':
                return model
            return 'Unknown Device'
        except FileNotFoundError:
            return 'Unknown Device'