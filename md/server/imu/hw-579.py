import smbus
import time
import math

# =========================
# I2C
# =========================

bus = smbus.SMBus(4)

ADXL = 0x53
GYRO = 0x68
MAG = 0x1E

# =========================
# ADXL345 초기화
# =========================

bus.write_byte_data(ADXL, 0x2D, 0x08)

# =========================
# ITG3200 초기화
# =========================

bus.write_byte_data(GYRO, 0x3E, 0x00)
time.sleep(0.1)

# DLPF + Full Scale
bus.write_byte_data(GYRO, 0x16, 0x18)

# =========================
# HMC5883L 초기화
# =========================

bus.write_byte_data(MAG, 0x00, 0x70)
bus.write_byte_data(MAG, 0x01, 0x20)
bus.write_byte_data(MAG, 0x02, 0x00)

time.sleep(0.1)

print("MAG MODE =", hex(bus.read_byte_data(MAG, 0x02)))

# =========================
# 공통 함수
# =========================

def signed16(v):
    if v > 32767:
        v -= 65536
    return v


def read16_le(addr, reg):
    low = bus.read_byte_data(addr, reg)
    high = bus.read_byte_data(addr, reg + 1)

    value = (high << 8) | low
    return signed16(value)


def read16_be(addr, reg):
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg + 1)

    value = (high << 8) | low
    return signed16(value)

# =========================
# ADXL345
# =========================

def read_accel():

    ax = read16_le(ADXL, 0x32)
    ay = read16_le(ADXL, 0x34)
    az = read16_le(ADXL, 0x36)

    ax *= 0.0039
    ay *= 0.0039
    az *= 0.0039

    return ax, ay, az

# =========================
# ITG3200
# =========================

def read_gyro_raw():

    gx = read16_be(GYRO, 0x1D)
    gy = read16_be(GYRO, 0x1F)
    gz = read16_be(GYRO, 0x21)

    return gx, gy, gz


def read_gyro():

    gx, gy, gz = read_gyro_raw()

    gx = (gx - gyro_offset_x) / 14.375
    gy = (gy - gyro_offset_y) / 14.375
    gz = (gz - gyro_offset_z) / 14.375

    return gx, gy, gz

# =========================
# HMC5883L
# =========================

def read_mag():

    mx = read16_be(MAG, 0x03)
    mz = read16_be(MAG, 0x05)
    my = read16_be(MAG, 0x07)

    return mx, my, mz

# =========================
# Roll / Pitch
# =========================

def calc_roll_pitch(ax, ay, az):

    roll = math.degrees(
        math.atan2(ay, az)
    )

    pitch = math.degrees(
        math.atan2(
            -ax,
            math.sqrt(ay * ay + az * az)
        )
    )

    return roll, pitch

# =========================
# Heading
# =========================

def calc_heading(mx, my):

    heading = math.degrees(
        math.atan2(my, mx)
    )

    if heading < 0:
        heading += 360

    return heading

# =========================
# 자이로 오프셋 보정
# =========================

print("Gyro calibration...")
print("센서를 움직이지 마세요.")

sum_x = 0
sum_y = 0
sum_z = 0

samples = 300

for _ in range(samples):

    gx, gy, gz = read_gyro_raw()

    sum_x += gx
    sum_y += gy
    sum_z += gz

    time.sleep(0.01)

gyro_offset_x = sum_x / samples
gyro_offset_y = sum_y / samples
gyro_offset_z = sum_z / samples

print("Gyro Offset")
print(gyro_offset_x)
print(gyro_offset_y)
print(gyro_offset_z)

# =========================
# 메인 루프
# =========================

while True:

    try:

        ax, ay, az = read_accel()

        gx, gy, gz = read_gyro()

        mx, my, mz = read_mag()

        roll, pitch = calc_roll_pitch(
            ax, ay, az
        )

        heading = calc_heading(
            mx,
            my
        )

        print("\n-----------------------")

        print(
            f"ACC[g] "
            f"X={ax:6.3f} "
            f"Y={ay:6.3f} "
            f"Z={az:6.3f}"
        )

        print(
            f"GYRO[d/s] "
            f"X={gx:7.2f} "
            f"Y={gy:7.2f} "
            f"Z={gz:7.2f}"
        )

        print(
            f"MAG "
            f"X={mx:6d} "
            f"Y={my:6d} "
            f"Z={mz:6d}"
        )

        print(
            f"ROLL ={roll:7.2f}°"
        )

        print(
            f"PITCH={pitch:7.2f}°"
        )

        print(
            f"YAW  ={heading:7.2f}°"
        )

        time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n종료")
        break

    except Exception as e:
        print("에러:", e)
        time.sleep(1)