import os,sys

# import RPi.GPIO as GPIO
import serial
# from serial import serial
import time
import struct
import numpy as np

def encode_speed(speed):
    if speed < 0:
        speed = 128 - speed
    speed = np.uint8(speed)
    return speed


def pack_lora_msg(Cam_Sign, Cam_Flag, Cam_Angle, Cam_Err):
    """
    input:
    Cam_Sign: 1 byte
    Cam_Flag: 1 byte
    Cam_Angle: 2 bytes
    Cam_Err: 2 bytes
    output:
    msg: 10 bytes，可以直接用于 serial.Serial() 的发送

    巡线任务当中AUV可能状态有巡线状态和识别状态
    Cam_Sign: int->1Byte 规定0代表巡线(正常运动)1代表识别(需要停止运行并且对应LED闪烁，LORA回传)巡线状态:
    Cam_Flag: int->1Byte 规定 1:纯绿色 2:纯蓝色 3:海洋动物-章鱼 4:海洋动物-鲨鱼 5:海洋动物-海龟
    机体坐标系右向为正
    Cam_Angle: float->2Byte 16Bit 65536取整数 单位为° 有符号数 -65536/2 对应-180°,，65536/2对应180°线性
    Cam_Err: float->2Byte 待确定 可能是导线距离图片中心的大小，单位为像素值 -240到240?
    识别状态:
    数据包传输格式:
    开头(0xFE)+ Cam Sign + Cam Flag + Cam Angle(2)+Cam Err(2)+ 累加和校验(2)+结束标志(0xFF)
    return example: b"\xFE\x01\x04\x00\x00\x00\x00\x00\x05\xFF"
    """
    a = np.uint8(Cam_Sign)
    b = np.uint8(Cam_Flag)
    Cam_Angle = Cam_Angle * 65536 / 2 / 180
    c = np.int16(Cam_Angle)
    Cam_Err = Cam_Err * 65536 / 2 / 240
    d = np.int16(Cam_Err)
    # print(f"a: {a}, b: {b}, c: {c}, d: {d}")

    cc = c.tobytes()
    dd = d.tobytes()
    # print(f"cc: {cc[0]} {cc[1]}, dd: {dd[0]} {dd[1]}")

    sum = np.uint16(a + b + cc[0] + cc[1] + dd[0] + dd[1])
    sum_bytes = sum.tobytes()
    # print(f"sum: {sum}, sum_bytes: {sum_bytes[0]} {sum_bytes[1]}")

    msg = (
        bytes(b"\xFE")
        + a.tobytes()
        + b.tobytes()
        # + c.tobytes()
        + np.uint8((c.tobytes())[1]).tobytes()
        + np.uint8((c.tobytes())[0]).tobytes()
        # + d.tobytes()
        + np.uint8((d.tobytes())[1]).tobytes()
        + np.uint8((d.tobytes())[0]).tobytes()
        # + sum_bytes
        + np.uint8(sum_bytes[1]).tobytes()
        + np.uint8(sum_bytes[0]).tobytes()
        + bytes(b"\xFF")
    )
    # print(f"msg: {msg}, len(msg): {len(msg)}")

    return msg


def test_pack_lora_msg():
    Cam_Sign = 3
    Cam_Flag = 4
    Cam_Angle = 55.5
    Cam_Err = 66.6
    msg = pack_lora_msg(Cam_Sign, Cam_Flag, Cam_Angle, Cam_Err)
    print(msg)
    for i in range(len(msg)):
        print(hex(msg[i]), end="\t")
    print()


def test_communication():
    port = "/dev/ttyAMA0"  # "/dev/ttyS0"

    # 配置串口
    com = serial.Serial(port, 9600)  # 发送提示字符
    com.stopbits = 1
    com.bytesize = 8
    com.parity = "N"

    Cam_Sign = 2
    Cam_Flag = 0
    Cam_Angle = 0
    Cam_Err = 0

    msg = pack_lora_msg(
        Cam_Sign=Cam_Sign, Cam_Flag=Cam_Flag, Cam_Angle=Cam_Angle, Cam_Err=Cam_Err
    )
    # print(f"CAM_Sign: {Cam_Sign}, CAM_Flag: {Cam_Flag}, CAM_Angle: {Cam_Angle}, CAM_Err: {Cam_Err}")
    for m in msg:
        print(hex(m), end=" ")
    print()

    # com.write(b'underwater robot')
    # com.write(b'\xFE\x01\x04\x00\x00\x00\x00\x00\x05\xFF')
    com.write(msg)

    # while True:
    #     count = com.inWaiting()
    #     if count != 0:
    #         recv = com.read(count)
    #         print(recv)  # 发回数据
    #         com.write(recv)
    #         com.flushInput()  # 清空接收缓冲区
    #         time.sleep(0.1)


if __name__ == "__main__":
    test_communication()
    # test_pack_lora_msg()
